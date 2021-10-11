"""
train_cgn_ours.py
Script to train contact graspnet on our data.
"""

## CGN imports
import typing
import contact_graspnet.pointnet2.models
import contact_graspnet.pointnet2.utils
from contact_graspnet.contact_graspnet import config_utils
from contact_graspnet.contact_graspnet.data import PointCloudReader, load_scene_contacts, center_pc_convert_cam
from contact_graspnet.contact_graspnet.summaries import build_summary_ops, build_file_writers
from contact_graspnet.contact_graspnet.tf_train_ops import load_labels_and_losses, build_train_op, get_bn_decay
from contact_graspnet.contact_graspnet.contact_grasp_estimator import GraspEstimator
import contact_graspnet.contact_graspnet.contact_graspnet as cgn_module

## Tensorflow imports
import tensorflow.compat.v1 as tf
tf.disable_eager_execution()
TF2 = True
physical_devices = tf.config.experimental.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(physical_devices[0], True)

## tsgrasp imports
from tsgrasp.data.lit_acronymvid import LitAcronymvidDataset
import torch

def get_cgn_config():
    cfg = config_utils.load_config(
        checkpoint_dir="/home/tim/Research/tsgrasp/ckts/cgn_ckpts"
    )
    return cfg

def train_dataloader(batch_size: int) -> LitAcronymvidDataset:
    from hydra import compose, initialize
    from torch.utils.data import DataLoader

    with initialize(config_path="../conf/data"):
        cfg = compose(config_name="acronymvid")

    lds = LitAcronymvidDataset(cfg.data.data_cfg, batch_size=batch_size)
    lds.prepare_data()
    lds.setup()

    # create DataLoader with special collate function, not minkowski_collate,
    # so we don't have to import ME
    def collate_fn(list_data):
        from tsgrasp.data.acronymvid import collate_control_points

            ## Each batch may have different numbers of ground truth grasps, resulting in ragged tensors. We require even, rectangular tensors for calculating the ADD-S loss, so we collate them into rectangular tensors.
        pos_control_points, sym_pos_control_points, gt_grasps_per_batch = \
            collate_control_points(
            batch = torch.arange(len(list_data)),
            time = torch.stack([d["coordinates"][:,0] for d in list_data]),
            pos_cp_list = [d["pos_control_points"] for d in list_data],
            sym_pos_cp_list = [d["sym_pos_control_points"] for d in list_data]
        )

        return {
            "positions": torch.stack([d["positions"] for d in list_data]),
            "labels": torch.stack([d["labels"] for d in list_data]),
            "pos_control_points": pos_control_points,
            "sym_pos_control_points": sym_pos_control_points,
            "gt_grasps_per_batch": gt_grasps_per_batch,
            "cam_frame_pos_grasp_tfs": [d["cam_frame_pos_grasp_tfs"] for d in list_data]
        }

    dl = DataLoader(
        lds.dataset_train,
        batch_size=batch_size,
        shuffle=True, 
        collate_fn=collate_fn
    )
    return dl

class CGNWrapper:

    def __init__(self, global_config, batch_size=3, num_input_points=20000, num_target_points=2048):
        self.global_config = global_config
        self.input_pl = self.placeholder_inputs(
            batch_size=batch_size,
            num_input_points=num_input_points,
            input_normals=False
        )
        self.step = tf.Variable(0)
        self.end_points = cgn_module.get_model(
            point_cloud = self.input_pl['pointclouds_pl'],
            is_training = self.input_pl['is_training_pl'],
            global_config = self.global_config,
            bn_decay = get_bn_decay(self.step, global_config['OPTIMIZER'])
        )
        self.loss_pl = self.placeholder_data(
            b = batch_size,
            N = num_target_points
        )
        self.losses = self.get_losses()
        self.train_op = build_train_op(self.losses['loss'], self.step, global_config)

    def get_losses(self):
        (
            dir_loss, 
            bin_ce_loss, 
            offset_loss, 
            approach_loss, 
            adds_loss, 
            adds_loss_gt2pred 
        )= cgn_module.get_losses(
            self.end_points['pred_points'],
            self.end_points,
            self.loss_pl['dir_labels_pc_cam'],
            self.loss_pl['offset_labels_pc'],
            self.loss_pl['grasp_success_labels_pc'],
            self.loss_pl['approach_labels_pc_cam'],
            self.global_config
        )

        total_loss = 0
        if self.global_config['MODEL']['pred_contact_base']:
            total_loss += self.global_config['OPTIMIZER']['dir_cosine_loss_weight'] * dir_loss
        if self.global_config['MODEL']['pred_contact_success']:
            total_loss += self.global_config['OPTIMIZER']['score_ce_loss_weight'] * bin_ce_loss
        if self.global_config['MODEL']['pred_contact_offset']:
            total_loss += self.global_config['OPTIMIZER']['offset_loss_weight'] * offset_loss
        if self.global_config['MODEL']['pred_contact_approach']:
            total_loss += self.global_config['OPTIMIZER']['approach_cosine_loss_weight'] * approach_loss
        if self.global_config['MODEL']['pred_grasps_adds']:
            total_loss += self.global_config['OPTIMIZER']['adds_loss_weight'] * adds_loss
        if self.global_config['MODEL']['pred_grasps_adds_gt2pred']:
            total_loss += self.global_config['OPTIMIZER']['adds_gt2pred_loss_weight'] * adds_loss_gt2pred

        return {
            'loss': total_loss,
            'dir_loss': dir_loss,
            'bin_ce_loss': bin_ce_loss,
            'offset_loss': offset_loss,
            'approach_loss': approach_loss,
            'adds_loss': adds_loss,
            'adds_gt2pred_loss': adds_loss_gt2pred
        }

    @staticmethod
    def placeholder_inputs(batch_size, num_input_points, input_normals):
        """
        Creates placeholders for input pointclouds and training/eval mode 

        Arguments:
            batch_size {int} -- batch size
            num_input_points {int} -- number of input points to the network (default: 20000)

        Keyword Arguments:
            input_normals {bool} -- whether to use normals as input (default: {False})

        Returns:
            dict[str:tf.placeholder] -- dict of placeholders
        """
        pl_dict = {}
        dim = 6 if input_normals else 3
        pl_dict['pointclouds_pl'] = tf.placeholder(
            tf.float32, shape=(batch_size, num_input_points, dim)
        )
        pl_dict['is_training_pl'] = tf.placeholder(tf.bool, shape=())

        return pl_dict
    
    @staticmethod
    def placeholder_data(b, N):
        """ Create placeholders for the information needed to compute loss,
        such as the camera-frame contact point positions."""

        return {
            'grasp_success_labels_pc': tf.placeholder(
                tf.float32, shape=(b, N), name='grasp_success_labels_pc'
            ),
            'offset_labels_pc': tf.placeholder(
                tf.float32, shape=(b, N, 10), name='offset_labels_pc'
            ),
            'approach_labels_pc_cam': tf.placeholder(
                tf.float32, shape=(b, N, 3), name='approach_labels_pc_cam'
            ),
            'dir_labels_pc_cam':  tf.placeholder(
            tf.float32, shape=(b, N, 3), name='dir_labels_pc_cam'
            )
        }

def train(global_config):

    dl = train_dataloader(batch_size = 3)
    with tf.Graph().as_default(): # create a new graph with this scope

        cgn = CGNWrapper(global_config)

        # Add ops to save and restore all the variables.
        saver = tf.train.Saver(save_relative_paths=True, keep_checkpoint_every_n_hours=4)

        # Create a session
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        config.allow_soft_placement = True
        # config = tf.ConfigProto(
        #     device_count = {'GPU': 0}
        # )
        sess = tf.Session(config=config)
        # from tensorflow.python import debug as tf_debug
        # sess = tf_debug.LocalCLIDebugWrapperSession(sess)

        # Log summaries
        # This logs everything, but I don't want to right now.
        # TODO: create our own "build_summary_ops" with just what we need.
        # summary_ops = build_summary_ops(ops, sess, global_config)

        # Init/Load weights
        # TODO: add weight loading
        sess.run(tf.global_variables_initializer())

    ## Run training op

    for i, data in enumerate(dl):
        print(f"**** Iteration {i} ****")
        # Inference on a batch
        data = next(iter(dl))

        import torch.nn.functional as f
        feed_dict = {
            cgn.input_pl['pointclouds_pl']: data['positions'][:, 0, range(20000)], # TODO: use proper idxs
            cgn.input_pl['is_training_pl']: True,
            # variables that we turned into placeholders
            cgn.loss_pl['grasp_success_labels_pc']: data['labels'].reshape(3, 10, -1)[:, 0, range(2048)], # TODO: use proper 2048 labeled points
            cgn.loss_pl['approach_labels_pc_cam']: f.normalize(torch.rand(3, 2048, 3), dim=2, p=2), # TODO: unrandom
            cgn.loss_pl['dir_labels_pc_cam']: f.normalize(torch.rand(3, 2048, 3), dim=2, p=2), # TODO: unrandom
            cgn.loss_pl['offset_labels_pc']: torch.rand(3, 2048, 10) # TODO: unrandom
        }
        (   
            # scene_idx, 
            step, 
            loss_val, 
            dir_loss, 
            bin_ce_loss, 
            # offset_loss, 
            # approach_loss,
            adds_loss, 
            adds_gt2pred_loss
        ) = sess.run([
            cgn.train_op,
            cgn.step, 
            cgn.losses['loss'], 
            cgn.losses['dir_loss'], 
            cgn.losses['bin_ce_loss'], 
            cgn.losses['adds_loss'],
            cgn.losses['adds_gt2pred_loss']], 
            feed_dict=feed_dict
        )

        print(
            (   
            # scene_idx, 
            step, 
            loss_val, 
            dir_loss, 
            bin_ce_loss, 
            # offset_loss, 
            # approach_loss,
            adds_loss, 
            adds_gt2pred_loss
        )
        )
        breakpoint()
        print("cat")
        # Compute loss

        # Update parameters

        # Log, etc.

def main():

    ge_global_config = get_cgn_config()
    ge_global_config['MODEL']['model'] = 'contact_graspnet.contact_graspnet.contact_graspnet'
    
    train(ge_global_config)

if __name__ == "__main__":
    main()