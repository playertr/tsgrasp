import hydra
from hydra.utils import instantiate
from omegaconf import DictConfig
import h5py
import os
from tqdm import tqdm
import torch

from contact_graspnet.contact_grasp_estimator import GraspEstimator
from contact_graspnet import config_utils
import tensorflow.compat.v1 as tf
tf.disable_eager_execution()
physical_devices = tf.config.experimental.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(physical_devices[0], True)

import time
import numpy as np

from tsgrasp.net.tsgrasp_super import TSGraspSuper

@hydra.main(config_path="../conf", config_name="scripts/save_contact_graspnet_outputs")
def main(cfg : DictConfig):
        
    pl_data = instantiate(cfg.data, batch_size=cfg.training.batch_size)
    pl_data.setup()

    global_config = config_utils.load_config(cfg.CGN_FLAGS.ckpt_dir, batch_size=cfg.CGN_FLAGS.forward_passes, arg_configs=cfg.CGN_FLAGS.arg_configs)

    grasp_estimator = GraspEstimator(global_config)
    grasp_estimator.build_network()

    # Add ops to save and restore all the variables.
    saver = tf.train.Saver(save_relative_paths=True)

    # Create a session
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    config.allow_soft_placement = True
    sess = tf.Session(config=config)

    # Load weights
    grasp_estimator.load_weights(sess, saver, cfg.CGN_FLAGS.ckpt_dir, mode='test')

    os.makedirs(os.path.dirname(cfg.output_path), exist_ok=True)
    ds = h5py.File(cfg.output_path,'w')
    ds["description"] = cfg.description
    ds.create_group("outputs")


    evaluate(grasp_estimator, sess, pl_data.test_dataloader(), ds, cfg.pt_radius)
    ds.close()

@torch.inference_mode()
def evaluate(grasp_estimator, sess, dl, h5_ds, pt_radius):
    """Iterate through the dataloader, saving outputs."""
    times = []
    example_num = 0
    for batchnum, batch in enumerate(tqdm(dl)):

        B, T, N, _3 = batch['positions'].shape

        # Contact graspnet processes point clouds one at a time, returning numpy arrays.
        # We use nested lists to turn the numpy arrays into torch.Tensors

        class_logits = []
        baseline_dir = []
        approach_dir = []
        grasp_offset = []
        positions = []
        for b in tqdm(range(B)):

            class_logits_t = []
            baseline_dir_t = []
            approach_dir_t = []
            grasp_offset_t = []
            positions_t = []

            for t in range(T):

                start = time.time()
                ## Actual inference
                pc_full = batch['positions'][b][t].numpy()
                pred_grasps_cam, pred_scores, pred_points, offset_pred = grasp_estimator.predict_grasps_from_pcl(sess, pc_full, forward_passes=1)

                end = time.time()
                if batchnum > 2:
                    times.append(end - start)

                    print(f"mean time: {np.mean(times)}")
                    print(f"std. dev time: {np.std(times)}")

                class_logits_t.append(torch.logit(torch.Tensor(pred_scores))) # inverse logistic function
                baseline_dir_t.append(torch.Tensor(pred_grasps_cam[:,:3,0]))
                approach_dir_t.append(torch.Tensor(pred_grasps_cam[:,:3,2]))
                grasp_offset_t.append(torch.Tensor(offset_pred))
                positions_t.append(torch.Tensor(pred_points))

            class_logits.append(torch.stack(class_logits_t))
            baseline_dir.append(torch.stack(baseline_dir_t))
            approach_dir.append(torch.stack(approach_dir_t))
            grasp_offset.append(torch.stack(grasp_offset_t))
            positions.append(torch.stack(positions_t))
        
        class_logits = torch.stack(class_logits)
        baseline_dir = torch.stack(baseline_dir)
        approach_dir = torch.stack(approach_dir)
        grasp_offset = torch.stack(grasp_offset)
        positions = torch.stack(positions)

        ## Save each example within the batch into its own group
        for b in range(len(class_logits)):
            h5_ds['outputs'].create_group(str(example_num))
            grp = h5_ds['outputs'][str(example_num)]
            grp.create_dataset(
                "class_logits", class_logits[b].shape, data=class_logits[b].detach().cpu(), compression="gzip", compression_opts=9
            )
            grp.create_dataset(
                "baseline_dir", baseline_dir[b].shape, data=baseline_dir[b].detach().cpu(), compression="gzip", compression_opts=9
            )
            grp.create_dataset(
                "approach_dir", approach_dir[b].shape, data=approach_dir[b].detach().cpu(), compression="gzip", compression_opts=9
            )
            grp.create_dataset(
                "grasp_offset", grasp_offset[b].shape, data=grasp_offset[b].detach().cpu(), compression="gzip", compression_opts=9
            )
            grp.create_dataset(
                "positions", positions[b].shape, data=positions[b].detach().cpu(), compression="gzip", compression_opts=9
            )

            ## Retrieve contact points
            cp = unstack_contact_points(batch['pos_contact_pts_cam'][b])
            T, N, _3 = cp.shape
            cp = cp[~cp.isnan()].reshape(T, -1, 3)

            ## Compute distance-based point labels
            if cp.shape[1] > 0:
                dists, _ = TSGraspSuper.closest_points(
                    positions[b],
                    cp
                )
                pt_labels = (dists < pt_radius).unsqueeze(-1)
            else:
                pt_labels = torch.zeros_like(class_logits[b], dtype=bool)

            grp.create_dataset(
                "gt_contact_pts", cp.shape, data=cp.cpu(), compression="gzip", compression_opts=9
            )

            grp.create_dataset(
                "pt_labels", pt_labels.shape, data=pt_labels.cpu(), compression="gzip", compression_opts=9
            )
            
            example_num += 1

def unstack_contact_points(cp):
    """Change contact point tensor from (T, N_GT_GRASPS, 2, 3) to (T, 2*G_GT_GRASPS, 3"""
    return torch.cat([cp[...,0,:], cp[...,1,:]], dim=-2)

if __name__ == "__main__":
    main()