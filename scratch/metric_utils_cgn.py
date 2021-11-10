import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sys, os
WORK_DIR = "/home/tim/Research"
# WORK_DIR = "/scratch/playert/workdir"
ROOT_DIR = os.path.join(WORK_DIR, "tsgrasp")
sys.path.insert(0, ROOT_DIR)

import hydra
import torch
from torch.utils.data import DataLoader
from omegaconf import DictConfig
from tqdm import tqdm

from collections import deque

class TensorMemoize:
    """Decorator to memoize a function with Tensor arguments.
    
    Arguments are not hashed, so this is slow and memory intensive."""

    # I haven't figured out how to pass parameters to decorators.
    def __init__(self, f, maxlen=1):
        self.f = f
        self.memo = deque(maxlen=maxlen)
    
    def __call__(self, *args):
        for (cargs, val) in self.memo:
            # assumes len(args) == len(cargs)
            if all(torch.equal(cargs[i], args[i]) for i in range(len(args))):
                return val

        val = self.f(*args)
        self.memo.appendleft((args, val))
        return val

def success_from_labels(pred_classes, actual_classes):
    """Return success, as proportion, from ground-truth point labels.
    
    (correct positive predictions) / (total positive predictions)"""
    all_positives = pred_classes == 1
    true_positives = torch.logical_and(all_positives, (pred_classes == actual_classes))
    return true_positives.sum() / all_positives.sum()

def coverage(pred_classes, pred_grasps, gt_labels, gt_grasps, radius=0.005):
    """Get the proportion of ground truth grasps within epsilon of predicted.
    
    (gt close to predicted) / (total gt)
    """
    if pred_classes.sum() == 0: return torch.Tensor([0.0])
    # for each grasp coordinate in gt_grasps, find the distance to every grasp grasp_coordinate in pred_grasps
    pos_gt_grasps = gt_grasps[torch.where(gt_labels.squeeze() == 1)]
    dists = distances(pos_gt_grasps, pred_grasps) # (n_gt, n_pred)

    # Remove the columns of the distance matrix from points that are predicted to fail
    dists = dists[:, torch.where(pred_classes.squeeze() == 1)[0]]

    closest_dists, idxs = dists.min(axis=1)
    return torch.mean((closest_dists < radius).float())

@TensorMemoize
def distances(grasps1, grasps2):
    """Finds the L2 distances from 3D points in grasps1 to points in grasps2.

    Returns a (grasp1.shape[0], grasps3.shape[0]) ndarray."""

    diffs = torch.unsqueeze(grasps1, 1) - torch.unsqueeze(grasps2, 0)
    return torch.linalg.norm(diffs, axis=-1)

def success_coverage_curve(confs, pred_grasps, gt_grasps, gt_labels):
    """Determine the success and coverage at various threshold confidence values."""

    res = []
    thresholds = torch.linspace(0, 1, 10000)

    # quantiles = torch.linspace(0, 1, 1000)
    # thresholds = [torch.quantile(confs, q) for q in quantiles]

    for t in thresholds:
        pred_classes = confs > t

        res.append({
            "confidence": t,
            "success": success_from_labels(pred_classes, gt_labels),
            "coverage": coverage(pred_classes, pred_grasps, gt_labels, gt_grasps)
        })
    
    return pd.DataFrame(res)

@hydra.main(config_path=os.path.join(ROOT_DIR, "conf"), config_name="config")
def main_cgn(cfg : DictConfig):
    pl_dataset = hydra.utils.instantiate(cfg.data, batch_size=1)
    pl_dataset.prepare_data()
    pl_dataset.setup()

    # create DataLoader with default collate function, not minkowski_collate,
    # so we don't have to import ME
    dl = DataLoader(
        pl_dataset.dataset_val,
        batch_size=1,
        shuffle=True
    )
    
    sess, cgn = load_cgn()
    cgn.select_grasps = select_all_grasps # hacky override
    
    s_c_curves = []
    for i, data in enumerate(tqdm(dl)):

        pc_full = data['positions'][0][0].numpy()
        orig_pc_full = pc_full.copy()

        pc_full = pc_full[np.random.choice(range(len(pc_full)), 20000)]

        # # plotly(pc_full)
        # plot(pc_full)
        # plt.imshow(depth)

        pred_grasps_cam, scores, contact_pts, _ = cgn.predict_scene_grasps(sess, pc_full, pc_segments=None)  

        # predict_scene_grasps downsamples the points.Let's find the indices of the original points to learn their labels.
        # Note: the labels were based on an arbitrary distance epsilon.
        cp = contact_pts[-1] # output 3D point locations
        idxs = np.nan * np.ones((len(cp)))
        for i in range(len(cp)):
            idx = np.where(np.isclose(orig_pc_full, cp[i]).all(axis=1))
            idxs[i], = idx
        idxs = idxs.astype(int)

        labels = data['labels'][0].numpy()
        labels_2048 = labels[idxs].reshape(-1, 1)
        pos_2048 = orig_pc_full[idxs]

        df = success_coverage_curve(confs=torch.Tensor(scores[-1]).reshape(-1, 1), pred_grasps=torch.Tensor(pos_2048), gt_grasps=torch.Tensor(pos_2048), gt_labels=torch.Tensor(labels_2048))

        df.to_csv(f"/home/tim/Research/tsgrasp/scratch/cgn_csvs/{i}.csv")
        s_c_curves.append(df)

    super_df = pd.concat(s_c_curves).astype(np.float)
    super_df2 = super_df.groupby('confidence').mean()

    super_df2.to_csv("cgn_sc.csv")
    # plot_s_c_curve(super_df2, title="Entire Test Set")
    # plt.show()
    # breakpoint()
    print("done")

@hydra.main(config_path=os.path.join(ROOT_DIR, "conf"), config_name="config")
def main_ours(cfg: DictConfig):

    cfg.data.data_cfg.points_per_frame = 45000
    cfg.training.batch_size=1
    # ckpt = '/home/tim/Research/tsgrasp/ckpts/45000_1/model.ckpt'
    ckpt = '/home/tim/Research/tsgrasp/outputs/2021-10-26/12-25-08/default_TSGrasp/0_12n2kq3e/checkpoints/epoch=24-step=6349.ckpt'

    pl_dataset = hydra.utils.instantiate(cfg.data, batch_size=1)
    pl_dataset.prepare_data()
    pl_dataset.setup()

    dl = pl_dataset.test_dataloader()

    pl_module = hydra.utils.instantiate(cfg.model, training_cfg=cfg.training)
    pl_module = pl_module.load_from_checkpoint(ckpt)
    pl_module.eval()

    s_c_curves = []
    for i, data in enumerate(tqdm(dl)):
        # if i < 78:
        #     continue
        if i > 128 and i < 140:
            continue

        if i > 12:
            break

        out = pl_module._step(data, i)

        class_logits, baseline_dir, approach_dir, grasp_offset = out['outputs']

        positions = data['positions'].reshape(-1, 3)
        labels=data['labels']
        idxs = torch.arange(0, 45000) # only take points from first timestep
        # idxs = np.random.choice(idxs, 2048) # only select 2048 of them, for fair coverage comparison with CGN
        positions = positions[idxs]
        class_logits = class_logits[idxs]
        labels = labels[idxs]

        # device = torch.device('cuda')
        device = torch.device('cpu')
        df = success_coverage_curve(confs=class_logits.to(device), pred_grasps=positions.to(device), gt_grasps=positions.to(device), gt_labels=labels.to(device))

        df = df.astype(float)
        df.to_csv(f"/home/tim/Research/tsgrasp/scratch/our_csvs/{i}.csv")
        s_c_curves.append(df)

    super_df = pd.concat(s_c_curves).astype(np.float)
    super_df2 = super_df.groupby('confidence').mean()

    super_df2.to_csv("ours_sc.csv")
    # plot_s_c_curve(super_df2, title="Entire Test Set")
    # plt.show()
    # breakpoint()
    print("done")


def plot(pc):
    import matplotlib.pyplot as plt
    fig = plt.figure()
    ax = fig.gca(projection='3d')
    ax.scatter(*pc.T)
    plt.show()

def plotly(pc):
    import plotly.graph_objects as go
    x,y,z=pc.T
    marker_data = go.Scatter3d(
        x=x,y=y,z=z,
        marker=go.scatter3d.Marker(size=3), 
        opacity=0.8, 
        mode='markers'
    )
    fig=go.Figure(data=marker_data)
    fig.show()

def select_all_grasps(self, contact_pts, contact_conf, max_farthest_points = 150, num_grasps = 200, first_thres = 0.25, second_thres = 0.2, with_replacement=False):
    """ Overrides GraspEstimator::select_grasps(), so that all grasps are returned without thresholding."""
    return range(len(contact_pts))

# @Timer(text="model_and_loader: {:0.4f} seconds")
def model_and_loader(ckpt_dir, check_name, device, yaml_config=None):
    """Create a pretrained model and Dataloader."""

    dataset = create_dataset(yaml_config)
    model_ckpt = ModelCheckpoint(ckpt_dir, check_name, "test", resume=False)
    model = model_ckpt.create_model(dataset, weight_name="best").to(device)
    model.eval()
    dataset.create_dataloaders(model,
        batch_size=2,
        shuffle=False,
        num_workers=6,
        precompute_multi_scale=False
    )
    loader = dataset.test_dataloaders[0]
    return model, loader

def plot_s_c_curve(df, ax=None, title="Coverage vs. Success"):
    if ax is None:
        fig, ax = plt.subplots()
    
    plot = ax.plot(df['coverage'], df['success'])
    ax.set_xlabel("Coverage")
    ax.set_ylabel("Success")
    ax.set_title(title)

    return plot

def load_cgn():
    ########################### Import CGN stuff ###############################
    sys.path.insert(0, os.path.join(WORK_DIR, "contact_graspnet/contact_graspnet"))
    from contact_grasp_estimator import GraspEstimator
    from visualization_utils import visualize_grasps, show_image
    from data import load_available_input_data
    import config_utils

    import tensorflow.compat.v1 as tf
    tf.disable_eager_execution()
    physical_devices = tf.config.experimental.list_physical_devices('GPU')
    tf.config.experimental.set_memory_growth(physical_devices[0], True)

    ########################### Load CGN ###############################

    # argparse params from contact_graspnet/inference.py
    ckpt_dir = os.path.join(WORK_DIR, "contact_graspnet/checkpoints/scene_test_2048_bs3_hor_sigma_001")
    forward_passes = 1
    arg_configs = []

    global_config = config_utils.load_config(ckpt_dir, batch_size=forward_passes, arg_configs=arg_configs)
     # Build the model
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
    grasp_estimator.load_weights(sess, saver, ckpt_dir, mode='test')

    print("done")

    return sess, grasp_estimator

if __name__ == "__main__":
    # plot_success_coverage_curve_on_testset()
    main_ours()

