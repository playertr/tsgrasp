import sys, os
ROOT_DIR = "/home/tim/Research/tsgrasp"
sys.path.insert(0, ROOT_DIR)

import hydra
import torch
from torch.utils.data import DataLoader
import numpy as np, pandas as pd, matplotlib.pyplot as plt
from omegaconf import OmegaConf, DictConfig
from tqdm import tqdm

from tsgrasp.utils.timer.timer import Timer
from collections import deque
from itertools import islice

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

# @Timer(text="success_from_labels: {:0.4f} seconds")
def success_from_labels(pred_classes, actual_classes):
    """Return success, as proportion, from ground-truth point labels.
    
    (correct positive predictions) / (total positive predictions)"""
    all_positives = pred_classes == 1
    true_positives = torch.logical_and(all_positives, (pred_classes == actual_classes))
    return true_positives.sum() / all_positives.sum()

# @Timer(text="coverage: {:0.4f} seconds")
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

# @Timer(text="distances: {:0.4f} seconds")
@TensorMemoize
def distances(grasps1, grasps2):
    """Finds the L2 distances from points in grasps1 to points in grasps2.

    Returns a (grasp1.shape[0], grasps3.shape[0]) ndarray."""

    diffs = torch.unsqueeze(grasps1, 1) - torch.unsqueeze(grasps2, 0)
    return torch.linalg.norm(diffs, axis=-1)

# @Timer(text="sucess_coverage_curve: {:0.4f} seconds")
def success_coverage_curve(confs, pred_grasps, gt_grasps, gt_labels):
    """Determine the success and coverage at various threshold confidence values."""

    res = []
    thresholds = torch.linspace(0, 1, 100)

    for t in thresholds:
        pred_classes = confs > t

        res.append({
            "confidence": t,
            "success": success_from_labels(pred_classes, gt_labels),
            "coverage": coverage(pred_classes, pred_grasps, gt_labels, gt_grasps)
        })
    
    return pd.DataFrame(res)

@hydra.main(config_path=os.path.join(ROOT_DIR, "conf"), config_name="config")
def main(cfg : DictConfig):
    pl_dataset = hydra.utils.instantiate(cfg.data, batch_size=2)
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
    
    for i, data in enumerate(dl):

        depth = data['depth'][0][0]

        # TODO: Remove this magic value
        K = np.array([
            [912.72143555, 0.0, 649.00366211], 
            [0.0, 912.7409668, 363.25247192], 
            [0.0, 0.0, 1.0]])

        pc_full, pc_segments, pc_colors = cgn.extract_point_clouds(np.array(depth), K=K)

        pc_full = pc_full[np.random.choice(range(len(pc_full)), 10000)]

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

        # plotly(pc_full)
        plot(pc_full)
        plt.imshow(depth)

        pred_grasps_cam, scores, contact_pts, _ = cgn.predict_scene_grasps(sess, pc_full, pc_segments=pc_segments)  

    breakpoint()
    print("cat")

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

def plot_success_coverage_curve_on_testset():
    """ Plots success/coverage curve."""

    # Define terms the user might want to change
    device = torch.device("cuda")
    # device = torch.device("cpu")
    # ckpt_dir = "/home/tim/Research/torch-points3d/outputs/2021-08-21/14-39-53"
    # ckpt_dir = "/home/tim/Research/torch-points3d/outputs/2021-08-24/15-14-53"
    # ckpt_dir = "/home/tim/Research/torch-points3d/outputs/2021-08-25/09-09-54"
    ckpt_dir = "/home/tim/Research/torch-points3d/outputs/2021-08-29/00-34-19"
    check_name = "GraspMinkUNet14A"

    # Create dataloader and model
    model, loader = model_and_loader(ckpt_dir, check_name, device)

    # Generate success/coverage curves for each batch in the test dataset
    s_c_curves = []
    for i, data in enumerate(tqdm(loader)):
        # Process a single batch
        with torch.no_grad():
            model.set_input(data, device)
            model.forward()

        # At each of the (batch, time, x, y, z) coordinates, get the output
        # confidence and ground truth label
        out_coords = model.input.coordinates[model.input.inverse_mapping].detach()#.cpu().numpy()
        out_confs = torch.sigmoid(model.class_logits.detach())#.cpu().numpy()
        out_positions = model.positions.detach()#.cpu().numpy()
        grasp_labels = model.labels.detach()#.cpu().numpy()

        batches = torch.unique(out_coords[:,0]) # batch dim is first column
        times = torch.unique(out_coords[:,1]) # time is second column
        for b in batches:
            for t in times:
                idxs = torch.where(
                    torch.logical_and(out_coords[:,0] == b, out_coords[:,1] ==t)
                )
                pos = out_positions[idxs]
                confs = out_confs[idxs]
                labels = grasp_labels[idxs]

                s_c_curves.append(success_coverage_curve(confs, pos, pos, labels))

    print(s_c_curves[-1])
    plot_s_c_curve(s_c_curves[-1], title="Last Example")

    super_df = pd.concat(s_c_curves).astype(np.float)
    super_df2 = super_df.groupby('confidence').mean()
    plot_s_c_curve(super_df2, title="Entire Test Set")
    plt.show()
    breakpoint()
    print("done")

def load_cgn():
    ########################### Import CGN stuff ###############################
    sys.path.insert(0, "/home/tim/Research/contact_graspnet/contact_graspnet")
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
    ckpt_dir = '/home/tim/Research/contact_graspnet/checkpoints/scene_test_2048_bs3_hor_sigma_001'
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


def test_success_coverage():
    pass
def try_cgn(pc_full):
    pass

    ########################### Run network  ###################################





if __name__ == "__main__":
    # plot_success_coverage_curve_on_testset()
    main()

