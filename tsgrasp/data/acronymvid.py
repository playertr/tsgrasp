import h5py
import os
import numpy as np
import torch
from dataclasses import dataclass
from tsgrasp.utils.mesh_utils import create_gripper
import MinkowskiEngine as ME
from omegaconf import DictConfig

@dataclass
class Data:
    time : torch.Tensor
    pos: torch.Tensor
    x: torch.Tensor
    y: torch.Tensor
    pos_control_points: np.ndarray
    sym_pos_control_points: np.ndarray
    single_gripper_pts: torch.Tensor

class AcronymVidDataset(torch.utils.data.Dataset):
    """
    A torch.geometric.Dataset for loading from files.
    """

    AVAILABLE_SPLITS = ["train", "val", "test"]

    def __init__(self, cfg : DictConfig, split="train"):

        self.root = cfg.dataroot

        # Find the raw filepaths. For now, we're doing no file-based preprocessing.
        if split in ["train", "val", "test"]:
            folder = os.path.join(self.root, split)
            self._paths = [os.path.join(folder, f) for f in os.listdir(folder) if f.endswith('.h5')]
        else:
            raise ValueError("Split %s not recognised" % split)

        # Make a list of tuples (name, path) for each trajectory.
        self._trajectories = []
        for path in self._paths:
            with h5py.File(path) as ds:
                keys = {k for k in ds.keys() if k.startswith('pitch')} # a Set
            self._trajectories += [(k, path) for k in keys]

    def download(self):
        if len(os.listdir(self.raw_dir)) == 0:
            print(f"No files found in {self.raw_dir}. Please create the dataset using the scripts in GraspRefinement and ACRONYM.")

    def len(self):
        return len(self._trajectories)

    def __getitem__(self, idx):
        traj_name, path = self._trajectories[idx]

        with h5py.File(path) as ds:
            # `depth` and `labels` are (B, 300,300) arrays.
            # `depth` contains the depth video values, and `labels` is a binary mask indicating
            # whether a given pixel's 3D point is within data_generation.params.EPSILON of a positive grasp contact.
            depth = np.asarray(ds[traj_name]["depth"])
            labels = np.asarray(ds[traj_name]["grasp_labels"])
            # nearest_grasp_idxs = np.asarray(ds[traj_name]["nearest_grasp_idx"])
            success = np.asarray(ds["grasps/qualities/flex/object_in_gripper"])
            pos_grasp_tfs = np.asarray(ds["grasps/transforms"])[success==1]
            tfs_from_cam_to_obj = np.asarray(ds[traj_name]["tf_from_cam_to_obj"])
            # grasp_contact_points = np.asarray(ds["grasps/contact_points"])

        ## Make data shorter via temporal decimation
        depth = depth[::3, :, :]
        labels = labels[::3, :, :]
        tfs_from_cam_to_obj = tfs_from_cam_to_obj[::3,:,:]

        pcs = [depth_to_pointcloud(d) for d in depth]
        pcs = multi_pointcloud_to_4d_coords(pcs)

        ## Calculate gripper widths
        # cp = grasp_contact_points
        # widths = np.linalg.norm(cp[:,0,:] - cp[:,1,:], axis=1)

        ## Generate camera-frame grasp poses corresponding to closest points
        obj_frame_grasp_tfs = pos_grasp_tfs # (2000, 4, 4)
        tfs_from_obj_to_cam = np.array([inverse_homo(t) for t in tfs_from_cam_to_obj])

        # gross numpy broadcasting
        # https://stackoverflow.com/questions/32171917/copy-2d-array-into-3rd-dimension-n-times-python
        cam_frame_grasp_tfs =  np.matmul(tfs_from_obj_to_cam[:,np.newaxis,:,:], obj_frame_grasp_tfs[np.newaxis,:,:,:])
        # (30, 2000, 4, 4)

        ## Create 3D tensor of (num_good_grasps, 5, 3) control points
        gripper = create_gripper('panda', root_folder=self.root)
        gripper_control_points = gripper.get_control_point_tensor(max(1, pos_grasp_tfs.shape[0]), use_tf=False) # num_gt_grasps x 5 x 3

        gripper_control_points_homog =  np.concatenate([gripper_control_points, np.ones((gripper_control_points.shape[0], gripper_control_points.shape[1], 1))], axis=2)  # b x 5 x 4

        control_points = np.matmul(gripper_control_points_homog, cam_frame_grasp_tfs.transpose((0, 1, 3, 2)))[:,:,:,:3]

        ## Create flipped symmetric control points for symmetric ADD-S loss
        sym_gripper_control_points = gripper.get_control_point_tensor(max(1, pos_grasp_tfs.shape[0]), symmetric=True, use_tf=False) # num_gt_grasps x 5 x 3

        sym_gripper_control_points_homog =  np.concatenate([sym_gripper_control_points, np.ones((sym_gripper_control_points.shape[0], sym_gripper_control_points.shape[1], 1))], axis=2)  # b x 5 x 4

        sym_control_points = np.matmul(sym_gripper_control_points_homog, cam_frame_grasp_tfs.transpose((0, 1, 3, 2)))[:,:,:,:3]

        coords4d = torch.Tensor(pcs)
        data = Data(
            time=coords4d[:,0],     # First col is time
            pos = coords4d[:, 1:],  # Last 3 cols are x, y, z
            x=torch.ones((len(pcs), 1)), 
            y=torch.Tensor(labels).view(-1 ,1),
            pos_control_points = control_points,
            sym_pos_control_points = sym_control_points,
            single_gripper_pts = torch.Tensor(gripper_control_points[0])
        )

        return data

    @property
    def raw_dir(self):
        return self.root

def minkowski_collate_fn(list_data):
    coordinates_batch, features_batch, labels_batch = ME.utils.sparse_collate(
        [d["coordinates"] for d in list_data],
        [d["features"] for d in list_data],
        [d["label"] for d in list_data],
        dtype=torch.float32,
    )
    return {
        "coordinates": coordinates_batch,
        "features": features_batch,
        "labels": labels_batch,
    }

# def downsample(d : dict, pts_per_frame : int) -> dict:
#     """Randomly downsample coordinates or features."""
#     # Find how many 

def multi_pointcloud_to_4d_coords(pcs):
    """Convert a list of N (L, 3) ndarrays into a single (N*L, 4) ndarray, where the first dimension becomes the time dimension."""
    num_pcs = len(pcs)
    time_coords = np.repeat(np.arange(num_pcs), len(pcs[0]))
    
    pcs = np.concatenate(pcs)
    pcs = np.column_stack([time_coords, pcs])

    return pcs

def depth_to_pointcloud(depth, fov=np.pi/6):
    """Convert depth image to pointcloud given camera intrinsics, from acronym.scripts.acronym_render_observations

    Args:
        depth (np.ndarray): Depth image.

    Returns:
        np.ndarray: Point cloud.
    """
    fy = fx = 0.5 / np.tan(fov * 0.5)  # aspectRatio is one.

    height = depth.shape[0]
    width = depth.shape[1]

    mask = np.where(depth > 0)

    x = mask[1]
    y = mask[0]

    normalized_x = (x.astype(np.float32) - width * 0.5) / width
    normalized_y = (y.astype(np.float32) - height * 0.5) / height

    world_x = normalized_x * depth[y, x] / fx
    world_y = normalized_y * depth[y, x] / fy
    world_z = depth[y, x]

    return np.vstack((world_x, world_y, world_z)).T

def inverse_homo(tf):
    """Compute inverse of homogeneous transformation matrix.

    The matrix should have entries
    [[R,       Rt]
     [0, 0, 0, 1]].
    """
    R = tf[0:3, 0:3]
    t = R.T @ tf[0:3, 3].reshape(3, 1)
    return np.block([
        [R.T, -t],
        [0, 0, 0, 1]
    ])

if __name__ == "__main__":
    # Test loading dataset.
    gds = AcronymVidDataset(root="/home/tim/Research/GraspRefinement/data/acronymvid", split="test")

    print(gds)
    print(gds[0])

    breakpoint()

    print("done")

def minkowski_collate_fn(list_data):
    r"""
    Collation function for MinkowskiEngine.SparseTensor that creates batched
    cooordinates given a list of dictionaries.
    """
    coordinates_batch, features_batch, labels_batch = ME.utils.sparse_collate(
        [d["coordinates"] for d in list_data],
        [d["features"] for d in list_data],
        [d["labels"] for d in list_data],
        dtype=torch.float32,
    )
    return {
        "coordinates": coordinates_batch,
        "features": features_batch,
        "labels": labels_batch,
    }