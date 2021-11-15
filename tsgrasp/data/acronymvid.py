import h5py
import os
import numpy as np
import torch
from tsgrasp.utils.mesh_utils.mesh_utils import create_gripper
from omegaconf import DictConfig
from functools import reduce

class AcronymVidDataset(torch.utils.data.Dataset):
    """
    A torch.geometric.Dataset for loading from files.
    """

    AVAILABLE_SPLITS = ["train", "val", "test"]

    def __init__(self, cfg : DictConfig, split="train"):

        self.root = cfg.dataroot
        self.pts_per_frame = cfg.points_per_frame
        self.grid_size = cfg.grid_size
        self.time_decimation_factor = cfg.time_decimation_factor

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

        self._trajectories = np.array(self._trajectories).astype(np.string_)
        # cannot be list of tuples, must be contiguous array due to memory leak
        #https://github.com/pytorch/pytorch/issues/13246

    def download(self):
        if len(os.listdir(self.raw_dir)) == 0:
            print(f"No files found in {self.raw_dir}. Please create the dataset using the scripts in GraspRefinement and ACRONYM.")

    def __len__(self):
        return len(self._trajectories)

    def __getitem__(self, idx):
        traj_name, path = self._trajectories[idx]

        with h5py.File(path) as ds:
            # `depth` and `labels` are (B, 300,300) arrays.
            # `depth` contains the depth video values, and `labels` is a binary mask indicating
            # whether a given pixel's 3D point is within data_generation.params.EPSILON of a positive grasp contact.
            depth = np.asarray(ds[traj_name]["depth"])
            labels = np.asarray(ds[traj_name]["grasp_labels"])
            nearest_grasp_idx = np.asarray(ds[traj_name]["nearest_grasp_idx"])
            success = np.asarray(ds["grasps/qualities/flex/object_in_gripper"])
            pos_grasp_tfs = np.asarray(ds["grasps/transforms"])[success==1]
            tfs_from_cam_to_obj = np.asarray(ds[traj_name]["tf_from_cam_to_obj"])
            grasp_contact_points = np.asarray(ds["grasps/contact_points"])

        ## Make data shorter via temporal decimation
        depth = depth[::self.time_decimation_factor, :, :]
        labels = labels[::self.time_decimation_factor, :, :]
        tfs_from_cam_to_obj = tfs_from_cam_to_obj[::self.time_decimation_factor,:,:]
        nearest_grasp_idx = nearest_grasp_idx[::self.time_decimation_factor,:,:]

        pcs = [depth_to_pointcloud(d) for d in depth]
        orig_pcs = torch.Tensor(pcs)
        labels = [label_frame for label_frame in labels]
        nearest_grasp_idx = [idcs for idcs in nearest_grasp_idx]

        ## Downsample points
        for i in range(len(pcs)):
            idxs = torch.randperm(len(pcs[i]), dtype=torch.int32, device='cpu')[:self.pts_per_frame].sort()[0].long()
            
            pcs[i] = pcs[i][idxs]
            labels[i] = labels[i].ravel()[idxs]
            nearest_grasp_idx[i] = nearest_grasp_idx[i].ravel()[idxs]

        ## Quantize points to grid
        positions = torch.Tensor(pcs) # save positions prior to truncation

        ## Generate camera-frame grasp poses corresponding to closest points
        obj_frame_pos_grasp_tfs = pos_grasp_tfs # (2000, 4, 4)
        tfs_from_obj_to_cam = np.array([inverse_homo(t) for t in tfs_from_cam_to_obj])

        # transform object-frame grasp poses into camera frame
        # gross numpy broadcasting
        # https://stackoverflow.com/questions/32171917/copy-2d-array-into-3rd-dimension-n-times-python
        cam_frame_pos_grasp_tfs =  np.matmul(tfs_from_obj_to_cam[:,np.newaxis,:,:], obj_frame_pos_grasp_tfs[np.newaxis,:,:,:])
        # (30, 2000, 4, 4)

        pos_contact_pts_mesh = grasp_contact_points[np.where(success)]
        data = {
            "positions" : positions,
            "cam_frame_pos_grasp_tfs": torch.Tensor(cam_frame_pos_grasp_tfs),
            "pos_contact_pts_mesh": torch.Tensor(pos_contact_pts_mesh.astype(np.float32)),
        }

        return data

    @property
    def raw_dir(self):
        return self.root

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