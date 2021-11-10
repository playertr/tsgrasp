# lit_temporal_contact_torchnet.py
# Tim Player, 2 November 2021, playert@oregonstate.edu
# A Pytorch Lightning module to wrap Contact Torchnet, so that
# the module can train and test using the the identical temporal batching to
# our temperospatial grasp network.

from omegaconf.dictconfig import DictConfig
import torch
import numpy as np
import pytorch_lightning as pl
import torchmetrics
from contact_torchnet.contact_torchnet.net.lit_contact_torchnet import LitContactTorchNet

class LitTemporalContactTorchNet(LitContactTorchNet):
    def __init__(self, model_cfg : DictConfig, training_cfg : DictConfig):
        super().__init__(model_cfg, training_cfg)

        # Deactivate the automatic optimization step, so that we can
        # train for multiple timesteps sequentially in step().
        self.automatic_optimization = False

    def _step(self, batch, batch_idx, stage=None):
        opt = self.optimizers()

        batch_nums = batch['coordinates'][:,0].unique().long()
        assert len(batch_nums) == 1, "Only batch size of one supported for CTN."

        # Pass each timestep's inputs through one frame at a time.
        all_res = []
        times = batch['coordinates'][:,1].unique().long()
        for time in times:
            opt.zero_grad()

            # ONLY batch size of one is allowed for CTN.
            # TODO change 0 -> `time` in a few places you goofazoid Timothy
            ctn_batch = {
                'all_pos': batch['all_pos'][:, time, ...],
                'cam_frame_pos_grasp_tfs': batch['cam_frame_pos_grasp_tfs'][0][time, ...].unsqueeze(0),
                'pos_contact_pts_mesh': batch['pos_contact_pts_mesh'][0].unsqueeze(0),
                'pos_finger_diffs': batch['pos_finger_diffs'][0].unsqueeze(0),
                'camera_pose': batch['camera_pose'][0][time].unsqueeze(0),
                'single_gripper_points': batch['single_gripper_points'].unsqueeze(0),
                'sym_single_gripper_points': batch['single_gripper_points'].unsqueeze(0),
            }
            results = super()._step(ctn_batch, batch_idx, stage=None)

            if stage == "train":
                self.manual_backward(results['loss'])
                opt.step()
            
            results['loss'] = results['loss'].detach().cpu()

            # append the result dictionary 
            all_res.append(results)
        
        # return a dictionary with the outputs from each step
        def stack_irreg(ls):
            """Attempt to stack items from a list"""
            try:
                return torch.stack(ls)
            except TypeError:
                return ls

        di = {k: stack_irreg([d[k] for d in all_res]) for k in all_res[0].keys()}
        di['loss'] = torch.mean(di['loss'])
        return di

    # override test_step, which was doing extra calculation in LCTN
    def test_step(self, batch, batch_idx):

        batch_nums = batch['coordinates'][:,0].unique().long()
        assert len(batch_nums) == 1, "Only batch size of one supported for CTN."

        # Pass each timestep's inputs through one frame at a time.
        all_res = []
        times = batch['coordinates'][:,1].unique().long()
        for time in times:

            # ONLY batch size of one is allowed for CTN.
            # TODO change 0 -> `time` in a few places you goofazoid Timothy
            ctn_batch = {
                'all_pos': batch['all_pos'][:, time, ...],
                'cam_frame_pos_grasp_tfs': batch['cam_frame_pos_grasp_tfs'][0][time, ...].unsqueeze(0),
                'pos_contact_pts_mesh': batch['pos_contact_pts_mesh'][0].unsqueeze(0),
                'pos_finger_diffs': batch['pos_finger_diffs'][0].unsqueeze(0),
                'camera_pose': batch['camera_pose'][0][time].unsqueeze(0),
                'single_gripper_points': batch['single_gripper_points'].unsqueeze(0),
                'sym_single_gripper_points': batch['single_gripper_points'].unsqueeze(0),
            }
            results = super()._step(ctn_batch, batch_idx, stage=None)
            
            results['loss'] = results['loss'].detach().cpu()

            # append the result dictionary 
            all_res.append(results)
        
        # return a dictionary with the outputs from each step
        def stack_irreg(ls):
            """Attempt to stack items from a list"""
            try:
                return torch.stack(ls)
            except TypeError:
                return ls

        di = {k: stack_irreg([d[k] for d in all_res]) for k in all_res[0].keys()}
        di['loss'] = torch.mean(di['loss'])
        return di