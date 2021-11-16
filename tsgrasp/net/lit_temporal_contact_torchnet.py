# lit_temporal_contact_torchnet.py
# Tim Player, 2 November 2021, playert@oregonstate.edu
# A Pytorch Lightning module to wrap Contact Torchnet, so that
# the module can train and test using the the identical temporal batching to
# our temperospatial grasp network.

from omegaconf.dictconfig import DictConfig
import torch
from tsgrasp.net.contact_torchnet import ContactTorchNet
import pytorch_lightning as pl
import numpy as np

class LitTemporalContactTorchNet(pl.LightningModule):
    def __init__(self, model_cfg : DictConfig, training_cfg : DictConfig):
        super().__init__()

        self.save_hyperparameters()
        self.model = ContactTorchNet(global_config=model_cfg)

        self.learning_rate = training_cfg.optimizer.learning_rate
        self.lr_decay = training_cfg.optimizer.lr_decay

        # Deactivate the automatic optimization step, so that we can
        # train for multiple timesteps sequentially in step().
        self.automatic_optimization = False

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(
            [{
                'params': [p for p in self.parameters()],
                'name': 'minkowski_graspnet'
            }],
            lr=self.learning_rate
        )
        lr_scheduler = {
            'scheduler': torch.optim.lr_scheduler.ExponentialLR(optimizer, 
                self.lr_decay),
            'name': 'learning_rate'
        }
        return [optimizer], [lr_scheduler]

    def forward(self,x):
        return self.model.forward(x)

    def _step(self, batch, batch_idx, stage=None):
        opt = self.optimizers()

        ## Unpack data from dictionary
        positions = batch['positions'] 
        # (B, T, N_PTS, 3) cam point cloud
        grasp_tfs = batch['cam_frame_pos_grasp_tfs'] 
        # (B,) list of (T, N_GT_GRASPS_i, 4, 4) tensors of homogeneous grasp poses (from positive grasps) in the camera frame
        contact_pts = batch['pos_contact_pts_cam'] 
        # (B,) list of (T, N_GT_GRASPS_i, 2, 3) tensors of gripper contact points (for left and right fingers) in the mesh frame

        grasp_widths = [
            torch.linalg.norm(
                cp[...,0, :] - cp[...,1, :],
                dim=-1
            ).unsqueeze(-1)
            for cp in contact_pts
        ]
        # (B,) list of 10, N_GT_GRASPS_i, 1)

        B, T, N_PTS, D = positions.shape
        assert B == 1, "Only batch size of one supported for CTN."

        (baseline_dir, class_logits, grasp_offset, approach_dir, pred_points
        ) = self.model.forward(positions[0])
        # class_logits is (B, N_PRED_PTS)
        # baseline_dir is (B, N_PRED_PTS, 3)
        # approach_dir is (B, N_PRED_PTS, 3)
        # grasp_offset is (B, N_PRED_PTS, 10)
        # pred_points is (B, N_PRED_PTS, 3)

        opt.zero_grad()
        ## Compute labels and losses. Do this in series over batches, because each batch might have different numbers of contact points.
        add_s_loss, class_loss, width_loss = 0., 0., 0.
        for b in range(B):

            ## Compute labels
            pt_labels_b, width_labels_b = self.model.class_width_labels(
                contact_pts[b], pred_points, grasp_widths[b], 
                self.model.pt_radius
            )

            ## Compute losses
            add_s_loss_b, width_loss_b, class_loss_b = self.model.losses(
                class_logits.unsqueeze(-1),
                baseline_dir,
                approach_dir,
                grasp_offset,
                pred_points,
                grasp_tfs[b],
                self.model.top_conf_quantile,
                pt_labels_b,
                width_labels_b
            ) 

            add_s_loss += add_s_loss_b / B
            width_loss += width_loss_b / B
            class_loss += class_loss_b / B

        ## Combine loss components
        loss = 0.0
        loss += self.model.add_s_loss_coeff * add_s_loss 
        loss += self.model.bce_loss_coeff * class_loss 
        loss += self.model.width_loss_coeff * width_loss

        
        if stage == "train":
            self.manual_backward(loss)
            opt.step()

        return {"loss": loss}

    def _epoch_end(self, outputs, stage=None):
        if stage:
            loss = np.mean([float(x['loss']) for x in outputs])
            self.logger.log_metrics(
                {f"{stage}/loss": loss}, self.current_epoch + 1)

    def training_step(self, batch, batch_idx):
        return self._step(batch, batch_idx, "train")

    def validation_step(self, batch, batch_idx):
        return self._step(batch, batch_idx, "val")

    def test_step(self, batch, batch_idx):
        return self._step(batch, batch_idx, "test")

    def training_epoch_end(self, outputs):
        self._epoch_end(outputs, "train")

    def validation_epoch_end(self, outputs):
        self._epoch_end(outputs, "val")

    def test_epoch_end(self, outputs):
        self._epoch_end(outputs, "test")