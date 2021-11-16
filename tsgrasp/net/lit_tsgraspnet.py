from omegaconf.dictconfig import DictConfig
import torch
import numpy as np
import pytorch_lightning as pl
import MinkowskiEngine as ME
from tsgrasp.net.tsgraspnet import multi_pointcloud_to_4d_coords, discretize, TSGraspNet
import torch.nn.functional as F

class LitTSGraspNet(pl.LightningModule):
    def __init__(self, model_cfg : DictConfig, training_cfg : DictConfig):
        super().__init__()
        self.save_hyperparameters()
        self.model = TSGraspNet(model_cfg)
        self.learning_rate = training_cfg.optimizer.learning_rate
        self.lr_decay = training_cfg.optimizer.lr_decay

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

    def _step(self, batch: dict,  batch_idx: int, stage: str=None) -> dict:
        """Run inference, compute labels, and compute a loss for a given batch of data.

        Args:
            batch (dict): batch of data containing input and grasp info needed to compute labels.
            batch_idx (int): Index of this batch
            stage (str, optional): "train", "val", or "test". Defaults to None.

        Returns:
            dict: dictionary of tensors, including `loss` (with gradient) and other values for logging or analysis
        """

        if batch_idx % 10 == 0:
            torch.cuda.empty_cache() # recommended for Minkowski

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

        ## Make predictions. Do this in parallel.
        # Convert point positions into a Minkowski SparseTensor
        voxelized_pos = discretize(positions, grid_size=self.model.grid_size)
        coords = multi_pointcloud_to_4d_coords(voxelized_pos)
        feats = torch.ones((len(coords), 1), device=coords.device)
        stensor = ME.SparseTensor(
            coordinates = coords,
            features = feats
        )

        # Make predictions
        class_logits, baseline_dir, approach_dir, grasp_offset = self.model.forward(stensor)
        class_logits = class_logits.reshape(B, T, N_PTS, 1)
        baseline_dir = baseline_dir.reshape(B, T, N_PTS, 3)
        approach_dir = approach_dir.reshape(B, T, N_PTS, 3)
        grasp_offset = grasp_offset.reshape(B, T, N_PTS, 1)

        ## Compute labels and losses. Do this in series over batches, because each batch might have different numbers of contact points.
        add_s_loss, class_loss, width_loss = 0., 0., 0.
        for b in range(B):

            ## Compute labels
            pt_labels_b, width_labels_b = self.model.class_width_labels(
                contact_pts[b], positions[b], grasp_widths[b], 
                self.model.pt_radius
            )

            ## Compute losses
            add_s_loss_b, width_loss_b, class_loss_b = self.model.losses(
                class_logits[b],
                baseline_dir[b],
                approach_dir[b],
                grasp_offset[b],
                positions[b],
                grasp_tfs[b],
                self.model.top_conf_quantile,
                pt_labels_b,
                width_labels_b
            ) 

            add_s_loss += add_s_loss_b / B
            width_loss += width_loss_b / B
            class_loss += class_loss_b / B

        ## Log loss components
        self.log(f"{stage}_add_s_loss", add_s_loss, on_step=True, on_epoch=True, sync_dist=True)
        self.log(f"{stage}_bce_loss", class_loss, on_step=True, on_epoch=True, sync_dist=True)
        self.log(f"{stage}_width_loss", width_loss, on_step=True, on_epoch=True, sync_dist=True)

        ## Combine loss components
        loss = 0.0
        loss += self.model.add_s_loss_coeff * add_s_loss 
        loss += self.model.bce_loss_coeff * class_loss 
        loss += self.model.width_loss_coeff * width_loss

        return {
            'loss': loss
        }

    # def training_step_end(self, outputs):
    #     self.log('train_pt_acc', accuracy(
    #         outputs['pt_preds'], outputs['pt_labels']), on_step=True, on_epoch=True, sync_dist=True)
    #     self.log('train_pt_true_pos', true_positive(
    #         outputs['pt_preds'], outputs['pt_labels']), on_step=True, on_epoch=True, sync_dist=True)
    #     self.log('train_pt_false_pos', false_positive(
    #         outputs['pt_preds'], outputs['pt_labels']), on_step=True, on_epoch=True, sync_dist=True)
    #     self.log('train_pt_true_neg', true_negative(
    #         outputs['pt_preds'], outputs['pt_labels']), on_step=True, on_epoch=True, sync_dist=True)
    #     self.log('train_pt_false_neg', false_negative(
    #         outputs['pt_preds'], outputs['pt_labels']), on_step=True, on_epoch=True, sync_dist=True)
    #     self.log('training_loss', float(outputs['loss']), on_step=True, on_epoch=True, sync_dist=True)

    # def validation_step_end(self, outputs):
    #     self.log('val_pt_acc', accuracy(
    #         outputs['pt_preds'], outputs['pt_labels']), on_step=True, on_epoch=True, sync_dist=True)
    #     self.log('val_pt_true_pos', true_positive(
    #         outputs['pt_preds'], outputs['pt_labels']), on_step=True, on_epoch=True, sync_dist=True)
    #     self.log('val_pt_false_pos', false_positive(
    #         outputs['pt_preds'], outputs['pt_labels']), on_step=True, on_epoch=True, sync_dist=True)
    #     self.log('val_pt_true_neg', true_negative(
    #         outputs['pt_preds'], outputs['pt_labels']), on_step=True, on_epoch=True, sync_dist=True)
    #     self.log('val_pt_false_neg', false_negative(
    #         outputs['pt_preds'], outputs['pt_labels']), on_step=True, on_epoch=True, sync_dist=True)
    #     self.log('val_loss', float(outputs['loss']), on_step=True, on_epoch=True, sync_dist=True)

    # def test_step_end(self, outputs):
    #     self.log('test_pt_acc', accuracy(
    #         outputs['pt_preds'], outputs['pt_labels']), on_step=True, on_epoch=True, sync_dist=True)
    #     self.log('test_pt_true_pos', true_positive(
    #         outputs['pt_preds'], outputs['pt_labels']), on_step=True, on_epoch=True, sync_dist=True)
    #     self.log('test_pt_false_pos', false_positive(
    #         outputs['pt_preds'], outputs['pt_labels']), on_step=True, on_epoch=True, sync_dist=True)
    #     self.log('test_pt_true_neg', true_negative(
    #         outputs['pt_preds'], outputs['pt_labels']), on_step=True, on_epoch=True, sync_dist=True)
    #     self.log('test_pt_false_neg', false_negative(
    #         outputs['pt_preds'], outputs['pt_labels']), on_step=True, on_epoch=True, sync_dist=True)
    #     self.log('test_loss', float(outputs['loss']), on_step=True, on_epoch=True, sync_dist=True)

    def _epoch_end(self, outputs, stage=None):
        if stage:
            loss = np.mean([float(x['loss']) for x in outputs])
            self.logger.log_metrics(
                {f"{stage}/loss": loss}, self.current_epoch + 1)
    
    # on_train_start()
    # for epoch in epochs:
    #   on_train_epoch_start()
    #   for batch in train_dataloader():
    #       on_train_batch_start()
    #       training_step()
    #       ...
    #       on_train_batch_end()
    #       on_validation_epoch_start()
    #
    #       for batch in val_dataloader():
    #           on_validation_batch_start()
    #           validation_step()
    #           on_validation_batch_end()
    #       on_validation_epoch_end()
    #
    #   on_train_epoch_end()
    # on_train_end

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

def accuracy(pred, des):
    return float(torch.mean((pred == des).float()))

def true_positive(pred, des):
    return float(torch.mean((des.bool()[pred.bool()].float())))

def false_positive(pred, des):
    return float(torch.mean(((~des.bool()[pred.bool()]).float())))

def true_negative(pred, des):
    return float(torch.mean((~des.bool()[~pred.bool()]).float()))

def false_negative(pred, des):
    return float(torch.mean((des.bool()[~pred.bool()]).float()))