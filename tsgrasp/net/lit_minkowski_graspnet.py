from omegaconf.dictconfig import DictConfig
import torch
import numpy as np
import pytorch_lightning as pl
import torchmetrics
import torch.nn.functional as F
from tsgrasp.net.minkowski_graspnet import MinkowskiGraspNet
import MinkowskiEngine as ME

class LitMinkowskiGraspNet(pl.LightningModule):
    def __init__(self, training_cfg : DictConfig, model_cfg : DictConfig):
        super().__init__()
        self.save_hyperparameters()
        self.model = MinkowskiGraspNet(model_cfg)
        self.loss = self.model.loss
        self.learning_rate = 0.001 # TODO pass as cfg
        self.data_len = training_cfg.data_len

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(
            [{
                'params': [p for p in self.parameters()],
                'name': 'minkowski_graspnet'
            }],
            lr=self.learning_rate
        )
        warmup_factor = 1.0 / 1000
        warmup_iters = min(1000, self.data_len - 1)
        def fun(iter_num: int) -> float:
            if iter_num >= warmup_iters:
                return 1
            alpha = float(iter_num) / warmup_iters
            return warmup_factor * (1 - alpha) + alpha
        lr_scheduler = {
            'scheduler': torch.optim.lr_scheduler.LambdaLR(optimizer, fun),
            'name': 'learning_rate'
        }
        return [optimizer], [lr_scheduler]

    def forward(self,x):
        return self.model.forward(x)

    def _step(self, batch, stage=None):
        stensor = ME.SparseTensor(
        coordinates = batch['coordinates'],
        features = batch['features']
        )

        class_logits, baseline_dir, approach_dir, grasp_offset = self.model.forward(stensor)

        labels = batch['labels']

        loss = self.model.loss(
            class_logits, labels,
            baseline_dir, approach_dir, grasp_offset,
            positions = batch['positions'],
            pos_control_points = batch['pos_control_points'], 
            sym_pos_control_points = batch['sym_pos_control_points'],
            gt_grasps_per_batch = batch['gt_grasps_per_batch'], 
            single_gripper_points = batch['single_gripper_points']
        )

        return {'loss': loss}

    def _epoch_end(self, outputs, stage=None):
        if stage:
            acc = np.mean([float(x['acc']) for x in outputs])
            loss = np.mean([float(x['loss']) for x in outputs])
            self.logger.log_metrics(
                {f"{stage}/acc": acc}, self.current_epoch + 1)
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
        return self._step(batch, "train")

    def validation_step(self, batch, batch_idx):
        return self._step(batch, "val")

    def test_step(self, batch, batch_idx):
        return self._step(batch)

    def training_epoch_end(self, outputs):
        self._epoch_end(outputs, "train")

    def validation_epoch_end(self, outputs):
        self._epoch_end(outputs, "val")

    def test_epoch_end(self, outputs):
        self._epoch_end(outputs)