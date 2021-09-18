from omegaconf.dictconfig import DictConfig
import torch
import numpy as np
import pytorch_lightning as pl
import torchmetrics
import torch.nn.functional as F
from tsgrasp.net.minkowski_graspnet import MinkowskiGraspNet
import MinkowskiEngine as ME

class LitMinkowskiGraspNet(pl.LightningModule):
    def __init__(self, model_cfg : DictConfig, training_cfg : DictConfig):
        super().__init__()
        self.save_hyperparameters()
        self.model = MinkowskiGraspNet(model_cfg)
        self.loss = self.model.loss
        self.learning_rate = 0.001 # TODO pass as cfg
        self.data_len = training_cfg.data_len

        self.train_pt_acc = torchmetrics.Accuracy()
        self.val_pt_acc = torchmetrics.Accuracy()

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

    def _step(self, batch,  batch_idx, stage=None):

        torch.cuda.empty_cache() # recommended for Minkowski

        stensor = ME.SparseTensor(
        coordinates = batch['coordinates'],
        features = batch['features']
        )

        class_logits, baseline_dir, approach_dir, grasp_offset = self.model.forward(stensor)

        pt_labels = batch['labels']

        loss = self.model.loss(
            class_logits, pt_labels,
            baseline_dir, approach_dir, grasp_offset,
            positions = batch['positions'],
            pos_control_points = batch['pos_control_points'], 
            sym_pos_control_points = batch['sym_pos_control_points'],
            gt_grasps_per_batch = batch['gt_grasps_per_batch'], 
            single_gripper_points = batch['single_gripper_points']
        )

        pt_preds = class_logits > 0

        return {'loss': loss, 'pt_preds': pt_preds, 'pt_labels': pt_labels}

    # def training_step_end(self, outputs):
    #     self.train_pt_acc(outputs['pt_preds'], outputs['pt_labels'].int())
    #     self.log('train_pt_acc', self.train_pt_acc)

    # def validation_step_end(self, outputs):
    #     self.val_pt_acc(outputs['pt_preds'], outputs['pt_labels'].int())
    #     self.log('val_pt_acc', self.val_pt_acc)

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
        return self._step(batch, batch_idx,)

    def training_epoch_end(self, outputs):
        self._epoch_end(outputs, "train")

    def validation_epoch_end(self, outputs):
        self._epoch_end(outputs, "val")

    def test_epoch_end(self, outputs):
        self._epoch_end(outputs)