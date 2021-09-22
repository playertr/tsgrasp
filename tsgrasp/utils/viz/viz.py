import torch
import wandb
import pytorch_lightning as pl
from pytorch_lightning.callbacks import Callback

class ImagePredictionLogger(Callback):
    def __init__(self, val_samples, num_samples=32):
        super().__init__()
        self.num_samples = num_samples
        self.val_imgs, self.val_labels = val_samples
    def on_validation_epoch_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule):
        val_imgs = self.val_imgs.to(device=pl_module.device)

        logits = pl_module(val_imgs)
        preds = torch.argmax(logits, 1)
        trainer.logger[1].experiment.log({
            "examples": [wandb.Image(x, caption=f"Pred:{pred}, Label:{y}") 
                            for x, pred, y in zip(val_imgs, preds, self.val_labels)],
            "global_step": trainer.global_step
            })

def inference_gif(class_logits, baseline_dir, approach_dir, grasp_offset):
    pass
