## Load model
import wandb
api = wandb.Api()
artifact = api.artifact('playertr/TSGrasp/model-oq9t9qwj:v65', type='model')
artifact_dir = artifact.download(root='ckpts/45000_1')

## Load config
# TODO: use same config from wandb run
from hydra import compose, initialize
from omegaconf import open_dict

with initialize(config_path="../conf"):
    cfg = compose(config_name="config")

## Override config items to match wandb run
cfg.data.data_cfg.points_per_frame = 45000
cfg.training.batch_size=1
ckpt = '/home/tim/Research/tsgrasp/ckpts/45000_1/model.ckpt'
with open_dict(cfg):
    cfg.training.resume_from_checkpoint=ckpt

## Create Trainer
from tsgrasp.training.trainer import Trainer
trainer = Trainer(cfg)
trainer.pl_model = trainer.pl_model.load_from_checkpoint(ckpt)

trainer.test()
