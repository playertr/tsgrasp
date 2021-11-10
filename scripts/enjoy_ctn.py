## Load model
import wandb
api = wandb.Api()
artifact = api.artifact('playertr/ContactTorchNet/model-2tnd0sd2:v29', type='model')
artifact_dir = artifact.download(root='ckpts/ctn_10-29-2021')

## Load config
# TODO: use same config from wandb run
from hydra import compose, initialize
from omegaconf import open_dict

with initialize(config_path="../contact_torchnet/conf"):
    cfg = compose(config_name="config")

## Override config items to match wandb run
cfg.training.batch_size=1
# ckpt = '/home/tim/Research/tsgrasp/ckpts/45000_1/model.ckpt'
ckpt = 'ckpts/ctn_10-29-2021'

# # DEBUG -- USE SUBSET OF DATA
# cfg.data.data_cfg.subset_factor=16

with open_dict(cfg):
    cfg.training.resume_from_checkpoint=ckpt

## Create Trainer
from tsgrasp.training.trainer import Trainer
trainer = Trainer(cfg)
trainer.pl_model = trainer.pl_model.load_from_checkpoint(ckpt)

trainer.test()
