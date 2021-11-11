import hydra
from omegaconf import DictConfig
from tsgrasp.training.trainer import Trainer
import wandb

@hydra.main(config_path="conf", config_name="config")
def train(cfg : DictConfig):

    if cfg.training.use_wandb:
        wandb.init()
        
    trainer = Trainer(cfg)
    trainer.train()

if __name__ == "__main__":
    train()