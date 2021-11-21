## Load model
import wandb
import hydra
import os

@hydra.main(config_path="../conf", config_name="download_model")
def main(cfg):
    api = wandb.Api()
    artifact = api.artifact(cfg.artifact_name, type='model')
    artifact_dir = artifact.download(root=cfg.download_path)

if __name__ == "__main__":
    main()