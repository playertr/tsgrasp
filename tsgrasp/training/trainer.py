from omegaconf import DictConfig
import importlib
import pytorch_lightning as pl
import tensorboard
import os
import wandb
from pytorch_lightning import loggers
from pytorch_lightning.callbacks import ModelCheckpoint,LearningRateMonitor

class Trainer:
    def __init__(self, cfg : DictConfig):

        ## Lightning module and datamodule
        PLModuleClass = dynamic_import(
            cfg.model.pl.module_path, 
            cfg.model.pl.module_name
        )

        self.pl_model = PLModuleClass(
            training_cfg = cfg.training,
            model_cfg = cfg.model
        )

        PLDataModuleClass = dynamic_import(cfg.data.pl.datamodule_path, cfg.data.pl.datamodule_name)
        self.pl_dataset = PLDataModuleClass(
            data_cfg = cfg.data,
            batch_size = cfg.training.batch_size
        )

        ## Loggers
        tb_dir = os.path.join(os.getcwd(), "tensorboard")
        os.makedirs(tb_dir, exist_ok=True)
        tb_logger = loggers.TensorBoardLogger(tb_dir)
        _loggers = [tb_logger]

        if cfg.training.use_wandb:
            wandb.login()
            wandb_logger = loggers.WandbLogger(
                project=cfg.training.wandb.project, log_model="all", name=cfg.training.wandb.experiment
            )
            wandb_logger.watch(self.pl_model)
            _loggers.append(wandb_logger)

        _callbacks = [ModelCheckpoint(), LearningRateMonitor(logging_interval='step')]

        ## Lightning Trainer
        self.trainer = pl.Trainer(
            gpus=cfg.training.gpus,
            logger=_loggers,
            log_every_n_steps=10,
            callbacks=_callbacks,
            max_epochs=cfg.training.max_epochs)

    def train(self):
        self.trainer.fit(self.pl_model, self.pl_dataset)

def dynamic_import(module_path, class_name):
    """Dynamically import the class given from the module path."""
    try:
        module = importlib.import_module(module_path)
        Class = getattr(module, class_name)
    except (ImportError, AttributeError):
        raise NotImplementedError(
            f"In {module_path}.py, there should be a Python class with name that matches {class_name}."
        )
    return Class