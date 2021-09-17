from omegaconf import DictConfig
import importlib
import pytorch_lightning as pl

class Trainer:
    def __init__(self, cfg : DictConfig):

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
        self.trainer = pl.Trainer(gpus=cfg.training.gpus)

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