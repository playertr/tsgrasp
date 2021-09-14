from omegaconf import DictConfig
import torch
import importlib

class Trainer:
    def __init__(self, cfg : DictConfig):
        self.model = instantiate_model(
            cfg.model_path, 
            cfg.model_name, 
            cfg.models[cfg.model_name],
            cfg.data.feature_dimension)

    def train(self):
        pass

def instantiate_model(model_path, class_name, model_cfg, feature_dimension) -> torch.nn.Module:
    """Dynamically import the network given from the config."""
    try:
        module = importlib.import_module(model_path)
        model = getattr(module, class_name)
    except (ImportError, AttributeError):
        raise NotImplementedError(
            f"In {model_path}.py, there should be a torch.nn.Module with class name that matches {class_name}."
        )
    return model(model_cfg, feature_dimension)