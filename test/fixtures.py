import pytest
from omegaconf import OmegaConf
from tsgrasp.data.acronymvid import AcronymVidDataset
from tsgrasp.training.trainer import instantiate_model
from tsgrasp.data.acronymvid import minkowski_collate_fn

from torch.utils.data import DataLoader

@pytest.fixture
def cfg():
    s = """
    model_path: tsgrasp.net.minkowski_graspnet
    model_name: MinkowskiGraspNet
    models:
        MinkowskiGraspNet:
            class: minkowski_graspnet.Minkowski_Baseline_Model
            conv_type: "SPARSE"
            model_name: "MinkUNet14A"
            D: 4
            backbone_out_dim: 128
            add_s_loss_coeff: 10
            bce_loss_coeff: 1
            points_per_frame: 1000
            grid_size: 0.005
            parallel_add_s: True
    data:
        feature_dimension: 1
    """
    return OmegaConf.create(s)

@pytest.fixture
def acronymvid_cfg():
    s = """
    dataroot : /home/tim/Research/tsgrasp/data/acronymvid
    points_per_frame: 1000
    grid_size: 0.05
    """
    return OmegaConf.create(s)

@pytest.fixture
def acronymvid_dataset(acronymvid_cfg):
    return AcronymVidDataset(acronymvid_cfg)

@pytest.fixture
def acronymvid_dataloader(acronymvid_dataset):
    return DataLoader(
        acronymvid_dataset,
        batch_size=2,
        collate_fn = minkowski_collate_fn,
    )

@pytest.fixture
def minkowski_graspnet_cfg():
    s = """
    class: minkowski_graspnet.Minkowski_Baseline_Model
    conv_type: "SPARSE"
    model_name: "MinkUNet14A"
    D: 4
    backbone_out_dim: 128
    add_s_loss_coeff: 10
    bce_loss_coeff: 1
    points_per_frame: 1000
    grid_size: 0.005
    use_parallel_add_s: True
    """
    return OmegaConf.create(s)

@pytest.fixture
def minkowski_graspnet(minkowski_graspnet_cfg):
    model = instantiate_model(
        "tsgrasp.net.minkowski_graspnet",
        "MinkowskiGraspNet",
        minkowski_graspnet_cfg,
        1
    )
    return model