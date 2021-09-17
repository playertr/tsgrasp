import pytest
from omegaconf import OmegaConf
from tsgrasp.data.acronymvid import AcronymVidDataset
from tsgrasp.net.minkowski_graspnet import MinkowskiGraspNet
from tsgrasp.data.acronymvid import minkowski_collate_fn

from torch.utils.data import DataLoader

@pytest.fixture
def cfg():
    s = """
    model:
        model_name: MinkowskiGraspNet
        backbone_model_name: "MinkUNet14A"
        D: 4
        backbone_out_dim: 128
        add_s_loss_coeff: 10
        bce_loss_coeff: 1
        points_per_frame: 1000
        grid_size: 0.005
        use_parallel_add_s: True
        feature_dimension: 1

        pl:
            module_path: tsgrasp.net.lit_minkowski_graspnet
            module_name: LitMinkowskiGraspNet
    data:
        dataroot: /home/tim/Research/tsgrasp/data/acronymvid
        points_per_frame: 1000
        grid_size: 0.05
        num_workers: 4

        # Lightning trainer config
        pl:
            datamodule_path: tsgrasp.data.lit_acronymvid
            datamodule_name: LitAcronymvidDataset
    training:
        batch_size: 2
        data_len: 50
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
        collate_fn=minkowski_collate_fn,
    )

@pytest.fixture
def minkowski_graspnet_cfg():
    s = """
    model_name: MinkowskiGraspNet
    backbone_model_name: "MinkUNet14A"
    D: 4
    backbone_out_dim: 128
    add_s_loss_coeff: 10
    bce_loss_coeff: 1
    points_per_frame: 1000
    grid_size: 0.005
    use_parallel_add_s: True
    feature_dimension: 1
    """
    return OmegaConf.create(s)

@pytest.fixture
def minkowski_graspnet(minkowski_graspnet_cfg):
    model = MinkowskiGraspNet(minkowski_graspnet_cfg)
    return model