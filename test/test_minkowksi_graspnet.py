from fixtures import cfg
import pytest
from tsgrasp.net.minkowski_graspnet import MinkowskiGraspNet
from omegaconf import OmegaConf
from tsgrasp.training.trainer import instantiate_model

@pytest.fixture
def minkowski_graspnet(minkowski_graspnet_cfg):
    model = instantiate_model(
        "tsgrasp.net.minkowski_graspnet",
        "MinkowskiGraspNet",
        minkowski_graspnet_cfg,
        1
    )
    return model

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
    parallel_add_s: True
    """
    return OmegaConf.create(s)

def test_minkowski_graspnet_init(minkowski_graspnet):
    assert isinstance(minkowski_graspnet, MinkowskiGraspNet)