import pytest
from omegaconf import OmegaConf

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
