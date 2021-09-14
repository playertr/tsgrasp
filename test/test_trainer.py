from tsgrasp.training.trainer import Trainer
from fixtures import cfg

def test_trainer_construction(cfg):
    trainer = Trainer(cfg)
    assert isinstance(trainer,  Trainer)