from yacs.config import CfgNode
import os

from iopath.common.file_io import PathManagerFactory

pathmgr = PathManagerFactory.get()
_C = CfgNode()
cfg = _C

_C.FEATURE_DIM = 32

_C.DATA = CfgNode()
_C.DATA.HBN_PATH = 'HBN'
_C.DATA.MIPDB_PATH = 'MIPDB'
_C.DATA.SOONCHEONHYANG_PATH = 'sooncheonhyang'

_C.PRETRAIN = CfgNode()
_C.PRETRAIN.MAX_EPOCH = 30000
_C.PRETRAIN.SAVE_EPOCH = 5000
_C.PRETRAIN.BETA = 2.5
_C.PRETRAIN.GAMMA = 0.001
_C.PRETRAIN.DELTA = 25.
_C.PRETRAIN.NOISE_STD = 0.

_C.TRAIN = CfgNode()
_C.TRAIN.MAX_EPOCH = 20000
_C.TRAIN.SAVE_EPOCH = 2000
_C.TRAIN.BETA = 2.5
_C.TRAIN.GAMMA = 0.001
_C.TRAIN.DELTA = 25.
_C.TRAIN.NOISE_STD = 0.

_C.TEST = CfgNode()
_C.TEST.EPOCH = 20000

# ---------------------------------- Default config ---------------------------------- #
_CFG_DEFAULT = _C.clone()
_CFG_DEFAULT.freeze()


def assert_cfg():
    """Checks config values invariants."""
    assert True


def dump_cfg():
    """Dumps the config to the output directory."""
    cfg_file = os.path.join(_C.OUT_DIR, _C.CFG_DEST)
    with pathmgr.open(cfg_file, "w") as f:
        _C.dump(stream=f)
    return cfg_file


def load_cfg(cfg_file):
    """Loads config from specified file."""
    with pathmgr.open(cfg_file, "r") as f:
        _C.merge_from_other_cfg(_C.load_cfg(f))


def reset_cfg():
    """Reset config to initial state."""
    _C.merge_from_other_cfg(_CFG_DEFAULT)
