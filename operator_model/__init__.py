"""Per-device Ascend operator latency estimates for system simulators."""
from .chips import ChipConfig, load_chip
from .backend import Estimate, estimate
from .stream import OperatorStream
from .rms_history import RmsHistoryBand, estimate_rms_history_band
from .matmul_splitk import estimate_matmul_splitk
from .rmsnorm_pipeline import RmsGeometry, rmsnorm_pipeline
from .post_history import PostHistory, estimate_post_history
from .kimi_kda import plan_kimi_kda_prefill

__version__ = '0.29.0'
__all__ = ['ChipConfig', 'load_chip', 'Estimate', 'estimate', 'OperatorStream',
           'RmsHistoryBand', 'estimate_rms_history_band', 'plan_kimi_kda_prefill',
           'PostHistory', 'estimate_post_history', 'RmsGeometry', 'rmsnorm_pipeline', 'estimate_matmul_splitk']
