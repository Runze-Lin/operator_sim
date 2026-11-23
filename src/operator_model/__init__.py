"""Per-device Ascend operator latency estimates for system simulators."""
from operator_model.chips import ChipConfig, load_chip
from operator_model.backend import Estimate, estimate
from operator_model.stream import OperatorStream
from operator_model.operators.rmsnorm.history import RmsHistoryBand, estimate_rms_history_band
from operator_model.operators.matmul.split_k import estimate_matmul_splitk
from operator_model.operators.matmul.supply import estimate_matmul_supply
from operator_model.operators.rmsnorm.pipeline import RmsGeometry, rmsnorm_pipeline
from operator_model.operators.mhc.history import PostHistory, estimate_post_history
from operator_model.operators.kda.prefill import plan_kimi_kda_prefill
from operator_model.operators.kda.recurrent_chains import estimate_recurrent_chains

__version__ = '1.2.0'
__all__ = ['ChipConfig', 'load_chip', 'Estimate', 'estimate', 'OperatorStream',
           'RmsHistoryBand', 'estimate_rms_history_band', 'plan_kimi_kda_prefill',
           'PostHistory', 'estimate_post_history', 'RmsGeometry', 'rmsnorm_pipeline',
           'estimate_matmul_splitk', 'estimate_matmul_supply', 'estimate_recurrent_chains']
