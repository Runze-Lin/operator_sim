"""FA fill / initiation-interval / drain model for the evidenced Q9 D64 route.

Calibration consumes existing DB aggregates offline. Prediction is O(1) in KV
length and never replays instructions. Other routes retain frozen v0.4 behavior.
"""
from collections import OrderedDict
import json
from pathlib import Path

from .model import freeze, ceildiv
from .model_v4 import Case, Predictor as V4
from .model_v2 import plan


def supported(case):
    return case.op == 'fa' and case.shape[0] == 9 and case.shape[1] >= 4096 and case.shape[2:] == (64, 1, 1)


def schedule(length, costs):
    iterations = ceildiv(length, 256)
    # Initial K/V read streams are staggered. Their windows may overlap.
    read_fill = max(costs['first_k_latency_us'], costs['first_v_start_lag_us'] + costs['first_v_latency_us'])
    # Initiation intervals already include steady scheduling/backpressure.
    interval = max(costs['k_initiation_interval_us'], costs['v_initiation_interval_us'],
                   costs['cube_service_per_tile_us'], costs['vector_service_per_tile_us'],
                   costs['mte1_service_per_tile_us'], costs['fix_service_per_tile_us'])
    total = costs['prefix_us'] + read_fill + costs['compute_fill_us'] + (iterations-1)*interval + costs['drain_us']
    return {'predicted_us': total, 'iterations': iterations, 'prefix_us': costs['prefix_us'],
            'read_fill_us': read_fill, 'compute_fill_us': costs['compute_fill_us'],
            'steady_interval_us': interval, 'steady_total_us': (iterations-1)*interval,
            'drain_us': costs['drain_us'], 'tail_tokens': length % 256,
            'tail_policy': 'charge a full 256-token tile; independently check tail dispatch'}


class Predictor:
    def __init__(self, model, cache_size=4096):
        self.model = freeze(json.loads(Path(model).read_text()) if isinstance(model, (str, Path)) else model)
        if self.model.get('schema_version') != 5 or self.model['soc_version'] != 'Ascend950DT_9582':
            raise ValueError('Expected v0.5 950DT model')
        if cache_size < 0: raise ValueError('cache_size must be nonnegative')
        self.base = V4(self.model['base_v4_model'], cache_size=0)
        self.cache = OrderedDict(); self.cache_size = cache_size

    def predict(self, case, use_cache=True):
        if use_cache and case in self.cache:
            self.cache.move_to_end(case); return self.cache[case]
        if supported(case):
            p = plan(case); s = schedule(case.shape[1], self.model['fa_long']['costs'])
            output = freeze({'op': 'fa', 'shape': case.shape, 'soc_version': self.model['soc_version'],
                'model_version': '0.5.0', 'predicted_us': s['predicted_us'], 'plan': p, 'phase_schedule': s,
                'coverage': 'single_shape_calibrated_route; length extrapolation',
                'reference_scope': 'cross-core instruction makespan',
                'constant_baseline_us': self.model['base_v4_model']['constant_baselines']['fa'],
                'estimated_max_core_selected_opcode_count': 2*ceildiv(case.shape[1], 128),
                'instruction_count_coverage': 'structural_q64_d64',
                'expected_kernel_candidates': ('entry_kernel_2050000040140220',),
                'kernel_selection_verified': False, 'calibration_shape_count': 1,
                'ideal_compute_efficiency': None,
                'efficiency_unavailable_reason': 'Effective initiation interval includes scheduling; not independent ISA peak'})
        else:
            output = self.base.predict(case, use_cache=False)
        if use_cache and self.cache_size:
            self.cache[case] = output
            if len(self.cache) > self.cache_size: self.cache.popitem(last=False)
        return output

    def predict_us(self, case, use_cache=True):
        if use_cache: return self.predict(case)['predicted_us']
        if supported(case): return schedule(case.shape[1], self.model['fa_long']['costs'])['predicted_us']
        return self.base.predict_us(case, use_cache=False)

    def predict_many(self, cases): return [self.predict(c) for c in cases]

    def instruction_counts(self, case):
        return {'family': 'RV_* including load/store/control' if case.op == 'quant' else 'MMAD',
                'estimated_max_core_count': self.predict(case)['estimated_max_core_selected_opcode_count']}
