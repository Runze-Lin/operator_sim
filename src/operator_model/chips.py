"""Versioned chip scenarios. The online backend only reads portable JSON."""
from operator_model.resources import chip_file
from operator_model.operators.attention.vector_classes import VF_CHIP_FIELDS
from collections.abc import Mapping
from dataclasses import dataclass, field
from functools import lru_cache
import json
import math
from pathlib import Path
from types import MappingProxyType

REFERENCE_NAME = 'Ascend950DT_9582'
_BUILTINS = (REFERENCE_NAME, 'Ascend950PR_957b', 'Ascend970DT')
_BITS = {'32', '16', '8', '4'}
_DETAILS = {'ub_bytes', 'l1_bytes', 'l2_bytes', 'l0a_bytes', 'l0b_bytes', 'l0c_bytes',
            'l2_bandwidth_gbs', 'mte1_bandwidth_gbs', 'mte2_bandwidth_gbs', 'mte3_bandwidth_gbs',
            'rms_first_scalar_cycles', 'rms_first_payload_cycles', 'rms_output_retained_fraction',
            'cast_initial_read_pair_cycles',
            'fa_read_latency_cycles', 'fa_issue_cycles', 'fa_mte_bytes_per_cycle',
            'matmul_launch_cycles', 'matmul_read_latency_cycles', 'matmul_read_ii_cycles',
            'matmul_mmad_latency_cycles', 'matmul_mmad_ii_cycles', 'matmul_l0_latency_cycles',
            'matmul_drain_cycles', 'quant_launch_cycles', 'quant_arrival_log2_cycles',
            'quant_read_latency_cycles', 'quant_vf_dispatch_cycles', 'quant_vf_dependency_cycles',
            'quant_fragment_launch_cycles', 'quant_fragment_spread_cycles', 'quant_fragment_frontend_cycles',
            'quant_fragment_vector_cycles', 'quant_fragment_grain_ii_cycles', 'quant_fragment_fabric_ii_cycles',
            'quant_write_latency_cycles', 'quant_end_cycles', 'fa_aiv_per_aic',
            'quant_prefetch_launch_cycles', 'quant_loop_work_ops', 'quant_output_retained_fraction',
            'matmul_hbm_efficiency', 'matmul_l2_efficiency', 'matmul_output_retained_fraction',
            'matmul_initial_read_gbs'} | VF_CHIP_FIELDS


def _number(x, label, *, integer=False):
    if (type(x) not in (int, float) or not math.isfinite(x) or x <= 0 or
            (integer and type(x) is not int)):
        raise ValueError(f'{label} must be a positive finite ' + ('integer' if integer else 'number'))
    return x


@dataclass(frozen=True)
class ChipConfig:
    name: str
    frequency_ghz: float
    cube_cores: int
    vector_cores: int
    cube_tops: Mapping
    vector_tops: Mapping
    hbm_capacity_gb: float
    hbm_bandwidth_gbs: float
    details: Mapping
    # Provenance does not change the hardware cost of a dispatch.
    sources: Mapping = field(compare=False)
    inherited_fields: tuple = field(default=(), compare=False)

    # Scheduling policy outside the CA device window. Hardware equivalence
    # selects a device schedule even if a caller changes its dispatch cost.
    kernel_overhead_us: float = field(default=0.0, compare=False)

    def __post_init__(self):
        overhead = self.kernel_overhead_us
        if type(overhead) not in (int, float) or not math.isfinite(overhead) or overhead < 0:
            raise ValueError('kernel_overhead_us must be finite and nonnegative')
        if not isinstance(self.name, str) or not self.name.strip():
            raise ValueError('Chip name must be a non-empty string')
        for field in ('cube_cores', 'vector_cores', 'frequency_ghz', 'hbm_capacity_gb', 'hbm_bandwidth_gbs'):
            _number(getattr(self, field), field, integer=field.endswith('_cores'))
        for field in ('cube_tops', 'vector_tops'):
            rates = getattr(self, field)
            if not isinstance(rates, Mapping) or set(rates) != _BITS:
                raise ValueError(f'{field} requires keys 32, 16, 8, 4 (use null for unknown)')
            for key, rate in rates.items():
                if rate is not None:
                    _number(rate, field + '.' + key)
        if not isinstance(self.details, Mapping) or set(self.details) - _DETAILS:
            raise ValueError('Unknown chip detail field')
        for field, value in self.details.items():
            if value is not None:
                if field.endswith('_output_retained_fraction'):
                    if type(value) not in (int, float) or not 0 <= value <= 1:
                        raise ValueError(f'{field} must be in [0,1]')
                else:
                    _number(value, field, integer=field.endswith('_bytes') or field == 'fa_aiv_per_aic')
                if field.endswith('_efficiency') and value > 1:
                    raise ValueError(f'{field} must be in (0,1]')
                if field == 'fa_aiv_per_aic' and value > 2:
                    raise ValueError('FA supports at most two AIV per AIC')
        if not isinstance(self.sources, Mapping) or any(type(k) is not str or type(v) is not str
                                                        for k, v in self.sources.items()):
            raise ValueError('sources must map names to source descriptions')
        if not isinstance(self.inherited_fields, (tuple, list)) or any(
                k not in {'details.'+field for field in self.details} for k in self.inherited_fields):
            raise ValueError('Invalid inherited detail fields')
        for field in ('details', 'sources', 'cube_tops', 'vector_tops'):
            object.__setattr__(self, field, MappingProxyType(dict(getattr(self, field))))
        object.__setattr__(self, 'inherited_fields', tuple(self.inherited_fields))


def _parse(doc, fallback):
    if not isinstance(doc, Mapping):
        raise ValueError('Expected chip configuration object')
    data = dict(doc)
    if type(data.get('schema_version')) is not int or data.pop('schema_version') != 1:
        raise ValueError('Expected chip schema_version 1')
    details = data.pop('details', {})
    if not isinstance(details, Mapping):
        raise ValueError('Expected chip details object')
    inherited = {k: v for k, v in fallback.items() if k not in details}
    if 'inherited_fields' in data:
        raise ValueError('inherited_fields is generated by the chip loader')
    data.setdefault('sources', {})
    return ChipConfig(**data, details={**inherited, **details},
                      inherited_fields=tuple('details.'+k for k in sorted(inherited)))


@lru_cache(maxsize=len(_BUILTINS))
def _builtin(name):
    fallback = {} if name == REFERENCE_NAME else _builtin(REFERENCE_NAME).details
    return _parse(json.loads(chip_file(name).read_text(encoding='utf-8')), fallback)


def load_chip(spec=REFERENCE_NAME):
    """Load a registered SKU, JSON path or mapping. Reuse the result across events."""
    if isinstance(spec, ChipConfig):
        return spec
    if isinstance(spec, str) and spec in _BUILTINS:
        return _builtin(spec)
    if isinstance(spec, (str, Path)):
        path = Path(spec).expanduser()
        if path.suffix.lower() != '.json':
            raise ValueError(f'Unknown chip {spec}. Supply a chip JSON file or mapping.')
        spec = json.loads(path.read_text(encoding='utf-8'))
    return _parse(spec, _builtin(REFERENCE_NAME).details)
