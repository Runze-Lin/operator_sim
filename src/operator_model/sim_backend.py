"""Optional prediction backend for the sim-workspace BackendRouter."""
from collections.abc import Mapping
from dataclasses import fields, replace
from functools import lru_cache
from hashlib import sha256
from importlib.resources import files
import json

from operator_performance_models import Estimate, PredictionDetails, UnsupportedOperator

from operator_model import __version__, load_chip
from operator_model.system_adapter import ADAPTER_VERSION, estimate_features


_TEMPLATES = {
    "950DT": "Ascend950DT_9582",
    "950PR": "Ascend950PR_957b",
    "970DT": "Ascend970DT",
}
_TEMPLATES.update({sku: sku for sku in tuple(_TEMPLATES.values())})
_DTYPES = {
    "bf16": "bfloat16", "fp16": "float16", "fp32": "float32",
    "fp8_e4m3fn": "float8_e4m3fn", "mxfp4": "float4_e2m1fn_x2",
}


def _target(name, capabilities, base):
    """Overlay the system's nominal resources on a SKU with known kernel geometry."""
    def peaks(resource):
        declared = dict(resource.tflops_by_bits)
        return {bits: declared.get(int(bits)) for bits in base.cube_tops}

    scenario = replace(
        base,
        cube_tops=peaks(capabilities.cube),
        vector_tops=peaks(capabilities.vec),
        hbm_capacity_gb=capabilities.hbm.capacity_gb,
        hbm_bandwidth_gbs=1000 * capabilities.hbm.bandwidth_tb_per_second,
    )
    if scenario != base:
        scenario = replace(scenario, name=f"{name}-system",
                           sources={**base.sources, "timing_template": base.name})
    return scenario


@lru_cache(maxsize=1)
def _resource_digest():
    digest = sha256()
    root = files("operator_model").joinpath("data")
    for group in ("chips", "parameters"):
        items = sorted(root.joinpath(group).iterdir(), key=lambda path: path.name)
        for path in items:
            if path.name.endswith(".json"):
                digest.update(path.name.encode() + b"\0" + path.read_bytes())
    return digest.hexdigest()


def _typed_work(values):
    return tuple(sorted((_DTYPES.get(dtype, dtype), work) for dtype, work in values.items()))


class OperatorSimBackend:
    """Return one complete device window; the host router owns fallback and DES."""

    name = "operator_sim"
    version = f"1/{__version__}/{ADAPTER_VERSION}"

    def __init__(self, *, chips=None):
        # User-supplied templates are resolved once. Include their effective
        # contents in the platform's persistent-cache identity, not their paths.
        self._templates = {name: load_chip(sku) for name, sku in _TEMPLATES.items()}
        overrides = {}
        for name, spec in (chips or {}).items():
            if not isinstance(name, str) or not name.strip():
                raise ValueError("Expected a nonempty system hardware name")
            template = load_chip(spec)
            self._templates[name] = template
            values = {f.name: getattr(template, f.name) for f in fields(template)}
            overrides[name] = {
                k: dict(v) if isinstance(v, Mapping) else v for k, v in values.items()
            }
        overrides_json = json.dumps(overrides, sort_keys=True, allow_nan=False).encode()
        self.configuration = (("resource_sha256", _resource_digest()),
                              ("template_sha256", sha256(overrides_json).hexdigest()))
        self._target_chip = lru_cache(maxsize=32)(self._resolve_chip)

    def _resolve_chip(self, name, capabilities):
        if name not in self._templates or capabilities is None:
            raise UnsupportedOperator(
                f"No operator-sim chip template or hardware capabilities for {name}"
            )
        return _target(name, capabilities, self._templates[name])

    def _estimate(self, features, target, hardware):
        return estimate_features(features, chip=target)

    def predict(self, features, hardware):
        target = self._target_chip(hardware.name, hardware.chip)
        result = self._estimate(features, target, hardware)
        if result.latency_us is None:
            raise UnsupportedOperator(result.reason or f"No qualified {features.raw_name} contract")
        work = result.work
        calibrated_target = target in (load_chip("Ascend950DT_9582"), load_chip("Ascend950PR_957b"))
        status = result.status if calibrated_target or result.status == "structural" else "extrapolated"
        details = PredictionDetails(
            model_name=result.contract,
            model_version=self.version,
            hardware_name=target.name,
            compute_bits=None,
            cube_flops=work["cube_ops"],
            cube_work_by_dtype=_typed_work(work["cube_ops_by_dtype"]),
            vec_flops=work["vec_ops"],
            vec_work_by_dtype=_typed_work(work["vec_ops_by_dtype"]),
            read_bytes=work["logical_read_bytes"],
            write_bytes=work["logical_write_bytes"],
            cube_us=None, cube_time_by_dtype=None,
            vec_us=None, vec_time_by_dtype=None,
            memory_us=None, overhead_us=None, cube_ratio=None, hbm_ratio=None,
            cube_peak_flops_per_second=None, cube_effective_flops_per_second=None,
            hbm_peak_bytes_per_second=hardware.chip.hbm.peak_bytes_per_second,
            hbm_effective_bytes_per_second=None,
            composition="complete device schedule",
            adaptor_name=self.name,
            adaptor_version=ADAPTER_VERSION,
            schema_version="resource-schedule-v1",
            model_parameters=tuple(sorted(
                (key, value if type(value) in (int,float,str) else json.dumps(value, sort_keys=True))
                for key,value in (work.get("schedule") or {}).items() if value is not None
            )) if self.name == "operator_sim_full" else (),
            communication_access=work.get("communication_access"), communication_us=work.get("communication_us"),
            assumptions=(
                status, f"SKU template: {self._templates[hardware.name].name}",
                "Clock, core counts, buffers and stage prices from the SKU template.",
                "Cube/Vector peaks and HBM capacity/bandwidth from system hardware.",
                "Zero device window for metadata." if result.status == "structural" else
                "Device schedule includes its modeled startup.",
                "Work counts useful operations and logical tensor bytes.",
                *((result.reason,) if result.reason else ()),
            ),
        )
        return Estimate(result.latency_us, details=details)


class FullOperatorSimBackend(OperatorSimBackend):
    """Rank-local device and interconnect schedules, including structural zero-time operations."""

    name = "operator_sim_full"
    prices_metadata = True

    def __init__(self, *, chips=None, fabric=None, memory_policy="cold_io"):
        from operator_model.system_routes.communication import FabricScenario
        super().__init__(chips=chips)
        self.fabric = fabric if fabric is not None else FabricScenario()
        if not isinstance(self.fabric, FabricScenario):
            raise TypeError("fabric must be a FabricScenario")
        if memory_policy not in ("cold_io", "ca_window"):
            raise ValueError("memory_policy must be cold_io or ca_window")
        self.memory_policy = memory_policy
        self.configuration += (("fabric_gbs", self.fabric.injection_gbs), ("fabric_step_us", self.fabric.step_us),
                               ("memory_policy", memory_policy))

    def _estimate(self, features, target, hardware):
        from operator_model.system_routes import estimate_full_features, communication
        if features.raw_name in communication.NAMES and hardware.topology is not None:
            raise UnsupportedOperator("Configured topology needs an explicit topology-aware communication backend")
        from operator_model.roofline import cold_io_window
        result = estimate_full_features(features,chip=target,fabric=self.fabric)
        if (self.memory_policy == "ca_window" or result.latency_us is None or
                result.status == "structural" or features.raw_name in communication.NAMES):
            return result
        io = cold_io_window(result.latency_us, result.work, target)
        schedule = {**(result.work.get("schedule") or {}), "memory_window":io}
        status = "extrapolated" if io["exposed_io_us"] > 0 else result.status
        return replace(result, status=status, latency_us=io["duration_us"], work={**result.work, "schedule":schedule})
