"""Common rates, tensor byte counts and wave scheduling for attention programs."""

from math import prod
from operator_model.system_adapter import NodeEstimate

BYTES = {
    "bfloat16": 2,
    "float32": 4,
    "float8_e4m3fn": 1,
    "float8_e8m0fnu": 1,
    "float4_e2m1fn_x2": 1,
    "int64": 8,
    "int32": 4,
    "uint8": 1,
}
DTYPE = {4: "mxfp4", 8: "fp8_e4m3fn", 16: "bf16"}


def divup(n, d):
    return (n + d - 1) // d


def nbytes(t):
    # Packed storage dimensions from SIMPlatform, not capture's separate logical_shape.
    return prod(t.shape) * BYTES[t.dtype.removeprefix("torch.")]


def _check(condition, reason):
    if not condition:
        raise ValueError(reason)


def _rate(tops, cores, clock):
    """One core's operations/cycle from a full chip peak in TOP/s."""
    return tops * 1000 / (cores * clock)


def _schedule(
    chip,
    name,
    cube,
    vec,
    bits,
    reads,
    writes,
    tiles,
    tile,
    pipeline,
    launch,
    evidence,
    *,
    cv=True,
    phases=None,
    hbm=None,
    status="extrapolated",
    active=None,
    vector_partners=None,
):
    clock = chip.frequency_ghz
    cores = chip.cube_cores if cv else chip.vector_cores
    available = min(cores, chip.vector_cores) if cv else cores
    live = min(available, tiles) if active is None else active
    waves = divup(tiles, live)
    total_bytes = reads + writes if hbm is None else hbm
    hbm_us = total_bytes / (chip.hbm_bandwidth_gbs * 1000)
    local = waves * pipeline / (clock * 1000)
    startup = launch / (clock * 1000)
    duration = chip.kernel_overhead_us + startup + max(local, hbm_us)
    work = {
        "cube_ops": cube,
        "vec_ops": vec,
        "cube_ops_by_dtype": {DTYPE[bits]: cube} if cube else {},
        "vec_ops_by_dtype": {"fp32": vec} if vec else {},
        "logical_read_bytes": reads,
        "logical_write_bytes": writes,
        "physical_hbm_bytes": None,
        "schedule": {
            "tile": tile,
            "tiles": tiles,
            "waves_on_critical_core": waves,
            "active_aic": live if cv else 0,
            "active_aiv": min(
                chip.vector_cores,
                live
                * (
                    vector_partners
                    if vector_partners is not None
                    else min(2, max(1, chip.vector_cores // chip.cube_cores))
                    if cv
                    else 1
                ),
            ),
            "dependency_path_cycles_per_wave": pipeline,
            "launch_us": startup,
            "additional_kernel_overhead_us": chip.kernel_overhead_us,
            "local_pipeline_us": local,
            "assumed_shared_hbm_us": hbm_us,
            "stage_budget": phases or {},
            "evidence": evidence,
        },
    }
    return NodeEstimate("system_attention/" + name, status, duration, evidence, work)
