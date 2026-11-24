"""Resource schedules for rank-local operator contracts in the decode captures."""
from operator_model.chips import load_chip
from operator_model.system_adapter import NodeEstimate, estimate_features

from . import communication

VIEWS = frozenset({
    "aten.view.default", "aten.unsqueeze.default", "aten.expand.default", "aten.select.int",
    "aten.alias.default", "aten.transpose.int", "aten.slice.Tensor", "aten.permute.default",
    "aten._unsafe_view.default", "aten.squeeze.dim", "aten.t.default", "aten.split_with_sizes.default",
})
STRUCTURAL = VIEWS | {"aten.empty.memory_format"}


def estimate_full_features(features, *, chip, fabric=None):
    """Price a complete operator or return an explicit unsupported result for a new contract."""
    from . import attention, cube, vector
    chip = load_chip(chip)
    name = features.raw_name
    if name in STRUCTURAL:
        kind = "allocator_reservation" if name == "aten.empty.memory_format" else "strided_view"
        return NodeEstimate(kind, "structural", 0., "Metadata only in the captured device graph.",
                            dict(cube_ops=0, cube_ops_by_dtype={}, vec_ops=0, vec_ops_by_dtype={},
                                 logical_read_bytes=0, logical_write_bytes=0))
    if name in communication.NAMES:
        return communication.price(features, chip, fabric or communication.FabricScenario())
    strict = estimate_features(features, chip=chip)
    if strict.latency_us is not None:
        return strict
    for family in (attention, cube, vector):
        if name in family.NAMES:
            devices = {t.device for t in (*features.inputs, *features.outputs)}
            if len(devices) != 1 or any(d.split(":")[0] not in ("meta", "npu", "privateuseone") for d in devices):
                return NodeEstimate(None,"unsupported",reason="Compute schedule requires rank-local device tensors")
            try:
                return family.price(features, chip)
            except (KeyError, ValueError, IndexError, ZeroDivisionError) as error:
                return NodeEstimate(None,"unsupported",reason=f"{name}: incompatible capture or chip resource ({error})")
    return strict
