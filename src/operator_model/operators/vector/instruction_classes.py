"""Register-class service for three contiguous arch35 SIMD programs, per active AIV."""
from math import ceil


def select_class_model(attrs):
    """Consume the optional public selector from an already private attrs copy."""
    mode = attrs.pop('vector_model', 'auto')
    if mode not in ('auto', 'calibrated', 'instruction_classes'):
        raise ValueError('vector_model must be auto, calibrated or instruction_classes')
    return mode == 'instruction_classes'


def tile_service(program, elements, chip, *, rows=1):
    """Return per-tile cycles and counts. A 256 B register holds 64 FP32 lanes."""
    width = elements // rows
    groups = rows * ceil(width / 64)
    cfg = chip.details
    if program == 'cast':
        counts = dict(predicate=groups, load=groups, convert=groups, store=groups)
    elif program == 'sigmoid':
        counts = dict(predicate=groups, load=groups, arithmetic=2*groups,
                      exp=groups, divide=groups, store=groups)
    elif program == 'token_quant':
        # First pass widens/absolutes/maxes; the second widens/divides/narrows.
        # Scale reduction, multiplication and scale store are paid once per row.
        counts = dict(predicate=2*groups, load=2*groups, arithmetic=2*groups+rows,
                      reduce=rows, convert=3*groups, divide=groups, store=groups+rows)
    else:
        raise ValueError('Unknown contiguous SIMD program')
    widths = cfg.get('vf_issue_width') or 2
    rates = {'exp':4, 'divide':4}
    service = {key: n * (cfg.get('vf_' + key + '_ii_cycles') or rates.get(key, 1))
               for key, n in counts.items()}
    # Predication and load/ALU can pair in this ideal design. A single store
    # initiates per cycle. Shared register and ALU ports can become limiting.
    bounds = {key: demand for key, demand in service.items()}
    bounds.update(issue=sum(counts.values())/widths,
                  ub=(service['load']+service['store'])/(cfg.get('vf_ub_ports') or 2),
                  arithmetic_cluster=sum(v for k,v in service.items()
                                         if k not in ('load','store','predicate'))/(cfg.get('vf_alu_lanes') or 2))
    return max(bounds.values()), dict(groups=groups, counts=counts, bounds_cycles=bounds)
