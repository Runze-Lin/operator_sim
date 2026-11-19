"""Cold contiguous Cast candidate, exposed for prospective replay only."""
from operator_model import estimate


def query(shape, chip, read_factor, write_factor):
    estimate0=estimate('cast_bf16_fp32',shape,chip=chip)
    diag=estimate0.projection['timing_diagnostics']
    stages=diag['stages_us']
    plan=diag['target_plan']
    local=sum(value for key,value in stages.items() if key!='shared_io')
    waves=plan['busy_core_waves']
    stronger_ii=None
    if waves>1:
        # The two input/output slots are reusable independently.
        stronger_ii=max((read_factor*stages['read']+stages['vector'])/2,
                        (write_factor*stages['write']+stages['vector'])/2)
        extra=(plan['busy_core_elements']-plan['ub_tile_elements'])/plan['ub_tile_elements']
        local+=max(0,extra*stronger_ii-stages['steady_tiles'])
    return dict(baseline_us=estimate0.latency_us,candidate_us=max(local,estimate0.latency_us),
                active_aiv=plan['active_aiv'],waves=waves,slot_floor_us=stronger_ii,
                local_two_slot_us=local)
