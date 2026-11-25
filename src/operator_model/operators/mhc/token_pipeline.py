"""Per-row pipeline candidate for BF16 aclnnMhcPost on the token-partitioned path."""
from math import ceil, log2


def _native_price(chip, name, fallback):
    bandwidth = chip.details.get(name + '_bandwidth_gbs')
    return 1024 * chip.frequency_ghz / bandwidth if bandwidth else fallback


def token_pipeline(shape, chip, plan, state, model, shared_floor_us=0., inspect_rows=(), *, vf_price=None):
    """Cost a named dispatch state. An optional bounded list exposes actual row predictions."""
    tokens, streams, width = shape
    if plan['route'] != 'post_token' or streams != 4 or state not in ('first', 'expand', 'reuse', 'repeat'):
        raise ValueError('Expected the four-stream token kernel and a known launch state')
    groups = ceil(width / 64)
    # Dense FP32 vector operations are only a service floor; DMA/dispatch can dominate.
    cycles_per_us = 1000 * chip.frequency_ghz
    vector_per_cycle = chip.vector_tops['32'] * 1000 / (chip.vector_cores * chip.frequency_ghz)
    vector_ratio = (27000 / (64 * 1.65)) / vector_per_cycle
    read_price = _native_price(chip, 'mte2', model['read_cycles_per_kib'])
    write_price = _native_price(chip, 'mte3', model['write_cycles_per_kib'])
    phases = model['by_state'][state]
    size = [64, 16, 2 * width, 8 * width]
    out_bytes = 8 * width
    body = (vf_price('post_mix',1,width,streams) if vf_price else model['ready_to_write_per_group'] * groups * vector_ratio)
    work = max(body, 0 if vf_price else 36 * width / vector_per_cycle, sum(size) * read_price / 1024, out_bytes * write_price / 1024)
    arrival = phases['arrival_per_log2_aiv'] * log2(plan['active_aiv'])
    first_read = arrival + phases['control_to_operand']
    wr_service = out_bytes * write_price / 1024
    response = {}
    for label in ('first', 'steady'):
        p = phases['ports'][label]
        issue = [0.]
        for gap in p['gaps']:
            issue.append(issue[-1] + gap)
        latency = [f + read_price*b/1024 for f,b in zip(p['fixed'], size)]
        ready = max(a+l for a,l in zip(issue, latency))
        after = p['ready_to_write_fixed'] + body
        response[label] = dict(issue=issue, latency=latency, ready=ready, to_write=ready+after)
    first_write = first_read + response['first']['to_write']
    gaps = {key:max(work, phases[key]) for key in ('read_gap_first','read_ii','write_gap_first','write_ii')}
    second_read = first_read + gaps['read_gap_first']
    second_write = max(first_write + gaps['write_gap_first'], second_read + response['steady']['to_write'])

    def times(i):
        if i == 0:
            return first_read, first_write, response['first']
        row_read = second_read + (i-1) * gaps['read_ii']
        row_write = max(second_write + (i-1) * gaps['write_ii'], row_read + response['steady']['to_write'])
        return row_read, row_write, response['steady']

    count = plan['rows_per_core']
    _, last_write, _ = times(count-1)
    write_response = phases['write_fixed'] + wr_service
    total_cycles = last_write + write_response + phases['exit_after_write']
    total = max(total_cycles/cycles_per_us, shared_floor_us)
    stages = dict(arrival=arrival, control_to_first_read=first_read-arrival,
        first_read_to_ready=response['first']['ready'],
        ready_to_first_write=response['first']['to_write']-response['first']['ready'],
        overlapping_rows=last_write-first_write, last_write_response=write_response,
        exit_after_write=phases['exit_after_write'])
    stages = {k:x/cycles_per_us for k,x in stages.items()}
    stages['shared_link_floor'] = max(0., total-total_cycles/cycles_per_us)
    trace = []
    for i in inspect_rows:
        if not 0 <= i < count:
            raise ValueError('row outside one longest AIV')
        start, end, p = times(i)
        trace.append(dict(row=i, reads_us=[[(start+a)/cycles_per_us,(start+a+b)/cycles_per_us] for a,b in zip(p['issue'],p['latency'])],
                          write_us=[end/cycles_per_us,(end+write_response)/cycles_per_us]))
    return dict(us=total, stages_us=stages, active_aiv=plan['active_aiv'], rows_per_core=count,
                state=state, row_work_floor_cycles=work, gap_cycles=gaps, timeline=trace)
