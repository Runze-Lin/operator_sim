"""Bounded L1 producer/consumer model for the V4 ND weight prologue. Units are cycles."""


# Observed 950 ND VF loop and port prices (cycles). See docs/examples/mx-handoff.md.
VF_ROW_CYCLES = 11
VF_REPEAT_CYCLES = 7.5
VF_REGISTER_BYTES = 256
WARM_READ_RESIDUAL = 150
FULL_OCCUPANCY_ISSUE_SPREAD = 2680


def _div(x, y):
    return (x+y-1)//y


def _periods(state, transition, count):
    """Apply an identical whole-ring transition count times in the max-plus algebra."""
    if count <= 8:
        for _ in range(count):
            state = transition(state)
        return state
    # All state updates use max or addition of fixed service costs. Applying a
    # period to each basis vector therefore gives its exact transition matrix.
    n = len(state)
    neg_inf = float('-inf')
    columns = []
    for i in range(n):
        basis = [neg_inf]*n
        basis[i] = 0.0
        columns.append(transition(basis))
    rows = list(zip(*columns))
    while count:
        if count & 1:
            state = [max(x+y for x,y in zip(row,state)) for row in rows]
        count //= 2
        if count:
            columns = tuple(zip(*rows))
            rows = [[max(x+y for x,y in zip(row,col)) for col in columns] for row in rows]
    return state


def vf_service(width, depth, vector_scale):
    return width*max(VF_ROW_CYCLES,VF_REPEAT_CYCLES*_div(depth,2*VF_REGISTER_BYTES))*vector_scale


def _run_tile(height, width, k, plan, rates, fast):
    page_k = plan['l1_k']
    base_k = plan['base_tile'][2]
    slots = plan['l1_slots']
    pages = _div(k,page_k)
    # Two buffers alternate complete pages. Four buffers concurrently split N
    # over the cooperating AIVs; charge the larger subtile on a 1:2 target.
    lanes = rates['aiv'] if slots == 2 else 1
    producer_n = width
    if slots == 4 and rates['aiv'] == 2:
        producer_n = max(min(width,plan['ub_n']),width-plan['ub_n'])

    def service(depth):
        scales = 2*_div(depth,64)
        a_bytes = height*(depth+scales)+width*scales
        w_bytes = producer_n*_div(depth,2)
        per_byte = max(1/rates['mte2'],rates['shared_per_byte'])
        # The VF row loop and its inner 256-B packed register repeats have
        # independent issue constraints. Both prices describe the 950 ISA.
        vf = rates['setup'] + vf_service(producer_n,depth,rates['vector_scale'])
        hand = rates['handoff_lat'] + producer_n*depth/rates['mte3']
        def fragment(d):
            ld = (height+width)*(d+2*_div(d,64))/rates['mte1']
            cube = 2*(_div(height,16)*16)*(_div(width,16)*16)*(_div(d,64)*64)/rates['cube']
            return ld, cube
        ld, compute = fragment(min(depth,base_k))
        last_ld, last_compute = fragment((depth-1)%base_k+1)
        return (a_bytes*per_byte,w_bytes*per_byte,vf,hand,ld,last_compute,
                _div(depth,base_k),max(ld,compute,rates['mmad_ii']),
                max(last_ld,last_compute,rates['mmad_ii']))

    # State entries are release times for A-read issue, next MMAD issue,
    # each L1 slot, and each lane's GM issue, Vector and MTE3 issue.
    def step(state, index, cost, a_latency, first=None):
        a_read,w_read,vf,hand,ld,last_compute,fragments,interval,last_interval = cost
        new = list(state)
        slot = 2+index%slots
        lane = 2+slots+3*(index%lanes)
        a_issue = max(state[0],state[slot])
        new[0] = a_issue+a_read
        a_ready = new[0]+a_latency
        new[lane] = state[lane]+w_read
        vec_ready = max(state[lane+1],new[lane]+rates['read_lat'])+vf
        new[lane+1] = vec_ready
        hand_ready = max(state[slot],state[lane+2],vec_ready)+hand
        new[lane+2] = hand_ready
        ready = max(a_ready,hand_ready)
        issue = max(state[1],ready+rates['load_lat']+ld)
        last_issue = issue+(fragments-1)*interval
        new[1] = last_issue+last_interval
        new[slot] = last_issue
        if first is not None:
            first.update(read_fill=max(0,ready-vf-hand),inline_unpack=vf,ub_to_l1=hand,l1_to_l0=issue-ready)
        return new

    full = service(min(k,page_k))
    tail = service((k-1)%page_k+1)
    state = [0.0]*(2+slots+3*lanes)
    fill = {}
    warm = min(WARM_READ_RESIDUAL,rates['read_lat'])
    # Peel the first ring (cold A reads), preserve the final page even if full.
    head = min(slots,pages-1)
    for i in range(head):
        state = step(state,i,full,rates['read_lat'],fill if i==0 else None)
    repeats, residual = divmod(pages-1-head,slots)
    def ring(s):
        for i in range(slots):
            s = step(s,i,full,warm)
        return s
    if fast:
        state = _periods(state,ring,repeats)
    else:
        for _ in range(repeats):
            state = ring(state)
    for i in range(residual):
        state = step(state,i,full,warm)
    idx = pages-1
    state = step(state,idx,tail,rates['read_lat'] if idx<slots else warm,fill if idx==0 else None)
    first_issue = sum(fill.values())
    last_issue = state[2+idx%slots]
    tail_compute = tail[5]+rates['cube_lat']
    fill.update(page_intervals=max(0,last_issue-first_issue),cube_tail=tail_compute)
    return fill, last_issue+tail_compute, state[1]-first_issue


def schedule(k, h, w, waves, plan, rates, *, fast=True):
    stages, _, steady = _run_tile(h,w,k,plan,rates,fast)
    stages['additional_waves'] = (waves-1)*steady
    if waves>1 and plan['tail_split'] != [1,1]:
        sm,sn = plan['tail_split']
        th = min(h, _div(_div(plan['base_tile'][0],sm),16)*16)
        tw = min(w, _div(_div(plan['base_tile'][1],sn),16)*16)
        _,tail,_ = _run_tile(th,tw,k,plan,rates,fast)
        boundary = stages['l1_to_l0']+stages['cube_tail']
        stages['additional_waves'] = (waves-2)*steady+max(boundary,tail-rates['read_lat'])
    return stages
