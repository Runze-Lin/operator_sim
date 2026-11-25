"""Work accounting for the rank-local DeepSeek candidate indexer capture."""


def describe(shape, attrs):
    if len(shape) != 7 or any(type(d) is not int or d < 1 for d in shape):
        raise ValueError('Expected positive B,Q,H,D,L,C,K for candidate_filtered_indexer')
    b, q, h, d, length, slots, top_k = shape
    rows = b * q
    policy = attrs.get('score_policy', 'full_scan')
    if set(attrs) - {'score_policy', 'block_size', 'scored_keys_per_query'}:
        raise ValueError('Unknown candidate_filtered_indexer attribute')

    # The integer input indexes blocks. Its width is not a count of unique keys.
    if policy == 'full_scan':
        if 'block_size' in attrs or 'scored_keys_per_query' in attrs:
            raise ValueError('block_size/scored_keys_per_query apply only to candidate_first')
        pairs = upper_pairs = rows * length
        key_read = 2 * b * length * d
        traffic_scope = 'Full key tensor read once per request, as in the captured system reference.'
    elif policy == 'candidate_first':
        span = attrs.get('block_size')
        if type(span) is not int or span < 1:
            raise ValueError('candidate_first requires positive integer block_size')
        limit = min(length, slots * span)
        upper_pairs = rows * limit
        counts = attrs.get('scored_keys_per_query')
        if counts is None:
            pairs = None
        elif type(counts) is int and 0 <= counts <= limit:
            pairs = rows * counts
        elif (isinstance(counts, (list, tuple)) and len(counts) == rows
              and all(type(x) is int and 0 <= x <= limit for x in counts)):
            pairs = sum(counts)
        else:
            raise ValueError('Expected a valid scored-key count per query within the block and table bounds')
        key_read = None if pairs is None else 2 * pairs * d
        traffic_scope = 'Logical key gathers per query; inter-query reuse and physical traffic are unknown.'
    else:
        raise ValueError('score_policy must be full_scan or candidate_first')

    fixed_read = 2 * rows * h * d + 2 * rows * h + 8 * rows + 4 * rows * slots
    output = 4 * rows * top_k
    work = {
        'dtype': 'bf16', 'cube_ops': None, 'vec_ops': None,
        'cube_ops_by_dtype': None, 'vec_ops_by_dtype': None,
        'score_dot_ops': None if pairs is None else 2 * h * d * pairs,
        'score_dot_ops_upper_bound': 2 * h * d * upper_pairs,
        'candidate_slots': rows * slots,
        'logical_read_bytes': None if key_read is None else fixed_read + key_read,
        'logical_write_bytes': output, 'physical_hbm_bytes': None,
        'key_table_capacity_bytes': 2 * b * length * d,
        'scope': traffic_scope,
        'notes': ['FMA counts as two. Device-unit assignment, head weighting, mask and top-k instruction costs are unknown.'],
    }
    return work, {
        'score_policy': policy, 'query_rows': rows, 'scored_query_key_pairs': pairs,
        'scored_query_key_pairs_upper_bound': upper_pairs,
        'input_shapes': [[rows, h, d], [b, length, d], [rows, h], [rows], [rows, slots]],
        'output_shape': [rows, top_k], 'timing_evidence': 'system shape capture only',
    }


# This describes a proposed BF16 implementation, not the name of an installed operator.
PIPELINE = 'li_pipeline_projection'


def _clipped_ramp(start, count, limit):
    """Sum min(start+i,limit) over i=1..count."""
    rising = min(count,max(0,limit-start))
    return rising*start+rising*(rising+1)//2+(count-rising)*limit


def _candidate_prefix(visible, slots, span, limit=None):
    """Sum retained counts for causal prefix lengths 1..visible.

    Before C*S every visible key fits. Afterwards the pinned partial block
    contributes 1..S keys per period, alongside C-1 complete blocks.
    """
    cap = slots*span
    dense = min(visible,cap)
    whole, tail = divmod(visible-dense,span)
    if limit is None:
        limit = cap
    return (_clipped_ramp(0,dense,limit) + whole*_clipped_ramp(cap-span,span,limit)
            + _clipped_ramp(cap-span,tail,limit))


def _tail_visits(lane, steps, groups, common, inverse):
    """Hits on the last group along lane+t*stride, independently of batch size."""
    delta = groups-1-lane
    if delta % common:
        return 0
    period = groups//common
    first = ((delta//common)*inverse) % period
    return (steps+period-1-first)//period


def _geometry(shape, attrs):
    if len(shape) != 7 or any(type(d) is not int or d < 1 for d in shape):
        raise ValueError('Expected positive B,Q,H,D,L,C,K for candidate_filtered_indexer')
    b, q, h, d, length, slots, k = shape
    if (attrs.get('implementation') != PIPELINE or
            set(attrs) - {'implementation', 'block_size', 'score_policy'}):
        raise ValueError('Select li_pipeline_projection with block_size and optional score_policy')
    policy, span = attrs.get('score_policy', 'full_scan'), attrs.get('block_size')
    if policy not in ('full_scan', 'candidate_first') or type(span) is not int or span < 1:
        raise ValueError('Expected full_scan or candidate_first and a positive block_size')
    if h not in (8, 32) or d != 128 or k != 512 or length < q or (q > 17 and policy != 'full_scan'):
        raise ValueError('Projection requires BF16 H=8/32, D=128, K=512 and Q<=L; candidate_first supports Q<=17')
    if q > 17:
        if length < k:
            raise ValueError('Long full_scan needs a key table with L>=K for a K-wide masked top-k')
        # A row near the start of the sequence may have fewer than K visible
        # keys. It still scores L positions and writes a K-wide padded output.
        pairs = _candidate_prefix(length,slots,span) - _candidate_prefix(length-q,slots,span)
        return policy,span,None,None,pairs
    # One document per batch row; target and draft both verify its FINAL Q consecutive positions.
    visible = tuple(range(length-q+1, length+1))
    # Distinct source-selected blocks with the newest block pinned: if the last
    # block is partial it displaces a full block. Identities are still unknown.
    kept = tuple(min(v, slots*span - (-v % span)) for v in visible)
    if min(kept) < k:
        raise ValueError('Fewer than top-k valid scores: padding/tie semantics need a separate contract')
    return policy, span, visible, kept, sum(kept)


def projected_work(shape, attrs):
    policy, span, visible, kept, kept_per_request = _geometry(shape, attrs)
    b, q, h, d, length, slots, k = shape
    rows = b*q
    pairs = rows*length if policy == 'full_scan' else b*kept_per_request
    cube, vec = 2*h*d*pairs, (3*h-1)*pairs
    read = 2*rows*h*(d+1) + 8*rows + 4*rows*slots
    read += 2*d*(b*length if policy == 'full_scan' else pairs)
    return {'dtype':'bf16', 'cube_ops':cube,'cube_ops_by_dtype':{'bf16':cube},
            'vec_ops':vec,'vec_ops_by_dtype':{'fp32':vec}, 'score_dot_ops':cube,
            'logical_read_bytes':read,'logical_write_bytes':4*rows*k,'physical_hbm_bytes':None,
            'key_table_capacity_bytes':2*b*length*d, 'candidate_slots':rows*slots,
            'notes':['Vector work counts ReLU, weight, and head reductions as FP32-equivalent; '
                     'integer mask, histogram and position selection are priced separately. '
                     'Interface reads do not measure HBM traffic.']}


def price(shape, chip, attrs, *, programs=None):
    """Query-tile resource schedule; only ordinary 950 LI supplied timing anchors.

    A persistent AIC with one or two AIVs owns a group. Request groups never
    cross document boundaries. Periodic round-robin allocation is independent
    of B, including when B is too large to enumerate all rows online.
    """
    from math import gcd
    from operator_model.operators.attention.indexer import _ceil, _prices, _sort_row
    policy, span, visible, kept, kept_per_request = _geometry(shape, attrs)
    work = projected_work(shape, attrs)
    b, q, h, d, length, slots, k = shape
    p = _prices()
    if not chip.cube_tops.get('16') or not chip.vector_tops.get('32'):
        raise ValueError('BF16 Cube and FP32-equivalent Vector rates are required')
    ghz = chip.frequency_ghz
    rc = chip.cube_tops['16']*1000/(ghz*chip.cube_cores)
    rv = chip.vector_tops['32']*1000/(ghz*chip.vector_cores)
    # One or two AIV partners per AIC, following the target's *own* CV ratio.
    partners = min(2, max(1, chip.vector_cores//chip.cube_cores))
    lanes = min(chip.cube_cores, chip.vector_cores//partners)
    # CANN 9.2.1 arch35 LI: s1BaseSize=4 for K<=2048 (K=512 here).
    # Compact rows have different key addresses and cannot share an MMAD.
    group_q = 4 if policy == 'full_scan' else 1
    groups = _ceil(q, group_q)
    # Steady 32KiB/416 cycle service is a grouped, aligned read proxy on DT.
    # Compact uses the same optimistic per-byte service plus per-block address work.
    dma = chip.details.get('mte2_bandwidth_gbs')
    page_service = 2*128*d*ghz/dma if dma else p['page_ii_cycles']
    stages = ('candidate_setup','first_mmad','page_intervals','last_mmad_reduce','sort_handoff',
              'histogram_and_store','position_order_and_map')
    prototypes = []
    key_local = 0
    long_scan = q > 17
    # Long full scans price a full query tile and (if present) a final
    # partial tile once, regardless of the size of Q or of the KV table.
    starts = (0, q-q%group_q) if long_scan and q%group_q else (0,)
    for j in (starts if long_scan else range(0,q,group_q)):
        nr = min(group_q,q-j)
        av = min(nr, partners)
        n = length if policy == 'full_scan' else kept[j]
        # ServiceCube aligns query-head rows to 2*G in L1 (including a
        # one-query tail); MmadParams then aligns the physical M to 16.
        m = 16*_ceil(2*h*_ceil(nr,2),16)
        pages, tail = _ceil(n,128), (n-1)%128+1
        key_local += n*d*2
        blocks = _ceil(length,span) if policy == 'full_scan' else 0
        # Full scan: clear block flags, set each selected flag, then apply one
        # membership and one causal predicate per key. Compact: one integer
        # multiply/add pair per slot to derive source addresses (no dense mask).
        setup = _ceil(nr,av)*(blocks+2*slots)/rv
        ops_per_score = 3*h-1+(2 if policy == 'full_scan' else 0)
        def vector_cycles(width):
            if programs: return programs.score(width,h,_ceil(nr,av),policy=='full_scan')
            return _ceil(nr,av)*width*ops_per_score/rv
        if programs:
            g=_ceil(blocks+2*slots,programs.fp32)
            setup=_ceil(nr,av)*programs.phase('candidate setup',dict(load=g,arithmetic=2*g,store=g),dict(load=1,arithmetic=2,store=1))
        interval = max(page_service, 2*m*128*d/rc, vector_cycles(128))
        # One AIV processes a row's histogram passes; different rows on its
        # AIC may run on separate partners. _sort_row accounts for 512 carry-in.
        aiv_load = [0.]*av
        for row in range(nr):
            sort, last_blocks, _ = _sort_row(n,p,p['vector_ops_cycle']/rv)
            if programs: sort=programs.sort_row(n)
            c0,c1 = p['store_drain']
            aiv_load[row%av] += sort + c0 + c1*last_blocks*p['vector_ops_cycle']/rv
        # Position order (ascending) is independent of score ranking. Exact
        # comparator count of a padded bitonic network. Three lane-equivalents
        # per compare/exchange (compare + two selects) is an uncalibrated proxy.
        levels = (k-1).bit_length(); swaps = (1<<levels)*levels*(levels+1)//4
        output_ops = k*(4 if policy == 'candidate_first' else 2) + 3*swaps
        prototypes.append((setup,p['first_issue_cycles'],(pages-1)*interval,
                           p['mmad_fixed_cycles']+2*m*tail*d/rc+vector_cycles(tail),
                           p['sort_handoff_cycles'],max(aiv_load),_ceil(nr,av)*(programs.sort_map(k,policy=='candidate_first') if programs else output_ops/rv)))
    ng = b*groups; live = min(lanes, ng); period = groups//gcd(groups,lanes)
    common = gcd(groups,lanes)
    inverse = pow(lanes//common,-1,period) if long_scan and period>1 else 0
    if long_scan:
        key_local = groups*length*d*2
    critical = None
    for lane in range(live):
        visits = 1+(ng-1-lane)//lanes
        if long_scan:
            tails = _tail_visits(lane,visits,groups,common,inverse) if q%group_q else 0
            totals = [(visits-tails)*full + tails*tail for full,tail in zip(prototypes[0],prototypes[-1])]
        else:
            rounds, rest = divmod(visits,period)
            totals = [0.]*len(stages)
            for t in range(min(visits,period)):
                weights = rounds + int(t<rest)
                for i, ticks in enumerate(prototypes[(lane+t*lanes)%groups]):
                    totals[i] += weights*ticks
        if critical is None or sum(totals)>sum(critical): critical = totals
    # Shared bandwidth can only hide behind the schedule up to its total local
    # time. Distinct tables are consumed per request. Reread when the concurrently
    # live dense requests do not fit L2; no cross-query gather reuse assumed.
    concurrent = min(b, _ceil(live+groups-gcd(lanes,groups),groups))
    unique_key = length*d*2
    reuse = (policy == 'full_scan' and concurrent*unique_key <= (chip.details.get('l2_bytes') or 0))
    hbm_keys = b*(unique_key if reuse else key_local)
    metadata = b*q*(2*h*(d+1)+8+4*slots+4*k)
    hbm = (hbm_keys+metadata)/(1000*chip.hbm_bandwidth_gbs)
    l2_bw = chip.details.get('l2_bandwidth_gbs')
    l2 = key_local*b/(1000*l2_bw) if l2_bw else 0.
    costs = {s:v/(1000*ghz) for s,v in zip(stages,critical)}
    local = sum(costs.values()); us = max(local,hbm,l2)
    costs['additional_shared_service'] = us-local
    # Keep decode's per-row diagnostics. Long scans expose a reproducible
    # arithmetic progression / block-period and tile histogram without Q objects.
    def cube_m(nr):
        return 16*_ceil(2*h*_ceil(nr,2),16)
    valid_output = (_candidate_prefix(length,slots,span,k)-_candidate_prefix(length-q,slots,span,k)
                    if long_scan else q*k)
    full_groups, tail_rows = divmod(q,group_q)
    tile_runs = [(cube_m(group_q),full_groups)] if full_groups else []
    if tail_rows:
        tile_runs.append((cube_m(tail_rows),1))
    shape_facts = dict(
        query_rows_per_request=q,
        visibility_first_last=(length-q+1,length),
        retained_count_rule='min(v, C*S - ((-v) mod S))',
        padded_output_slots=b*(q*k-valid_output),
        mmads_m_group_runs_per_request=tile_runs,
    )
    return us,work,{'timing_qualification':'extrapolated','implementation':PIPELINE,
        'reason':'Hypothetical BF16 LI-style fused schedule; candidate consumer has no qualified native 950 CA window.',
        'score_policy':policy,'block_size':span,'visible_keys_per_request':visible,
        'candidate_keys_per_request':kept,'candidate_key_pairs':b*kept_per_request,
        'scored_query_key_pairs':b*q*length if policy=='full_scan' else b*kept_per_request,
        'query_tile':group_q,'kv_page':128,'query_groups':ng,'waves':_ceil(ng,lanes),
        'mmad_m_per_request_group':None if long_scan else [cube_m(min(group_q,q-j)) for j in range(0,q,group_q)],
        **shape_facts,
        'stages_us':costs,'assumed_hbm_bytes':hbm_keys+metadata,'hbm_service_us':hbm,'l2_service_us':l2,
        'target_plan':{'active_aic':live,'active_aiv':sum(min(partners, min(group_q, q-(i%groups)*group_q)) for i in range(live))},
        'assumptions':['Uniform contiguous suffix, distinct blocks and latest visible block pinned.',
           'Kernel-local BF16 input/FP32-equivalent epilogue, on-chip intermediate scores, ascending output.',
           'DT ordinary LightningIndexer cycle anchors, projected to different H, B, L and candidate logic; '
           'not an installed binary, no collective or separate API launch charged.']}
