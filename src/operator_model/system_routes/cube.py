"""Rank-local dense, grouped head and block-quantized batch products.

The explicit layouts are the capture contract.  The custom products select an
implementable design; none claims that a capture identified a binary.
"""
from math import ceil

from . import vf
from operator_model.system_adapter import (
    BF16, E8, F32, FP8, NodeEstimate, _Mismatch, _arity, _dtype, _fields, _layout,
    _packed, _require, _shape,
)
from operator_model.workload import _counts

NAMES = frozenset({
    'aten.mm.default', 'aten.addmm.default', 'npu::npu_quant_matmul',
    'npu::npu_grouped_matmul', 'inference::mixed_precision_matmul',
    'inference::matmul_rhs_transposed', 'inference::quantized_batched_matmul',
})
_TRANS = 'inference::matmul_rhs_transposed'
_BATCH = 'inference::quantized_batched_matmul'
_NATIVES = ('aten.mm.default', 'aten.addmm.default')


def _align(x, width):
    return ((x + width - 1) // width) * width


def _dense(f, sem):
    trans = f.raw_name == _TRANS
    bias = f.raw_name == 'aten.addmm.default'
    _arity(f, 3 if bias else 2, 1)
    a, b = f.inputs[-2:]
    m, k = _shape(a.shape, 2)
    n = _shape(b.shape, 2)[0 if trans else 1]
    dtype = _dtype(a)
    _require(dtype in (BF16, F32), 'Expected BF16 or strict FP32 dense product')
    row_pitch = a.stride[0] if len(a.stride) == 2 else 0
    _layout(a, dtype, (m, k), strides=(row_pitch, 1) if not trans and row_pitch >= k else None)
    _layout(b, dtype, (n, k) if trans else (k, n), strides=None if trans else (1, k))
    output = _dtype(f.outputs[0])
    if trans:
        _require(dtype == BF16 and output == F32 and sem.get('accumulator_dtype') == F32
                 and sem.get('output_dtype') == F32, 'Expected a BF16 router with FP32 output')
    else:
        _require(output == dtype, 'Expected dense output to have the operand dtype')
    if bias:
        _layout(f.inputs[0], dtype, (n,))
        kw = dict(f.attributes).get('kwargs', ('mapping', ()))
        _require(kw == ('mapping', ()) or (isinstance(kw, tuple) and len(kw) == 2 and
                 kw[0] == 'mapping' and all(name in ('alpha','beta') and value == 1
                                           for name, value in kw[1])), 'Expected unit alpha and beta')
    _layout(f.outputs[0], output, (m, n))
    width = 4 if dtype == F32 else 2
    return ((m,), n, k, 'dense32' if dtype == F32 else ('router16' if trans else 'dense16'),
            (m*k + n*k + n*int(bias))*width, m*n*(4 if output == F32 else 2),
            {'bias_add': m*n} if bias else {})


def _mixed(f, sem):
    _arity(f, 4, 1)
    grouped = f.raw_name == 'npu::npu_grouped_matmul'
    _require(sem.get('activation_precision') == 'fp8' and sem.get('weight_precision') == 'fp4',
             'Expected token-FP8 activation and MXFP4 weight')
    roles = _fields(dict(f.attributes).get('tensor_semantics', ())).get('inputs', ())
    named_roles = {record.get('index'):record.get('role') for record in map(_fields, roles)}
    expected_roles = ('activation','weight','weight_scale','activation_scale')
    _require(named_roles == dict(enumerate(expected_roles)), 'Missing indexed activation/weight/scale roles')
    _require(sem.get('group_size', 32) == 32 and sem.get('padding', (0,0)) == (0,0)
             and sem.get('accumulator_dtype', F32) == F32
             and sem.get('activation_scale_layout','token_fp32') == 'token_fp32',
             'Mismatched scales, roles, padding or accumulation')
    m, k = _shape(sem.get('logical_input_shape'), 2)
    weight = _shape(sem.get('logical_weight_shape'), 3 if grouped else 2)
    prefix, n = weight[:-2], weight[-2]
    _require(weight[-1] == k, 'Weight reduction dimension mismatch')
    if grouped:
        counts = sem.get('group_counts')
        _require(sem.get('gemm_grouping') == 'output_head_groups' and isinstance(counts,tuple)
                 and len(counts) == prefix[0] and all(type(c) is int and c>=0 for c in counts)
                 and sum(counts) == m, 'Missing output-head group histogram')
    else:
        counts = (m,)
    groups = sum(c > 0 for c in counts)
    pages = ceil(k/64)
    _layout(f.inputs[0], FP8, (m,k))
    _packed(f, 'inputs', 1, (*prefix,k,n), axis=-2,
            strides=(*((n*(k//2),) if grouped else ()),1,k//2))
    _layout(f.inputs[2], E8, (*prefix,pages,n,2),
            strides=(*((2*n*pages,) if grouped else ()),2,2*pages,1))
    _layout(f.inputs[3], F32, (m,))
    _layout(f.outputs[0], BF16, (m,n))
    reads = m*k + groups*n*(k//2+2*pages) + 4*m
    return counts,n,k,'token_mx',reads,2*m*n,{'unpack_weight': groups*n*k,
                                               'token_rescale': m*n, 'output_cast': m*n}


def _batched(f, sem):
    _arity(f, 4, 1)
    _require(sem.get('group_size') == 32 and sem.get('compute_precision') == 'fp4'
             and sem.get('input_layout') == 'values_then_corresponding_scales'
             and sem.get('accumulator_dtype') == F32, 'Expected two last-axis MXFP4 tensors')
    shapes = sem.get('logical_input_shapes')
    _require(isinstance(shapes, tuple) and len(shapes) == 2, 'Missing packed bmm logical shapes')
    g,m,k = _shape(shapes[0], 3)
    g2,k2,n = _shape(shapes[1], 3)
    _require((g,k) == (g2,k2), 'Invalid batch or reduction dimension')
    for index, dims in enumerate(shapes):
        _packed(f, 'inputs', index, dims)
        _layout(f.inputs[index+2], E8, (*dims[:-1],ceil(dims[-1]/64),2))
    _layout(f.outputs[0], BF16, (g,m,n))
    elements = g*k*(m+n)
    reads = elements//2 + 2*g*(m*ceil(k/64) + k*ceil(n/64))
    return (m,)*g,n,k,'batch_mx',reads,2*g*m*n,{'unpack': elements,'scale_bf16': elements}


def _vector_work(mode, classes):
    if mode == 'token_mx':
        return {'fp8_e4m3fn': classes['unpack_weight'],
                'fp32': classes['token_rescale'] + classes['output_cast']}
    dtype = 'bf16' if mode == 'batch_mx' else 'fp32'
    return {dtype: sum(classes.values())} if classes else {}


def _schedule(rows, n, k, mode, reads, writes, classes, chip, add_bias):
    rows = tuple(c for c in rows if c)
    groups, total = len(rows), sum(rows)
    f = chip.frequency_ghz
    cores, vec = chip.cube_cores, chip.vector_cores
    bits = '32' if mode == 'dense32' else ('8' if mode == 'token_mx' else '16')
    rates = chip.cube_tops
    rate16 = (rates['16'] or 432) * 1000 / (cores*f)
    if mode == 'dense32':
        # Observed strict FP32 16x192 K32 MMADs take 409 cycles.  At 12 cycles
        # per K plus a 25-cycle issue shell this gives 512 aligned ops/cycle.
        rate = 512 * rate16 / (432000 / (32*1.65))
        if rates['32']:
            rate = min(rate, rates['32']*1000/(cores*f))
    else:
        rate = (rates[bits]*1000/(cores*f)) if rates[bits] else rate16*16/int(bits)
    ub = chip.details.get('ub_bytes') or 253952
    l1 = chip.details.get('l1_bytes') or 524288
    l0a = chip.details.get('l0a_bytes') or 65536
    l0b = chip.details.get('l0b_bytes') or 65536
    l0c = chip.details.get('l0c_bytes') or 262144
    bm = min(128,_align(max(rows),16))
    if mode in ('dense32','router16'):
        bn = min(256,_align(ceil(n/ceil(n/256)),16))
    else:
        mtiles = sum(ceil(r/bm) for r in rows)
        bn = min(128,max(16,_align(ceil(n/max(1,cores//mtiles)),16)))
    bm = min(bm, max(16, (l0c//(4*bn*16))*16))
    jobs = sum(ceil(r/bm) for r in rows)*ceil(n/bn)
    element = int(bits)//8
    k_align = 16 if mode == 'dense32' else 32
    limits = [l0a//(2*bm*element),l0b//(2*bn*element),l1//(2*(bm+bn)*element)]
    if mode in ('batch_mx','token_mx'):
        limits.append(ub//(4*(bm+bn)*element))
    page_k = min(k,max(k_align,(min(limits)//k_align)*k_align))
    split = min(max(1, cores//jobs),max(1,k//k_align)) if mode in ('dense32','router16') else 1
    active = min(cores,jobs*split)
    waves = ceil(jobs*split/cores)
    local_k = ceil(k/split)
    pages = ceil(local_k/page_k)
    core_k_cost = max(3.09 if mode == 'dense32' else 0, 2*bm*bn/rate)
    cube = waves*(core_k_cost*local_k + pages*(25 if mode == 'dense32' else 32))
    # One input footprint fetched cold from HBM. Redundant panels can be
    # served on chip; use an additional bound on traffic into each AIC.
    hbm = (reads+writes)*f/min(chip.hbm_bandwidth_gbs,active*128*f)
    panel_bytes = waves*(min(max(rows),bm)+min(n,bn))*local_k*element
    panel_cycles = panel_bytes/min(128, (chip.details.get('mte2_bandwidth_gbs') or 128*f)/f)
    vactive = min(vec, max(1,ceil(active*vec/cores)))
    issued = _vector_work(mode, classes)
    if mode == 'token_mx':
        # Distinct output M tiles may repeat unpacking of the W panel in UB.
        issued['fp8_e4m3fn'] += (sum(ceil(r/bm) for r in rows)-groups)*n*k
    if mode == 'batch_mx':
        repeats = (ceil(n/bn)-1)*total*k + (sum(ceil(r/bm) for r in rows)-groups)*n*k
        issued['bf16'] += 2*repeats
    vector_cycles = {}
    for precision, count in issued.items():
        width = {'fp8_e4m3fn':'8', 'bf16':'16', 'fp32':'32'}[precision]
        peak = chip.vector_tops.get(width)
        _require(peak, f'Missing {width}-bit Vector peak for {precision} service')
        vector_cycles[precision] = count * vec*f / (peak*1000*vactive)
    for precision, old in tuple(vector_cycles.items()):
        count=issued[precision]/vactive
        if mode=='token_mx' and precision=='fp8_e4m3fn':
            recipe=dict(load=count,permute=count,convert=count,store=count)
        elif mode=='batch_mx':
            recipe=dict(load=count,permute=count/2,convert=count/2,arithmetic=count/2,store=count/2)
        elif mode=='token_mx':
            recipe=dict(load=count,arithmetic=count/2,convert=count/2,store=count/2)
        else:
            recipe=dict(load=2*count,arithmetic=count,store=count)
        vector_cycles[precision]=vf.stage(chip,'gemm/'+mode+'/'+precision,old,count,bits={'fp8_e4m3fn':8,'bf16':16,'fp32':32}[precision],
                                          known=mode.startswith('dense'),**recipe)
    vector = sum(vector_cycles.values())
    if mode == 'dense32':
        # Distinct launch/retirement mechanisms: beta=1 broadcasts a vector
        # bias, while un-biased transposed Stream-K has an extra rendezvous.
        front, drain = (1932,2240) if add_bias else (4273,3335)
    else:
        front, drain = 1794,520  # measured BF16 transposed first MMAD/retire
    transfer = max(hbm,panel_cycles)
    if mode in ('batch_mx','token_mx'):
        # Dependency chain for the first panel; the next panels double buffer.
        steps = max(1,pages*waves)
        stages = (transfer, vector, cube)
        body = max(stages) + (sum(stages)-max(stages))/steps
    else:
        body = max(transfer,cube,vector)
    if mode=='dense32' and split>1:
        # Stream-K output reduction on AIV; native drain contains its normal
        # small-shape service. Keep this floor and expose a larger merge only
        # when the issued reduction itself becomes limiting.
        merge=(split-1)*total*n/max(1,vactive)
        merge_cycles=vf.stage(chip,'gemm/split_k_merge',0.,merge,known=True,load=2*merge,arithmetic=merge,store=merge)
        drain=max(drain,merge_cycles)
    router_stages = {}
    if mode == 'router16':
        # First/last timestamps in the 950 BF16-input FP32-output Stream-K
        # kernel. HBM reads begin before MMAD. AIV reduces the partial tiles.
        pre_dma, read_to_mmad, drain = 2900.,2886.,3736.
        spread = 3264*(active-1)/max(1,cores-1)
        port = min(100., (chip.details.get('mte2_bandwidth_gbs') or 100*f)/f)
        local_stream = panel_bytes/port
        front = pre_dma
        merge_ops = (split-1)*total*n
        # Retirement already includes the small-case synchronization. For
        # large partial sums price the FP32 reduction and its workspace too.
        partial_bytes = split*total*n*4 if split>1 else 0
        merge_l2 = 2*partial_bytes*f/(chip.details.get('l2_bandwidth_gbs') or
                                      2*chip.hbm_bandwidth_gbs)
        merge_rate = (chip.vector_tops['32'] or 0.5*(chip.vector_tops['16'] or 54))*1000/f
        per=merge_ops/max(1,vec)
        merge=vf.stage(chip,'gemm/partial_reduction',merge_ops/merge_rate,per,known=True,load=2*per,arithmetic=per,store=per)
        drain = max(drain,merge_l2,merge)
        body = max(hbm,read_to_mmad+spread+max(local_stream,cube))
        router_stages = dict(read_to_first_mmad_cycles=read_to_mmad,
                             arrival_spread_cycles=spread,local_panel_cycles=local_stream,
                             partial_workspace_bytes=partial_bytes,merge_l2_cycles=merge_l2)
    unit = f*1000
    schedule = dict(tile_mnk=(bm,bn,page_k),output_tiles=jobs,split_k=split,
                    active_aic=active,active_aiv=None if split>1 else (vactive if classes else 0),waves=waves,
                    k_pages_per_split=pages,core_ops_per_cycle=rate,
                    resource_cycles=dict(first_issue=front,cube=cube,hbm=hbm,
                                         panel_mte2=panel_cycles,vector=vector,drain=drain),
                    vector_issued_ops_by_dtype=issued, vector_cycles_by_dtype=vector_cycles,
                    body_cycles=body, groups=groups, **router_stages)
    return (front+body+drain)/unit, schedule


def price(features, chip):
    try:
        sem = _fields(dict(features.attributes).get('semantic', ()))
        if features.raw_name in (*_NATIVES, _TRANS):
            parsed = _dense(features, sem)
        elif features.raw_name == _BATCH:
            parsed = _batched(features, sem)
        elif features.raw_name in NAMES:
            parsed = _mixed(features, sem)
        else:
            return NodeEstimate(None,'unsupported',reason='Unregistered product')
        rows,n,k,mode,reads,writes,classes = parsed
        t,schedule = _schedule(*parsed,chip,features.raw_name == 'aten.addmm.default')
    except _Mismatch as err:
        return NodeEstimate(None,'unsupported',reason=str(err))
    dtype = {'dense32':'fp32','dense16':'bf16','router16':'bf16','token_mx':'fp8_e4m3fn','batch_mx':'bf16'}[mode]
    contract = {'aten.mm.default':'matmul/transposed_weight_'+dtype,
                'aten.addmm.default':'addmm/transposed_weight_fp32',
                _TRANS:'matmul/router_bf16_to_fp32',
                _BATCH:'batch_matmul/last_axis_mxfp4_to_bf16'}.get(
                    features.raw_name,'grouped_matmul/token_fp8_mxfp4' if len(rows)>1 or
                    features.raw_name=='npu::npu_grouped_matmul' else 'matmul/token_fp8_mxfp4')
    if schedule['split_k'] > 1:
        # 950 Stream-K shards show RV_VADD on a subset of the launched AIVs.
        # Count intermediate partial-sum additions; their communication and
        # execution tail is already represented by the CA retirement budget.
        classes = dict(classes, split_k_combine=(schedule['split_k']-1)*sum(rows)*n)
        schedule['active_aiv'] = None  # exact set depends on the reduction partition
        schedule['split_k_merge'] = 'FP32 partial sums; AIV subset, priced in retirement'
    vector_ops = _vector_work(mode, classes)
    work = _counts(dtype,reads,writes,{dtype: 2*sum(rows)*n*k},vector_ops,classes=classes)
    measured_family = features.raw_name in (*_NATIVES,_TRANS)
    evidence = ('native_development_ca' if measured_family else 'analytical_design')
    note = ('aclnnMm/aclnnAddmm transposed ND, verified representative shapes on 950DT/PR; '
            'split geometry and instruction service are development-calibrated, other shapes/projected chips are extrapolations.'
            if measured_family else
            'Explicit layout-specific CV schedule; representative transfer/issue shell from BF16 MatMulV3; no complete matching CA kernel.')
    if mode == 'router16':
        note = ('aclnnMm BF16 transposed rhs with FP32 output; split-K and AIV merge anchored '
                'to native 950DT/PR router CA. Other workloads/projected chips use the same resource rules.')
    work.update(schedule=schedule,evidence=evidence,logical_shape_MNK=(sum(rows),n,k),
                evidence_detail=note)
    if features.raw_name in _NATIVES:
        a = features.inputs[-2]
        if a.shape[0] > 1 and a.stride[0] > k:
            # Last-token activations are contiguous within each request but their
            # base addresses are a whole prefill chunk apart. Pack just those rows.
            from .vector import Tiles, _schedule as vector_schedule
            count = sum(rows)*k
            size = count*(4 if mode == 'dense32' else 2)
            pack = vector_schedule(Tiles("pack_strided_activation_rows",count,k,size,size,0,setup="slice"),chip)
            work["logical_read_bytes"] += size
            work["logical_write_bytes"] += size
            schedule.update(row_pitch_elements=a.stride[0],pack_us=pack.latency_us,
                            pack_active_aiv=pack.work["schedule"]["active_aiv"])
            note = "Proposed materialization of selected activation rows before the dense kernel; no matched composed CA window."
            work.update(evidence="analytical_design",evidence_detail=note)
            return NodeEstimate(contract+"/pack_rows","extrapolated",t+pack.latency_us,note,work)
    stock = chip.name in ('Ascend950DT_9582','Ascend950PR_957b')
    return NodeEstimate(contract,'estimated' if stock and measured_family else 'extrapolated',t,note,work)
