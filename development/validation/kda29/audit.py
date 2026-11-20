"""Explain source-level Kimi V1 Finalize ownership and check it against official 950 CA."""
from collections import Counter
from datetime import datetime,timezone
import gzip
import hashlib
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[3]
OUT=ROOT/'results/v0.29/kda'
PRIOR=ROOT/'results/v0.27/kda/host-control'
FLA=ROOT/'results/chunk-recovery/fla2691/raw'
KERNEL=FLA/'fla_npu/opp/vendors/fla_npu_transformer/op_impl/ai_core/tbe/fla_npu_transformer_impl/ascendc/chunk_kda_fwd'
SOURCE=KERNEL/'arch35/chunk_kda_fwd_finalize.h'


def read(p):return json.loads(p.read_bytes())
def sha(p):
    d=hashlib.sha256()
    with p.open('rb') as f:
        while block:=f.read(2**20):d.update(block)
    return d.hexdigest()
def rel(p):return str(p.relative_to(ROOT))
def write(name,obj):
    tmp=(OUT/name).with_suffix('.tmp');tmp.write_text(json.dumps(obj,indent=2,ensure_ascii=False)+'\n');tmp.replace(OUT/name)


def plan(lengths,heads,aic):
    """Canonical Kimi BF16 K=V=128, block 64, explicit cu. O(rank batch + AIC) time."""
    if not isinstance(lengths,(list,tuple)) or any(type(l)!=int or l<0 for l in lengths):
        raise ValueError('lengths must be nonnegative integers')
    if type(heads)!=int or heads<=0 or type(aic)!=int or aic<=0:
        raise ValueError('invalid rank-local heads or AIC count')
    active=[s for s in lengths if s]
    full=sum(s//64 for s in active);tails=Counter(s%64 for s in active if s%64)
    c=full+sum(tails.values()); tasks=c*heads
    common=dict(lengths=list(lengths),rank_local_heads=heads,physical_aic=aic,chunks=c,work_items=tasks,
                automatic_route='no_launch' if not c else 'v2' if tasks>=4096 else 'v1',
                v1_phase_device_us={},v1_total_device_us=None)
    if not c:return {**common,'v1_phases':[],'selection_reason':'No active sequence.'}
    if tasks>=4096:return {**common,'v1_phases':[],'selection_reason':'Official default stable dispatcher selects a different ABI (V2).'}
    if c==1:return {**common,'v1_phases':['full'],'selection_reason':'One chunk selects the monolithic key-2 V1 specialization; its standalone phase-3 mapping is inapplicable.'}
    hist={k:heads*count for k,count in tails.items()}
    if full:hist[64]=heads*full
    ops=sum(count*(2*t*128*128+2*t*128*t) for t,count in hist.items())
    # A contiguous run of equally priced task IDs gives each core a quotient
    # and a circular interval of cores one extra task. Carry the first core
    # across packed sequence boundaries instead of silently resetting it.
    base=cursor=0
    differences=[0]*(aic+1)
    for length in active:
        for rows,num in ((64,length//64),(length%64,int(length%64!=0))):
            count=num*heads
            if not count:continue
            price=2*rows*128*(128+rows)
            quotient,remainder=divmod(count,aic);base+=quotient*price
            stop=cursor+remainder
            if remainder:
                differences[cursor]+=price;differences[min(aic,stop)]-=price
                if stop>aic:differences[0]+=price;differences[stop-aic]-=price
            cursor=stop%aic
    per_core=[]
    for delta in differences[:-1]:
        base+=delta;per_core.append(base)
    assert sum(per_core)==ops
    return {**common,'v1_phases':['gate_prepare','post_wu','fwd_h','finalize'],
            'finalize_work':dict(valid_rows_per_owner_task=hist,full_block_tasks=full*heads,
             tail_tasks=sum(tails.values())*heads,logical_cube_ops=ops,
             logical_state_product_ops=2*sum(active)*heads*128*128,
             logical_local_product_ops=sum(2*t*128*t*count for t,count in hist.items()),
             predicted_aic_with_output_tasks=min(aic,tasks),tasks_per_aic=[tasks//aic+int(core<tasks%aic) for core in range(aic)],
             logical_ops_per_aic=per_core,critical_owner_logical_ops=max(per_core)),
            'selection_reason':'Nonempty explicit-length 950 V1 multi-chunk dispatch. Finalize uses the ordinary strided queue; each valid full and tail branch constructs an inner Resource.'}


def source():
    anchors=[
        (FLA/'aclnn_chunk_kda_fwd.cpp',['const bool useDenseA5FastPath','const bool splitStages','for (int64_t stage = KDA_STAGE_GATE_PREPARE']),
        (ROOT/'results/v0.24/kda-v1/raw/chunk_kda_fwd_tiling.cpp',['const uint32_t blockDim =','const bool useChunk64K128V128Template =','context->SetTilingKey(']),
        (KERNEL/'chunk_kda_fwd.cpp',['tiling.stage == KDA_STAGE_FINALIZE','KdaFinalize::RunChunkKdaOutput','TILING_KEY_IS(2)']),
        (KERNEL/'chunk_kda_fwd_common.h',['#include "arch35/chunk_kda_fwd_finalize.h"']),
        (SOURCE,['void ComputeOutputCubeStagedArch35','void ComputeOutputCube(','BT_ == 64 && curT == BT_','void ProcessOutAic()',
                 'uint64_t task = GetBlockIdx(); task < taskNum; task += coreNum','hv = task % HV_;','flatChunk = task / HV_',
                 'void ProcessChunkOutAic','Catlass::Arch::Resource<KdaArchTag> resource;','GemmCoord shapeQH','GemmCoord shapeAV']),
        (KERNEL.parent/'common/catlass/arch/resource.hpp',['AscendC::TPipe pipe','pipe.Destroy()'])]
    result={}
    for p,keys in anchors:
        lines=p.read_text().splitlines();loc={needle:[i for i,line in enumerate(lines,1) if needle in line] for needle in keys}
        if not all(loc.values()):raise ValueError((p,loc))
        result[rel(p)]={'sha256':sha(p),'line_numbers':loc}
    return result


def releases(api_dump):
    raw=api_dump.read_bytes();raw=gzip.decompress(raw) if api_dump.suffix=='.gz' else raw
    api=json.loads(raw);old=read(ROOT/'results/v0.25/kda-v1/packages.json')
    kinds={r['asset']['name']:(ROOT/'results/v0.25/kda-v1/raw'/(kind+'.whl'),r['unique_text_count']) for kind,r in old['versions'].items()}
    kinds[old['reference_kimi_wheel']['name']]=(FLA/'fla-a5.whl',4)
    hits=[]
    for r in api:
        for asset in r['assets']:
            if '_a5-' not in asset['name'] or not asset['name'].endswith('aarch64.whl'):continue
            p,types=kinds.get(asset['name'],(None,None))
            actual=sha(p) if p and p.exists() else None
            hits.append(dict(tag=r['tag_name'],url=r['html_url'],asset=asset['name'],published=r['published_at'],size=asset['size'],
                             api_digest=asset['digest'],verified_locally=bool(actual and asset['digest']=='sha256:'+actual and p.stat().st_size==asset['size']),
                             local=rel(p) if p else None,fused_v1_unique_device_texts=types))
    roots=[ROOT/'toolchains/cann/latest/opp/built-in/op_impl/ai_core/tbe/kernel',
           Path('/usr/local/Ascend/cann-8.5.0/opp/built-in/op_impl/ai_core/tbe/kernel')]
    return dict(checked_utc=datetime.now(timezone.utc).isoformat(),release_api='https://api.github.com/repos/flashserve/flash-linear-attention-npu/releases?per_page=100',
                api_sha256=hashlib.sha256(raw).hexdigest(),releases=hits,no_unchecked_official_a5_wheel=all(x['verified_locally'] for x in hits),
                local_kernel_roots={str(p):sorted(q.name for q in p.glob('ascend*') if q.is_dir()) for p in roots},
                qualified_v1_full_us=None)


def payload_events(p):
    for e in read(p)['traceEvents']:
        if e['ph']!='X' or e['name'] not in ('MMAD','MOV_OUT_TO_L1_MULTI_ND2NZ','FIX_L0C_TO_DST'):continue
        d=json.loads(e['args']['detail'])
        if any(d.get(k) for k in ('is_wait','is_sync','is_duration_excluded')):continue
        yield dict(op=e['name'],pc=e['args']['pc_addr'],start_tick=round(e['ts']*1650),end_tick=round((e['ts']+e['dur'])*1650),
                   **{k:d[k] for k in ('src_addr','dst_addr','m','n','k','d','processed_bytes') if k in d})


def physical():
    r=read(PRIOR/'dt.json');fault=read(PRIOR/'fault-sites.json')
    assert r['actual_sku_verified'] and r['four_release_v1_phases'] and r['golden'] and r['full_windows'] and r['chronological'] and not r['qualified']
    assert sha(PRIOR/'dt.json')==fault['source_receipt_sha256']
    pack=read(PRIOR/'register.json');lengths=[b-a for a,b in zip(pack['cu'],pack['cu'][1:])]
    assert lengths==[1,1]
    p=plan(lengths,1,32);stages=[]
    for j,f in zip(r['stages'][2:6],fault['phases']):
        base=ROOT/j['raw_path'];assert sha(ROOT/j['aggregate'])==j['aggregate_sha256']
        observed=[]
        for cf,receipt in sorted(j['cores'].items()):
            if 'cubecore' not in cf:continue
            file=base/'simulator'/cf/'trace.json'
            if sha(file)!=receipt['shard_sha256']:raise ValueError(cf)
            instructions=sorted(payload_events(file),key=lambda w:w['start_tick'])
            mms=[w for w in instructions if w['op']=='MMAD']
            if len(mms)!=receipt['mmads']:raise ValueError(('MMAD mismatch',cf))
            cert=f['cores'][cf]['evidence']
            if cert or mms:
                observed.append(dict(core=cf,shard_sha256=receipt['shard_sha256'],family_ordinal=j['ordinal'],window=[receipt['first_tick'],receipt['last_tick']],
                                     mmads=mms,earliest_repeated_set=cert,
                                     transfer_details=[w for w in instructions if w['op']!='MMAD'] if j['ordinal']==3 else []))
        assert sum(len(z['mmads']) for z in observed)==j['mmads']
        stages.append(dict(stage=f['source_stage_order'],kernel=j['kernel'],capture=rel(base),trace_sha256=j['aggregate_sha256'],text_sha256=j['device_text_sha256'],
                           observed_window_us=j['ca_us'],first_tick=j['first_tick'],last_tick=j['last_tick'],active_aic=j['active_aic'],active_aiv=j['active_aiv'],
                           useful_mmad_aics=sum(bool(o['mmads']) for o in observed),callback_count=f['duplicate_callbacks'],prefix_certified_aics=f['aic_with_any_two_sets_before_first_matching_wait'],
                           status='phase in faulted measurement',formal_timing_us=None,reference_for_individual_phase_outputs=False,
                           resource=j['resource'],owners=observed))
    final=stages[-1];owners=final['owners']
    assert final['resource']['issued_mmad_ops']==p['finalize_work']['logical_cube_ops']==66048
    assert [o['core'] for o in owners]==['core0.cubecore0','core1.cubecore0']
    for o in owners:
        assert [(x['m'],x['n'],x['k']) for x in o['mmads']]==[(1,128,128),(1,128,1)]
        assert set(o['earliest_repeated_set'])=={'0','1','2'}
        assert all(x['second_pc_in_actual_callback'] and x['second_tick']<o['mmads'][0]['start_tick'] for x in o['earliest_repeated_set'].values())
    ptrdiff=[]
    assert len(owners[0]['transfer_details'])==len(owners[1]['transfer_details'])==6
    for e1,e2 in zip(owners[0]['transfer_details'],owners[1]['transfer_details']):
        assert e1['op']==e2['op'] and e1['pc']==e2['pc']
        field='src_addr' if e1['op']=='MOV_OUT_TO_L1_MULTI_ND2NZ' else 'dst_addr'
        ptrdiff.append(dict(op=e1['op'],pc=e1['pc'],core0=e1[field],core1=e2[field],delta_bytes=e2[field]-e1[field],descriptor_bytes=e1['processed_bytes']))
    expected_source_reads=[256,32768,128,256]
    assert [x['delta_bytes'] for x in ptrdiff if x['op'].startswith('MOV_')]==expected_source_reads
    assert [x['delta_bytes'] for x in ptrdiff if x['op'].startswith('FIX')]==[512,512]
    counts=Counter(z['source_stage_order'] for z in fault['phases'])
    assert len(counts)==4
    return dict(scope='Measured V1 phase intervals inside an invalid full dispatch. Finalize source-derived work and nominal owner identities compared to actual instructions/GM displacements. Whole V1 and standalone subphase latency calibration remain unavailable.',
                target='Ascend950DT_9582',inputs=pack['shape'],cu=pack['cu'],four_groups=r['output_groups'],
                register_sha256=sha(PRIOR/'register.json'),capture_sha256=sha(PRIOR/'dt.json'),fault_sha256=sha(PRIOR/'fault-sites.json'),
                model_work=p,phases=stages,finalize_cross_owner_pointers=ptrdiff,formal_time_us=None)


def main():
    OUT.mkdir(exist_ok=True,parents=True)
    if sys.argv[1:2]==['plan']:
        print(json.dumps(plan([int(x) for x in sys.argv[4:]],int(sys.argv[2]),int(sys.argv[3])),indent=2));return
    if sys.argv[1:2]!=['audit']:raise SystemExit('audit <official release API dump>  |  plan HEADS AIC LENGTH...')
    write('release-inventory.json',releases(Path(sys.argv[2])))
    write('source-map.json',source())
    write('partial-stages.json',physical())
    scenarios=[plan([1,1],1,32),plan([64,1],1,32),plan([64,64],1,32),plan([128],1,32),
               plan([512]*8,12,32),plan([2048]*8,12,32),plan([2688]*8,12,32),plan([2689]*8,12,32),
               plan([32768]*8,12,32),plan([131072]*8,12,32),plan([262144]*8,12,32)]
    rates={}
    for name in ('Ascend950DT_9582','Ascend950PR_957b','Ascend970DT'):
        p=ROOT/'configs/chips'/(name+'.json');chip=read(p)
        rates[name]=dict(aic=chip['cube_cores'],bf16_cube_tops=chip['cube_tops']['16'],source=rel(p),sha256=sha(p),
                         contract_status='future configuration; same Kimi and library code assumed' if name=='Ascend970DT' else '950 chip configuration')
    for s in scenarios:
        bounds={}
        for name,r in rates.items():
            q=plan(s['lengths'],s['rank_local_heads'],r['aic'])
            if 'finalize_work' not in q:continue
            work=q['finalize_work']['critical_owner_logical_ops']
            bounds[name]=dict(critical_owner_logical_ops=work,useful_work_peak_floor_us=work*r['aic']/(r['bf16_cube_tops']*1e6))
        s['finalize_useful_cube_floor']=bounds
    write('system-contracts.json',dict(scope='Source- and shape-derived work; optional useful-work floor for Finalize assuming symmetric per-AIC peak and the same vendor schedule. Not full stage or kernel latency.',chip_rates=rates,rows=scenarios))


if __name__=='__main__':main()
