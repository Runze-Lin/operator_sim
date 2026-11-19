"""Map one three-chunk recurrent chain from a qualified, independently frozen FLA V2 run."""
from collections import defaultdict
from hashlib import sha256
import json
from pathlib import Path
import sys

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[2]
OUT=ROOT/'results/v0.23/kda-multiblock'
CAMPAIGN=ROOT/'results/kda-v23'
HOLDOUT=CAMPAIGN/'supplement-ascend950pr-t129-h3.json'
FROZEN=CAMPAIGN/'supplement-predictions.json'
REG=CAMPAIGN/'precision-supplement-register.json'
sys.path.insert(0,str(HERE.parent/'kda_resource23'))
import model
from plot_schedule import Recorder

ROWS=(64,64,1)
CYCLE=1650
# Match ELF-relative PCs, all address spaces and per-chunk counts on this exact ELF only.
# Names match arch35 ProcessSingleHeadPipeline, the real public-release kernel.
MAP={
 0x67a98:('init','OUT->UB','input state'),0x67c38:('init','VF','narrow H'),0x67c8c:('init','UB->OUT','initial H'),
 0x289d0:('S0','OUT->L1','W'),0x28ab4:('S0','OUT->L1','H'),0x28d5c:('S0','L1->L0A','W → A'),0x28d78:('S0','L1->L0B','H → B'),
 0x28da4:('S0','MMAD','W × H'),0x28ec8:('S0','L0C->UB','P'),
 0x67e8c:('S1','OUT->UB','U'),0x68194:('S1','VF','correct U'),0x681dc:('S1','UB->OUT','v_new'),
 0x68210:('S1','UB->OUT','right scratch'),
 0x28f64:('S2','OUT->L1','gated K'),0x291b8:('S2','OUT->L1','right scratch'),0x293b0:('S2','L1->L0A','K → A'),
 0x293b8:('S2','L1->L0B','right → B'),0x28870:('S2','MMAD','Kᵀ × right'),0x288e8:('S2','L0C->UB','state delta'),
 0x6846c:('S2','OUT->UB','input state for update'),0x684b4:('S2','OUT->UB','gate'),
 0x68598:('S3','VF','update state'),0x68600:('S3','VF','final state update'),
 0x683e0:('S3','UB->OUT','H for next chunk'),0x68650:('S3','UB->OUT','final state')}
PHASES={'init':'FP32 initial state to BF16 H','S0':'S0 W times H','S1':'S1 corrected value','S2':'S2 outer product','S3':'S3 update recurrent state'}


def engine(route):
    if route.startswith('OUT->'):return 'MTE2'
    if route.startswith('L1->'):return 'MTE1'
    if route.startswith('L0C->'):return 'FIXP'
    return {'UB->OUT':'MTE3','VF':'Vector','MMAD':'Cube'}[route]


def file_hash(p):
    h=sha256()
    with p.open('rb') as f:
        for b in iter(lambda:f.read(1024**2),b''):h.update(b)
    return h.hexdigest()


def measured(stage):
    parent=(ROOT/stage['aggregate']).parent
    base=int((parent.parent/'dump/pc_start_addr.txt').read_text(),16)
    last=next(c for c,r in stage['cores'].items() if r['last_tick']==stage['last_tick'])
    group=last.split('.')[0]
    loads={};finals={};subset=defaultdict(list);shards={}
    ids=set(c.split('.')[0] for c in stage['cores'])
    for core in sorted(stage['cores']):
        path=parent/core/'trace.json';digest=file_hash(path)
        if digest!=stage['cores'][core]['shard_sha256']:raise ValueError('core shard changed')
        shards[core]=dict(path=str(path.relative_to(ROOT)),sha256=digest)
        for e in json.loads(path.read_bytes())['traceEvents']:
            if e['ph']!='X':continue
            desc=json.loads(e['args']['detail'] or '{}')
            if any(desc.get(k) for k in ('is_sync','is_wait','is_duration_excluded')):continue
            route=e['name'] if e['name'] in ('MMAD','VF') else None
            if all(k in desc for k in ('processed_bytes','src_mem','dst_mem')):route=desc['src_mem']+'->'+desc['dst_mem']
            if not route:continue
            pc=int(e['args']['pc_addr'],16)-base
            if pc==0x67a98:loads[core.split('.')[0]]=desc['src_addr']
            if pc==0x68650:finals[core.split('.')[0]]=desc['dst_addr']
            if core.split('.')[0]!=group:continue
            if pc not in MAP:raise ValueError('unmapped payload PC '+hex(pc))
            phase,wanted,label=MAP[pc];assert route==wanted
            q=dict(core=core,phase=phase,operation=label,route=route,engine=engine(route),pc_offset=hex(pc),physical_opcode=e['name'],
              start_tick=round(e['ts']*CYCLE)-stage['first_tick'],end_tick=round((e['ts']+e['dur'])*CYCLE)-stage['first_tick'])
            for k in ('src_addr','dst_addr','src_mem','dst_mem','processed_bytes','m','n','k'):
                if k in desc:q[k]=desc[k]
            subset[pc].append(q)
    assert set(loads)==set(finals)==ids and len(ids)==3
    peers=sorted((addr,c) for c,addr in loads.items())
    for h,(addr,c) in enumerate(peers):
        assert addr==peers[0][0]+65536*h and finals[c]==min(finals.values())+65536*h
    head=[c for _,c in peers].index(group)
    for pc in MAP:
        series=sorted(subset[pc],key=lambda v:v['start_tick'])
        which=([0] if MAP[pc][0]=='init' or pc==0x6846c else [0,1] if pc in (0x683e0,0x68598) else [2] if pc in (0x68600,0x68650) else [0,1,2])
        if len(series)!=len(which):raise ValueError('per-PC ordinal mismatch '+hex(pc))
        for seg,chunk in zip(series,which):seg['chunk']=chunk
        subset[pc]=series
    leading=[subset[0x67c8c][0],*subset[0x683e0]]
    dependencies=[]
    for i,(producer,consumer) in enumerate(zip(leading,subset[0x28ab4])):
        assert producer['dst_addr']==consumer['src_addr'] and producer['processed_bytes']==consumer['processed_bytes']==32768
        assert producer['end_tick']<=consumer['start_tick']
        dependencies.append(dict(produced_by='initial conversion' if i==0 else f'chunk {i}',read_by=f'chunk {i+1}',
           address=producer['dst_addr'],size=32768,producer_end_tick=producer['end_tick'],consumer_start_tick=consumer['start_tick']))
        valid=ROWS[i];padded=((valid+15)//16)*16
        for pc,mnk in ((0x28da4,(padded,128,128)),(0x28870,(128,128,padded))):
            q=subset[pc][i];assert tuple(q[k] for k in ('m','n','k'))==mnk
        assert subset[0x289d0][i]['processed_bytes']==256*valid
        for src_pc,dst_pc in ((0x68210,0x291b8),):
            src,dst=subset[src_pc][i],subset[dst_pc][i]
            assert src['dst_addr']==dst['src_addr'] and src['end_tick']<=dst['start_tick'] and src['processed_bytes']==dst['processed_bytes']==valid*256
    all_rows=sorted((q for seq in subset.values() for q in seq),key=lambda q:q['start_tick'])
    return dict(head=head,physical_group=group,last_core=last,shards=shards,
      head_proof=[dict(head=i,physical_group=c,initial_fp32_gm=addr,final_fp32_gm=finals[c]) for i,(addr,c) in enumerate(peers)],
      initial=[q for q in all_rows if q['phase']=='init'],chunks=[dict(chunk=i,rows=n,resources=[q for q in all_rows if q['chunk']==i and q['phase']!='init']) for i,n in enumerate(ROWS)],state_handoffs=dependencies,
      timing_unit='device ticks, 1650 ticks/us. Zero is the earliest instruction on any core of the FwdH device launch.',
      omitted='Control, sync and waits retained in the full window. VF spans denote dispatched vector functions; internal retirements overlap. The second AIV of this workgroup has zero shown payload events.')


def predicted(frozen,sku):
    chip=model.load_chip(sku);a=frozen['active_aic'];end=frozen['first_payload_cycles']
    payloads=[];front=dict(start_tick=0,end_tick=end,engine='Control',phase='front',route='launch')
    for i,n in enumerate(ROWS):
        original=model.Tile;model.Tile=Recorder
        try:tile=model.recurrence(n,chip,a,i==0,i==2)
        finally:model.Tile=original
        calls=iter(tile.calls);events=[]
        def step(expect,t):
            route,_,cost=next(calls)
            assert route==expect
            events.append(dict(chunk=i,phase=phase,route=route,engine=engine(route),start_tick=t,end_tick=t+cost))
            return t+cost
        def line(routes,t):
            for r in routes:
                t=max(step(v,t) for v in r) if isinstance(r,tuple) else step(r,t)
            return t
        stored=frozen['representative_tile_phase_cycles'][str((n,i==0,i==2))]
        for phase,long in PHASES.items():
            if long not in stored:continue
            t=end
            if phase=='init':end=line(['OUT->UB','VF','UB->OUT'],t)
            if phase=='S0':end=line(['OUT->L1',('L1->L0A','L1->L0B'),'MMAD','L0C->UB'],t)
            if phase=='S1':end=line(['OUT->UB','VF','UB->OUT'],t)
            if phase=='S2':end=max(line(['OUT->L1',('L1->L0A','L1->L0B'),'MMAD','L0C->UB'],t),line(['OUT->UB'],t))
            if phase=='S3':end=line(['VF','UB->OUT'],t)
            assert abs(end-t-stored[long])<1e-6
        assert next(calls,None) is None;payloads+=events
    assert abs(end/CYCLE-frozen['device_us'])<1e-8
    return dict(front=front,initial=[q for q in payloads if q['phase']=='init'],chunks=[dict(chunk=i,rows=n,resources=[q for q in payloads if q['phase']!='init' and q['chunk']==i]) for i,n in enumerate(ROWS)],
                end_tick=end,front_end_cycles=frozen['first_payload_cycles'],cost_unit='cycles of the 1.65 GHz profile; stage rates and response overhead frozen before target observation',
                model_view='Critical path of one head. Three heads have equal cost; no prediction about which physical core finishes last.')


def bundle():
    r=json.loads(HOLDOUT.read_bytes());f=json.loads(FROZEN.read_bytes());reg=json.loads(REG.read_bytes())
    assert r['qualified'] and r['full_windows'] and r['exact_stage_elfs'] and len(r['output_groups'])==4 and all(n and bad==0 for n,bad in r['output_groups'].values())
    assert r['registration_sha256']==file_hash(REG)
    for k,v in reg['source_hashes'].items():
        if 'kda_resource23/model.py' in k or k.endswith('supplement-predictions.json'):assert file_hash(ROOT/k)==v
    p=next(q for q in f['predictions'] if q['shape']==r['case']['shape'] and q['chip']==r['case']['sku'])
    obs=r['stages'][1];fc=p['stages'][1];assert obs['family']==fc['family']=='chunk_fwd_h' and obs['active_aic']==3 and fc['active_aic']==3
    pack=dict(case=r['case'],family=obs['family'],actual_kernel=obs['kernel'],device_text_sha256=obs['device_text_sha256'],stage_first_tick=obs['first_tick'],
       stage_last_tick=obs['last_tick'],ca_stage_us=obs['ca_us'],model_stage_us=fc['device_us'],ca_three_stage_us=r['serialized_device_us'],model_three_stage_us=p['serialized_device_us'],
       three_stage_ca_wall_seconds=r['cost']['seconds'],new_ca_seconds=0,golden=r['output_groups'],mathematical_vs_published_checks=r['auxiliary_diagnostics'],
       role='frozen independent holdout; direct public V2, default small Kimi route is V1',reference_receipt=str(HOLDOUT.relative_to(ROOT)),receipt_sha256=file_hash(HOLDOUT),
       prediction_file=str(FROZEN.relative_to(ROOT)),prediction_sha256=file_hash(FROZEN),registration_sha256=r['registration_sha256'],model_freeze_sha256=f['model_freeze_sha256'],
       phase_window_sha256=obs['aggregate_sha256'],cost_input=dict(model_primitive_sha256=file_hash(ROOT/'results/kda-resource23/dt-primitive-services.json'),
          base_rates_sha256=file_hash(ROOT/'results/v0.16/matmul.json'),target_profile_sha256=file_hash(ROOT/'configs/chips/Ascend950PR_957b.json')),
       prediction=predicted(fc,p['chip']),observation=measured(obs))
    project_sources=json.loads((ROOT/'results/kda-resource23/source-evidence.json').read_bytes())
    pack['released_kernel_sources']=[source for source in project_sources if source.get('file','').endswith(('arch35/chunk_fwd_h_cube.h','arch35/chunk_fwd_h_vec.h'))]
    assert len(pack['released_kernel_sources'])==2
    for source in pack['released_kernel_sources']:
        assert file_hash(ROOT/source['file'])==source['sha256']
    out=OUT/'950pr-t129-h3-fwdh.json';tmp=out.with_suffix('.tmp');tmp.write_text(json.dumps(pack,indent=2,ensure_ascii=False)+'\n');tmp.replace(out)
    return pack

if __name__=='__main__':
    from render import draw
    draw(bundle(),OUT/'950pr-t129-h3-fwdh.png')
