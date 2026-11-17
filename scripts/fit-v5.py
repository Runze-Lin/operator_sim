#!/usr/bin/env python3
"""Freeze a long-FA model using only the existing 32k DB and old v0.4 model."""
from datetime import datetime, timezone
import json
from pathlib import Path
import sqlite3
import statistics
import sys

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from operator_model.ca import sha256
from operator_model.npusim import extract_compact
from operator_model.model_v5 import Predictor, Case
from operator_model.model_v4 import Predictor as V4


def slope(values, start=8, stop=120):
    xs=list(range(start,stop));ys=values[start:stop]
    xm=statistics.mean(xs);ym=statistics.mean(ys)
    return sum((x-xm)*(y-ym) for x,y in zip(xs,ys))/sum((x-xm)**2 for x in xs)


def analyze(folder):
    m=extract_compact(folder/'profile',folder/'run.log')
    assert m['active_aic']==1 and m['active_aiv']==2 and m['max_core_mmad']==512
    with sqlite3.connect(Path(m['db_path']).resolve().as_uri()+'?mode=ro',uri=True) as db:
        begin,end=db.execute('SELECT MIN(ExecInstrTickStart),MAX(ExecInstrTickEnd) FROM ExecutedInstructions').fetchone()
        reads=[{'start_us':(a-begin)/1650,'end_us':(z-begin)/1650,**json.loads(e)} for a,z,e in db.execute(
            "SELECT ExecInstrTickStart,ExecInstrTickEnd,ExtendParams FROM ExecutedInstructions WHERE ExecInstrName='MOV_OUT_TO_L1_MULTI_ND2NZ' ORDER BY ExecInstrTickStart")]
        mmads=[{'pc':pc,'start_us':(a-begin)/1650,'end_us':(z-begin)/1650,**json.loads(e)} for pc,a,z,e in db.execute(
            "SELECT SourceInstrAddr,ExecInstrTickStart,ExecInstrTickEnd,ExtendParams FROM ExecutedInstructions WHERE ExecInstrName='MMAD' ORDER BY ExecInstrTickStart")]
    q=[r for r in reads if r['processed_bytes']==1152]
    kv=[r for r in reads if r['processed_bytes']==32768]
    assert len(q)==1 and len(kv)==256
    base=min(r['src_addr'] for r in kv);tensor_bytes=32768*64*2
    k=sorted([r for r in kv if r['src_addr']<base+tensor_bytes],key=lambda r:r['src_addr'])
    v=sorted([r for r in kv if r['src_addr']>=base+tensor_bytes],key=lambda r:r['src_addr'])
    assert len(k)==len(v)==128
    for stream,offset in ((k,0),(v,tensor_bytes)):
        assert [r['src_addr'] for r in stream]==[base+offset+i*32768 for i in range(128)]
    # Missing MMAD dimensions are not imputed. Known rows identify phase PCs;
    # all events at those PCs are grouped for timing, only within this DB.
    qkpcs={r['pc'] for r in mmads if r.get('k')==64 and r.get('n')==128}
    pvpcs={r['pc'] for r in mmads if r.get('k')==128 and r.get('n')==64}
    assert len(qkpcs)==len(pvpcs)==1 and not qkpcs & pvpcs
    qk=[r for r in mmads if r['pc'] in qkpcs];pv=[r for r in mmads if r['pc'] in pvpcs]
    assert len(qk)==len(pv)==256
    first_done=max(r['end_us'] for r in pv[:2]);last_done=max(r['end_us'] for r in pv[-2:])
    costs={'prefix_us':k[0]['start_us'], 'first_k_latency_us':k[0]['end_us']-k[0]['start_us'],
        'first_v_start_lag_us':v[0]['start_us']-k[0]['start_us'],'first_v_latency_us':v[0]['end_us']-v[0]['start_us'],
        'compute_fill_us':first_done-max(k[0]['end_us'],v[0]['end_us']),
        'drain_us':m['reference_us']-last_done,
        'k_initiation_interval_us':slope([r['start_us'] for r in k]),
        'v_initiation_interval_us':slope([r['start_us'] for r in v])}
    for name,queue,kind in [('cube','CUBE',1),('vector','SIMD',2),('mte1','MTE1',1),('fix','FIXP',1)]:
        cores=[c for c in m['cores'] if (c['core_type_id']==1)==(kind==1)]
        costs[name+'_service_per_tile_us']=max(c['nonexcluded_instruction_union_us_by_queue'].get(queue,0) for c in cores)/128
    assert min(costs.values())>=0
    trace={'calibration_shape':[9,32768,64,1,1],'reference_us':m['reference_us'],'db_sha256':m['db_sha256'],
        'db_path':m['db_path'],'kernel':m['kernel'],'costs':costs,'steady_fit_indices':[8,120],
        'known_phase_pcs':{'qk':sorted(qkpcs),'pv':sorted(pvpcs)},
        'missing_mmad_dimension_rows':sum('k' not in r or 'n' not in r for r in mmads),
        'read_streams':{name:[{f:r[f] for f in ('start_us','end_us','src_addr','dst_addr','processed_bytes')} for r in stream] for name,stream in [('q',q),('k',k),('v',v)]},
        'qk_pairs':[[min(r['start_us'] for r in qk[i:i+2]),max(r['end_us'] for r in qk[i:i+2])] for i in range(0,256,2)],
        'pv_pairs':[[min(r['start_us'] for r in pv[i:i+2]),max(r['end_us'] for r in pv[i:i+2])] for i in range(0,256,2)],
        'limitations':['One long calibration shape; intra-trace blocks are not independent cases.',
            'Initiation intervals include buffer/control/backpressure effects, not isolated memory throughput.',
            'Phase grouping uses same-DB PCs with known shapes; missing dimensions remain missing.',
            'Full tile cost is used for tails; independent tail validation required.']}
    # Temporal block slopes diagnose drift; never call these independent shape validation.
    trace['block_slopes_us']={name:[slope([r['start_us'] for r in stream],a,b) for a,b in [(8,40),(40,80),(80,120)]] for name,stream in [('k',k),('v',v)]}
    return trace


def main():
    out=ROOT/'results/v0.5'
    if (out/'frozen-predictions.json').exists():raise SystemExit('Already frozen; never refit after holdout')
    trace=analyze(ROOT/'runs/long-fa/kv32k-compact-complete')
    (out/'trace-analysis.json').write_text(json.dumps(trace,indent=2)+'\n')
    base=json.loads((ROOT/'results/v0.4/model.json').read_text())
    model={'schema_version':5,'version':'0.5.0','soc_version':'Ascend950DT_9582','base_v4_model':base,
        'fa_long':{'costs':trace['costs'],'domain':{'q':9,'d':64,'b':1,'h':1,'kv_min':4096},
            'calibration_shape':trace['calibration_shape'],'calibration_db_sha256':trace['db_sha256']},
        'fit_utc':datetime.now(timezone.utc).isoformat(),'qualification':'Separated fill latency and effective initiation interval; one long training shape; no peak-throughput claim'}
    (out/'model.json').write_text(json.dumps(model,indent=2)+'\n')
    spec=json.loads((ROOT/'configs/sampling-v5.json').read_text());p=Predictor(model);old=V4(base)
    seen={(tuple(r['shape'])) for r in json.loads((ROOT/'results/long-fa/evaluation.json').read_text())['historical'] if r['op']=='fa'}|{tuple(trace['calibration_shape'])}
    predictions=[]
    for c in spec['cases']:
        if tuple(c['shape']) in seen:raise ValueError('Holdout has been used')
        case=Case('fa',c['shape']);predictions.append({**c,'prediction':p.predict(case),'v04_prediction':old.predict(case)})
    paths=['operator_model/model_v5.py','operator_model/model_v4.py','operator_model/model_v3.py','operator_model/model_v2.py','operator_model/model.py',
        'operator_model/npusim.py','operator_model/ca.py','scripts/fit-v5.py','scripts/collect-v5.py','configs/sampling-v5.json',
        'results/v0.5/trace-analysis.json','results/v0.4/model.json','smoke/three_ops.cpp','build/smoke/three_ops',
        'capture/sim_soc_adapter.cpp','build/capture/libsim_soc_adapter.so','scripts/cann-run.sh']
    frozen={'frozen_utc':datetime.now(timezone.utc).isoformat(),'model_sha256':sha256(out/'model.json'),
        'artifact_sha256':{f:sha256(ROOT/f) for f in paths},'predictions':predictions}
    (out/'frozen-predictions.json').write_text(json.dumps(frozen,indent=2)+'\n')
    print('Development32k',p.predict(Case('fa',trace['calibration_shape']))['predicted_us'],trace['reference_us'])
    for r in predictions:print(r['id'],r['prediction']['predicted_us'],r['v04_prediction']['predicted_us'])

if __name__=='__main__':main()
