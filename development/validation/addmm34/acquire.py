"""Prospectively freeze and acquire a real BF16 column-bias Addmm in a retired CA lane."""
import argparse
import fcntl
import json
import math
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import time

from certificate import ROOT, PKG, NATIVE, sha, certify, require, no_bias_rejections

sys.path[:0] = [str(ROOT/'development/validation/large29/cube'),str(ROOT)]
import campaign as original
from operator_model import estimate, load_chip
from operator_model.matmul_supply32 import resident_schedule
from operator_model.matmul_resource import _costs

OUT=ROOT/'results/v0.34/addmm'; RAW=ROOT/'runs/addmm34'; REG=OUT/'freeze.json'
SLOT=ROOT/'results/v0.26/sink/raw/lock'; PREVIOUS=ROOT/'results/v0.32/large/recurrent-state.json'
OLD=ROOT/'results/v0.29/large/cube'; REPLAY=ROOT/'results/v0.32/replay.json'
SHAPE=[65536,128,512]; ATTRS=dict(self_shape=[128],alpha=1,beta=1)
DT,PR,FUT=original.CHIPS
read,now=original.read,original.stamp


def rel(path): return str(path.relative_to(ROOT))


def save(path, obj):
    path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_name(f'.{path.name}.{os.getpid()}.tmp')
    tmp.write_text(json.dumps(obj,ensure_ascii=False,indent=2,allow_nan=False)+'\n');tmp.replace(path)


def files():
    return [Path(__file__), Path(__file__).with_name('certificate.py'), NATIVE, PKG, original.BIN/'addmm',
            original.BIN/'addmm.cpp',Path(original.__file__), ROOT/'scripts/cann-run.sh',
            ROOT/'configs/models/DeepSeek-V4.1-Flash.json',ROOT/'development/workloads/prefill_shapes.py',
            OLD/'freeze.json',REPLAY, *[ROOT/'configs/chips'/f'{s}.json' for s in (DT,PR,FUT)],
            *[ROOT/'operator_model'/f'{s}.py' for s in ('addmm_v9','matmul_supply32','matmul_resource','matmul_v15','matmul_waves')]]


def library_inventory():
    rows=[r for r in read(REPLAY)['rows'] if r['op']=='addmm']; existing={}
    for sku in (DT,PR):
        group=[r for r in rows if r['chip']==sku]; large=max(group,key=lambda r:math.prod(r['shape']))
        existing[sku]=dict(windows=len(group),shapes=len({tuple(r['shape']) for r in group}),
            max_rows=max(r['shape'][0] for r in group),largest_volume_shape=large['shape'],largest_ca_us=large['ca_us'],
            largest_trace_sha256=large['identity'])
    require(not any(r['shape']==SHAPE for r in rows), 'already scored')
    native=[]; logs=0; no_bias=0; sizeable=[]
    sig=re.compile(rb'case=addmm api=aclnnAddmm M=(\d+) N=(\d+) K=(\d+) self=(bias|matrix)')
    for folder,dirs,names in os.walk(ROOT/'runs'):
        dirs[:]=[x for x in dirs if x not in ('bin','sources','profile','simulator','log','log_ca','npusim','dump','ca_workspace','record','golden') and not x.startswith(('OPPROF_','PROF_'))]
        if 'run.log' not in names: continue
        path=Path(folder)/'run.log';logs+=1
        with path.open('rb') as stream:
            for line in stream:
                hit=sig.search(line)
                if hit:
                    a,b,c,contract=hit.groups();shape=[int(a),int(b),int(c)];native.append((shape,contract.decode(),rel(path)))
                    require(shape!=SHAPE or contract!=b'bias','same Addmm already captured')
                if b'case=matmul api=aclnnMatmul ' in line:
                    no_bias+=1
                    if all(s.encode() in line for s in ('N=128 ','K=512 ')):
                        hit=re.search(rb' M=(\d+)',line)
                        if hit and int(hit[1])>=32768:sizeable.append(rel(path))
    return dict(raw_logs=logs,native_host_case_lines=len(native),raw_largest_by_volume=max(native,key=lambda x:math.prod(x[0])) if native else None,
        raw_large_bias_logs=[x for x in native if x[0][0]>=32768 and x[1]=='bias'],archive=existing,
        bias_free_matmul_lines=no_bias,large_bias_free_examples=sizeable,source=rel(REPLAY),
        scope='Run logs under runs/ (data payloads/OPPROF trees excluded); standard qualified score checked separately. A host launch log alone is not an admitted timing.')


def predict(shape,sku):
    start=time.perf_counter_ns(); p=estimate('addmm',shape,ATTRS,chip=sku,dtype='bf16'); ms=(time.perf_counter_ns()-start)/1e6
    tile=p.projection['target_plan']; hw=load_chip(sku); resident=resident_schedule(shape,hw); alt=None
    if resident:
        us,cores,_,analysis=resident; plan=analysis['target_plan']; bias=2*shape[1]
        mte1=hw.details.get('mte1_bandwidth_gbs') or _costs()['mte1_bytes_per_cycle']*hw.frequency_ghz
        hbm=max(1.,hw.hbm_bandwidth_gbs); first=cores*bias/(1000*hbm); bt=plan['waves']*bias/(1000*mte1)
        alt=dict(overlapped_us=us,extra_bandwidth_service_us=first+bt,serialized_service_us=us+first+bt,
            waves=plan['waves'],mmads=plan['mmads_total'],tile=plan['tile'],full_rhs_bytes=2*shape[1]*shape[2],
            requires='Same B-resident body as isolated no-bias MatMulV3: one full KxN B load per core and one bias move per output fragment. Separate setup/queue latency unresolved. Two cost scenarios, not a certified interval.')
    return dict(prediction_us=p.latency_us,active_aic=p.active_aic,active_aiv=p.active_aiv,waves=tile['waves'],
                tile=tile['tile'],mmads=tile['mmads_total'],query_ms=ms,extrapolation=sku==FUT,b_resident_scenario=alt)


def register():
    require(not REG.exists(),'already frozen'); require(read(PREVIOUS)['phase']=='withdrawn before acquisition' and read(PREVIOUS)['terminal'], 'lane still requested')
    require(all((ROOT/f'results/v0.32/large/sink1024-{s}.json').exists() for s in (DT,PR)),'prior endpoints still pending')
    require(not any((RAW/tag/'run.log').exists() for tag in ('dt','pr')), 'prospect violated')
    cfg=read(ROOT/'configs/models/DeepSeek-V4.1-Flash.json')['text_config']; require(SHAPE==[8*8192,cfg['index_head_dim'],cfg['head_dim']], 'resolved geometry drift')
    inventory=library_inventory();old=next(j for j in read(OLD/'freeze.json')['jobs'] if j['id']=='addmm')
    require(old['shape']==[262144,128,512], 'full reservation changed')
    j=dict(id='addmm',host='addmm',hours=4,gib=16,args=['addmm',*SHAPE,'bias']); preds={s:predict(SHAPE,s) for s in (DT,PR,FUT)}
    reg=dict(utc=now(),shape=SHAPE,attrs=ATTRS,forecasts=preds,job=j,archive_inventory=inventory,
        scenario='Flash B8×32768, 4 equal chunks/request, 8192 query rows each => 65536 local rows, replicated indexer.wk width from real config. Explicit BF16 column bias is a standalone variant; model node bias not established.',
        logical_tensor_bytes=dict(lhs=2*SHAPE[0]*SHAPE[2],rhs=2*SHAPE[1]*SHAPE[2],self=2*SHAPE[1],output=2*SHAPE[0]*SHAPE[1]),
        expected_output_elements=SHAPE[0]*SHAPE[1],outputs_rejecting_omitted_bias=no_bias_rejections(SHAPE),
        existing_full=dict(shape=old['shape'],registration=rel(OLD/'freeze.json'),state=rel(OLD/'state.json'),prices={s:old['predictions']['addmm'][s]['us'] for s in (DT,PR,FUT)}),
        lane=rel(SLOT),retired_predecessor=rel(PREVIOUS),acquire='Official msopprof simulator, --dump=false, mandatory pipe collection only, one process/SKU, serial DT then PR. Each endpoint releases flock before qualification so queued AttnRes can use the same physical lane. No retry. Four hours and 16 GiB per target; minimum 48 GiB free. Preserve partials.',
        qualification='Actual aclnnAddmm host and dtype/scalars/self rank, full output golden observable against zero bias, one STARS task, correct SoC/DRAM, installed ELF, all active shard retirements and ENDs matching vendor table. Never price with wrapper or no-bias Matmul.',
        criterion='Publish current point signed error and pair speedup. Resident-B branch is a pre-priced hypothesis pending source/structural and independent timing agreement; no default change on this geometry alone.',
        source_sha256={rel(p):sha(p) for p in files()})
    with REG.open('x') as stream:json.dump(reg,stream,indent=2,ensure_ascii=False,allow_nan=False);stream.write('\n')


def score():
    reg=read(REG); rows=[]
    for label,shape,pred,paths in [('independent_chunk',SHAPE,{s:p['prediction_us'] for s,p in reg['forecasts'].items()},[OUT/'dt.json',OUT/'pr.json']),
        ('existing_full',reg['existing_full']['shape'],reg['existing_full']['prices'],[OLD/'addmm-dt.json',OLD/'addmm-pr.json'])]:
        for sku,path in zip((DT,PR),paths):
            r=read(path) if path.exists() else {}; v=r.get('verdict',{}); us=v.get('ca_us') if v.get('qualified') else None
            rows.append(dict(op='addmm',role=label,shape=shape,chip=sku,prediction_us=pred[sku],ca_us=us,qualified=us is not None,
                status='measured' if us is not None else ('failed' if r else 'pending'),signed_error_pct=(pred[sku]/us-1)*100 if us else None,
                future_970dt_extrapolated_us=pred[FUT],receipt=rel(path) if r else None,capture_s=r.get('cost',{}).get('seconds')))
    pairs=[]
    for a,b in zip(rows[::2],rows[1::2]):
        pred=a['prediction_us']/b['prediction_us'];actual=a['ca_us']/b['ca_us'] if a['qualified'] and b['qualified'] else None
        pairs.append(dict(role=a['role'],predicted_dt_over_pr=pred,measured_dt_over_pr=actual,signed_speedup_error_pct=(pred/actual-1)*100 if actual else None))
    save(OUT/'score.json',dict(freeze_sha256=sha(REG),rows=rows,pairs=pairs))


def run():
    reg=read(REG);require(all(sha(ROOT/p)==h for p,h in reg['source_sha256'].items()), 'frozen source changed')
    RAW.mkdir(parents=True,exist_ok=True);os.nice(15);os.environ.update(OMP_WAIT_POLICY='PASSIVE',GOMP_SPINCOUNT='0')
    def status(phase,**extra):save(OUT/'state.json',dict(phase=phase,utc=now(),pid=os.getpid(),freeze_sha256=sha(REG),**extra))
    with (RAW/'owner.lock').open('a') as owner:
        fcntl.flock(owner,fcntl.LOCK_EX|fcntl.LOCK_NB)
        for sku,tag in ((DT,'dt'),(PR,'pr')):
            raw=RAW/tag; dest=OUT/(tag+'.json')
            if dest.exists():continue
            if not (raw/'cost.json').exists():
                require(not raw.exists(),'partial launch exists, do not repeat');status('waiting for disk',chip=sku)
                while shutil.disk_usage(ROOT).free<48*2**30:time.sleep(30)
                status('waiting for retired standalone lane',chip=sku,lane=rel(SLOT))
                with SLOT.open('r+') as slot:
                    fcntl.flock(slot,fcntl.LOCK_EX)
                    raw.mkdir();status('acquiring',chip=sku,raw=rel(raw),lane=rel(SLOT));cost=original.capture(reg['job'],sku,raw)
            else:cost=read(raw/'cost.json')
            status('qualifying',chip=sku)
            try: v=certify(raw,sku,reg['shape'],cost)
            except (OSError,ValueError,KeyError,IndexError,RuntimeError,subprocess.CalledProcessError) as exc:v=dict(qualified=False,ca_us=None,reason=str(exc))
            save(dest,dict(utc=now(),freeze_sha256=sha(REG),sku=sku,shape=reg['shape'],cost=cost,verdict=v));score()
        status('finished');score()


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('action',choices=['freeze','score','run']);arg=parser.parse_args()
    {'freeze':register,'score':score,'run':run}[arg.action]()
