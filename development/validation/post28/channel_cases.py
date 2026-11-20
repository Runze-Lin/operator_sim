"""Register, run once, and gate the three small channel-partitioned Post holdouts."""
import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import time
from datetime import datetime,timezone
ROOT=Path(__file__).resolve().parents[3]
sys.path.insert(0,str(ROOT))
from development.validation.post24.collect import LIB,BIN,capture,qualify
from development.validation.kda27.quant_transfer import headroom
from operator_model import estimate
from operator_model.post_history import estimate_post_history

DST=ROOT/'results/v0.28/post'
RAW=ROOT/'runs/post28'
REG=DST/'registration.json'
SKUS=('Ascend950DT_9582','Ascend950PR_957b')
SHAPES=((7,4,3328),(11,4,1152),(23,4,1984))


def now():return datetime.now(timezone.utc).isoformat()
def read(p):return json.loads(p.read_text())
def digest(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def save(p,v):
    temp=p.with_suffix('.tmp');temp.write_text(json.dumps(v,indent=2,ensure_ascii=False)+'\n');temp.replace(p)
def novelty():
    checks=[re.compile(rb'\[\s*'+rb'\s*,\s*'.join(str(v).encode() for v in shape)+rb'\s*\]') for shape in SHAPES]
    files=0; found=[]
    for root,children,names in os.walk(ROOT/'results'):
        children[:]=[n for n in children if n not in ('raw','profile','v0.28','.git')]
        for name in names:
            if not name.endswith('.json') or name=='trace.json':continue
            path=Path(root)/name;data=path.read_bytes();files+=1
            for shape,pattern in zip(SHAPES,checks):
                if pattern.search(data):found.append(dict(shape=shape,source=str(path.relative_to(ROOT))))
    if found:raise ValueError(found)
    return dict(manifests=files,exact_shape_tuples_found=found)


def register():
    assert not REG.exists() and not RAW.exists()
    novel=novelty()
    reg=dict(utc=now(),novelty=novel,miB_cap_each=1024,seconds_cap_each=600,
             selection='N4 real channel route on both targets, core-count boundaries and three fragment widths; previously unmeasured shapes. Two calls same BF16 contract, per shape per SKU.',
             scores='Keep original v0.24 frozen prices and qualification (aggregate END_LABEL, per-lane physical END, reference ELF, full golden and write volume). Gate per SKU unique qualified channel repeats >=10; all-shape repeat WAPE <=25%, max <=50% and WAPE below old same-set occupancy. No replacement registrations. First/paired ordinal also reported; no common CA across chips or backfilling any absent time.',
             legacy_cohort='results/v0.24/post/evaluation.json',jobs=[],hashes={})
    for shape in SHAPES:
        for sku in SKUS:
            s=estimate_post_history(shape,chip=sku); assert s.route=='post_channel'
            prefix=0
            rows=[]
            for i in (0,1):
                future=estimate_post_history(shape,chip='Ascend970DT')
                model_state='repeat' if i else 'first'
                pred=s.scenarios[model_state]
                rows.append(dict(i=i,shape=list(shape),active=s.active_aiv,role='repeat' if i else 'entry',
                                 model_state=model_state,sku=sku,previous_shape=list(shape) if i else None,
                                 prior_aiv=prefix,public_known_us=estimate('mhc_post',shape,{'resident_aiv':prefix},chip=sku).latency_us,
                                 public_shape_only_us=estimate('mhc_post',shape,chip=sku).latency_us,mode=pred.model,
                                 candidate_us=pred.latency_us,future970_extrapolation_us=future.scenarios[model_state].latency_us))
                prefix=s.active_aiv
            reg['jobs'].append(dict(id=f'{shape[0]}x{shape[2]}-{sku}',sku=sku,rows=rows))
    names=[Path(__file__).resolve(),LIB,BIN,DST/'parameters.json',ROOT/'operator_model/post_history.py',
           ROOT/reg['legacy_cohort'],ROOT/'results/v0.24/post/freeze.json',ROOT/'development/validation/post24/collect.py',
           ROOT/'development/validation/rms_ordered_verify/verify.py',ROOT/'development/validation/kda27/quant_transfer.py']
    reg['hashes']={str(p.relative_to(ROOT)):digest(p) for p in names}
    save(REG,reg)


def data(receipt,rows):
    if not (receipt['cost']['exit_code']==0 and not receipt['cost']['stop_reason']):raise ValueError('process or budget failure')
    if receipt['cost']['started_utc']<=read(REG)['utc']:raise ValueError('predictions did not predate the CA')
    out=qualify(receipt['chip'],ROOT/receipt['raw'],rows)
    for v in out:
        if v['timing_qualified'] and ('_5' != v['stage']['kernel'][-2:] or not v['active_matches']):
            v['timing_qualified']=False;v['reason']='unexpected installed entry or actual AIV partition'
    return out


def ratios(rows):
    if not rows:return None
    return dict(n=len(rows),distinct_shapes=len({tuple(r['shape']) for r in rows}),
                **{k:dict(wape=100*sum(abs(r[k]-r['ca_us']) for r in rows)/sum(r['ca_us'] for r in rows),
                          maximum=max(100*abs(r[k]/r['ca_us']-1) for r in rows))
                   for k in ('candidate_us','public_known_us','public_shape_only_us')})


def score():
    reg=read(REG)
    out=[];missing=[]
    for r in read(ROOT/reg['legacy_cohort'])['rows']:
        if r['role']=='repeat' and r['route']=='post_channel':
            assert r['timing_qualified'] and r['kernel'].endswith('_5') and r['measured_aiv']==r['predicted_aiv']
            actual=estimate_post_history(r['shape'],chip=r['chip']).scenarios['repeat'].latency_us
            assert abs(actual-r['candidate_us'])<1e-8
            out.append(dict(role='repeat',chip=r['chip'],shape=r['shape'],cohort='v0.24 pre-CA channel schedule',ca_us=r['ca_us'],
                            candidate_us=actual,public_known_us=r['baseline_us'],public_shape_only_us=r['unknown_us'],
                            trace_sha=r['trace_sha'],trace=r['trace'],kernel=r['kernel']))
    complete=True
    for j in reg['jobs']:
        receipt=DST/'receipts'/(j['id']+'.json')
        if not receipt.exists():complete=False;continue
        rec=read(receipt)
        if not rec.get('rows'):missing.append(dict(id=j['id'],reason=rec.get('reason')))
        for r in rec.get('rows',[]):
            if not r['timing_qualified']:
                missing.append(dict(id=j['id'],i=r['i'],reason=r.get('reason')));continue
            stage=r['stage']
            out.append(dict(chip=j['sku'],shape=r['shape'],role=r['role'],cohort='v0.28 new pre-CA',
                            ca_us=stage['ca_us'],candidate_us=r['candidate_us'],public_known_us=r['public_known_us'],
                            public_shape_only_us=r['public_shape_only_us'],future970_extrapolation_us=r['future970_extrapolation_us'],
                            trace_sha=stage['trace_sha256'],trace=stage['trace'],kernel=stage['kernel'],
                            active_aiv=stage['active_aiv'],bytes_written=stage['writes_bytes'],tested_outputs=r['all_output_elements']))
    groups={};good=complete
    for sku in SKUS:
        allrows=[r for r in out if r['chip']==sku and r['role']=='repeat']
        merged=ratios(allrows)
        groups[sku]={key:ratios(vals) for key,vals in [('all_repeat',allrows),('old',[r for r in allrows if r['cohort'].startswith('v0.24')]),
            ('new',[r for r in allrows if r['cohort'].startswith('v0.28')]),('new_first',[r for r in out if r['chip']==sku and r['role']=='entry'])]}
        good &= bool(merged and merged['distinct_shapes']>=10 and merged['candidate_us']['wape']<=25 and merged['candidate_us']['maximum']<=50
                     and merged['candidate_us']['wape']<merged['public_known_us']['wape'])
    result=dict(utc=now(),complete=complete,qualifies_to_enable_on_ordered_channel_repeat=good,metrics=groups,rows=out,missing=missing)
    save(DST/'channel-score.json',result)
    return result


def run():
    reg=read(REG);RAW.mkdir(exist_ok=True);(DST/'receipts').mkdir(exist_ok=True)
    with (RAW/'serial.lock').open('a') as lk:
        fcntl.flock(lk,fcntl.LOCK_EX|fcntl.LOCK_NB)
        for name,h in reg['hashes'].items():
            if digest(ROOT/name)!=h:raise ValueError('frozen source changed '+name)
        os.nice(10);allowed=os.sched_getaffinity(0)
        for j in reg['jobs']:
            dest=DST/'receipts'/(j['id']+'.json');folder=RAW/j['id']
            if dest.exists():continue
            if folder.exists():raise FileExistsError(folder)
            while True:
                mem=headroom()
                if mem['eligible'] and mem['idle_physical']>=104:break
                save(DST/'state.json',dict(utc=now(),phase='queued for spare physical CPUs',pid=os.getpid(),case=j['id'],host=mem))
                time.sleep(20)
            os.sched_setaffinity(0,mem['chosen_cpus'][:32])
            state=dict(utc=now(),phase='running',pid=os.getpid(),case=j['id'],cpus=sorted(os.sched_getaffinity(0)),host=mem)
            save(DST/'state.json',state)
            res=dict(chip=j['sku'],cost=capture(j['sku'],folder,j['rows'],dict(wall_s=reg['seconds_cap_each'],mib=reg['miB_cap_each'])),
                     raw=str(folder.relative_to(ROOT)),scope='v0.24 original per-lane END and END_LABEL inclusive',freeze_sha256=digest(REG))
            try:res['rows']=data(res,j['rows'])
            except (AssertionError,ValueError,OSError,KeyError,IndexError) as exc:res['reason']=repr(exc)
            save(dest,res);score();os.sched_setaffinity(0,allowed)
        final=score();save(DST/'state.json',dict(utc=now(),phase='complete',pid=os.getpid(),passed_gate=final['qualifies_to_enable_on_ordered_channel_repeat']))


def start():
    RAW.mkdir(exist_ok=True)
    with (RAW/'supervisor.log').open('a') as out:
        p=subprocess.Popen([sys.executable,str(Path(__file__).resolve()),'run'],cwd=ROOT,
                           stdin=subprocess.DEVNULL,stdout=out,stderr=subprocess.STDOUT,start_new_session=True)
        print(p.pid,flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('action',choices=('register','run','start','score'))
    {'register':register,'run':run,'start':start,'score':score}[p.parse_args().action]()
