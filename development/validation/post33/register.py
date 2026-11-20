"""Freeze shape-only prices and acquire two new shapes on an existing Post slot."""
import argparse
import fcntl
import json
import os
from pathlib import Path
import re
import shutil
from time import sleep
from datetime import datetime,timezone

from prices import ROOT,OUT,read,sha
from candidate import predict
from operator_model import estimate
from development.validation.post24.collect import LIB,BIN,capture,qualify
from development.validation.kda27.quant_transfer import headroom

DT,PR,FUT='Ascend950DT_9582','Ascend950PR_957b','Ascend970DT'
RAW=ROOT/'runs/post33-small';REG=OUT/'freeze.json'
SEQ=[[1,4,208],[1,4,208],[3,4,240],[3,4,240]]
ARCHIVE=ROOT/'results/v0.28/post/state.json'
SLOT=ROOT/'runs/post28/serial.lock'

def utc():return datetime.now(timezone.utc).isoformat()
def save(p,v):
    t=p.with_suffix('.tmp');t.write_text(json.dumps(v,indent=2,allow_nan=False,ensure_ascii=False)+'\n');t.replace(p)
def rel(p):return str(p.relative_to(ROOT))


def dedup(shapes):
    catalog=[]
    for parent in (ROOT/'results').glob('v*'):
        for base in ('post','mhc_post'):
            for path in (parent/base).glob('*.json'):
                if path.stat().st_size>16*2**20 or path==REG or parent==OUT.parent:continue
                ids=[tuple(map(int,d)) for d in re.findall(rb'\[\s*(\d+)\s*,\s*(4)\s*,\s*(\d+)\s*\]',path.read_bytes())]
                if any(tuple(s) in ids for s in shapes):raise ValueError('previously reserved: '+str(path))
                catalog.append(rel(path))
    raw=[]
    for top in (ROOT/'runs',ROOT/'development/validation'):
        for folder,dirs,files in os.walk(top):
            dirs[:]=[v for v in dirs if v not in ('sources','simulator','log','log_ca','profile','record','dump','ca_workspace','__pycache__') and not v.startswith(('OPPROF_','PROF_','npusim_'))]
            for fn in ('batch.txt','run.log'):
                if fn not in files:continue
                p=Path(folder)/fn
                if p.stat().st_size>12*2**20:continue
                with p.open(errors='replace') as f:
                    for ln in f:
                        sig=re.search(r'(?:case=mhc_post api=aclnnMhcPost T=(\d+) N=4 D=(\d+)|^mhc_post \d+ (\d+) 4 (\d+)\s*$)',ln)
                        if not sig:continue
                        t,w=[int(x) for x in sig.groups() if x is not None]
                        if [t,4,w] in shapes:raise ValueError('previously launched: '+str(p))
                raw.append(rel(p))
    return dict(result_files_checked=len(catalog),host_logs_and_batches_checked=len(raw),hits=[])


def freeze():
    checks=dedup(SEQ)
    params=read(OUT/'prices.json');rows={}
    for chip in (DT,PR,FUT):
        rows[chip]=[]
        for i,shape in enumerate(SEQ):
            old=estimate('mhc_post',shape,chip=chip)
            rows[chip].append(dict(i=i,shape=shape,active=old.active_aiv,role=['first','repeat','expand from one','repeat'][i],
                              v032_shape_us=old.latency_us,**predict(shape,chip,params)))
        assert rows[chip][0]['us']==rows[chip][1]['us'] and rows[chip][2]['us']==rows[chip][3]['us']
    srcs=[Path(__file__),Path(__file__).with_name('prices.py'),Path(__file__).with_name('candidate.py'),LIB,BIN,OUT/'prices.json',
          ROOT/'development/validation/post24/collect.py',ROOT/'operator_model/post_resource.py',ROOT/'operator_model/post_history.py',
          ROOT/'operator_model/post_token_pipeline.py',ROOT/'results/v0.28/post/parameters.json',ROOT/'results/v0.25/post/prices.json',
          *[ROOT/'configs/chips'/(s+'.json') for s in (DT,PR,FUT)]]
    more=[]
    legacy=ROOT/'results/v0.32/post/forward.json'
    for r in read(legacy)['rows']:
        if r['ca_us'] is not None:continue
        receipt=ROOT/('results/v0.31/post/mhc_post-'+r['chip']+'.json') if r['role']=='isolated_large' else (
            ROOT/('results/v0.30/post/channel-'+r['chip']+'.json') if r['role']=='candidate_channel_repeat'
            else ROOT/('results/v0.29/post/split-'+r['chip']+'.json'))
        if receipt.exists():continue
        more.append(dict(chip=r['chip'],shape=r['shape'],role=r['role'],receipt=rel(receipt),us=predict(r['shape'],r['chip'],params)['us']))
    out=dict(utc=utc(),status='diagnostic proposal, production default unchanged',rows=rows,additional_prepriced=more,
        upper_tail_underestimation_tolerance=.5,budget=dict(wall_s=1800,mib=1536),slot=rel(SLOT),predecessor=rel(ARCHIVE),dedup=checks,
        checks='Four ordered launches/SKU, every element, released ELF, actual SKU and memory, all lanes physical END / terminal label / last instruction, and exact output write volume. One attempt per physical SKU; missing/partial raw kept.',
        dependencies={rel(p):sha(p) for p in srcs})
    with REG.open('x') as f:json.dump(out,f,indent=2,ensure_ascii=False);f.write('\n')


def run():
    reg=read(REG)
    if not all(sha(ROOT/p)==v for p,v in reg['dependencies'].items()):raise ValueError('changed pricing or inputs')
    RAW.mkdir(exist_ok=True)
    def state(phase,**more):save(OUT/'state.json',dict(utc=utc(),pid=os.getpid(),phase=phase,freeze_sha256=sha(REG),**more))
    os.nice(12);allowed=os.sched_getaffinity(0)
    with (RAW/'lock').open('a') as mine,SLOT.open('a') as slot:
        fcntl.flock(mine,fcntl.LOCK_EX|fcntl.LOCK_NB)
        state('queued behind v0.28 last registered Post endpoint',predecessor=rel(ARCHIVE),slot=rel(SLOT))
        fcntl.flock(slot,fcntl.LOCK_EX)
        for sku in (DT,PR):
            dst=OUT/(sku+'.json');folder=RAW/sku
            if dst.exists():continue
            jobs=reg['rows'][sku];costfile=folder/'cost.json'
            if not costfile.exists():
                if folder.exists():state('preserved interrupted acquisition',chip=sku);return
                state('checking free resources for retired slot',chip=sku)
                while True:
                    os.sched_setaffinity(0,allowed)
                    info=headroom()
                    if info['eligible'] and shutil.disk_usage(ROOT).free>=32*2**30:break
                    sleep(45)
                os.sched_setaffinity(0,info['chosen_cpus'][:32]);state('acquiring',chip=sku)
                cost=capture(sku,folder,jobs,reg['budget'])
            else:cost=read(costfile)
            try:
                if cost['exit_code'] or cost['stop_reason'] or cost['started_utc']<=reg['utc']:raise ValueError('incomplete execution')
                windows=qualify(sku,folder,jobs);result=dict(rows=windows,qualified=all(r['timing_qualified'] for r in windows))
            except (ValueError,AssertionError,KeyError,IndexError,OSError) as e:result=dict(rows=[],qualified=False,reason=repr(e))
            save(dst,dict(chip=sku,freeze_sha256=sha(REG),cost=cost,**result))
            from score import forward
            forward()
        state('finished')


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('job',choices=['freeze','run'])
    {'freeze':freeze,'run':run}[p.parse_args().job]()
