"""Pre-CA registration and one resumable low-priority experiment for two ordinary split-width shapes."""
import argparse
import fcntl
from hashlib import sha256
import json
import os
from pathlib import Path
import re
import sys
import time
from datetime import datetime, timezone
ROOT=Path(__file__).resolve().parents[3];sys.path.insert(0,str(ROOT))
from development.validation.post24.collect import BIN,LIB,capture,qualify
from development.validation.kda27.quant_transfer import headroom
from operator_model import OperatorStream,estimate
from operator_model.post_history import estimate_post_history
from operator_model.post_repeat import channel_tiles

OUT=ROOT/'results/v0.29/post'
RAW=ROOT/'runs/post29-split'
REG=OUT/'split-register.json'
SOCS=['Ascend950DT_9582','Ascend950PR_957b']
PREFIX=[112,4,768]
HOLDOUT=[[67,4,4912],[69,4,5680]]
END_OF_WORK={
    'results/v0.28/post/state.json': 'complete',
    'results/v0.28/matmul/narrow/state.json': 'finished',
    'results/v0.28/matmul/multi/state.json': 'finished',
}
# Also wait for the in-flight shapes on the large Matmul lanes at registration.
LARGE={
    'results/v0.28/matmul/prefill-status.json':'262144-128-512-dt',
    'results/v0.28/matmul/long-prefill/state.json':'131072-dt',
    'results/v0.28/matmul/wide-prefill/state.json':'DeepSeek-V4-Pro-dt',
}


def now():return datetime.now(timezone.utc).isoformat()
def read(p):return json.loads(p.read_text())
def checksum(p):return sha256(p.read_bytes()).hexdigest()
def save(p,data):
    temp=p.with_suffix('.tmp');temp.write_text(json.dumps(data,indent=2,ensure_ascii=False)+'\n');temp.replace(p)


def novel():
    patterns=[re.compile(rb'\[\s*'+rb'\s*,\s*'.join(str(v).encode()for v in shape)+rb'\s*\]') for shape in HOLDOUT]
    checked=0
    for root,dirs,files in os.walk(ROOT/'results'):
        dirs[:]=[d for d in dirs if d not in ('raw','profile','runs','.git')]
        for name in files:
            path=Path(root)/name
            if not name.endswith('.json') or name=='trace.json':continue
            if any(p.search(path.read_bytes())for p in patterns):
                raise ValueError('Prior geometry '+str(path))
            checked+=1
    target={tuple(s)for s in HOLDOUT}
    logs=list((ROOT/'runs').glob('post*/**/batch.txt'))
    for log in logs:
        for line in log.read_text().splitlines():
            if line.startswith('mhc_post ') and tuple(map(int,line.split()[2:])) in target:
                raise ValueError('Prior dispatch '+str(log))
    return dict(result_json=checked,local_post_batch_rosters=len(logs),prior_targets=[])


def register():
    if REG.exists() or RAW.exists():raise FileExistsError('Prior registration or raw path')
    novel_status=novel();targets={}
    for sku in SOCS+['Ascend970DT']:
        stream=OperatorStream(sku,track_post_history=True);previous=None;h=0;rr=[]
        for i,shape in enumerate([PREFIX]+HOLDOUT):
            forecast=stream.advance('mhc_post',shape)
            history=estimate_post_history(shape,chip=sku)
            pred=history.for_predecessor(h,previous)
            tiles=channel_tiles(shape[2],stream.chip)
            if i:
                assert tiles['count']==2 and tiles['inner']!=tiles['tail']
                assert h>=forecast.active_aiv and history.route=='post_token'
                assert forecast.projection['timing_diagnostics']['model']=='post_target_critical_aiv'
            rr.append(dict(i=i,shape=shape,active=forecast.active_aiv,prior_aiv=h,previous_shape=previous,
                           role='ordinary split width after different token shape' if i else 'prefix only',
                           state='reuse_after_token' if i else 'first',
                           aggregate_us=pred.latency_us,public_us=forecast.latency_us,
                           shape_only_us=estimate('mhc_post',shape,chip=sku).latency_us,
                           stages_us=dict(pred.stages_us),tiles=tiles))
            previous=shape;h=max(h,forecast.active_aiv)
        targets[sku]=rr
    dependencies=[Path(__file__),LIB,BIN,ROOT/'scripts/cann-run.sh',
          ROOT/'development/validation/post24/collect.py',ROOT/'development/validation/rms_ordered_verify/verify.py',
          ROOT/'development/validation/post_candidate/extract.py',ROOT/'development/validation/kda27/quant_transfer.py']
    model_files=['operator_model/'+n+'.py' for n in ('post_resource','post_repeat','post_token_pipeline','post_history','post_ordered','stream')]
    model_files+=['results/v0.25/post/prices.json','results/v0.28/post/parameters.json']+['configs/chips/'+s+'.json' for s in targets]
    save(REG,dict(utc=now(),observed=SOCS,extrapolated='Ascend970DT',novelty=novel_status,order=[PREFIX]+HOLDOUT,
                  predecessor_states=END_OF_WORK,also_wait_for_active_large_matmuls=LARGE,
                  compute_sha256={p:checksum(ROOT/p)for p in model_files},
                  capture_sha256={str(p.relative_to(ROOT)):checksum(p)for p in dependencies},
                  rows=targets,wall_seconds_per_sku=2400,raw_mib_per_sku=6144,
                  objective='Same-register honest check of the frozen ordinary issue model as an aggregate for two fragments. Use the complete correctness-checked parent and targets, matching source ELF, official inclusive envelope/retired roster, predicted active AIV and MTE3 write volume. No replacement. Hypothesis on these two shapes only if both WAPE<=25%, max<=50% and better than public on both SKUs. Production admission still requires >=10 independent geometries per SKU.'))


def status(phase,**more):
    save(OUT/'split-state.json',dict(utc=now(),phase=phase,pid=os.getpid(),registration_sha256=checksum(REG),**more))


def wait_ready():
    while True:
        pending=[]
        for name,terminal in END_OF_WORK.items():
            f=ROOT/name
            if not f.exists() or read(f).get('phase')!=terminal:pending.append(name)
        for name,active in LARGE.items():
            f=ROOT/name
            if f.exists():
                prev=read(f)
                if prev.get('case')==active and prev.get('phase') in ('acquiring','running'):
                    pending.append(name)
        if pending:
            status('queued after registered Post and Matmul work',waiting_for=pending);time.sleep(55);continue
        mem=headroom()
        # Leave >=64 continuously idle physical cores for existing higher-priority work.
        chosen=mem['chosen_cpus'][:72]
        if mem['eligible'] and mem['idle_physical']>=len(chosen)+64 and mem['free_disk_gib']>=64:
            return mem,chosen
        status('waiting for idle physical cores and disk',host=mem);time.sleep(55)


def score():
    reg=read(REG);final={};tests=[]
    for sku in SOCS:
        target=OUT/('split-'+sku+'.json')
        if not target.exists():continue
        rec=read(target)
        rr=[]
        for r in rec.get('rows',[]):
            if not r.get('i') or not r.get('timing_qualified'):continue
            stage=r['stage']
            rr.append(dict(shape=r['shape'],ca_us=stage['ca_us'],aggregate_us=r['aggregate_us'],public_us=r['public_us'],
                           shape_only_us=r['shape_only_us'],trace=stage['trace'],sha256=stage['trace_sha256']))
        stats={}
        if rr:
            for key in ('aggregate_us','public_us','shape_only_us'):
                stats[key]=dict(wape=100*sum(abs(r[key]-r['ca_us'])for r in rr)/sum(r['ca_us'] for r in rr),
                                maximum=max(100*abs(r[key]/r['ca_us']-1)for r in rr))
        passed=len(rr)==2 and stats['aggregate_us']['wape']<=25 and stats['aggregate_us']['maximum']<=50 and stats['aggregate_us']['wape']<stats['public_us']['wape']
        tests.append(passed);final[sku]=dict(n=len(rr),hypothesis_on_pair_passed=passed,stats=stats,rows=rr)
    save(OUT/'split-score.json',dict(utc=now(),by_chip=final,complete=len(final)==2,preregistered_pair_passed=len(tests)==2 and all(tests),registration_sha256=checksum(REG)))


def run():
    RAW.mkdir(exist_ok=True);reg=read(REG)
    with (RAW/'serial.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        for path,h in reg['capture_sha256'].items():
            if checksum(ROOT/path)!=h:raise ValueError('Changed collection source '+path)
        os.nice(10);allowed=os.sched_getaffinity(0)
        for sku in SOCS:
            f=OUT/('split-'+sku+'.json');folder=RAW/sku
            if f.exists():continue
            cost=folder/'cost.json'
            if folder.exists() and not cost.exists():
                status('existing raw without cost; never relaunch',sku=sku);return
            if not cost.exists():
                host,cpus=wait_ready()
                os.sched_setaffinity(0,cpus);status('capturing',sku=sku,cpus=cpus,host=host)
                try:expense=capture(sku,folder,reg['rows'][sku],dict(wall_s=reg['wall_seconds_per_sku'],mib=reg['raw_mib_per_sku']))
                finally:os.sched_setaffinity(0,allowed)
            else:expense=read(cost)
            rec=dict(sku=sku,raw=str(folder.relative_to(ROOT)),cost=expense,registration_sha256=checksum(REG))
            try:
                if expense['started_utc']<=reg['utc'] or expense['exit_code']!=0 or expense['stop_reason']:
                    raise ValueError('process, registration or budget')
                rec['rows']=qualify(sku,folder,reg['rows'][sku])
                for r in rec['rows']:
                    if r['timing_qualified'] and (not r['active_matches'] or not r['stage']['kernel'].endswith('_6')):
                        r.update(timing_qualified=False,reason='token entry or active AIV mismatch')
            except (AssertionError,OSError,ValueError,KeyError,IndexError) as e:rec['reason']=repr(e)
            save(f,rec);score()
        status('finished')


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('action',choices=['register','run','score'])
    {'register':register,'score':score,'run':run}[p.parse_args().action]()
