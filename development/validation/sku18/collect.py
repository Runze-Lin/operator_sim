"""Capture both fresh CA labels independently, then join them on the registered inputs."""
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta
import json
import math
from pathlib import Path
import statistics
import sys
import threading
import time
from register import DT, PR, OUT, RAW, ROOT, digest, read, utc, write

sys.path.insert(0,str(ROOT/'development/validation/sku17'))
from campaign_collect import capture, qualify


def make_pair(j,dt,pr):
    frozen=j['outputs'][0]
    pair={k:v for k,v in frozen.items() if k!='indices'}
    observed=dt['ca_us']/pr['ca_us']
    projected=pair['dt_us']/pair['pr_us']
    return dict(**pair,id=j['id'],regime=j['regime'],dtype=j['dtype'],
                dt=dt,pr=pr,dt_ca_us=dt['ca_us'],pr_ca_us=pr['ca_us'],speedup_ca=observed,
                speedup_pred=projected,speedup_old=pair['old_dt_us']/pair['old_pr_us'],
                speedup_reuse_dt=1.,pr_us_reuse_dt=pair['dt_us'],speedup_relative_error=projected/observed-1.,
                role='both DT and PR fresh frozen holdout',backend_match=True)


def summarize(records):
    start=[datetime.fromisoformat(x['started_utc'])for x in records]
    end=[dt+timedelta(seconds=x['seconds'])for dt,x in zip(start,records)]
    return dict(attempts=len(records),eligible=sum(x['eligible']for x in records),wall_s=(max(end)-min(start)).total_seconds(),
                simulator_wall_s_sum=sum(x['seconds']for x in records),cpu_s_accounted_lower_bound=sum(x['cpu_seconds'] or 0 for x in records),
                cpu_missing=sum(x['cpu_seconds'] is None for x in records),raw_bytes=sum(x['raw_bytes']for x in records))


def error_metrics(pairs):
    vals={}
    for col,ref in [('dt_us','dt_ca_us'),('pr_us','pr_ca_us'),('old_dt_us','dt_ca_us'),('old_pr_us','pr_ca_us'),
                    ('pr_us_reuse_dt','pr_ca_us'),('speedup_pred','speedup_ca'),('speedup_old','speedup_ca'),('speedup_reuse_dt','speedup_ca')]:
        abs_errs=[abs(p[col]-p[ref]) for p in pairs]
        vals[col]=dict(n=len(pairs),wape=sum(abs_errs)/sum(p[ref]for p in pairs),max_ape=max(e/p[ref]for p,e in zip(pairs,abs_errs)))
    return vals


def publish(freeze, terminal=False):
    ps=[];rs=[];costs=[];reject=[]
    for j in freeze['jobs']:
        both={}
        for chip in (DT,PR):
            d=RAW/(j['id']+'_'+chip)
            if not(d/'outcome.json').exists():continue
            o=read(d/'outcome.json');costs.append(o)
            if o['eligible']:
                r=read(d/'qualified.json');rs.append(r);both[chip]=r
            else:reject.append(o)
        if len(both)==2:ps.append(make_pair(j,both[DT],both[PR]))
    for filename,data in [('pairs',ps),('records',rs),('all-costs',costs),('rejections',reject)]:write(OUT/(filename+'.json'),data)
    if not terminal:return
    assert len(costs)==2*len(freeze['jobs']) and len({(x['id'],x['chip'])for x in costs})==len(costs)
    by_op={}
    for op in ('quant','rmsnorm'):
        subset=[p for p in ps if p['op']==op]
        if subset:
            by_op[op]={'all':error_metrics(subset)}
            for group in sorted({p['regime']for p in subset}):
                by_op[op][group]=error_metrics([p for p in subset if p['regime']==group])
    write(OUT/'paired-errors.json',by_op)
    write(OUT/'campaign-completion.json',dict(utc=utc(),**summarize(costs),pairs=len(ps),new_both_ends=True,
          counts={op:sum(p['op']==op for p in ps)for op in by_op},files={n:digest(OUT/n)for n in ['registration.json','freeze.json','pairs.json','records.json','all-costs.json','rejections.json','paired-errors.json','query-wall.json']}))


def collect():
    b=read(OUT/'freeze.json');registration=read(OUT/'registration.json')
    assert digest(OUT/'registration.json')==b['registration_sha256']
    assert all(digest(ROOT/n)==v for n,v in b['refs'].items()), 'Frozen inputs changed before capture'
    RAW.mkdir(exist_ok=True)
    lock=threading.Lock();start=time.monotonic()
    jobs=[(j,c)for j in b['jobs']for c in (DT,PR)]
    assert len(jobs)==registration['max_physical_attempts']
    retained=sum(read(p)['raw_bytes']for p in RAW.glob('*/cost.json'));reserved=0

    def measure(args):
        nonlocal retained,reserved
        j,chip=args;d=RAW/(j['id']+'_'+chip)
        if (d/'outcome.json').exists():return
        if not(d/'cost.json').exists():
            amount=j['raw_mib']*2**20
            with lock:
                if time.monotonic()-start+j['wall_s']>registration['total_wall_s']:raise RuntimeError('global wall cap')
                if retained+reserved+amount>registration['total_raw_gib']*2**30:raise RuntimeError('global raw reservation ceiling')
                reserved+=amount
            cost=capture(j,chip,d,b)
            with lock:retained+=cost['raw_bytes'];reserved-=amount
        else:cost=read(d/'cost.json')
        outcome=dict(id=j['id'],op=j['op'],shape=j['shape'],chip=chip,**cost)
        try:
            r,=qualify(j,chip,d)
            contract=[l for l in (d/'run.log').read_text(errors='replace').splitlines()if l.startswith('case=')]
            assert len(contract)==1
            r.update(role='both DT and PR fresh frozen holdout',backend='msopprof official full trace',dtype='bf16',
                     input_contract_log=contract,runner_sha256=j['runner_sha256'],host_args=j['args'],run_log=str((d/'run.log').relative_to(ROOT)))
            outcome.update(eligible=True,reason=None);write(d/'qualified.json',r)
        except (ValueError,AssertionError,OSError,KeyError,IndexError)as exc:
            outcome.update(eligible=False,reason=repr(exc))
        with lock:
            write(d/'outcome.json',outcome);publish(b);print(j['id'],chip,outcome['reason'] or 'qualified',flush=True)

    with ThreadPoolExecutor(registration['workers'])as pool:list(pool.map(measure,jobs))
    publish(b,True)

if __name__=='__main__':collect()
