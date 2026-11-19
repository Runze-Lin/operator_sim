"""Freeze published backend forecasts before paired isolated CA runs; resume from receipts."""
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import json
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[3]
sys.path[:0] = [str(ROOT), str(ROOT/'development/validation/sku17'), str(ROOT/'development/workloads')]
# This script shares a basename with sku17/campaign.py; insert sku17 before the script directory.
from campaign_collect import capture, stages, sha, utc, DT, PR
import attn_res_ca
from operator_model import estimate

OUT = ROOT/'results/v0.21/fusion'
RAW = ROOT/'runs/fusion21'
FUT = 'Ascend970DT'
VENDOR = ROOT/'runs/workloads19/sources/fastbuild/vendors/custom_transformer'
CASES = {
    'attn_res_fwd': [(2,7168,8,3),(3,512,8,3),(2,4096,8,1),(4,256,8,6),(1,1024,8,7),
                     (2,2048,8,3),(5,512,8,4),(3,7168,8,1),(4,1024,8,2),(3,2048,8,4)],
    'mhc_sinkhorn': [(t,4) for t in (57,6,14,28,54,55,56,68,113,193)],
    'mhc_pre_sinkhorn': [(t,4,4096) for t in (20,33,22,26,30,36,48,56,64,80)]}


def write(path, data):
    tmp = path.with_suffix('.tmp')
    tmp.write_text(json.dumps(data, indent=2, allow_nan=False)+'\n')
    tmp.replace(path)


def holdout():
    file = OUT/'freeze.json'
    if file.exists():
        raise FileExistsError(file)
    rows = []
    paths = [ROOT/'operator_model'/f for f in ('mhc_sink_resource.py','attention_residual_timing.py','attention_residual.py','fusion_v8.py','vector_v9.py')]
    paths += [ROOT/'results'/f for f in ('v0.8/fusion/model.json','v0.9/vector/profile.json','v0.19/workloads/attn-res-prices.json')]
    paths += [ROOT/'configs/chips'/f'{chip}.json' for chip in (DT,PR,FUT)]
    for i in range(10):
        for op, shapes in CASES.items():
            s = shapes[i]
            k = s[-1]
            runner = ROOT/'build/smoke/extension_indexer_mhc'
            if op=='attn_res_fwd':
                runner = ROOT/'build/smoke'/('attn_res_fwd19' if s[1]==7168 else f'attn_res_fwd19_h{s[1]}')
                attrs = dict(addend=True, materialize=True, write_idx=k)
                args = [s[0], k]
            elif op=='mhc_sinkhorn':
                iters = [20,2,7,15,20,32,4,10,20,12][i]
                attrs = dict(num_iters=iters, return_intermediates=False)
                args = [op,*s,iters]
            else:
                attrs = dict(num_iters=20, need_backward=False)
                args = [op,*s]
            paths.append(runner)
            for chip in (DT,PR,FUT):
                start = time.perf_counter()
                pred = estimate(op, s, attrs, chip=chip)
                query_ms = (time.perf_counter()-start)*1000
                assert pred.latency_us, (op,s,pred.reason)
                rows.append(dict(op=op,shape=s,attrs=attrs,chip=chip,latency_us=pred.latency_us,query_ms=query_ms,
                                 stages_us=pred.projection['stages_us'],plan=pred.projection['target_plan'],
                                 runner=str(runner.relative_to(ROOT)),args=list(map(str,args))))
    # Run baseline outside the checkout with an isolated released wheel.
    code = """import json,sys; from operator_model import estimate,__version__; assert __version__=='0.20.0'
print(json.dumps([estimate(r['op'],r['shape'],r['attrs'],chip=r['chip']).latency_us for r in json.load(sys.stdin)]))"""
    prior = json.loads(subprocess.check_output(['/tmp/operator-sim-v20-deploy/bin/python','-I','-c',code],
                       input=json.dumps(rows),text=True,cwd='/tmp'))
    for row, old in zip(rows,prior):
        row['v020_us'] = old
    write(file, dict(utc=utc(),scope='Independent dual-SKU CA. 970DT is projection only. Single isolated launch.',
                     per_attempt_wall_s=600,raw_mib=2400,workers=2,hashes={str(p.relative_to(ROOT)):sha(p) for p in paths},
                     cases=rows))


def key(r):
    return r['op']+'_'+'_'.join(map(str,r['shape']))+'_'+r['chip']


def run():
    reg = json.loads((OUT/'freeze.json').read_text())
    for name,digest in reg['hashes'].items():
        assert sha(ROOT/name)==digest, name
    RAW.mkdir(exist_ok=True)
    receipts = OUT/'receipts'
    receipts.mkdir(exist_ok=True)
    attn_res_ca.RAW = RAW
    def one(r):
        path = receipts/(key(r)+'.json')
        if path.exists():
            return
        op,s,chip=r['op'],r['shape'],r['chip']
        d=RAW/key(r)
        j = dict(id=key(r),op=op,shape=s,nchecks=3 if op=='mhc_pre_sinkhorn' else 1,kernels=1,
                 args=r['args'],batch=None,runner=r['runner'],runner_sha256=reg['hashes'][r['runner']],
                 wall_s=reg['per_attempt_wall_s'],raw_mib=reg['raw_mib'])
        if op=='attn_res_fwd':
            p=f'attnres_h{s[1]}'
            limits=dict(utc=reg['utc'],name=p,wall_s_per_attempt=j['wall_s'],max_raw_mib_per_attempt=j['raw_mib'],
                        profiler_timeout_minutes=max(1, j['wall_s']//60))
            item=dict(chip=chip,tokens=s[0],valid=s[-1],hidden=s[1],runner=r['runner'])
            d=RAW/f'{p}_{s[0]}_{s[-1]}_{chip}'
            if not d.exists():
                row=attn_res_ca.collect(item,VENDOR,limits)
                write(path,row)
                print(key(r),row['qualified'],flush=True)
                return
        elif not d.exists():
            capture(j,chip,d,reg)
        cost=json.loads((d/'cost.json').read_text())
        try:
            ks,checks=stages(j,chip,d)
            row=dict(qualified=True,kernel=ks[0],correctness=checks)
        except ValueError as exc:
            row=dict(qualified=False,reason=str(exc))
        write(path,dict(chip=chip,shape=s,cost=cost,run_log=str((d/'run.log').relative_to(ROOT)),**row))
        print(key(r),row['qualified'],flush=True)
    with ThreadPoolExecutor(reg['workers']) as pool:
        list(pool.map(one,[r for r in reg['cases'] if r['chip']!=FUT]))


if __name__=='__main__':
    {'freeze':holdout,'run':run}[sys.argv[1]]()
