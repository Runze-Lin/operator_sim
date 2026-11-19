"""Retry the exact frozen shapes once if simulator/host produced no numerical judgement."""
import json
import sys
import time
from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path

HERE=Path(__file__).resolve().parent
spec=spec_from_file_location('frozen_fusion21',HERE/'campaign.py')
camp=module_from_spec(spec);spec.loader.exec_module(camp)
OUT,RAW,DT,PR=camp.OUT,camp.RAW,camp.DT,camp.PR
def read(p):return json.loads(p.read_text())

def main():
    reg=read(OUT/'freeze.json')
    for name,digest in reg['hashes'].items():
        assert camp.sha(camp.ROOT/name)==digest
    jobs=[r for r in reg['cases'] if r['chip']!=camp.FUT]
    while len(list((OUT/'receipts').glob('*.json')))<len(jobs):
        time.sleep(40)
    dest=OUT/'retries'
    dest.mkdir(exist_ok=True)
    selected=[]
    for r in jobs:
        o=read(OUT/'receipts'/(camp.key(r)+'.json'))
        if o['qualified']:continue
        log=(camp.ROOT/o['run_log']).read_text(errors='replace')
        if 'SMOKE_FAIL' in log or ' SMOKE_PASS' in log or 'bad=0' in log:continue
        selected.append(r)
    policy=OUT/'retry-registration.json'
    if not policy.exists():
        camp.write(policy,dict(utc=camp.utc(),rule='One serial retry of the same public forecast, input, binary, vendor libraries and SKU for original runs without a numerical judgment. Physical profiler gets the full original 600 s cap. Original failures retained.',
                              cases=[camp.key(r) for r in selected]))
    policy=read(policy)
    camp.attn_res_ca.RAW=RAW
    for r in selected:
        name=camp.key(r);path=dest/(name+'.json')
        if path.exists():continue
        s=r['shape'];chip=r['chip'];assert name in policy['cases']
        if r['op']=='attn_res_fwd':
            limits=dict(utc=policy['utc'],name=f'retry_attnres_h{s[1]}',wall_s_per_attempt=reg['per_attempt_wall_s'],
                        max_raw_mib_per_attempt=reg['raw_mib'],profiler_timeout_minutes=10)
            item=dict(chip=chip,tokens=s[0],valid=s[-1],hidden=s[1],runner=r['runner'])
            row=camp.attn_res_ca.collect(item,camp.VENDOR,limits)
        else:
            d=RAW/(name+'-retry')
            j=dict(id=name,op=r['op'],shape=s,nchecks=3 if r['op']=='mhc_pre_sinkhorn' else 1,kernels=1,
                   batch=None,args=r['args'],runner=r['runner'],runner_sha256=reg['hashes'][r['runner']],
                   wall_s=reg['per_attempt_wall_s'],raw_mib=reg['raw_mib'])
            cost=camp.capture(j,chip,d,policy)
            row=dict(chip=chip,shape=s,cost=cost,run_log=str((d/'run.log').relative_to(camp.ROOT)))
            try:
                ks,correct=camp.stages(j,chip,d)
                row.update(qualified=True,kernel=ks[0],correctness=correct)
            except ValueError as exc:row.update(qualified=False,reason=str(exc))
        camp.write(path,row)
        print(name,row['qualified'],flush=True)

if __name__=='__main__':main()
