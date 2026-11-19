"""Bounded dual-SKU evidence for the pinned native operator in a local vendor package."""
from datetime import datetime,timezone
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import json
import os
import re
import signal
import subprocess
import sys
import time

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'development/validation/sku17'))
from campaign_collect import nbytes, stages, sha

OUT=ROOT/'results/v0.19/workloads'
RAW=ROOT/'runs/workloads19'


def collect(item,vendor,registration):
    chip=item['chip']; t=item['tokens']; valid=item['valid']
    hidden=item['hidden']
    prefix=registration.get('name','attnres')
    location=RAW/f'{prefix}_{t}_{valid}_{chip}'
    job=dict(id=f'attnres_{t}_{valid}',op='attn_res_fwd',shape=[t,hidden,8,valid],nchecks=1,kernels=1)
    location.mkdir(exist_ok=False)
    caps=registration['wall_s_per_attempt'],registration['max_raw_mib_per_attempt']*2**20
    begin=datetime.now(timezone.utc).isoformat()
    assert begin>registration['utc']
    # The vendor OPP is selected before launching msopprof and its child,
    # in the same isolated CANN environment used by the established campaigns.
    launch='export ASCEND_CUSTOM_OPP_PATH="$1"; export LD_LIBRARY_PATH="$1/op_api/lib:$LD_LIBRARY_PATH"; shift; exec "$@"'
    args=['/usr/bin/time','-f','%U %S','-o',str(location/'cpu.txt'),'bash',str(ROOT/'scripts/cann-run.sh'),
          'bash','-c',launch,'attn-res-vendor',str(vendor),'msopprof','simulator','--soc-version='+chip,
          '--dump=true','--launch-count=1','--timeout=3','--output='+str(location/'profile'),
          str(ROOT/item.get('runner','build/smoke/attn_res_fwd19')),str(t),str(valid)]
    start=time.monotonic(); reason=None
    with (location/'run.log').open('w') as f:
        proc=subprocess.Popen(args,cwd=location,stdout=f,stderr=subprocess.STDOUT,start_new_session=True)
        while proc.poll() is None:
            if time.monotonic()-start>caps[0] or nbytes(location)>caps[1]:
                reason='wall or raw cap';os.killpg(proc.pid,signal.SIGKILL);proc.wait();break
            time.sleep(1)
    cpu=(location/'cpu.txt').read_text().splitlines()
    cpu_s=sum(map(float,cpu[-1].split())) if cpu and re.fullmatch(r'[\d.]+ [\d.]+',cpu[-1]) else None
    cost=dict(seconds=time.monotonic()-start,cpu_seconds=cpu_s,raw_bytes=nbytes(location),exit_code=proc.returncode,
              stop_reason=reason,started_utc=begin,finished_utc=datetime.now(timezone.utc).isoformat(),command=args[4:])
    (location/'cost.json').write_text(json.dumps(cost,indent=2)+'\n')
    try:
        # Common qualifier checks exact runtime SoC and all numeric result groups.
        windows,correct=stages(job,chip,location)
        record=dict(qualified=True,kernel=windows[0],correctness=correct)
    except ValueError as e:
        record=dict(qualified=False,reason=str(e))
    record.update(shape=job['shape'],chip=chip,cost=cost,run_log=str((location/'run.log').relative_to(ROOT)))
    return record


if __name__=='__main__':
    reg=json.loads((OUT/(sys.argv[2] if len(sys.argv)>2 else 'attn-res-registration.json')).read_text())
    for name,digest in reg['hashes'].items():
        actual=ROOT/name
        if name=='operator_model/attention_residual.py' and sha(actual)!=digest:
            # Archived pre-fix source used by the original explicit-chip forecasts.
            frozen=ROOT/'development/validation/sku19/frozen-attn-source.txt'
            assert frozen.read_text().replace('target=load_chip(chip) if chip is None or isinstance(chip,str) else chip',
                    'target=load_chip() if chip is None else load_chip(chip)')==actual.read_text()
            actual=frozen
        assert sha(actual)==digest,name
    vendor=Path(sys.argv[1]).resolve()
    assert (vendor/'op_api/lib/libcust_opapi.so').is_file()
    jobs=[i for i in reg['scenarios'] if i['chip']!='Ascend970DT']
    with ThreadPoolExecutor(reg['max_workers']) as pool:
        rows=list(pool.map(lambda j:collect(j,vendor,reg),jobs))
    (OUT/reg.get('output','attn-res-ca.json')).write_text(json.dumps(rows,indent=2)+'\n')
