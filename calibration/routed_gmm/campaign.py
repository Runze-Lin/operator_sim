"""Serial grouped A4W4 CA campaign. Never consume an independent case before the freeze."""
import argparse
import os
import shutil
import signal
import json
import re
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path

from calibration.__main__ import require
from calibration.msopprof import CORE, sha
from calibration.mx_matmul.collect import core_windows, MEMORY_CONFIG
from calibration.mx_matmul.native_phases import summarize

ROOT = Path(__file__).resolve().parents[2]
CHIPS = {'dt':'Ascend950DT_9582', 'pr':'Ascend950PR_957b'}
# Contract: counts, N, K. Every entry prices a single grouped projection with no bias.
DEV = [([1,2],128,256), ([2,2,2,2],256,512), ([0,11,0,23],512,1024)]
HOLD = [([1,1,1],64,128), ([2,0,1],160,256), ([3,1,0,5],128,320),
        ([1]*8,256,512), ([9]*6,5120,2304), ([1]*6,128,768), ([7,2,9,4],192,384),
        ([16,16],256,640), ([9,3,0,1],384,1024), ([1]*12,128,1536),
        ([32,64,48,16],256,1024)]


def write(path, obj):
    path.write_text(json.dumps(obj, indent=2, allow_nan=False)+'\n')


def price(shapes):
    from operator_model import estimate
    return {f'{i:02d}-{sku}': {'counts':h,'shape':[sum(h),n,k],'sku':chip,
            'prediction':estimate('routed_grouped_matmul',(sum(h),n,k),
                {'expert_counts':h,'implementation':'aclnnGroupedMatmulV5'},dtype='mxfp4',chip=chip).latency_us}
            for i,(h,n,k) in enumerate(shapes) for sku,chip in CHIPS.items()}


def receipt(raw, counts, n, k, chip, opp):
    log=(raw/'run.log').read_text(errors='strict')
    ids=re.findall(r'Start profiling on kernel: (\S+)',log)
    require(len(ids)==1 and re.fullmatch(r'GroupedMatmul_[a-f0-9]{32}_\d+',ids[0]),'Unexpected kernel')
    h=','.join(map(str,counts)); m=sum(counts)
    tag=f'api=aclnnGroupedMatmulV5 M={m} N={n} K={k} groups={h} group_list_type=1 split_item=3 storage=GNK dtype=MXFP4xMXFP4_to_BF16 scale=E8M0 workspace='
    require(re.findall(r'^runtime_soc=(\S+)',log,re.M)==[chip] and MEMORY_CONFIG[chip] in log,'Wrong CA SKU/memory model')
    require(len(re.findall('^'+re.escape(tag)+r'\d+$',log,re.M))==1 and 'SMOKE_PASS' in log and
            'SMOKE_FAIL' not in log and re.findall(r'^correctness[^\r\n]+',log,re.M)==[f'correctness count={m*n} bad=0']
            and 'All task success.' in log and 'Profiling data parse finished.' in log,'Failed correctness/identity/profile')
    base,=(raw/'profile').glob('OPPROF_*')
    obj=opp/(ids[0].rsplit('_',1)[0]+'.o')
    require(sha(obj)==sha(base/'dump/aicore_binary.o'),'Runtime binary does not match independently installed OPP')
    rows=CORE.findall(log); roster={a:float(b) for a,b in rows}
    require(roster and len(roster)==len(rows),'Absent/duplicate core')
    cores,hashes=core_windows(base/'simulator',roster)
    phases=summarize(raw)
    require(phases['fragments'] and all(f['a']==f['b']=='fp4_e2m1' for f in phases['fragments']),'Incorrect MMAD precision')
    # Device matrix instruction output M/N/K can include padding or tail splitting.
    result={'schema':1,'provider':'CANN 9.2.1 ops-transformer','shape':[m,n,k],'expert_counts':counts,'chip':chip,
            'api':'aclnnGroupedMatmulV5','binary':obj.name,'kernel':ids[0],'elf_sha256':sha(obj),
            'log_sha256':sha(raw/'run.log'),'core_shard_hashes':hashes,'cores':cores,
            'ticks_per_us':1650,'output_checked':m*n,'mismatches':0,'phases':phases,
            'ca_us':(max(v['retire_tick'] for v in cores.values())-min(v['first_tick'] for v in cores.values()))/1650}
    require(abs(result['ca_us']-phases['full_window_us'])<.001,'Phase/receipt clock disagreement')
    return result


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('phase',choices=['dev','freeze','hold'])
    p.add_argument('--raw',type=Path,required=True); p.add_argument('--probe',type=Path,required=True)
    p.add_argument('--opp',type=Path,required=True);p.add_argument('--freeze',type=Path)
    p.add_argument('--launcher',type=Path,help='Optional shell wrapper that activates CANN before msopprof')
    args=p.parse_args(); root=args.raw.resolve();root.mkdir(parents=True,exist_ok=True)
    model=ROOT/'src/operator_model/operators/moe'
    fingerprint={p.name:sha(p) for p in sorted(model.glob('*.py'))}
    if args.phase=='freeze':
        require(args.freeze and not args.freeze.exists(),'Supply a new freeze file')
        frozen={'utc':datetime.now(timezone.utc).isoformat(),'model':fingerprint,'probe':sha(args.probe),'cases':price(HOLD)}
        require(all(c['prediction'] is not None for c in frozen['cases'].values()),'Need priced holdout predictions')
        write(args.freeze, frozen); return
    if args.phase=='hold':
        frozen=json.loads(args.freeze.read_text())
        require(frozen['model']==fingerprint and frozen['probe']==sha(args.probe) and frozen['cases']==price(HOLD),
                'Frozen implementation, geometry or prices changed')
    items=DEV if args.phase=='dev' else HOLD
    for i,(h,n,k) in enumerate(items):
        for sku,chip in CHIPS.items():
            raw=root/f'{args.phase}-{sku}-{i:02d}'
            if (raw/'receipt.json').exists():continue
            raw.mkdir(exist_ok=True)
            if not (raw/'run.log').exists():
                command=['msopprof','simulator',
                         '--soc-version='+chip,'--dump=false','--timeout=15','--output='+str(raw/'profile'),
                         str(args.probe.resolve()),','.join(map(str,h)),str(n),str(k)]
                if args.launcher:
                    command=['bash',str(args.launcher.resolve(strict=True)),*command]
                require(shutil.disk_usage(raw).free >= 75*(1 << 30),'Shared-disk reserve is below 75 GiB')
                start=datetime.now(timezone.utc).isoformat(); wall=time.monotonic()
                with (raw/'run.log').open('x') as f:
                    proc = subprocess.Popen(command,stdout=f,stderr=subprocess.STDOUT,cwd=raw,start_new_session=True)
                    stop = None
                    while proc.poll() is None:
                        if shutil.disk_usage(raw).free < 75*(1 << 30):
                            stop = '75 GiB shared-disk reserve'
                        elif time.monotonic()-wall > 16*60:
                            stop = '16 min wall limit (msopprof 15 min)'
                        if stop:
                            os.killpg(proc.pid, signal.SIGTERM)
                            try: proc.wait(timeout=8)
                            except subprocess.TimeoutExpired:
                                os.killpg(proc.pid, signal.SIGKILL); proc.wait()
                            break
                        time.sleep(2)
                    cost={'started_utc':start,'wall_seconds':time.monotonic()-wall,
                          'exit_code':proc.returncode,'stop_reason':stop}
                    write(raw/'cost.json',cost)
            if (raw/'cost.json').exists():
                cost=json.loads((raw/'cost.json').read_text())
                if cost.get('exit_code') not in (None,0) or cost.get('stop_reason'):
                    print(raw.name, 'unqualified', cost, flush=True)
                    continue
            row=receipt(raw,h,n,k,chip,args.opp)
            row['cost']=json.loads((raw/'cost.json').read_text()) if (raw/'cost.json').exists() else None
            if args.phase=='hold':
                row.update(freeze_sha256=sha(args.freeze),predicted_us=frozen['cases'][f'{i:02d}-{sku}']['prediction'])
            write(raw/'receipt.json',row)
            print(raw.name,row['ca_us'],row['phases']['active_mmad_aics'],flush=True)


if __name__=='__main__':
    main()
