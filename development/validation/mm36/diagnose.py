"""Audit a stopped Kimi CA and the existing large Matmul runs without launching a simulator."""
from pathlib import Path
from hashlib import sha256
from datetime import datetime, timezone
from collections import Counter
import json
import re
import sys

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
from development.validation.quant35.partial import extract

FREEZE = ROOT / 'results/v0.32/matmul/kimi-head/freeze.json'
RECEIPT = FREEZE.parent / 'dt.json'
OUT = ROOT / 'results/v0.36/matmul'
DIR = ROOT / 'runs/matmul32/kimi-head'
def read(p):
    return json.loads(p.read_bytes())

def sha(p):
    return sha256(p.read_bytes()).hexdigest()

def describe(log):
    text = log.read_text(errors='replace')
    case = re.search(r'case=matmul api=aclnnMatmul M=(\d+) N=(\d+) K=(\d+) dtype=BF16 layout=ND_ND', text)
    if case is None:
        return None
    shape = list(map(int, case.groups()))
    raw = log.parent
    task = raw/'stars_log0_0_task.dump'
    roster = {}
    for ev in ('start','end'):
        roster[ev] = sorted({int(c) for c in re.findall(
            rf'\[block_{ev}\]\s*: AIC, task_id=0, core_id=(\d+), block_id=\d+', text)})
    binary = list(raw.glob('profile/*/device0/MatMul*/0/dump/aicore_binary.o'))
    summaries = list(raw.glob('profile/*/device0/tmp_dump/core*_summary_log'))
    merged = list(raw.glob('profile/*/simulator/trace.json'))
    cubes = list(raw.glob('profile/*/simulator/core*.cubecore*/trace.json'))
    return dict(raw=str(raw.relative_to(ROOT)), shape=shape, sku=re.search(r'runtime_soc=(\S+)',text)[1],
                gold=f'correctness count={shape[0]*shape[1]} bad=0' in text and 'SMOKE_PASS' in text,
                kernel=re.findall(r'Start profiling on kernel: (\S+)', text), unique_stdout_aic_ids=roster,
                stars_done=task.exists() and 'TASK_DONE' in task.read_text(errors='replace'),
                temporary_summary_files=len(summaries), nonempty_temporary_summaries=sum(p.stat().st_size>0 for p in summaries),
                all_core_global_exports=len(merged), full_export_cube_shards=len(cubes),
                instruction_dump_files=sum(1 for _ in raw.glob('profile/**/*.instr_log.dump')),
                temporary_binary_hashes=[sha(p) for p in binary], cost_exists=(raw/'cost.json').is_file())

def memory():
    files = sorted((DIR/'dt/log').glob('lpddr_state*.log'))
    assert {p.name for p in files} == {f'lpddr_state{base+i}.log' for base in (0,131072) for i in range(64)}
    panels = [extract(p) for p in files]
    n = min(len(v) for v,_ in panels)
    periods = []
    for i in range(n):
        rd,wr,dmc,_ = map(sum, zip(*(v[i] for v,_ in panels)))
        periods.append(dict(end_dfi_us=i+1,read_bytes=32*rd,write_bytes=32*wr,dmc_bytes=dmc))
    return dict(all_128_closed_periods=n,read_bytes=sum(r['read_bytes'] for r in periods),
                write_bytes=sum(r['write_bytes'] for r in periods),periods=periods,
                channel_sha256={p.name:entry[1] for p,entry in zip(files,panels)},
                scope='DFI process time, common closed 1000 ns panels only. No cross-clock alignment or addresses.')

def audit():
    j, receipt = read(FREEZE),read(RECEIPT)
    assert receipt['freeze_sha256']==sha(FREEZE) and receipt['shape']==j['shape']==[32760,256,512]
    a = receipt['cost']; assert a==read(DIR/'dt/cost.json') and a['started_utc']>j['utc'] and a['exit_code']==-9 and a['stop_reason']=='wall cap'
    assert not receipt['qualified'] and receipt['ca_us'] is None
    assert a['command'][-4:]==[str(ROOT/'build/smoke/matmul28'),*map(str,j['shape'])]
    dt = describe(DIR/'dt/run.log')
    assert dt['shape']==j['shape'] and dt['sku']==receipt['chip'] and dt['unique_stdout_aic_ids']['start']==list(range(32))
    assert not dt['gold'] and not dt['stars_done'] and not dt['unique_stdout_aic_ids']['end'] and not dt['all_core_global_exports']
    assert dt['temporary_binary_hashes']==[j['installed_matmul_elf']] and dt['kernel'][0].endswith('_131074')
    large=[]
    for log in sorted((ROOT/'runs').glob('matmul*/**/run.log')):
        # For finished archives read only files containing this operator and a large M.
        tag=re.search(rb'case=matmul api=aclnnMatmul M=(\d+) ',log.read_bytes())
        if tag and int(tag[1])>=32000:
            large.append(describe(log))
    memory_sample=memory()
    preds={chip:dict(v032_us=v['us'],previous_us=v['old_us'],aic=v['active_aic'],waves=v['plan']['waves'],
                     expected_mmad=v['plan']['mmads_total']) for chip,v in j['predictions'].items()}
    return dict(utc=datetime.now(timezone.utc).isoformat(), shape=j['shape'], registration_sha256=sha(FREEZE),
                receipt_sha256=sha(RECEIPT), predictions=preds, dt=dt, cost=a,
                dt_run_log_hash=sha(DIR/'dt/run.log'),dt_ca_us=None,dt_error_pct=None,
                partial_950dt_controllers=memory_sample,by_run=large,
                kind_counts=dict(Counter(('full gold and merged file' if r['gold'] and r['all_core_global_exports'] else 'no full output') for r in large)))

if __name__=='__main__':
    data=audit()
    (OUT/'audit.json').write_text(json.dumps(data,indent=2)+'\n')
    print(data['kind_counts'], {k:v for k,v in data['partial_950dt_controllers'].items() if k not in ('periods','channel_sha256')})
