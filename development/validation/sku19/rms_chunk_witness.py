"""Verify the physical tail and logical payloads of a completed 2048x5120 RMS PR export."""
from collections import Counter
import json
from pathlib import Path
import re
import sys

ROOT=Path(__file__).resolve().parents[3]
sys.path.insert(0,str(ROOT))
from development.validation.sku19.campaign import digest, read, write
from development.validation.sku19.prefill_retry import REG,TARGET,FOLDER
from development.validation.sku19.collect import qualify,PR

def check(condition,why):
    if not condition:raise ValueError(why)

def inspect(folder):
    norm=read(REG)['job']
    reference=read(ROOT/'runs/sku19/rmsnorm_2048_5120_Ascend950DT_9582/qualified.json')['stages'][0]
    trace=list(folder.glob('profile/OPPROF_*/simulator/trace.json')) or list(folder.glob('profile/OPPROF_*/device0/RmsNorm*/0/simulator/trace.json'))
    check(len(trace)==1,'one full aggregated trace')
    sim=trace[0].parent
    binary=sim.parent/'dump/aicore_binary.o'
    check(reference['binary_sha256']==digest(binary),'kernel binary mismatch')
    base=int((sim.parent/'dump/pc_start_addr.txt').read_text().strip(),16)
    log=(folder/'run.log').read_text(errors='replace')
    vendor={c:float(v) for c,v in re.findall(r'^(core\d+\.veccore\d+)\s+([.\d]+)\s+[.\d]+\s*$',log,re.M)}
    complete=[]
    for core in sim.glob('core*.veccore*'):
        data=json.loads((core/'trace.json').read_bytes())['traceEvents']
        rows=0;rstd_bytes=0;start=float('inf');stop=-float('inf');first=None;last=None
        pc={}
        for e in data:
            if e['ph']!='X':continue
            a=e['ts'];b=a+e['dur']
            if a<start:start,first=a,e['name']
            if b>stop:stop,last=b,e['name']
            if e['name'] in ('DCI','END'):
                pc[e['name']]=int(e['args']['pc_addr'],16)-base
            if e['name']!='MOV_SRC_TO_DST_ALIGNv2':continue
            m=json.loads(e['args']['detail'])
            if m['queue_type']!='MTE3':continue
            if m['burst_len']==10240:
                rows+=m['burst_number']
            else:
                rstd_bytes+=m['burst_len']*m['burst_number']
        check(first=='MOV_XD_SPR' and last in ('END_LABEL','DCI'),f'{core.name}: partial last={last}')
        check(pc=={'DCI':0x51cc,'END':0x51d0},f'{core.name}: missing executable termination')
        check(rstd_bytes==rows*4,f'{core.name}: inconsistent output byte counts')
        check(abs(stop-start-vendor[core.name])<=.011,f'{core.name}: vendor duration')
        complete.append(dict(core=core.name,rows=rows,start_us=start,end_us=stop,last=last))
    check({c['core']for c in complete}==set(vendor),'active cores')
    check(Counter(x['rows'] for x in complete)==Counter({37:55,13:1}),'all 2048 rows')
    return dict(ca_us=max(x['end_us']for x in complete)-min(x['start_us']for x in complete),
                core_witnesses=complete,rows=sum(x['rows'] for x in complete),
                kernel_sha256=digest(binary),trace=str(trace[0].relative_to(ROOT)))

if __name__=='__main__':
    j=read(REG)['job']
    result=inspect(FOLDER)
    outcome=read(FOLDER/'outcome.json')
    if not outcome['eligible']:
        record,=qualify(j,PR,FOLDER)
        check(abs(record['ca_us']-result['ca_us'])<1e-8,'aggregate differs from core shards')
        record.update(dtype='bf16',host_args=j['args'],backend='normal official full export; hardware END/DCI and complete outputs verified',
                      role='supplemental complete prefill chunk',runner_sha256=j['runner_sha256'])
        write(FOLDER/'qualified.json',record)
        outcome.update(eligible=True,reason=None,ca_us=record['ca_us'],trace_sha256=record['trace_sha256'])
        write(FOLDER/'outcome.json',outcome)
        write(TARGET/'chunk-retry-outcome.json',outcome)
    else:check(abs(result['ca_us']-outcome['ca_us'])<1e-8,'aggregate differs from core shards')
    write(TARGET/'chunk-retry-tail-witness.json',result)
