"""Convert one native CANN 9.2.1 BF16 ND Matmul run into a portable CA receipt."""
import argparse
import hashlib
import json
import mmap
import re
from pathlib import Path
from .__main__ import ingest, read, require, unseal

NUMBER = rb'([-+.0-9eE]+)'
EVENT = re.compile(rb'"dur":'+NUMBER+rb',"name":"([^"]+)","ph":"X","pid":'
                   rb'"(core\d+\.(?:cube|vec)core\d+)","tid":"[^"]+","ts":'+NUMBER+rb'}')
CORE = re.compile(r'^(core\d+\.(?:cube|vec)core\d+)\s+([0-9.]+)\s+[0-9.]+\s*$',re.M)
KERNEL = re.compile(r'MatMulV3_ND_ND_ND_ND_BF16_BF16_BF16_BF16_high_performance_\d+(?:_mix_aic)?')


def sha(path):
    obj = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda:f.read(1<<20),b''):
            obj.update(block)
    return obj.hexdigest()


def boundaries(path, roster):
    with path.open('rb') as f, mmap.mmap(f.fileno(),0,access=mmap.ACCESS_READ) as data:
        require(data[-8:].rstrip().endswith(b']}'), 'Incomplete global trace')
        windows, count = {}, 0
        for m in EVENT.finditer(data):
            dur, op, pid, ts = m.groups()
            lo=float(ts); hi=lo+float(dur); name=pid.decode()
            require(0<=lo<=hi and name in roster, 'Invalid event or unlisted core')
            w=windows.setdefault(name, [lo,hi,None,0])
            w[0]=min(w[0],lo); w[1]=max(w[1],hi)
            if op==b'END':
                w[2]=lo
                w[3]+=1
            count+=1
        require(count==sum(1 for _ in re.finditer(rb'"ph"\s*:\s*"X"',data))
                and set(windows)==set(roster), 'Unsupported event format or missing core')
        result={}
        for name,(lo,hi,mark,total) in windows.items():
            require(total==1 and lo<=mark<=hi and abs(hi-lo-roster[name])<.011,
                    'Physical END or vendor duration mismatch on '+name)
            result[name]=dict(first_tick=round(lo*1650),end_tick=round(mark*1650),
                              retire_tick=round(hi*1650),end_count=total)
        return result,hashlib.sha256(data).hexdigest()


def convert(frozen, case_id, raw, library_object):
    unseal(frozen,'freeze')
    case=next((c for c in frozen['cases'] if c['id']==case_id),None)
    require(case and case['op']=='matmul' and case['api']=='aclnnMatmul' and case['attrs']=={}
            and case['dtype']=='bf16' and case['prefix'] is None, 'Adapter requires isolated BF16 ND Matmul')
    chip=case['chip']; m,n,k=case['shape']
    require(chip in ('Ascend950DT_9582','Ascend950PR_957b') and case['outputs']=={'Y':m*n},
            'Expected a 950DT/PR Matmul with all output elements')
    env={ 'Ascend950DT_9582': 'hbm3_8000M.ini', 'Ascend950PR_957b':'lpddr5_6400M_bailu.ini'}
    require(frozen['chips'][chip]['frequency_ghz']==1.65,'Unsupported CA clock')
    cost=read(raw/'cost.json')
    require(cost['exit_code']==0 and cost['stop_reason'] is None, 'Failed or incomplete run')
    log=(raw/'run.log').read_text(errors='strict')
    require(re.findall(r'^runtime_soc=(\S+)',log,re.M)==[chip] and env[chip] in log,'SKU or external-memory model mismatch')
    tag=f'case=matmul api=aclnnMatmul M={m} N={n} K={k} dtype=BF16 layout=ND_ND'
    require(re.findall(r'^case=[^\r\n]+',log,re.M)==[line for line in log.splitlines() if line.startswith(tag+' ')]
            and len([line for line in log.splitlines() if line.startswith(tag+' ')])==1,'Wrong input or multiple dispatches')
    gold=re.findall(r'^correctness[^\r\n]*',log,re.M)
    require(gold==[f'correctness count={m*n} bad=0'] and 'SMOKE_PASS' in log and 'SMOKE_FAIL' not in log
            and 'Profiling data parse finished.' in log and 'All task success.' in log, 'Missing full output or end of parse')
    kernels=re.findall(r'Start profiling on kernel: (\S+)',log)
    require(len(kernels)==1 and KERNEL.fullmatch(kernels[0]),'Unexpected runtime kernel')
    profiles=list((raw/'profile').glob('OPPROF_*'))
    require(len(profiles)==1, 'Expected one profile')
    base=profiles[0]
    actual=base/'dump/aicore_binary.o'
    require(library_object.resolve()!=actual.resolve() and library_object.name=='MatMulV3_ND_ND_ND_ND_BF16_BF16_BF16_BF16_high_performance.o',
            'Supply the independently installed OPP MatMulV3 object')
    reference_hash=sha(library_object)
    require(reference_hash==sha(actual),'Executed image differs from installed OPP')
    published=CORE.findall(log); roster={name:float(us) for name,us in published}
    require(published and len(published)==len(roster), 'Missing or duplicate official per-core rows')
    cores, trace_sha=boundaries(base/'simulator/trace.json',roster)
    # A collector may copy only this small stamp at run registration.
    binding=raw/'freeze.sha256'
    receipt=dict(schema=1,kind='ca-receipt',case={k:v for k,v in case.items() if k not in (
        'predicted_us','query_ms','extrapolated_970dt_us')},provider='CANN 9.2.1 ops-nn',timing='all-core-instruction',
        complete=True,started_utc=cost['started_utc'],finished_utc=cost['finished_utc'],run_log_sha256=sha(raw/'run.log'),
        correctness={'Y':{'checked':m*n,'bad':0}},stages=[dict(kernel=kernels[0],kernel_binary_sha256=reference_hash,
        trace_sha256=trace_sha,ticks_per_us=1650,launched_cores=sorted(roster),cores=cores)])
    if binding.exists():
        receipt['freeze_sha256']=binding.read_text().strip()
    ingest(frozen,receipt)
    return receipt


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    for field in ('freeze','raw','library_object','output'):
        parser.add_argument('--'+field.replace('_','-'),type=Path,required=True)
    parser.add_argument('--case',required=True)
    args=parser.parse_args()
    try:
        doc=convert(read(args.freeze),args.case,args.raw,args.library_object)
        with args.output.open('x') as f:json.dump(doc,f,indent=2); f.write('\n')
    except (OSError,KeyError,TypeError,ValueError) as err:
        parser.exit(1,str(err)+'\n')
