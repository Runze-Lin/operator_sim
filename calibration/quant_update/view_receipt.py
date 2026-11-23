"""Validate all kernels dispatched by the Kimi [9,128] noncontiguous call."""
import argparse
from pathlib import Path
import re

from calibration.__main__ import digest, encoded, read, require, seal, unseal
from calibration.msopprof import CORE, sha
from calibration.mx_matmul.collect import MEMORY_CONFIG, core_windows, write_json


def verify(raw, opp):
    contract=unseal(read(raw.with_suffix('.contract.json')),'layout-capability')
    require((raw/'freeze.sha256').read_text().strip()==contract['sha256'],'Capability run has wrong contract')
    cost=read(raw/'cost.json')
    log=(raw/'run.log').read_text()
    require(cost['exit_code']==0 and '--launch-count=3' in cost['command']
            and log.count('Multi SoC in one chip Created')==1
            and log.count('Start simulation with m_socCycle: 0',log.index('Start profiling on kernel:'))==0
            and re.findall(r'^runtime_soc=(\S+)',log,re.M)==[contract['chip']]
            and MEMORY_CONFIG[contract['chip']] in log and 'All task success.' in log and 'Profiling data parse finished.' in log,
            'Require one uninterrupted 950 CA run with all launches enabled')
    require(re.findall(r'^view_stride=.*$',log,re.M)==['view_stride=6288,1 storage_offset=6144 backing_bytes=113184']
            and re.findall(r'^correctness[^\r\n]+',log,re.M)==['correctness Y count=1152 bad=0','correctness scale count=36 bad=0']
            and 'SMOKE_PASS' in log and 'SMOKE_FAIL' not in log,'Incorrect view or output')
    kernels=re.findall(r'Start profiling on kernel: (\S+)',log)
    require(len(kernels)==2 and re.fullmatch(r'Slice_[0-9a-f]+_high_performance_\d+',kernels[0])
            and kernels[1]=='DynamicMxQuant_bf16toe2m1_high_performance_33_mix_aiv','Actual kernel order changed')
    tables=re.findall(r'core_name[^\n]+\n((?:core\d+\.[^\n]+\n)+)',log)
    require(len(tables)==len(kernels),'Missing per-kernel vendor durations')
    base,=list((raw/'profile').glob('OPPROF_*'))
    stages=[]
    for i,(kernel,table) in enumerate(zip(kernels,tables)):
        root=base/kernel/'0'
        obj=(opp/'ops_nn/dynamic_mx_quant/DynamicMxQuant_bf16toe2m1_high_performance.o' if i else
             opp/'ops_math/slice_apt'/(kernel.rsplit('_',1)[0]+'.o'))
        require(sha(obj)==sha(root/'dump/aicore_binary.o'),'Executed kernel differs from OPP')
        roster={n:float(us) for n,us in CORE.findall(table)}
        cores,hashes=core_windows(root/'simulator',roster)
        lo=min(c['first_tick'] for c in cores.values());hi=max(c['retire_tick'] for c in cores.values())
        stages.append(dict(kernel=kernel,library_object=str(obj.relative_to(opp)),object_sha256=sha(obj),
                           cores=cores,shard_sha256=hashes,trace_sha256=digest(encoded(hashes)),
                           first_tick=lo,retire_tick=hi,window_us=(hi-lo)/1650))
    interval=stages[1]['first_tick']-stages[0]['retire_tick']
    require(interval>=0,'Stage timestamps are out of order')
    return seal(dict(schema=1,kind='layout-observation',contract=contract,started_utc=cost['started_utc'],
                     finished_utc=cost['finished_utc'],wall_seconds=cost['wall_seconds'],probe_sha256=cost['probe_sha256'],
                     run_log_sha256=sha(raw/'run.log'),provider='CANN 9.2.1 ops_math and ops_nn',stages=stages,
                     ticks_per_us=1650,interkernel_idle_us=interval/1650,
                     device_span_us=(stages[-1]['retire_tick']-stages[0]['first_tick'])/1650,
                     sum_kernel_windows_us=sum(s['window_us'] for s in stages),
                     correctness={'Y':{'checked':1152,'bad':0},'scale':{'checked':36,'bad':0}},
                     status='measured-layout-only',predicted_us=None))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for a in ('raw','opp','output'):
        p.add_argument('--'+a,type=Path,required=True)
    args=p.parse_args()
    write_json(args.output,verify(args.raw.resolve(),args.opp.resolve()))
