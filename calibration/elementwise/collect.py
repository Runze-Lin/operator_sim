"""Qualify the actual per-rank residual Add and broadcast Mul. Save measured-only evidence."""
import argparse
import gzip
from pathlib import Path
import re
from calibration.__main__ import digest, encoded, read, require, stage_window
from calibration.msopprof import CORE, sha
from calibration.mx_matmul.collect import MEMORY_CONFIG, capture, core_windows

CASES = {'add': (9,7168,'aclnnAdd','bf16'), 'mul': (9,4,'aclnnMul','fp32')}


def collect(root,probe,launcher,opp):
    observed=[]
    for name,(r,d,api,dtype) in CASES.items():
        for chip,memory in MEMORY_CONFIG.items():
            folder=root/(name+'-'+chip)
            if not folder.exists():
                capture(dict(chip=chip,shape=[r,d]),name,'development',folder,probe,launcher,3)
            log=(folder/'run.log').read_text()
            cost=read(folder/'cost.json')
            require(cost['exit_code']==0 and not cost['stop_reason'] and memory in log and
                    re.findall(r'^runtime_soc=(\S+)',log,re.M)==[chip] and
                    re.findall(r'^case=.+',log,re.M)==[
                        f"case={name} rows={r} width={d} rhs_width={d if name=='add' else 1} api={api}"] and
                    re.findall(r'^correctness .+',log,re.M)==[f'correctness Y count={r*d} bad=0'] and
                    'SMOKE_PASS' in log and 'All task success.' in log and 'Profiling data parse finished.' in log,
                    f'Unqualified run {folder}')
            kernel,=re.findall(r'Start profiling on kernel: (\S+)',log)
            base,=(folder/'profile').glob('OPPROF_*')
            traced=sha(base/'dump'/'aicore_binary.o')
            matches=[p for p in (opp/(name+'_apt')).glob('*.o') if kernel.startswith(p.stem+'_') and sha(p)==traced]
            require(len(matches)==1,'Need unique installed OPP ELF match')
            roster=CORE.findall(log);times={k:float(v) for k,v in roster}
            require(times and len(times)==len(roster),'Invalid physical cores')
            cores,hashes=core_windows(base/'simulator',times)
            stage=dict(kernel=kernel,kernel_binary_sha256=traced,trace_sha256=digest(encoded(hashes)),
                       ticks_per_us=1650,launched_cores=sorted(cores),cores=cores,physical_shard_sha256=hashes)
            observed.append(dict(op='aten.'+name+'.Tensor',api=api,chip=chip,provider='CANN 9.2.1 ops-math',
                                dtype=dtype,lhs=[r,d],rhs=[r,d if name=='add' else 1],attrs={'alpha':1} if name=='add' else {},
                                output=[r,d],stage=stage,ca_us=stage_window(stage),predicted_us=None,
                                correctness={'Y':{'checked':r*d,'bad':0}},probe_sha256=cost['probe_sha256'],
                                log_sha256=sha(folder/'run.log'),wall_seconds=cost['wall_seconds']))
            print(name,chip,observed[-1]['ca_us'],flush=True)
    return observed


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for arg in ('raw','probe','launcher','opp','output'):
        p.add_argument('--'+arg,required=True,type=Path)
    args=p.parse_args()
    rows=collect(args.raw.resolve(),args.probe,args.launcher,args.opp)
    args.output.write_bytes(gzip.compress(encoded(rows),mtime=0))
