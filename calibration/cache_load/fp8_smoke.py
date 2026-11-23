"""Qualify an observed FP8/per-row cache-load implementation with no timing-model fit."""
import argparse
from pathlib import Path
import re

from calibration.__main__ import digest, encoded, read, require, stage_window
from calibration.msopprof import CORE, sha
from calibration.mx_matmul.collect import MEMORY_CONFIG, capture, core_windows, write_json

STEM='AscendAntiQuantV2_650baea70767c4bd7838a0d2f6bb5823_high_performance'
SHAPE=[1,128,512]


def qualify(raw, sku, opp):
    cost=read(raw/'cost.json')
    log=(raw/'run.log').read_text()
    require(cost['exit_code']==0 and cost['stop_reason'] is None, 'Unsuccessful capture')
    require(re.findall(r'^runtime_soc=(\S+)',log,re.M)==[sku] and MEMORY_CONFIG[sku] in log,'Wrong SKU')
    require(re.findall(r'^case=.+',log,re.M)==[
        'case=cache_load_fp8 api=aclnnAscendAntiQuant sqrt=false offset=null dst=bf16 shape=1x128x512'],
        'Different FP8 dispatch')
    require(re.findall(r'^correctness .+',log,re.M)==['correctness Y count=65536 bad=0'] and
            'SMOKE_FAIL' not in log and all(s in log for s in ('SMOKE_PASS','All task success.','Profiling data parse finished.')),
            'Incomplete host verification')
    kernel=re.findall(r'Start profiling on kernel: (\S+)',log)
    require(kernel==[STEM+'_2'],'Unexpected dispatched binary')
    folders=list((raw/'profile').glob('OPPROF_*'))
    require(len(folders)==1,'Expected a single profiling directory')
    obj=opp/'ascend_anti_quant_v2'/(STEM+'.o')
    image=folders[0]/'dump/aicore_binary.o'
    require(obj.resolve()!=image.resolve() and sha(obj)==sha(image),'Installed ELF differs')
    roster=CORE.findall(log)
    require(roster and len({k for k,_ in roster})==len(roster) and all('.veccore' in k for k,_ in roster),'Expected unique AIV workers')
    cores,hashes=core_windows(folders[0]/'simulator',{k:float(t) for k,t in roster})
    stage=dict(kernel=kernel[0],kernel_binary_sha256=sha(image),physical_shard_sha256=hashes,
               trace_sha256=digest(encoded(hashes)),ticks_per_us=1650,launched_cores=sorted(cores),cores=cores)
    result=dict(schema=1,scope='candidate implementation, CA-only; system capture has no dispatch identity',
                api='aclnnAscendAntiQuant',provider='CANN 9.2.1 ops-nn',chip=sku,shape=SHAPE,
                x_dtype='fp8_e4m3fn',scale_dtype='fp32',scale_shape=[128,1],output_dtype='bf16',
                sqrt_mode=False,offset=None,stages=[stage],ca_us=stage_window(stage),predicted_us=None,
                correctness={'Y':{'checked':65536,'bad':0}},started_utc=cost['started_utc'],finished_utc=cost['finished_utc'],
                wall_seconds=cost['wall_seconds'],probe_sha256=cost['probe_sha256'],run_log_sha256=sha(raw/'run.log'))
    write_json(raw/'fp8-smoke.json',result)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--chip',choices=MEMORY_CONFIG,required=True)
    for name in ('raw','opp'):
        p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--probe',type=Path)
    p.add_argument('--launcher',type=Path)
    p.add_argument('--minutes',type=int,default=4)
    p.add_argument('--convert-only',action='store_true')
    a=p.parse_args()
    raw=a.raw.resolve()
    if not a.convert_only:
        require(a.probe and a.minutes>=1,'Supply a compiled probe and a positive budget')
        capture({'chip':a.chip,'shape':SHAPE},'cache_load_fp8','development',raw,a.probe,a.launcher,a.minutes)
    qualify(raw,a.chip,a.opp)


if __name__=='__main__':
    main()
