"""Capture or reprocess a single contiguous AntiMxQuant call and its full BF16 golden."""
import argparse
from math import prod
from pathlib import Path
import re

from calibration.__main__ import CASE_FIELDS, digest, encoded, ingest, read, require, stamp, unseal
from calibration.msopprof import CORE, sha
from calibration.mx_matmul.collect import MEMORY_CONFIG, capture, core_windows, write_json
from calibration.quant_mxfp4 import terminal_phases

STEM = 'AntiMxQuant_e2m1tobf16_high_performance'


def collect(case, raw, opp, freeze=None):
    """Development runs can precede a model freeze; prospective runs must match it."""
    shape, chip = case['shape'], case['chip']
    require(case['op'] == 'cache_load' and case['dtype'] == 'mxfp4'
            and case['contract'] == 'cache_load_mxfp4_antimx' and case['api'] == 'aclnnAntiMxQuant'
            and case['prefix'] is None and case['attrs'] == {'implementation':'aclnnAntiMxQuant'}
            and case['outputs'] == {'Y':prod(shape)} and chip in MEMORY_CONFIG
            and 2 <= len(shape) <= 7 and all(d > 0 for d in shape) and shape[-1]%2 == 0,
            'Native probe input does not match the model contract')
    cost = read(raw/'cost.json')
    require(cost['exit_code'] == 0 and cost['stop_reason'] is None, 'Failed CA')
    log = (raw/'run.log').read_text()
    require(re.findall(r'^runtime_soc=(\S+)',log,re.M)==[chip] and MEMORY_CONFIG[chip] in log,
            'Invalid runtime or memory model')
    require(re.findall(r'^case=.+',log,re.M)==[
        'case=cache_load api=aclnnAntiMxQuant axis=-1 group=32 dst=bf16 shape='+'x'.join(map(str,shape))],
        'Shape or dispatch differs')
    require(re.findall(r'^correctness .+',log,re.M)==[f"correctness Y count={prod(shape)} bad=0"] and
            'SMOKE_FAIL' not in log and all(x in log for x in ('SMOKE_PASS','All task success.','Profiling data parse finished.')),
            'Incomplete numerical check or profiling')
    kernels=re.findall(r'Start profiling on kernel: (\S+)',log)
    require(len(kernels)==1 and re.fullmatch(STEM+r'_\d+(?:_mix_aiv)?',kernels[0]),'Unexpected installed kernel')
    folders=list((raw/'profile').glob('OPPROF_*'))
    require(len(folders)==1,'Expected an isolated profile')
    base=folders[0]
    obj=opp/'anti_mx_quant'/(STEM+'.o')
    require(obj.resolve()!=(base/'dump/aicore_binary.o').resolve() and sha(obj)==sha(base/'dump/aicore_binary.o'),'Different installed ELF')
    roster=CORE.findall(log)
    require(roster and len({name for name,_ in roster})==len(roster) and all('.veccore' in name for name,_ in roster),'Missing unique AIV roster')
    cores, hashes=core_windows(base/'simulator',{name:float(t) for name,t in roster})
    binding=(raw/'freeze.sha256').read_text().strip()
    if freeze is not None:
        require(freeze['chips'][chip]['frequency_ghz']==1.65,'Trace clock differs')
        if binding=='development':
            require(stamp(cost['finished_utc'])<=stamp(freeze['created_utc']),'Development probe postdates freeze')
        else:
            require(freeze['sha256']==binding,'Incorrect freeze')
    receipt=dict(schema=1,kind='ca-receipt',case={k:case[k] for k in CASE_FIELDS},
                 provider='CANN 9.2.1 ops-nn',timing='all-core-instruction',complete=True,
                 started_utc=cost['started_utc'],finished_utc=cost['finished_utc'],
                 run_log_sha256=sha(raw/'run.log'),correctness={'Y':{'checked':prod(shape),'bad':0}},
                 stages=[dict(kernel=kernels[0],kernel_binary_sha256=sha(obj),physical_shard_sha256=hashes,
                              trace_sha256=digest(encoded(hashes)),ticks_per_us=1650,launched_cores=sorted(cores),cores=cores)])
    if binding!='development':
        receipt['freeze_sha256']=binding
    write_json(raw/'receipt.json',receipt)
    write_json(raw/'phases.json',terminal_phases(base,cores))
    if freeze is not None:
        write_json(raw/'measurement.json',ingest(freeze,receipt))


def main():
    p=argparse.ArgumentParser(description=__doc__)
    g=p.add_mutually_exclusive_group(required=True)
    g.add_argument('--freeze',type=Path)
    g.add_argument('--development',type=Path,help='Development manifest; its observations are excluded from independent scores')
    p.add_argument('--case',required=True)
    p.add_argument('--raw',type=Path,required=True)
    p.add_argument('--opp',type=Path,required=True)
    p.add_argument('--probe',type=Path)
    p.add_argument('--launcher',type=Path)
    p.add_argument('--minutes',type=int,default=4)
    p.add_argument('--convert-only',action='store_true')
    a=p.parse_args()
    frozen=unseal(read(a.freeze),'freeze') if a.freeze else None
    doc=frozen or read(a.development)
    c=next(x for x in doc['cases'] if x['id']==a.case)
    raw=a.raw.resolve()
    if not a.convert_only:
        require(a.probe and a.minutes>=1,'Expected native probe and a positive timeout')
        capture(c,'cache_load',frozen['sha256'] if frozen else 'development',raw,a.probe,a.launcher,a.minutes)
    collect(c,raw,a.opp,frozen)
    print(c['id'],len(read(raw/'receipt.json')['stages'][0]['cores']),'AIV')
if __name__=='__main__':
    main()
