"""Single-process CANN 9.2.1 collector for BF16 -> E4M3FN plus FP32 row scales."""
import argparse
from pathlib import Path
import re
import json

from calibration.__main__ import CASE_FIELDS, digest, encoded, ingest, read, require, unseal
from calibration.msopprof import CORE, sha
from calibration.mx_matmul.collect import MEMORY_CONFIG, capture, core_windows, write_json
from calibration.mx_matmul.shards import trace_events


def terminal_phases(base, cores):
    core = max(cores, key=lambda k: cores[k]['retire_tick'])
    origin = min(c['first_tick'] for c in cores.values())
    dma, vector = [], []
    for e in trace_events(base/'simulator'/core/'trace.json'):
        if e['ph'] != 'X':
            continue
        if e['name'] != 'MOV_SRC_TO_DST_ALIGNv2' and not e['name'].startswith('RV_V'):
            continue
        spec = json.loads(e['args']['detail'])
        issue, retire = round(e['ts']*1650)-origin, round((e['ts']+e['dur'])*1650)-origin
        if e['name'] == 'MOV_SRC_TO_DST_ALIGNv2':
            dma.append(dict(kind=spec['instr_type'], bytes=spec['processed_bytes'], issue=issue, retire=retire))
        elif spec['instr_type']=='RVECEX':
            vector.append((issue,retire))
    return dict(core=core, dma=dma, vector_events=len(vector), vector_first=min(x[0] for x in vector),
                vector_last=max(x[1] for x in vector), end=cores[core]['retire_tick']-origin)


def collect_batch(manifest, raw, probe, opp, launcher, minutes):
    frozen = manifest.get('kind') == 'freeze'
    binding = unseal(manifest,'freeze')['sha256'] if frozen else digest(encoded(manifest))
    for case in manifest['cases']:
        directory = raw / case['id']
        rows, cols = case['shape']
        require(case['api'] == 'aclnnDynamicQuantV2' and case['op'] == 'quant_token_fp8'
                and case['contract'] == 'quant_token_fp8' and case['prefix'] is None
                and case['attrs'] == {} and case['dtype'] == 'bf16'
                and case['outputs'] == {'Y': rows*cols, 'scale': rows}
                and case['chip'] in MEMORY_CONFIG, 'Unsupported native case')
        if not directory.exists():
            capture(case,'token_quant',binding,directory,probe,launcher,minutes)
        require((directory/'freeze.sha256').read_text().strip() == binding, 'Existing run belongs to another plan')
        cost = read(directory/'cost.json')
        log = (directory/'run.log').read_text()
        require(cost['exit_code']==0 and cost['stop_reason'] is None and 'SMOKE_PASS' in log
                and 'Profiling data parse finished.' in log and 'All task success.' in log
                and re.findall(r'^runtime_soc=(\S+)',log,re.M)==[case['chip']]
                and MEMORY_CONFIG[case['chip']] in log, f'CA did not finish on requested SKU: {directory}')
        require(re.findall(r'^case=.*',log,re.M) == [f'case=token_quant rows={rows} cols={cols} api=aclnnDynamicQuantV2']
                and re.findall(r'^correctness.*',log,re.M) == [f'correctness {k} count={n} bad=0' for k,n in case['outputs'].items()],
                'Numerics or input shape mismatch')
        kernels = re.findall(r'Start profiling on kernel: (\S+)',log)
        stem = 'DynamicQuantV2_bf16_float8_e4m3fn_high_performance'
        require(len(kernels)==1 and re.fullmatch(stem+r'_\d+(?:_mix_aiv)?',kernels[0]), 'Unexpected kernel '+str(kernels))
        base, = (directory/'profile').glob('OPPROF_*')
        obj = opp/'dynamic_quant_v2'/(stem+'.o')
        require(sha(obj)==sha(base/'dump/aicore_binary.o'), 'Installed OPP binary differs')
        roster = CORE.findall(log)
        require(roster and len(dict(roster))==len(roster),'Missing per-core listing')
        cores, hashes = core_windows(base/'simulator',{k:float(us) for k,us in roster})
        stage = dict(kernel=kernels[0],kernel_binary_sha256=sha(obj),physical_shard_sha256=hashes,
                     trace_sha256=digest(encoded(hashes)),ticks_per_us=1650,launched_cores=sorted(cores),cores=cores)
        receipt = dict(schema=1,kind='ca-receipt',case={k:case[k] for k in CASE_FIELDS},provider='CANN 9.2.1 ops-nn',
                       timing='all-core-instruction',complete=True,started_utc=cost['started_utc'],finished_utc=cost['finished_utc'],
                       stages=[stage],run_log_sha256=sha(directory/'run.log'),
                       correctness={k:{'checked':n,'bad':0} for k,n in case['outputs'].items()})
        if frozen:
            receipt['freeze_sha256']=binding
            write_json(directory/'measurement.json',ingest(manifest,receipt))
        write_json(directory/'receipt.json',receipt)
        write_json(directory/'phases.json',terminal_phases(base,cores))
        print(case['id'],len(cores),'AIVs',flush=True)


if __name__ == '__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    for arg in ('manifest','raw','probe','opp'):
        parser.add_argument('--'+arg,type=Path,required=True)
    parser.add_argument('--launcher',type=Path)
    parser.add_argument('--minutes',type=int,default=3)
    args=parser.parse_args()
    collect_batch(read(args.manifest),args.raw.resolve(),args.probe,args.opp,args.launcher,args.minutes)
