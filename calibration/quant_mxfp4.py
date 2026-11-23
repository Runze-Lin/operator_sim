"""Capture one isolated, prospectively frozen block32 FP4 quantization (rank 2)."""
import argparse
from pathlib import Path
import re

from calibration.__main__ import CASE_FIELDS, digest, encoded, ingest, read, require, unseal
from calibration.msopprof import CORE, sha
from calibration.mx_matmul.collect import MEMORY_CONFIG, capture, core_windows, write_json


def terminal_phases(base, cores):
    import json
    from calibration.mx_matmul.shards import trace_events
    name = max(cores, key=lambda k: cores[k]['retire_tick'])
    origin = min(c['first_tick'] for c in cores.values())
    phases = {'core':name, 'origin_tick':origin, 'ticks_per_us':1650, 'dma':[], 'rvecex_count':0}
    vector = []
    for event in trace_events(base/'simulator'/name/'trace.json'):
        if event.get('ph')!='X':
            continue
        spec = json.loads(event['args']['detail'])
        start = round(1650*event['ts'])-origin
        end = round(1650*(event['ts']+event['dur']))-origin
        if event['name']=='MOV_SRC_TO_DST_ALIGNv2':
            phases['dma'].append(dict(kind=spec['instr_type'], bytes=spec['processed_bytes'],
                                      issue=start, retire=end))
        if spec['instr_type']=='RVECEX':
            vector.append((start, end))
    if vector:
        phases.update(rvecex_count=len(vector), first_rvecex=min(x[0] for x in vector),
                      last_rvecex_retire=max(x[1] for x in vector))
    return phases


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('freeze', 'raw', 'opp'):
        parser.add_argument('--'+name, type=Path, required=True)
    parser.add_argument('--case', required=True)
    parser.add_argument('--probe', type=Path)
    parser.add_argument('--launcher', type=Path)
    parser.add_argument('--minutes', type=int, default=5)
    parser.add_argument('--convert-only', action='store_true')
    args = parser.parse_args()
    frozen = unseal(read(args.freeze), 'freeze')
    case = next(c for c in frozen['cases'] if c['id'] == args.case)
    rows, cols = case['shape']
    chip = case['chip']
    outputs = {'Y': rows*cols, 'scale': rows*2*((cols+63)//64)}
    require(case['op']=='quant_mxfp4' and case['api']=='aclnnDynamicMxQuant' and case['dtype']=='bf16'
            and case['prefix'] is None and case['outputs']==outputs and chip in MEMORY_CONFIG
            and frozen['chips'][chip]['frequency_ghz']==1.65, 'Expected an isolated 950 BF16 block32 quant')
    fixed = {'group_size':32, 'scale_alg':0, 'round_mode':'round', 'axis':-1}
    require(all(k in fixed and type(v) is type(fixed[k]) and v==fixed[k]
                for k,v in case['attrs'].items()), 'Probe contract differs')
    raw = args.raw.resolve()
    if not args.convert_only:
        require(args.probe and args.minutes>=1, 'Provide --probe and a positive timeout')
        capture(case, 'quant_mxfp4', frozen['sha256'], raw, args.probe, args.launcher, args.minutes)
    cost = read(raw/'cost.json')
    require(cost['exit_code']==0 and cost['stop_reason'] is None, 'Missing successful completion')
    log = (raw/'run.log').read_text()
    require(re.findall(r'^runtime_soc=(\S+)', log, re.M)==[chip] and MEMORY_CONFIG[chip] in log,
            'Unexpected runtime SKU or memory model')
    require(re.findall(r'^case=[^\r\n]+', log, re.M)==[
        f'case=quant_mxfp4 rows={rows} cols={cols} api=aclnnDynamicMxQuant axis=-1 group=32 round=round scaleAlg=0'],
        'Run differs from case')
    require(re.findall(r'^correctness[^\r\n]+', log, re.M)==[
        f'correctness {k} count={n} bad=0' for k,n in outputs.items()]
        and all(s in log for s in ('SMOKE_PASS','All task success.','Profiling data parse finished.'))
        and 'SMOKE_FAIL' not in log, 'Incomplete output verification or profile')
    stem = 'DynamicMxQuant_bf16toe2m1_high_performance'
    kernels = re.findall(r'Start profiling on kernel: (\S+)', log)
    require(len(kernels)==1 and re.fullmatch(stem+r'_33(?:_mix_aiv)?', kernels[0]), 'Unexpected kernel')
    profiles = list((raw/'profile').glob('OPPROF_*'))
    require(len(profiles)==1, 'Expected exactly one profile')
    base = profiles[0]
    obj = args.opp/'dynamic_mx_quant'/(stem+'.o')
    actual = base/'dump/aicore_binary.o'
    require(obj.resolve()!=actual.resolve() and sha(obj)==sha(actual), 'CA image differs from installed OPP object')
    roster_rows = CORE.findall(log)
    roster = {name:float(us) for name,us in roster_rows}
    require(roster and len(roster)==len(roster_rows), 'Missing per-core statistics')
    cores, hashes = core_windows(base/'simulator', roster)
    result = dict(schema=1, kind='ca-receipt', case={k:case[k] for k in CASE_FIELDS},
                  provider='CANN 9.2.1 ops-nn', timing='all-core-instruction', complete=True,
                  started_utc=cost['started_utc'], finished_utc=cost['finished_utc'],
                  run_log_sha256=sha(raw/'run.log'), correctness={k:{'checked':n,'bad':0} for k,n in outputs.items()},
                  stages=[dict(kernel=kernels[0], kernel_binary_sha256=sha(obj), physical_shard_sha256=hashes,
                               trace_sha256=digest(encoded(hashes)), ticks_per_us=1650, launched_cores=sorted(roster),
                               cores=cores)], freeze_sha256=(raw/'freeze.sha256').read_text().strip())
    observation = ingest(frozen, result)
    write_json(raw/'receipt.json', result)
    write_json(raw/'measurement.json', observation)
    write_json(raw/'phases.json', terminal_phases(base, cores))
    print(args.case, observation['ca_us'])


if __name__ == '__main__':
    main()
