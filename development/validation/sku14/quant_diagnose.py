"""Identify quant critical-core phase intervals from the previous PR campaign."""
import hashlib
import json
import math
import statistics
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
from operator_model.chips import load_chip
from operator_model.predictor import _legacy
from operator_model.target_tiling import quant_plan

OUT = ROOT/'results/v0.14'


def dissect(record):
    path = ROOT / record['trace']
    data = path.read_bytes()
    assert hashlib.sha256(data).hexdigest() == record['trace_sha256']
    cores = defaultdict(list)
    for e in json.loads(data)['traceEvents']:
        if e['ph'] == 'X':
            cores[e['pid']].append(e)
    start = min(e['ts'] for es in cores.values() for e in es)
    ends = lambda e: e['ts'] + e['dur']
    per_core = {}
    for pid, events in cores.items():
        def moves(tid):
            return [e for e in events if e['tid']==tid and e['name']=='MOV_SRC_TO_DST_ALIGNv2']
        reads, writes = moves('MTE2'), moves('MTE3')
        assert len(reads) == len(writes) == 2
        read = max(reads, key=ends)
        simd = [e for e in events if e['tid'].startswith('RVEC')]
        boundaries = [start,read['ts'],ends(read),min(e['ts'] for e in simd),
                      min(e['ts'] for e in writes),max(ends(e) for e in writes),max(map(ends,events))]
        names = ['before_data_read','data_read','read_to_simd','simd_to_write','write_window','end']
        per_core[pid] = {k:b-a for k,a,b in zip(names,boundaries,boundaries[1:])}
        assert min(per_core[pid].values()) >= -1e-6
    critical=max(per_core,key=lambda core:sum(per_core[core].values()))
    assert abs(sum(per_core[critical].values())-record['ca_us']) < 1e-6
    return dict(source_sha256=record['trace_sha256'],critical_core=critical,active=len(cores),
                critical_us=per_core[critical],median_core_us={k:statistics.median(
                    core[k] for core in per_core.values()) for k in per_core[critical]},
                rvecex_events=sum(e['tid']=='RVECEX' for e in cores[critical]))


def main():
    cases=[]
    for file in ('pairs.json','holdout.json'):
        rows=json.loads((ROOT/'results/v0.13/sku'/file).read_text())
        row, = [r for r in rows if r['op']=='quant']
        cases.append(dict(shape=row['shape'], role='development', dt=dissect(row['dt']), pr=dissect(row['pr'])))
    (OUT/'quant-diagnosis.json').write_text(json.dumps(cases,indent=2)+'\n')
    # Keep original DT slopes; separate identifiable PR critical-core intervals.
    costs=_legacy().quant_model['quant']['costs']
    profile={'sku':load_chip('Ascend950PR_957b').name,'unit':'us','development_shapes':[r['shape'] for r in cases]}
    parameters=defaultdict(list)
    for row in cases:
        p=quant_plan(row['shape'],56,253952)
        obs=row['pr']['critical_us']
        for name, observed, base, units in [
            ('prefix',obs['before_data_read'],'prefix',math.log2(1+p['active_aiv'])),
            ('read',obs['data_read'],'first_read',p['units']['read']),
            ('vector',obs['simd_to_write'],'vector',p['units']['vector'])]:
            parameters[name+'_intercept_us'].append(observed-costs[base]['slope_us_per_unit']*units)
        for key in ['read_to_simd','write_window','end']:
            parameters[key+'_us'].append(obs[key])
    profile.update({k:statistics.median(x) for k,x in parameters.items()})
    (OUT/'quant-phases.json').write_text(json.dumps(profile,indent=2)+'\n')


if __name__ == '__main__':
    main()
