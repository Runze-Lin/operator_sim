"""Derive DynamicMxQuant resource defaults from the archived DT phase reduction."""
from hashlib import sha256
import json
from math import ceil, log2
from pathlib import Path
from statistics import median
import sys

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
from operator_model.chips import load_chip
from operator_model.model import fit_nonnegative_line


def main():
    chip = load_chip()
    clock = chip.frequency_ghz * 1000
    evidence = json.loads((ROOT/'results/v0.4/trace-analysis.json').read_text())['quant']
    runs = {r['measurement']['trace_sha256']: r for file in sorted(ROOT.glob('runs/model-v[123]/q*/result.json'))
            if (r := json.loads(file.read_text()))['status'] == 'ok'}
    samples = []
    for e in evidence:
        r = runs[e['trace_sha256']]
        m = r['measurement']
        assert r['identity']['soc_version'] == chip.name and 'bad=0' in str(m['correctness'])
        assert m['trace_sha256'] == e['trace_sha256']
        trace, = Path(m['report_path']).glob('simulator/trace.json')
        assert sha256(trace.read_bytes()).hexdigest() == e['trace_sha256']
        for i, tile in enumerate(e['iterations']):
            size = tile['read']['bytes']
            groups = ceil(size/64)
            samples.append(dict(id=e['id'], sha256=e['trace_sha256'], shape=e['shape'],
                tile=i, active=m['active_aiv'], prefix=e['prefix_us']*clock,
                read=(tile['read']['end_us']-tile['read']['start_us'])*clock,
                rd_bytes=size, loops=ceil(groups/8)+ceil(groups/128),
                vf=(min(w['start_us'] for w in tile['writes'])-tile['vf'][0]['start_us'])*clock,
                gap=tile['vector_ready_gap_us']*clock,
                write=tile['write_drain_us']*clock, wr_bytes=sum(w['bytes'] for w in tile['writes']),
                end=e['suffix_us']*clock, issue=None if not i else clock*(
                    tile['read']['start_us']-e['iterations'][i-1]['read']['start_us'])))
    first = [s for s in samples if not s['tile']]
    def fit(xs, ys):
        line = fit_nonnegative_line(xs, ys)
        return line['intercept_us'], line['slope_us_per_unit']
    launch, arrival = fit([log2(s['active']) for s in first], [s['prefix'] for s in first])
    latency, slope = fit([s['rd_bytes'] for s in first], [s['read'] for s in first])
    vf, vf_slope = fit([s['loops'] for s in samples], [s['vf'] for s in samples])
    # Payload issue and fixed read latency are distinct: double buffering can
    # overlap successive requests. Use the fastest observed large-read service
    # as the issue-rate envelope; retain measured first-read service for latency.
    repeated = [s for s in samples if s['issue'] is not None]
    issue_bpc = max(s['rd_bytes']/s['issue'] for s in repeated)
    writes = [s for s in samples if s['wr_bytes']>=4096]
    write_bpc = max(s['wr_bytes']/s['write'] for s in writes)
    write_latency = median(max(0,s['write']-s['wr_bytes']/write_bpc) for s in samples)
    p = dict(launch_cycles=launch, arrival_log2_cycles=arrival, read_latency_cycles=latency,
             read_bytes_per_cycle=1/slope, read_issue_bytes_per_cycle=issue_bpc,
             vf_dispatch_cycles=vf, vec_loop_cycles=vf_slope, default_vector_ops_per_cycle=
             chip.vector_tops['16']*1000/chip.vector_cores/chip.frequency_ghz,
             write_latency_cycles=write_latency, write_bytes_per_cycle=write_bpc,
             vf_dependency_cycles=median(s['gap'] for s in first), end_cycles=median(s['end'] for s in first))
    out=ROOT/'results/v0.16'
    for name,data in [('quant',dict(training_soc=chip.name,parameters=p)),
                      ('quant-dt-evidence',dict(source='results/v0.4/trace-analysis.json',cycles_per_us=clock,samples=samples))]:
        (out/(name+'.json')).write_text(json.dumps(data,indent=2)+'\n')

if __name__ == '__main__':
    main()
