"""Derive quant service costs from DT descriptors and two-buffer tile metadata."""
from math import log2
from pathlib import Path
import json
from statistics import median
import sys

ROOT=Path(__file__).resolve().parents[3]
sys.path.insert(0,str(ROOT))
from operator_model.model import fit_nonnegative_line
from operator_model.quant_scale import peak_scale_writers
from operator_model.target_tiling import quant_plan

OUT=ROOT/'results/v0.17'


def main():
    prior=json.loads((ROOT/'results/v0.16/quant.json').read_text())['parameters']
    traces=json.loads((ROOT/'results/v0.4/trace-analysis.json').read_text())['quant']
    dt=json.loads((ROOT/'configs/chips/Ascend950DT_9582.json').read_text())
    frequency=dt['frequency_ghz']*1000
    samples=[]
    for t in traces:
        shape=t['shape']
        p=quant_plan(shape,dt['vector_cores'],dt['details']['ub_bytes'])
        owners=peak_scale_writers(*shape,p['rows_per_core'],p['cols_per_core'])
        for idx,tile in enumerate(t['iterations']):
            rd=tile['read']
            y,s=sorted(tile['writes'],key=lambda w:w['start_us'])
            assert y['bytes']==32*s['bytes'] and len(tile['writes'])==2
            samples.append(dict(shape=shape,sha256=t['trace_sha256'],tile=idx,active=p['active_aiv'],
                     scale_granule_writers=owners,read_bytes=rd['bytes'],
                     read_cycles=frequency*(rd['end_us']-rd['start_us']),
                     data_bytes=y['bytes'],scale_bytes=s['bytes'],
                     data_cycles=frequency*(y['end_us']-y['start_us']),
                     scale_cycles=frequency*(s['end_us']-s['start_us']),
                     issue_gap=frequency*(s['start_us']-y['start_us'])))
    data=fit_nonnegative_line([r['data_bytes'] for r in samples],[r['data_cycles'] for r in samples])
    unshared=[r for r in samples if r['scale_granule_writers']==1]
    shared=[r for r in samples if r['scale_granule_writers']>1]
    scale_tail=median(max(0,r['scale_cycles']-r['data_cycles']) for r in unshared)
    scale_base=median(r['scale_cycles'] for r in unshared if r['scale_bytes']<=64)
    charge=lambda r:max(0,r['scale_cycles']-max(scale_base,r['data_cycles']+scale_tail))
    queue_slope=median(charge(r)/((r['scale_granule_writers']-1)*log2(r['active'])) for r in shared)
    # Long requests contain throughput information; the fastest slope bounds
    # local service without folding every slow memory queue into the port rate.
    read_bpc=max(r['read_bytes']/max(1,r['read_cycles']-prior['read_latency_cycles'])
                 for r in samples if r['read_bytes']>=16384 and r['tile']==0)
    params=dict(prior,read_bytes_per_cycle=read_bpc,write_latency_cycles=data['intercept_us'],
                write_bytes_per_cycle=1/data['slope_us_per_unit'],scale_tail_cycles=scale_tail,
                scale_base_cycles=scale_base,scale_peer_cycles=queue_slope,
                scale_issue_gap_cycles=median(r['issue_gap'] for r in samples))
    (OUT/'quant.json').write_text(json.dumps(dict(training_soc=dt['name'],parameters=params),indent=2)+'\n')
    evidence=dict(source='results/v0.4/trace-analysis.json',cycles_per_us=frequency,samples=samples)
    (OUT/'quant-dt-evidence.json').write_text(json.dumps(evidence,indent=2)+'\n')


if __name__=='__main__':
    main()
