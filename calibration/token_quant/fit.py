"""Extract per-stage resource coefficients from six development terminal-core timelines."""
import argparse
import json
from math import log2
from pathlib import Path
from statistics import median

from operator_model.chips import load_chip


def fit(samples):
    observations = []
    for case, stages in samples:
        reads = [d for d in stages['dma'] if d['kind'] == 'MTE2']
        value, scale = sorted((d for d in stages['dma'] if d['kind'] == 'MTE3'),
                              key=lambda d: d['bytes'], reverse=True)
        assert len(reads) == 2 and reads[0]['bytes'] == 256
        read = reads[1]
        assert read['bytes'] == 2*value['bytes']
        rows, width = case['shape']
        observations.append(dict(chip=case['chip'], rows=rows, width=width, n=value['bytes'],
            active=min(rows,load_chip(case['chip']).vector_cores), issue=read['issue'],
            read=read['retire']-read['issue'], vec=value['issue']-read['retire'],
            write=value['retire']-value['issue'], scale=scale['retire']-value['issue'],
            scale_bytes=scale['bytes'], end=stages['end']-max(scale['retire'],value['retire'])))
    small=[o for o in observations if o['rows']==9]
    slopes = {}
    for stage in ('read','vec','write'):
        delta=[]
        for name in {o['chip'] for o in small}:
            lo,hi=sorted([o for o in small if o['chip']==name],key=lambda o:o['width'])
            delta.append((hi[stage]-lo[stage])/(hi['n']-lo['n']))
        slopes[stage]=median(delta)
    rd,wr=2/slopes['read'],1/slopes['write']
    def arrivals(key, subtract):
        bases=[]; rises=[]
        for name in {o['chip'] for o in observations}:
            cases = [o for o in small if o['chip']==name]
            low = median(o[key]-subtract(o) for o in cases)
            high, = [o for o in observations if o['rows']==144 and o['chip']==name]
            rise=(high[key]-subtract(high)-low)/log2(high['active']/9)
            bases.append(low-rise*log2(9));rises.append(rise)
        return max(0,median(bases)),max(0,median(rises))
    issue,rise=arrivals('issue',lambda o:0)
    scale,scale_rise=arrivals('scale',lambda o:o['scale_bytes']/wr)
    chip=load_chip()
    return dict(reference_ops_per_aiv_cycle=chip.vector_tops['32']*1000/(chip.frequency_ghz*chip.vector_cores),
                read_B_per_cycle=rd, write_B_per_cycle=wr,
                read_setup_cycles=median(o['read']-2*o['n']/rd for o in observations),
                vector_setup_cycles=median(o['vec']-slopes['vec']*o['n'] for o in observations),
                vector_cycles_per_element=slopes['vec'],
                payload_setup_cycles=median(o['write']-o['n']/wr for o in observations),
                scale_retirement_cycles=scale, scale_arrival_cycles=scale_rise,
                issue_cycles=issue, issue_arrival_cycles=rise,end_cycles=median(o['end'] for o in observations))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('raw',type=Path)
    p.add_argument('parameters',type=Path)
    a=p.parse_args()
    samples=[(json.loads((folder/'receipt.json').read_text())['case'],json.loads((folder/'phases.json').read_text()))
             for folder in sorted(a.raw.iterdir()) if (folder/'receipt.json').is_file()]
    a.parameters.write_text(json.dumps(fit(samples),indent=2)+'\n')
