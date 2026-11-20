"""Calibrate service cycles at MMAD and histogram boundaries on four archived DT shapes."""
import json
from pathlib import Path
from statistics import median

ROOT=Path(__file__).resolve().parents[3]
OUT=ROOT/'results/v0.34/indexer'


def line(left, right, x0, x1):
    slope=(right-left)/(x1-x0)
    offset=left-slope*x0
    assert offset>=0 and slope>=0
    return [offset,slope]


def derive():
    all_rows=json.loads((OUT/'physical.json').read_bytes())
    dt={tuple(r['shape'][1:3]):r for r in all_rows if r['protocol']=='npusim'}
    a,b,u,v=[dt[k] for k in ((1,1024),(1,8192),(4,1024),(4,4096))]
    def span(r):return r['sort_ticks'][-1][1]-r['sort_ticks'][0][0]
    def end(r):return r['duration_ticks']-r['sort_ticks'][-1][1]
    def gap(r):
        s=r['sort_ticks'];half=len(s)//2
        return s[half][0]-s[half-1][1]
    for r in (a,b,u,v):
        assert len(r['sort_ticks'])==8*(r['shape'][2]//256)*((r['shape'][1]+1)//2)
    src=json.loads((ROOT/'configs/chips/Ascend950DT_9582.json').read_bytes());rate=lambda pipe,cores:src[pipe]['16' if pipe=='cube_tops' else '32']*1000/(src[cores]*src['frequency_ghz'])
    cube=rate('cube_tops','cube_cores')
    def response_less_compute(r):
        e=r['critical_mmad_ticks'];m,n,k,_=r['instruction_shapes'][0]
        return median(hi-lo for lo,hi in e)-2*m*n*k/cube
    gaps=[]
    for r in (b,v):
        starts=[x[0] for x in r['critical_mmad_ticks']]
        # The three rotating key buffers leave bursts of short / long issue gaps.
        gaps.extend((starts[i+3]-starts[i])/3 for i in range(3,len(starts)-3,3))
    return dict(reference='Ascend950DT_9582',cube_ops_cycle=cube,vector_ops_cycle=rate('vector_tops','vector_cores'),
                first_issue_cycles=median(r['critical_mmad_ticks'][0][0] for r in (a,b,u,v)),
                page_ii_cycles=median(gaps),mmad_fixed_cycles=median(response_less_compute(r) for r in (a,b,u,v)),
                sort_handoff_cycles=median(r['sort_ticks'][0][0]-r['last_mmad_retire'] for r in (a,b,u,v)),
                histogram_span=line(span(a),span(b),4,32),histogram_gap=line(gap(u),gap(v),4,16),
                store_drain=line(end(a),end(b),4,32),
                source_sha256=[r['sha'] for r in (a,b,u,v)])


if __name__=='__main__':
    (OUT/'prices.json').write_text(json.dumps(derive(),indent=2)+'\n')
