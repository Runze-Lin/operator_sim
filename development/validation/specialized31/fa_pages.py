"""Independent PR long-KV replay, matched to the archived DT case on the device clock."""
from collections import defaultdict
import hashlib
import json
from math import ceil, isclose
from pathlib import Path
from statistics import mean, median
import sys
from time import perf_counter

ROOT = Path(__file__).resolve().parents[3]
sys.path[:0] = [str(ROOT), str(ROOT/'development/validation/sku17')]
from development.validation.sku17.campaign_collect import events, sha
from development.validation.sku20.fa_pages import draw
from operator_model import estimate, load_chip

OUT = ROOT / 'results/v0.31/fa'
SOURCE = ROOT / 'results/v0.29/large/fa'
DT = 'Ascend950DT_9582'
PR = 'Ascend950PR_957b'
FUT = 'Ascend970DT'


def read(p):
    return json.loads(p.read_bytes())


def load_pr():
    job = next(j for j in read(SOURCE/'freeze.json')['jobs'] if j['chip']==PR and j['shape'][1]==131072)
    source = SOURCE / (job['id']+'.json')
    receipt = read(source)
    assert receipt['qualified'] and receipt['freeze_sha256']==sha(SOURCE/'freeze.json')
    stage, = receipt['stages']
    assert receipt['physical_core_count']==3 and stage['mmad']==2048
    cube, = (name for name in stage['windows'] if 'cube' in name)
    path = (ROOT/stage['trace']).parent/cube/'trace.json'
    lo,hi = float('inf'), 0
    groups = defaultdict(list)
    for e in events(path):
        if e['ph'] != 'X':
            continue
        start, end = e['ts'], e['ts']+e['dur']
        lo,hi = min(lo,start),max(hi,end)
        if e['name'] == 'MMAD':
            d = json.loads(e['args']['detail'])
            groups[e['args']['pc_addr']].append((start,end,tuple(d.get(x) for x in ('m','n','k'))))
    origin = lo-stage['windows'][cube][0]
    assert isclose(hi-origin,stage['windows'][cube][1],abs_tol=.001)
    shape = job['shape']
    p = estimate('fa',shape,chip=PR)
    assert isclose(p.latency_us,job['predictions'][PR]['us'])
    clock=load_chip(PR).frequency_ghz*1000
    di=p.projection['timing_diagnostics']
    w,=di['wave_types']
    pages=di['target_plan']['iterations']
    step=w['first']['ii_cycles']/clock
    launch=p.projection['stages_us']['launch']
    model={}
    observed={}
    pcs={}
    for pc, group in groups.items():
        group.sort()
        dims = {e[2] for e in group} - {(None,None,None)}
        dims, = dims
        name={(9,128,64):'QK',(9,64,128):'PV'}[dims]
        assert name not in observed and len(group)==pages*2
        assert all(group[i][2]==dims for i in range(0,len(group),2))
        observed[name]=[[[a-origin,b-origin] for a,b,_ in group[i:i+2]] for i in range(0,len(group),2)]
        pcs[name]=dict(pc=pc,missing_dimensions=sum(v[2]==(None,None,None) for v in group),fragments_per_page=2)
        first=launch+w[name.lower()+'_start_cycles']/clock
        duration=w['first'][name.lower()]/clock
        model[name]=[[first+i*step,first+i*step+duration] for i in range(pages)]
    assert set(observed)=={'QK','PV'} and pages==512
    assert isclose(model['PV'][-1][1]+p.projection['stages_us']['drain'],p.latency_us)
    endcore=max(stage['windows'],key=lambda s:stage['windows'][s][1])
    futures=job['predictions'][FUT]
    return dict(shape=shape,sku=PR,api='aclnnFlashAttentionScoreV2',kernel=stage['kernel'],
                ca_us=stage['ca_us'],model_us=p.latency_us,model=model,ca=observed,pc=pcs,cube=cube,
                terminal=dict(core=endcore,end_us=stage['windows'][endcore][1]),model_ii_us=step,
                cube_trace=str(path.relative_to(ROOT)),cube_sha256=sha(path),merged_sha256=stage['trace_sha256'],
                physical_end_count=receipt['physical_core_count'],correctness=receipt['correctness'],
                frozen_970dt_extrapolation_us=futures['us'],ca_wall_s=receipt['cost']['seconds'],
                peak_bytes=receipt['cost']['peak_bytes'],resource_demand_cycles=w['first']['demands_cycles'])


def profile(r):
    phases={}
    for key in ('QK','PV'):
        first=[page[0][0] for page in r['ca'][key]]
        deltas=[b-a for a,b in zip(first,first[1:])]
        average=mean(deltas)
        p=r['model'][key]
        ca=[first[0],first[-1]-first[0],r['ca_us']-first[-1]]
        model=[p[0][0],p[-1][0]-p[0][0],r['model_us']-p[-1][0]]
        phases[key]=dict(ca_us=ca,model_us=model,errors_us=[x-y for x,y in zip(model,ca)],
                         mean_ii_us=average,median_ii_us=median(deltas),
                         p95_ii_us=sorted(deltas)[ceil(.95*len(deltas))-1])
    return phases


def main():
    t=perf_counter()
    pr=load_pr()
    dt=read(ROOT/'results/v0.20/examples/fa-dt-128k-pages.json')
    assert dt['sku']==DT and dt['shape']==pr['shape']
    payload={'phase_order':['first_issue','first_to_last_page_issue','last_issue_to_device_retirement'],
             'by_chip':{DT:profile(dt),PR:profile(pr)},'trace_replay_s':perf_counter()-t}
    (OUT/'pages-pr.json').write_text(json.dumps(pr,separators=(',',':'))+'\n')
    (OUT/'phases.json').write_text(json.dumps(payload,indent=2)+'\n')
    draw(pr,'fa-pr-128k-pages')
    for sku,r in payload['by_chip'].items():
        print(sku,r['QK'])
    print('cost',pr['ca_wall_s'],pr['peak_bytes'],payload['trace_replay_s'])


if __name__=='__main__':
    main()
