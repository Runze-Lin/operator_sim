"""Convert qualified development descriptor endpoints into stage prices, in core clocks."""
from collections import defaultdict
from itertools import combinations
import json
from statistics import median

from extract import OUT, read


def line(points):
    # Equal weight per supported transfer length; Theil–Sen is robust to outlying responses.
    groups = defaultdict(list)
    for x,y in points:
        groups[x].append(y)
    centers = [(x, median(ys)) for x,ys in sorted(groups.items())]
    slope = median((y2-y1)/(x2-x1) for (x1,y1),(x2,y2) in combinations(centers,2))
    return dict(setup=median(y-slope*x for x,y in centers), ticks_per_unit=slope,
                medians=centers)


def duration(event):
    return event['end']-event['begin']


def build():
    files = sorted((OUT/'primitives').glob('*.json'))
    cases = [read(p) for p in files]
    if len(cases) != 9 or any('development' not in c['role'] for c in cases):
        raise ValueError('Exactly nine qualified development windows are required')
    controls = defaultdict(list)
    for c in cases:
        for core in c['cores'].values():
            tiles = [t for r in core['rows'] for t in r['tiles']]
            first = tiles[0]
            for k in ['x','res']:
                controls['first_'+k].append((first[k]['bytes'],duration(first[k])))
            controls['arithmetic'].append((first['width']//64, first['arith']['end']-core['first_rvec']))
            for t in tiles:
                for k in ['x','res','write']:
                    controls[k].append((t[k]['bytes'],duration(t[k])))
                controls['cast'].append(duration(t['cast']))
                controls['cast_overlap'].append(t['arith']['end']-t['cast']['begin'])
            for a,b in zip(tiles,tiles[1:]):
                controls['next_overlap'].append(a['cast']['end']-b['arith']['begin'])
            for a,b in zip(tiles,tiles[2:]):
                controls['slot'].append(b['x']['begin']-a['write']['begin'])
    ar = line(controls['arithmetic'])
    warm = []
    for c in cases:
        for core in c['cores'].values():
            for t in [t for r in core['rows'] for t in r['tiles']][1:]:
                warm.append(duration(t['arith'])-ar['setup']-ar['ticks_per_unit']*(t['width']//64))
    # Use the critical start cohort for the one-time fanout (select by first RV, never by kernel end).
    # Each calibration observation is a measured stage, with common DT/PR prices.
    low=[];many=[]
    for c in cases:
        cores=c['cores']; n=len(cores)
        k=max(cores.values(),key=lambda k:k['first_rvec'])
        post=k['rows'][0]['post']['begin']
        admission=k['first_rvec']-k['rows'][0]['tiles'][0]['arith']['begin']
        (low if n<=5 else many).append((n,post,admission))
    base=median(z[1] for z in low);first=median(z[2] for z in low)
    # The L2 and DRAM soft contention model below has no HcPost byte counters to fit it.
    prices=dict(clock_unit='one local core cycle', vector_reference_ops32_cycle=27e3/(64*1.65),
                first_coeff=base, issue_fanout_per_aiv=median((v-base)/(n-1) for n,v,_ in many),
                first_vf_admission=first, admission_fanout_per_aiv=median((v-first)/(n-1) for n,_,v in many),
                arith_body=ar, warm_dispatch=median(warm), cast_body=median(controls['cast']),
                cast_head_overlap=median(controls['cast_overlap']), next_head_overlap=median(controls['next_overlap']),
                triple_slot_release=median(controls['slot']), dma={k:line(controls[k]) for k in ['x','res','write','first_x','first_res']},
                # Instruction positions measured from flow-control issue, not independent service durations.
                coeff_issue_gap=47, payload_after_coeff=129, split_descriptor_extra=12, initial_read_issue_gap=47,
                full_read_issue_extra=2, second_fragment_issue=232, tail_descriptor_extra=20,
                next_row_coeff_issue=386, input_ready=3, cast_to_write=4, return_after_write=10,
                controls={'low_fanout':low,'high_fanout':many},
                inputs=[dict(file=p.name,trace_sha256=c['trace_sha256']) for p,c in zip(files,cases)])
    if prices['triple_slot_release']!=2:
        raise ValueError('Unexpected third-slot release')
    return prices


if __name__=='__main__':
    p=OUT/'parameters.json'
    v=json.dumps(build(),indent=2)+'\n'
    if p.exists() and p.read_text()!=v:
        raise RuntimeError('Existing calibration differs')
    p.write_text(v)
