"""Recover hot-channel front-end and single-resident expansion costs from prior DT events."""
import hashlib
import json
from pathlib import Path
from statistics import median
import sys

ROOT=Path(__file__).resolve().parents[3]
sys.path.insert(0,str(ROOT))
from operator_model import load_chip
from operator_model.post_repeat import _prices as prior
from operator_model.rms_resource import READ_PER_KIB
OUT=ROOT/'results/v0.33/post'
DT='Ascend950DT_9582'


def read(p):return json.loads((ROOT/p).read_bytes())
def sha(p):return hashlib.sha256((ROOT/p).read_bytes()).hexdigest()
def write(p,j):(OUT/p).write_text(json.dumps(j,ensure_ascii=False,indent=2,allow_nan=False)+'\n')


def prices():
    original=read('results/v0.32/post/channel-freeze.json')['parameters']
    rates=prior();v=original['reference_ops_per_cycle']
    rows=read('results/v0.32/post/wide-repeats.json')
    active={r['trace_sha']:r['measured_aiv'] for r in read('results/v0.24/post/evaluation.json')['rows'] if r.get('timing_qualified')}
    points=[]
    for r in rows:
        if r['chip']!=DT or r['fragment_width']>448 or active[r['ca_sha256']]==1:continue
        # Withhold the single-lane extreme and wider geometry; never use a total device duration.
        a=active[r['ca_sha256']];d=r['fragment_width']
        control=1650*r['measured_stages_us']['control']
        first=1650*r['writes_us_bytes'][0][0]
        timings=[]
        for event in read(r['shard'])['traceEvents']:
            if event.get('ph')!='X' or event['name']!='MOV_SRC_TO_DST_ALIGNv2':continue
            detail=json.loads(event['args']['detail'])
            if detail['queue_type']=='MTE2':
                timings.append((event['ts'],event['dur'],detail['processed_bytes']))
        timings.sort()
        op0,op1=timings[-2:]
        assert [op0[2],op1[2]]==[2*d,8*d]
        wr=r['writes_us_bytes'][-1]
        points.append(dict(read0_cycles=1650*op0[1],read1_cycles=1650*op1[1],
            read_gap_cycles=1650*(op1[0]-op0[0]),vector_handoff_cycles=1650*(r['first_rv_load_zero_us']-max(op0[0]+op0[1],op1[0]+op1[1])),
            hot_write_response_cycles=1650*(wr[1]-wr[0]),shape=r['shape'],active=a,pressure=(a-1)/63,
            setup_cycles=control+first-9*d/v,arrival_cycles=1650*r['measured_stages_us']['arrival'],
            control_cycles=control,first_write_after_vector_load_cycles=first,trace_sha=r['ca_sha256']))
    slopes=[(h['setup_cycles']-l['setup_cycles'])/(h['pressure']-l['pressure']) for h in points for l in points if h['pressure']>l['pressure']]
    slope=median(slopes);inter=median(p['setup_cycles']-slope*p['pressure'] for p in points)
    deviations=[p['setup_cycles']-inter-slope*p['pressure'] for p in points]
    records=[r for r in read('results/post-candidate/phase-evidence.json') if r['chip']==DT and r['state']=='expand'
             and r['prior_aiv']==1 and r['route']=='post_channel']
    cold=[]
    for r in records:
        d=r['fragment_width'];a,b=r['dma_read'][-2:];wr=r['dma_write'];rv=r['first_vector_operand_us']
        assert [a['bytes'],b['bytes']]==[2*d,8*d] and len(wr)==4 and all(w['bytes']==2*d for w in wr)
        cold.append(dict(shape=r['shape'],trace_sha=r['sha256'],arrival=1650*r['phase_us']['arrival'],
           control=1650*r['phase_us']['control'],read0=1650*(a['end_us']-a['start_us'])-a['bytes']*READ_PER_KIB/1024,
           gap=1650*(b['start_us']-a['start_us']),read1=1650*(b['end_us']-b['start_us'])-b['bytes']*READ_PER_KIB/1024,
           after_read=1650*(rv-max(a['end_us'],b['end_us'])),
           first_write=1650*(wr[0]['start_us']-rv)-9*d/v,
           fourth_write_extra=1650*(wr[-1]['start_us']-wr[0]['start_us'])-3*(original['scalar_issue_cycles']+original['group_cycles']*(d//64)),
           ack=1650*(wr[-1]['end_us']-wr[-1]['start_us'])-2*d*rates['write_cycles_per_kib']/1024,
           exit=1650*(r['phase_us']['retire']-(wr[-1]['end_us']-wr[-1]['start_us']))))
    assert len(cold)==2 and len(points)==5
    return dict(hot=dict(original,setup_base_cycles=inter,setup_per_pressure_cycles=slope,
            setup_mad_cycles=median(abs(z) for z in deviations),
            multi_aiv_arrival_cycles=median(p['arrival_cycles'] for p in points),
            **{key:median(p[key] for p in points) for key in ('read0_cycles','read1_cycles','read_gap_cycles','vector_handoff_cycles','hot_write_response_cycles')}),
        expansion={key:median(p[key] for p in cold) for key in cold[0] if key not in ('trace_sha','shape')},
        expansion_first_write_upper_cycles=max(p['first_write'] for p in cold),
        development=dict(hot=points,expand=cold),
        source=dict(warm='v0.24 immediate repeat DT with >1 AIV and critical width <=448',
            cold='v0.8-v0.9 two DT transitions from one previously resident AIV to a new physical core; independent operand and write events',
            clock_ghz=load_chip(DT).frequency_ghz,read_cycles_per_kib=READ_PER_KIB,
            write_cycles_per_kib=rates['write_cycles_per_kib']))


if __name__=='__main__':write('prices.json',prices())
