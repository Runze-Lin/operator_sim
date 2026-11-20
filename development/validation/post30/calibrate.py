"""Measure physical operands of the DT ordinary split-width entry; exclude PR and repeats."""
from collections import defaultdict
from hashlib import sha256
import json
from pathlib import Path
from statistics import median
import sys
ROOT=Path(__file__).resolve().parents[3];sys.path.insert(0,str(ROOT))
from development.validation.rms_ordered_verify.verify import events,digest,checked
OUT=ROOT/'results/v0.30/post'
DT='Ascend950DT_9582'
def read(p):return json.loads(p.read_bytes())
def save(p,v):p.write_text(json.dumps(v,indent=2,allow_nan=False)+'\n')

def extract():
    ref=ROOT/'results/v0.25/post/receipts/width-Ascend950DT_9582.json'
    allrows=read(ref)['rows'];r,=(r for r in allrows if r['shape']==[65,4,5120] and r['role']=='entry')
    stage=r['stage'];tr=ROOT/stage['trace']
    checked(r['timing_qualified'] and digest(tr)==stage['trace_sha256'] and stage['kernel'].endswith('_6'),'historical DT calibration identity')
    raw=defaultdict(lambda:{'read':[],'write':[],'core_start':float('inf'),'end':0})
    for event in events(tr):
        v=raw[event['pid']];lo=round(event['ts']*1650)-stage['first_tick'];hi=round((event['ts']+event['dur'])*1650)-stage['first_tick']
        v['core_start']=min(v['core_start'],lo);v['end']=max(v['end'],hi)
        if not event['name'].startswith('MOV_SRC_TO_DST'):continue
        d=json.loads(event['args']['detail']);q=d['queue_type']
        if q not in ('MTE2','MTE3') or any(d.get(k) for k in ('is_wait','is_sync','is_duration_excluded')):continue
        item=dict(a=lo,b=hi,size=d['processed_bytes'],src=d['src_addr'],dst=d['dst_addr'])
        v['read' if q=='MTE2' else 'write'].append(item)
    checked(len(raw)==33,'exact physical roster')
    p=read(ROOT/'results/v0.25/post/prices.json');rd=p['read_cycles_per_kib'];wr=p['write_cycles_per_kib'];group=p['ready_to_write_per_group']
    evidence={};observations=defaultdict(list)
    def observe(name,val):observations[name].append(val)
    for core,v in raw.items():
        rr=sorted(v['read'],key=lambda x:x['a']);yy=sorted(v['write'],key=lambda x:x['a'])
        count=len(yy)//2;checked(count in [1,2] and len(yy)==2*count and len(rr)==1+6*count,'source row schedule')
        checked([i['size']for i in rr]==[96]+count*[64,16,5120,20480,5120,20480] and all(i['size']==20480 for i in yy),'source descriptor lengths')
        hh=[rr[1+6*i+k]for i in range(count)for k in [2,4]];xx=[rr[1+6*i+k]for i in range(count)for k in [3,5]]
        checked(len(set(z['dst']for z in hh))==len(set(z['dst']for z in xx))==2,'two operand slots')
        checked(all(z['dst']==hh[k%2]['dst'] and xx[k]['dst']==xx[k%2]['dst'] for k,z in enumerate(hh)),'alternating reusable UB addresses')
        # Scalar/dispatch uses the existing arrival log2 law, calibrated independently.
        observe('first_coeff_after_lane_entry',rr[1]['a']-v['core_start'])
        observe('coeff_to_first_h',hh[0]['a']-rr[1]['a'])
        observe('coeff1_to_coeff2',rr[2]['a']-rr[1]['a'])
        observe('second_h_lookahead',hh[1]['a']-hh[0]['a'])
        early=max(hh[0]['b'],xx[0]['b'],rr[1]['b'],rr[2]['b'])
        observe('first_ready_to_write_over_vec',yy[0]['a']-early-40*group)
        observe('final_write_fixed',yy[-1]['b']-yy[-1]['a']-yy[-1]['size']*wr/1024)
        observe('exit_after_last_write',v['end']-yy[-1]['b'])
        for col,e in enumerate(rr[1:3]):observe('coeff'+str(col)+'_fixed',e['b']-e['a']-e['size']*rd/1024)
        for i,(h,x,y) in enumerate(zip(hh,xx,yy)):
            tag='prefetch' if i<2 else 'reuse'
            observe(tag+'_hx_issue',x['a']-h['a'])
            observe(tag+'_h_ack',h['b']-h['a']-h['size']*rd/1024)
            observe(tag+'_x_ack',x['b']-x['a']-x['size']*rd/1024)
            if i:
                observe('write_gap_over_vec',y['a']-yy[i-1]['a']-40*group)
            if i>=2:
                observe('slot_release_lead_to_store',yy[i-2]['a']-h['a'])
        evidence[core]=dict(rows=count,core_start=v['core_start'],last=v['end'],coefficient_read=[rr[1+6*i:3+6*i]for i in range(count)],h=hh,x=xx,y=yy)
    # Fixed waits from median of per-instruction durations, not total window regression.
    pool={k:dict(median=median(vals),minimum=min(vals),maximum=max(vals),n=len(vals)) for k,vals in observations.items()}
    prices={k:max(0.,v['median']) for k,v in pool.items()}
    params=dict(cycles=prices,units='950 reference issue/ack/retirement cycles; nonnegative median; one ordinary DT process only',
        source_sha256=stage['trace_sha256'],source=stage['trace'],phase_samples=pool,
        imported_prices='results/v0.25/post/prices.json',calibration_scope='Non-repeated post_token shape change after covered post_token. Exactly two alternating UB slots, four BF16 streams.')
    save(OUT/'operands-dt.json',dict(sha256=stage['trace_sha256'],lanes=evidence,starting_instruction=stage['first_tick'],shape=r['shape'],previous_shape=r['previous_shape'],known_history=r['prior_aiv'],ca_us=stage['ca_us']))
    save(OUT/'prices.json',params)
if __name__=='__main__':extract()
