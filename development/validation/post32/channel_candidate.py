"""Diagnostic four-output schedule with DT event prices; never changes the public price."""
from collections import defaultdict
from datetime import datetime, timezone
from hashlib import sha256
import json
from math import ceil, isclose
from pathlib import Path
from statistics import median
import sys
ROOT=Path(__file__).resolve().parents[3]
sys.path.insert(0,str(ROOT))
from operator_model import estimate_post_history, load_chip
from operator_model.post_repeat import _prices

OUT=ROOT/'results/v0.32/post'
DT,PR,FUT='Ascend950DT_9582','Ascend950PR_957b','Ascend970DT'

def read(p):return json.loads(p.read_bytes())

def prices():
    # The variable here is the actual terminal AIV fragment width, not the global hidden size.
    development=[r for r in read(OUT/'wide-repeats.json') if r['chip']==DT and r['fragment_width']<=448]
    x=[]
    for r in development:
        w=r['writes_us_bytes']; d=r['fragment_width']; issue=[i[0]*1650 for i in w]
        assert len(w)==4 and all(p[2]==2*d for p in w)
        delta=median(b-a for a,b in zip(issue,issue[1:]))
        x.append((r,d,ceil(d/64),delta))
    aligned=[(g,delta) for _,d,g,delta in x if d%64==0]
    rates=[(y1-y0)/(g1-g0) for g0,y0 in aligned for g1,y1 in aligned if g1>g0]
    slope=median(rates); fixed=median(y-g*slope for g,y in aligned)
    tail=median(y-fixed-g*slope for _,d,g,y in x if d%64)
    chip=load_chip(DT)
    v=chip.vector_tops['32']*1000/(chip.vector_cores*chip.frequency_ghz)
    write_price=_prices()['write_cycles_per_kib']
    first=[];ack=[];retire=[]
    for r,d,_,_ in x:
        w=r['writes_us_bytes']
        first.append(w[0][0]*1650-9*d/v)
        drain=(w[-1][1]-w[-1][0])*1650
        ack.append(drain-2*d*write_price/1024)
        retire.append(r['measured_stages_us']['retire']*1650-drain)
    return dict(reference_ops_per_cycle=v,scalar_issue_cycles=fixed,group_cycles=slope,aligned_tail_cycles=tail,
                first_fixed_cycles=median(first),last_ack_fixed_cycles=median(ack),exit_cycles=median(retire),
                training_sha256=[r['ca_sha256'] for r in development])


def predict(shape,chip,params):
    h=estimate_post_history(shape,chip=chip)
    original={k:v.latency_us for k,v in h.scenarios.items()}
    pred=dict(original);info=None
    if h.route=='post_channel':
        c=load_chip(chip)
        d=ceil(shape[2]/(h.active_aiv//shape[0]))
        per_cycle=c.vector_tops['32']*1000/(c.vector_cores*c.frequency_ghz)
        wr=c.details.get('mte3_bandwidth_gbs')
        service=(2*d*1024*c.frequency_ghz/wr/1024 if wr else 2*d*_prices()['write_cycles_per_kib']/1024)
        clk=c.frequency_ghz*1000
        g=ceil(d/64); p=params
        first=p['first_fixed_cycles']+9*d/per_cycle
        ii=max(9*d/per_cycle,service,p['scalar_issue_cycles']+g*p['group_cycles']*p['reference_ops_per_cycle']/per_cycle
               +(p['aligned_tail_cycles'] if d%64 else 0))
        st=dict(h.scenarios['repeat'].stages_us)
        st['body']=(first+3*ii)/clk
        st['retire']=(p['last_ack_fixed_cycles']+service+p['exit_cycles'])/clk
        st.pop('shared_link_floor')
        in_bytes=10*shape[0]*shape[2]+80*shape[0];total_bytes=in_bytes+8*shape[0]*shape[2]
        l2=c.details.get('l2_bandwidth_gbs') or 100*c.vector_cores*c.frequency_ghz
        hbm=max(in_bytes,total_bytes-(c.details.get('l2_bytes') or 134217728))
        floor=max(hbm/c.hbm_bandwidth_gbs,total_bytes/l2)/1000
        pred['repeat']=max(sum(st.values()),floor)
        info=dict(per_output_bytes=2*d,estimated_first_issue_cycles=first,estimated_issue_interval_cycles=ii,
                  modeled_repeat_us=pred['repeat'],development_shards=len(p['training_sha256']))
    lo,hi=min(pred.values()),max(pred.values())
    return dict(candidate_shape_us=2*lo*hi/(lo+hi), candidate_range_us=[lo,hi],
                candidate_repeat_us=pred['repeat'], stage_information=info)


def main():
    params=prices()
    src=ROOT/'results/v0.30/post/channel-register.json'
    reg=read(src)
    forward=[]; freeze=OUT/'channel-freeze.json'
    if not freeze.exists():
        assert not list((ROOT/'results/v0.30/post').glob('channel-Ascend*.json'))
        for chip,seq in reg['scenarios'].items():
            for t in seq:
                if t['role']=='exact-repeat':
                    forward.append(dict(chip=chip,shape=t['shape'],ordinal=t['i'],prior_shapes=[r['shape'] for r in seq[:t['i']]],
                            public_repeat_us=t['candidate_us'],**predict(t['shape'],chip,params)))
        freeze.write_text(json.dumps(dict(utc=datetime.now(timezone.utc).isoformat(),scope='diagnostic candidate; no change to default',
            training_role='all six DT channel repeats with terminal width <=448; both outcomes on wider DT/PR were visible and remain development checks',
            parameters=params, prospective_target='Original frozen and previously queued channel process forwards; both exact repeats per SKU require whole process and all original numeric/timing gates.',
            parent_registration_sha256=sha256(src.read_bytes()).hexdigest(),script_sha256=sha256(Path(__file__).read_bytes()).hexdigest(),
            new_ca_jobs=0,forward_rows=forward),indent=2)+'\n')
    else:
        assert read(freeze)['parameters']==params
    rows=read(OUT/'replay.json')['rows']; summary={}
    for chip in (DT,PR):
        rs=[r for r in rows if r['chip']==chip]
        values=[]
        for row in rs:
            out=predict(row['shape'],chip,params)
            values.append(dict(ca_us=row['ca_us'],shape=row['shape'],sha256=row['sha256'],**out))
        vals=[r['candidate_shape_us'] for r in values]; ys=[r['ca_us'] for r in values]
        wape=100*sum(abs(y-v) for y,v in zip(ys,vals))/sum(ys)
        worst=max(values,key=lambda r:abs(r['candidate_shape_us']/r['ca_us']-1))
        summary[chip]=dict(shape_only_all_windows=len(values),shape_only_wape_pct=wape,
                shape_only_max_pct=100*abs(worst['candidate_shape_us']/worst['ca_us']-1),worst=worst,rows=values)
    (OUT/'channel-diagnostic.json').write_text(json.dumps(summary,indent=2)+'\n')

if __name__=='__main__':main()
