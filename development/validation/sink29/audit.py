"""Audit live public routing and freeze the remaining no-additional-CA prefix probes."""
from collections import Counter,defaultdict
from datetime import datetime,timezone
from hashlib import sha256
import json
from math import ceil
from pathlib import Path
import re
import sys
from time import perf_counter_ns
from statistics import median
ROOT=Path(__file__).resolve().parents[3];sys.path.insert(0,str(ROOT))
V28=ROOT/'results/v0.28/fusion';V26=ROOT/'results/v0.26/sink'
OUT=ROOT/'results/v0.29/sink';REG=OUT/'freeze.json'
from operator_model import OperatorStream,load_chip,estimate
from context import envelope,scenario_mix
DT,PR,FUT='Ascend950DT_9582','Ascend950PR_957b','Ascend970DT'


def read(path):return json.loads(path.read_bytes())
def digest(path):return sha256(path.read_bytes()).hexdigest()
def save(path,obj):path.write_text(json.dumps(obj,indent=2,allow_nan=False)+'\n')
def utc():return datetime.now(timezone.utc).isoformat()


def row(sequence,sku,marker=None):
    c=load_chip(sku);plain=OperatorStream(c);s=OperatorStream(c,track_sinkhorn_history=True)
    lost=OperatorStream(c,track_sinkhorn_history=True);lost.forget_history()
    out=[];h=0
    for i,(t,it) in enumerate(sequence):
        shape=[t,4];opt={'num_iters':it}
        cur=plain.advance('mhc_sinkhorn',shape,opt,dtype='fp32')
        known=s.advance('mhc_sinkhorn',shape,opt,dtype='fp32')
        att=lost.advance('mhc_sinkhorn',shape,opt,dtype='fp32')
        new=envelope(shape,it,c,h);pure=envelope(shape,it,c)
        v=known.projection['target_plan'];assert v['resident_aiv']==h
        frozen=estimate('mhc_sinkhorn',shape,{**opt,'resident_aiv':h},chip=c,dtype='fp32')
        assert frozen.latency_us==known.latency_us
        out.append(dict(index=i,t=t,iters=it,sku=sku,H=h,active=v['active_aiv'],rows=v['rows_per_core'],G=v['register_passes'],
                        state=v['history_state'],default_us=cur.latency_us,ordered_us=known.latency_us,
                        lost_us=att.latency_us,prefix_guard_us=new['point_us'],unknown_guard_us=pure['point_us'],
                        partial=new,no_history=pure,epochs_us=known.projection['timing_diagnostics']['issue_epochs_us']))
        h=max(h,v['active_aiv'])
    return out


def sequence_of_legacy(v):
    path=ROOT/v['source'];p=next((a/'inputs.txt' for a in path.parents if (a/'inputs.txt').is_file()),None)
    if p is None:
        if v['h']!=0:raise ValueError('an unrecorded launch prefix cannot be inferred')
        log=next((a/'run.log' for a in path.parents if (a/'run.log').is_file()))
        text=log.read_text(errors='replace')
        assert text.count('Start profiling on kernel:')==1 and 'SMOKE_PASS' in text
        return [(v['t'],v['iters'])],0,str(log.relative_to(ROOT))
    jobs=[a.split() for a in p.read_text().splitlines()]
    assert all(j[0]=='mhc_sinkhorn' and int(j[1])==k and j[3]=='4' for k,j in enumerate(jobs))
    log=(p.parent/'run.log').read_text(errors='replace')
    observed=[tuple(map(int,q)) for q in re.findall(r'case=mhc_sinkhorn api=aclnnMhcSinkhorn T=(\d+) N=4 iters=(\d+) dtype=FP32',log)]
    seq=[(int(j[2]),int(j[4])) for j in jobs]
    assert observed==seq and 'SMOKE_PASS' in log
    index=int(path.name.split('_')[0]);assert seq[index]==(v['t'],v['iters'])
    return seq,index,str(p.relative_to(ROOT))


def original_groups(reg):
    seq=defaultdict(list)
    for j in reg['jobs']:
        if j['chip']==DT:seq[j.get('group','paired')].append((j['shape'][0],j['iters']))
    return seq


def register():
    if REG.exists():raise FileExistsError('freeze exists')
    paths=[V28/'register.json',V28/'eligibility.freeze.json']
    seen=[];future=[]
    for fn in paths:
        source=read(fn);groups=original_groups(source)
        for tag,seq in groups.items():
            for sku in (DT,PR,FUT):
                for v in row(seq,sku):
                    v['origin']=str(fn.relative_to(ROOT));v['group']=tag
                    if sku!=FUT:
                        path=V28/'receipts'/(f'{tag}-{sku}.json') if fn==paths[0] else V28/f'eligibility.receipt-{sku}.json'
                        v['receipt']=str(path.relative_to(ROOT))
                        if path.exists():seen.append(v);continue
                        # No open profile may be read by this study before freezing.
                        raw=V28/'raw'/f'{tag}-{sku}' if fn==paths[0] else ROOT/'runs/sink28-register-group'/sku
                        assert not raw.exists(),('target has already started',raw)
                    future.append(v)
    params=[ROOT/'operator_model/sink_history.py',ROOT/'operator_model/mhc_sink_resource.py',V26/'prices.json',
            ROOT/'results/v0.9/vector/profile.json',V28/'prices.json',Path(__file__).with_name('context.py'),*paths]
    record=dict(utc=utc(),method='no new physical coefficient; nearest existing point in the relative-error corridor over reachable source-proven G=1 prices. Exclude already seen calls, report overlap in geometry and groups separately. Exact history uses normal public stream estimate.',
        eligibility='No default change on old replays. New guard prices assessed against normal and previous registered points. Separate pristine, shared geometry (already observed T69 first), cold vs expanded/resident and partially joined streams; maximum 2 new T/I within support per SKU. Add no CA.',
        sources={str(p.relative_to(ROOT)):digest(p) for p in params},observed_before_freeze=seen,jobs=future)
    with REG.open('x') as f:json.dump(record,f,indent=2);f.write('\n')


def stat(rows,key):
    vals=[(v[key],v['ca_us']) for v in rows if v.get(key) is not None and v.get('ca_us') is not None]
    if not vals:return None
    return dict(n=len(vals),wape=100*sum(abs(x-y) for x,y in vals)/sum(y for x,y in vals),maximum=100*max(abs(x/y-1) for x,y in vals))


def summaries(rows):
    return {k:stat(rows,k) for k in ('default_us','ordered_us','lost_us','unknown_guard_us','prefix_guard_us','previous_v28_us')}


def run():
    reg=read(REG); allrows=[]
    historical=read(V28/'replay.json')['rows'];markers=read(V28/'landmarks.json')
    v26=read(V26/'register.json');obs26=read(V26/'independent.json')['rows'];lookup26={(r['chip'],r['group'],r['index']):r for r in obs26}
    for v in historical:
        if v['role']!='historical development' or v['iters']!=10 or v['t']<=64:continue
        seq,i,path=sequence_of_legacy(v); p=row(seq,v['sku'])[i];f=row(seq,FUT)[i]
        assert p['H']==v['h'] and abs(p['default_us']-v['old_unknown_us'])<1e-12 and abs(p['ordered_us']-v['old_known_us'])<1e-12
        z=next(x for x in markers if x['sku']==v['sku'] and x['shape'][0]==v['t'] and x['iters']==v['iters'] and x['history']==v['h'])
        assert digest(ROOT/z['trace'])==z['trace_sha256']
        m=z['milestones'];e=p['epochs_us']
        measured=[m['tiling_issue'],m['setup_issue'],m['read_issue'],m['soft_issue'],m['first_col_issue'],m['first_row_issue'],m['steady'],m['scatter_issue'],m['write_issue'],m['write_wait']]
        epoch=[1000*load_chip(v['sku']).frequency_ghz*e[k] for k in ('tiling_begin','setup_vf','read_begin','softmax_vf','first_col_vf','first_row_vf','steady_pair_period','scatter_vf','write_begin')]
        epoch.append((e['write_end']-e['write_begin'])*1650)
        marks=dict(zip(('tiling_issue','setup','input_issue','softmax_issue','first_column','first_row','steady_row_to_row','scatter','write_issue','write_lifetime'),zip(epoch,measured)))
        allrows.append(dict(**p,ca_us=v['ca_us'],role='original skewed slice, replay',sequence=path,sequence_inputs=seq,marker_trace=z['trace'],
            sha256=z['trace_sha256'],previous_v28_us=v['point_us'],marks_cycles=marks,per_core_vf=2*v['iters']+2,local_payload_bytes=z['dma_bytes'],
            future_970=dict(H=f['H'],default_us=f['default_us'],ordered_us=f['ordered_us'],no_history=f['no_history'])))
    for group,seq in original_groups(v26).items():
        future=row(seq,FUT)
        for sku in (DT,PR):
            for p in row(seq,sku):
                i=p['index'];v=lookup26[sku,group,i];f=future[i]
                assert p['H']==v['h'] and p['active']==v['active_ca'] and abs(p['ordered_us']-v['known_us'])<1e-12
                assert v['vf_ca_values']==[2*p['iters']+2] or set(v['vf_ca_values'])=={2*p['iters']+2}
                r=next(x for x in historical if x['sku']==sku and x['id']==v['raw_sha256'])
                allrows.append(dict(**p,ca_us=v['ca_us'],role='v0.26 prospective conditional model; replay for new guard',group=group,
                                    previous_v28_us=r['point_us'],receipt=v['receipt'],sha256=v['raw_sha256'],
                                    future_970=dict(H=f['H'],default_us=f['default_us'],ordered_us=f['ordered_us'],no_history=f['no_history'])))
    pending=[];probe=[]
    for v in reg['observed_before_freeze']+reg['jobs']:
        if v['sku']==FUT:continue
        receipt=ROOT/v['receipt'];r=dict(**v,ca_us=None,source_already_seen=v in reg['observed_before_freeze'])
        source=V28/'register.json' if v['group']!='paired' else V28/'eligibility.freeze.json'
        oldreg=read(source)
        j=next(j for j in oldreg['jobs'] if j['chip']==v['sku'] and (j.get('group','paired'),j['index'])==(v['group'],v['index']))
        r['previous_v28_us']=j.get('point_us')
        if receipt.exists():
            rc=read(receipt);obs=next((x for x in rc.get('rows',[]) if x['index']==v['index'] and x['qualified']),None)
            if obs:
                z=obs['stage'];assert obs['shape']==[v['t'],4] and obs['iters']==v['iters'] and obs['h']==v['H'] and z['active_aiv']==v['active']
                assert set(z['vf_calls_per_core'].values())=={2*v['iters']+2} and z['written_bytes']==64*v['t']
                r.update(ca_us=z['ca_us'],sha256=z['trace_sha256'])
        if r['ca_us'] is None:pending.append(r)
        else:probe.append(r)
    summary={}
    for sku in (DT,PR):
        old=[v for v in allrows if v['sku']==sku and v['role'].startswith('original')]
        ordered=[v for v in allrows if v['sku']==sku and v['role'].startswith('v0.26')]
        full=[v for v in ordered if v['t']<=256]
        summary[sku]={label:summaries(rs) for label,rs in [
            ('original-slice',old),('v26-ordered-T-le256',full),('v26-same-I10-T>64', [v for v in ordered if v['t']>64 and v['iters']==10]),
            ('v26-joined-H-partially-known', [v for v in full if v['H']>0 and len(v['partial']['states'])>1]),
            ('v26-joined-H-proves-reuse',[v for v in full if v['H']>0 and len(v['partial']['states'])==1]),
            ('v28-earlier-registration-complete',[v for v in probe if v['sku']==sku and v['group']!='paired']),
            ('v29-guard-eligible-previously-unseen',[v for v in probe if v['sku']==sku and not v['source_already_seen'] and v['unknown_guard_us'] is not None])]}
    return dict(utc=utc(),freeze_sha256=digest(REG),history=allrows,forward_complete=probe,pending=pending,metrics=summary)


if __name__=='__main__':
    if sys.argv[1]=='freeze':register()
    else:
        d=run();save(OUT/'score.json',d)
        from reporting import output
        output(d)
