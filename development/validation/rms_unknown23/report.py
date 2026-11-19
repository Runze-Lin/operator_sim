"""Retrospective scores, prospective matched-pair errors, and exact unknown-state feasibility."""
from collections import defaultdict
import json,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[3];sys.path.insert(0,str(ROOT))
from operator_model import load_chip
from development.validation.sku19.campaign import write
from development.validation.rms_unknown23.fast import forecast
OUT=ROOT/'results/rms-unknown23'


def scores(rows,columns=('previous_us','candidate_us','known_us')):
    if not rows:return {'n':0,'scores':{}}
    result={}
    for k in columns:
        result[k]=dict(wape=sum(abs(r[k]-r['ca_us']) for r in rows)/sum(r['ca_us'] for r in rows),max_ape=max(abs(r[k]/r['ca_us']-1) for r in rows))
    return dict(n=len(rows),scores=result)


def history():
    archive=json.loads((ROOT/'results/v0.22/rms-unknown/audit.json').read_text())['rows'];out=[];cohorts=defaultdict(list)
    for r in archive:
        shape=r['shape'];chip=load_chip(r['chip']);p=forecast(shape,chip);a=p['active_aiv'];h=r['prior_aiv'];state='first' if h==0 else 'hot' if a<=h else 'expand'
        val=p['conditional_us_by_H'][min(a,h)];assert abs(val-r['known_us'])<1e-8 and abs(p['previous_us']-r['unknown_us'])<1e-8
        s={k:r[k] for k in ('shape','chip','prior_aiv','source','trace','trace_id','ca_us','provider')};s.update(**p,known_us=val,state=state)
        out.append(s)
        for key in [r['provider'],r['source'],state,r['provider']+'/'+state,'all']:cohorts[(chip.name,key)].append(s)
    external={}
    isolated=json.loads((ROOT/'results/rms-dispatch-audit/independent.json').read_text());ordered=json.loads((ROOT/'results/rms-dispatch-audit/prefix-recovery/comparison.json').read_text())['rows']
    def records(obj):
        if isinstance(obj,dict):
            if 'candidate_known_us' in obj and 'ca_us' in obj and 'shape' in obj:yield obj
            else:
                for s in obj.values():yield from records(s)
        elif isinstance(obj,list):
            for s in obj:yield from records(s)
    for raw in [*[r for r in records(isolated) if r.get('protocol')=='isolated frozen v0.21'],*ordered]:
        if not raw.get('ca_us'):continue
        k=(raw['chip'],raw['protocol']);shape=tuple(raw['shape']);d=forecast(shape,load_chip(k[0]));h=raw['prior_aiv'];a=d['active_aiv'];d.update(shape=shape,ca_us=raw['ca_us'],known_us=d['conditional_us_by_H'][min(h,a)]);external.setdefault(k,{})[shape]=d
    write(OUT/'replay.json',dict(role='Historical replay only. Each actual distinct trace appears once in archive. Separate isolated/ordered reported on their registered ten targets, omitting repeated prefix controls.',rows=out,groups=[dict(chip=c,group=g,**scores(rs)) for (c,g),rs in cohorts.items()],paired_protocols=[dict(chip=c,group=g,**scores(list(rs.values()))) for (c,g),rs in external.items()]))


def interval(entry,repeat):
    low=min(entry,repeat);high=max(entry,repeat);l=max(low,.6*high);u=min(high,1.4*low);return dict(minimax=(high-low)/(high+low),allowed=[l,u] if l<=u else None,harmonic=2*low*high/(low+high),raw_bounds=[.6*high,1.4*low])


def feasibility(pair):
    if not pair:return None
    e_sum=sum(r['entry_us'] for r in pair);r_sum=sum(r['repeat_us'] for r in pair);D=sum(abs(r['entry_us']-r['repeat_us']) for r in pair)
    feasible_peak=all(r['allowed'] for r in pair);out=dict(n=len(pair),minimax_max=max(r['minimax'] for r in pair),max40_feasible=feasible_peak,both_protocol_minimax_wape_without_max=D/(e_sum+r_sum))
    if not feasible_peak:return {**out,'max40_and_each_wape20':False}
    # Every point in [min(E,R),max(E,R)] splits the unavoidable pair error D
    # between the two protocols. Choose total error on entries in this interval.
    min_e=sum(min(abs(x-r['entry_us']) for x in r['allowed']) for r in pair)
    max_e=sum(max(abs(x-r['entry_us']) for x in r['allowed']) for r in pair)
    a=max(min_e,D-.2*r_sum);b=min(max_e,.2*e_sum)
    return {**out,'sum_abs_entry_feasible_us':[min_e,max_e],'sum_abs_entry_joint_feasible_us':[a,b] if a<=b else None,'max40_and_each_wape20':a<=b}


def independent():
    receipts={};rows=[];failed=[];pairs=[];groups=defaultdict(list)
    for f in sorted((OUT/'verified').glob('*.json')):
        j=json.loads(f.read_text());rs=j['rows'];receipts[f.name]=dict(qualified=j['qualified'],reason=j.get('reason'),ca_wall_s=j['cost']['seconds'],raw_bytes=j['cost']['raw_bytes']);within={}
        for r in rs:
            if not r['timing_qualified']:
                failed.append(dict(chip=r['chip'],process=r['process'],i=r['i'],role=r['role'],shape=r['shape'],reason=r['reason']));continue
            if r['role']=='prefix':continue
            s={k:r[k] for k in ('chip','process','i','role','shape','prior_aiv','active_aiv','known_us','candidate_us','previous_us','correctness','host_task')};s.update({k:r['stage'][k] for k in ('kernel','trace','trace_sha256','binary_sha256','ca_us','first_tick','last_tick')});s['elf_text_sha256']=j['elf_text_sha256'];state='first' if s['prior_aiv']==0 else 'hot' if s['prior_aiv']>=s['active_aiv'] else 'expand';s['state']=state;rows.append(s)
            for g in ['all',s['role'],state]:groups[s['chip'],g].append(s)
            within[tuple(s['shape']),s['role']]=s
        for sh in sorted({k[0] for k in within}):
            entry=within.get((sh,'entry'));repeat=within.get((sh,'immediate_repeat'))
            if not entry or not repeat:continue
            assert entry['i']+1==repeat['i'] and entry['last_tick']<repeat['first_tick'] and entry['trace_sha256']!=repeat['trace_sha256']
            assert entry['candidate_us']==repeat['candidate_us'] and entry['previous_us']==repeat['previous_us'] and entry['elf_text_sha256']==repeat['elf_text_sha256']
            # NORM creates one aclrtStream outside the whole batch and passes it
            # to both checked calls. CA logs a different raw SQE stream_id for
            # each full_task_id; those values are preserved in the raw rows.
            assert entry['host_task']['full_task_id']+1==repeat['host_task']['full_task_id']
            e=entry['ca_us'];r=repeat['ca_us'];pairs.append(dict(chip=j['chip'],process=j['process'],shape=sh,entry_us=e,repeat_us=r,entry_H=entry['prior_aiv'],repeat_H=repeat['prior_aiv'],entry_trace=entry['trace_sha256'],repeat_trace=repeat['trace_sha256'],**interval(e,r)))
    write(OUT/'independent.json',dict(expected_shapes_per_chip=10,expected_scored_windows_per_chip=20,completed_processes=len(receipts),receipts=receipts,rows=rows,failed=failed,pairs=pairs,groups=[dict(chip=c,group=g,**scores(rs)) for (c,g),rs in groups.items()],bounds=[dict(chip=c,**feasibility([p for p in pairs if p['chip']==c])) for c in sorted({p['chip'] for p in pairs})]))

if __name__=='__main__':
    if 'independent-only' not in sys.argv:history()
    independent()
