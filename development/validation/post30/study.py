"""Reprice the complete historical cohort, and independently freeze new paired prices."""
from collections import defaultdict
from datetime import datetime,timezone
from hashlib import sha256
import json
from pathlib import Path
import sys
from statistics import median
from time import perf_counter_ns
from math import isclose
ROOT=Path(__file__).resolve().parents[3];sys.path.insert(0,str(ROOT))
from operator_model import OperatorStream,load_chip
from operator_model.post_fragment_pipeline import estimate_post_fragments
from operator_model.post_repeat import channel_tiles
from development.validation.memory27.accuracy import floor_metrics
from development.validation.post29 import replay
OUT=ROOT/'results/v0.30/post'
OLD=ROOT/'results/v0.29/post'
REG=OUT/'freeze.json'
DT,PR,FUT=replay.DT,replay.PR,replay.FUTURE
def read(f):return json.loads(f.read_bytes())
def h(f):return sha256(f.read_bytes()).hexdigest()
def now():return datetime.now(timezone.utc).isoformat()
def save(p,obj,new=False):
    if new:
        with p.open('x') as f:json.dump(obj,f,indent=2,allow_nan=False);f.write('\n')
    else:
        tmp=p.with_suffix('.tmp');tmp.write_text(json.dumps(obj,indent=2,allow_nan=False)+'\n');tmp.replace(p)


def freeze():
    src=OLD/'split-register.json'
    old=read(src);status=read(OLD/'split-state.json')
    assert 'queued' in status['phase'] and not any(OLD.glob('split-Ascend*.json')) and not REG.exists()
    plans=[]
    for chip,rows in old['rows'].items():
        for r in rows[1:]:
            p=estimate_post_fragments(r['shape'],chip=chip,resident_aiv=r['prior_aiv'],previous_shape=r['previous_shape'])
            plans.append(dict(chip=chip,ordinal=r['i'],previous_shape=r['previous_shape'],shape=r['shape'],
                resident_aiv=r['prior_aiv'],baseline_us=r['public_us'],frozen_aggregate_us=r['aggregate_us'],
                conditional_us=p['us'],stages_us=p['stages_us'],plan=p['channel_tiles'],critical_rows=p['critical_rows'],
                extrapolated=chip==FUT))
    inputs=[src,OUT/'prices.json',ROOT/'operator_model/post_fragment_pipeline.py',
            ROOT/'operator_model/post_resource.py',ROOT/'operator_model/post_repeat.py',ROOT/'results/v0.25/post/prices.json',
            *[ROOT/f'configs/chips/{c}.json'for c in (DT,PR,FUT)]]
    save(REG,dict(utc=now(),target_slots='Reuse the registered v0.29 official Post sequences, no new CA launches.',
                  physical_role='DT [65,4,5120] ordinary launch is development. Neither PR nor repeat sets fit prices; those historical outcomes were previously visible.',
                  test='Two frozen unequal-tail shapes, per SKU both WAPE <=25% and max <=50%, strictly better WAPE than v0.29 ordered and its pre-CA whole-row alternative. The original v0.29 eligibility gates remain in force. Two shapes alone do not qualify the conditional schedule for unrestricted default.',
                  archive_preregistered=old['utc'],state_at_freeze=status['phase'],plans=plans,
                  inputs={str(f.relative_to(ROOT)):h(f) for f in inputs}),True)


class ReconstructedDispatch(replay.Dispatch):
    def __init__(self,sku):
        super().__init__(sku)
        self.last_shape=None
        self.proposed=OperatorStream(sku,track_post_history=True,experimental_post_fragments=True)
        self.future_proposed=OperatorStream(FUT,track_post_history=True,experimental_post_fragments=True)

    def forget(self):
        super().forget();self.proposed.forget_history();self.future_proposed.forget_history();self.last_shape=None

    def call(self,shape,op=replay.OP,attrs=None):
        prev=self.last_shape
        start=perf_counter_ns()
        optional=self.proposed.advance(op,shape,attrs)
        elapsed=(perf_counter_ns()-start)/1e6
        future_optional=self.future_proposed.advance(op,shape,attrs)
        resident,exact=self.new._prefix('mhc_post')
        case=super().call(shape,op,attrs)
        self.last_shape=list(shape) if op==replay.OP else None
        if case is None:return
        case.update(previous_shape=prev,resident_lower_bound=resident,resident_exact=exact,
                    candidate_us=case['ordered_us'],decision='public ordered')
        if prev and case['target_plan']['route']=='post_token' and prev!=list(shape) and case['scenario'] in ('reuse','reuse_after_token'):
            c=load_chip(self.sku);s=channel_tiles(shape[2],c)
            if s and s['count']==2 and 2*prev[0]>=c.vector_cores and resident>=case['target_plan']['active_aiv']:
                t=perf_counter_ns()
                proposed=estimate_post_fragments(shape,chip=c,resident_aiv=resident,previous_shape=prev)
                case.update(candidate_us=proposed['us'],decision=proposed['model'],
                            candidate_query_ms=(perf_counter_ns()-t)/1e6)
        assert isclose(case['candidate_us'],optional.latency_us,abs_tol=1e-9)
        case['stream_query_ms']=elapsed
        case['candidate_970_us']=future_optional.latency_us
        case['candidate_970_model']=future_optional.projection['timing_diagnostics']['model']
        return case


def score(rows):
    def one(key):
        vals=[abs(r[key]/r['ca_us']-1)*100 for r in rows]
        return dict(wape=100*sum(abs(r[key]-r['ca_us'])for r in rows)/sum(r['ca_us']for r in rows),maximum=max(vals))
    return dict(windows=len(rows),shapes=len({tuple(r['shape'])for r in rows}),
            **{k:one(k) for k in ('ordered_us','candidate_us','stateless_us')}) if rows else None


def replay_all():
    replay.Dispatch=ReconstructedDispatch
    replay.mixed_archive();replay.registered()
    raw=list(replay.rows.values());physical=list(replay.physical.values())
    prior=read(OLD/'score.json');old={r['sha256']:r for r in prior['rows']}
    for r in raw:
        if r['sha256'] in old:
            assert isclose(r['ordered_us'],old[r['sha256']]['ordered_us'],abs_tol=1e-8)
    assert set(old).issubset({r['sha256'] for r in raw})
    # time/cohort membership of both source windows is recovered from the actual ordered host, never from CA latency.
    summaries={}
    for chip in (DT,PR):
        rs=[r for r in raw if r['chip']==chip]
        summaries[chip]={name:score(sub) for name,sub in [('all',rs),('changed',[r for r in rs if r['decision']!='public ordered']),
             ('unchanged',[r for r in rs if r['decision']=='public ordered']),
             ('channel',[r for r in rs if r['target_plan']['route']=='post_channel']),
             ('ordinary_single_fragment',[r for r in rs if r['model']=='post_token_ordered']),
             ('exact_token_repeat',[r for r in rs if r['model']=='post_token_immediate_repeat']),
             ('physical_retirements_separate',[r for r in physical if r['chip']==chip])]}
    return dict(rows=raw,physical_rows=physical,summary=summaries,source_sha256=replay.source_hashes,
                timing_scope='qualified published full windows. Physical-termination windows are extra, retained separately.')


def lower_bounds():
    inputs=read(ROOT/'results/v0.24/post/evaluation.json')['rows'];res={}
    for chip in (DT,PR):
        rows=[r for r in inputs if r.get('timing_qualified') and r['chip']==chip]
        by=defaultdict(list)
        for r in rows:by[tuple(r['shape'])].append(r)
        fixed=[]
        for shape,rs in by.items():
            assert len({r['kernel_sha']for r in rs})==len({r['kernel']for r in rs})==1
            xs=[r['ca_us']for r in rs];lo,hi=min(xs),max(xs);m=median(xs)
            left,right=.5*hi,1.5*lo
            fixed.append(dict(shape=shape,kernel=rs[0]['kernel'],released_sha=rs[0]['kernel_sha'],us=xs,
                        feasible_at_max50=left<=right,interval_at_max50=[left,right],best_WAPE_point=m,
                        best_WAPE_point_at_max50=max(left,min(right,m)) if left<=right else None,
                        minimum_worst_pct=100*(hi-lo)/(hi+lo),minimum_worst_point=2*lo*hi/(lo+hi)))
        res[chip]=dict(**floor_metrics(rows),by_shape=fixed)
    return res


def forward():
    reg=read(REG);rows=[]
    for sku in (DT,PR):
        path=OLD/('split-'+sku+'.json')
        if not path.is_file():continue
        j=read(path);assert j['cost']['started_utc']>reg['utc']
        for r in j.get('rows',[]):
            if not r.get('i') or not r.get('timing_qualified'):continue
            pred,=(p for p in reg['plans'] if p['chip']==sku and p['ordinal']==r['i'])
            assert r['shape']==pred['shape'] and r['stage']['kernel'].endswith('_6') and r['active_matches']
            rows.append(dict(**pred,ca_us=r['stage']['ca_us'],kernel=r['stage']['kernel'],trace_sha256=r['stage']['trace_sha256'],cost=j['cost']))
    return rows


if __name__=='__main__':
    if sys.argv[1]=='freeze':freeze()
    if sys.argv[1]=='score':
        frames=replay_all();save(OUT/'score.json',dict(**frames,bounds=lower_bounds(),
            independently_returned=forward(),updated_utc=now()))
