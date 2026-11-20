"""Reproduce the v0.34 installation score. Run with python -I from any directory."""
from pathlib import Path
import sys,os,json,tempfile
from math import isclose,ceil
from hashlib import sha256
from statistics import median
from datetime import datetime,timezone
from collections import defaultdict
from time import perf_counter_ns

ROOT=Path(__file__).resolve().parents[3];OUT=ROOT/'results/v0.34';HERE=Path(__file__).resolve().parent
DT,PR,FUT='Ascend950DT_9582','Ascend950PR_957b','Ascend970DT'

def save(name,value):
    p=OUT/name;t=p.with_name(f'.{name}.{os.getpid()}.tmp');t.write_text((value if isinstance(value,str) else json.dumps(value,indent=2,ensure_ascii=False,allow_nan=False))+'\n');t.replace(p)

def run(prefix):
    assert sys.flags.isolated;prefix=Path(prefix).resolve();sys.path.insert(0,str(prefix));import operator_model as api
    assert api.__version__=='0.34.0' and Path(api.__file__).is_relative_to(prefix)
    sys.path.append(str(HERE));sys.path.append(str(ROOT));from manifest import fingerprint
    from incoming import incoming,AT
    from ordered import ordered
    from tables import render
    man=fingerprint(ROOT,prefix);archive=ROOT/'results/v0.32/replay.json';man.update(archive_sha256=sha256(archive.read_bytes()).hexdigest(),scope_sha256=sha256((ROOT/'results/v0.33/acceptance-scope.json').read_bytes()).hexdigest());save('model.json',man)
    originals=json.loads(archive.read_bytes())['rows'];assert len(originals)==1979
    hist=ordered(api,ROOT,save);prices=[];rows={}
    def key(r):return r['op'],r['chip'],tuple(r['identity'])
    def public(r,sku):
        route=r['op'];base=route.partition(' · ')[0];op={'vllm_hc_post':'mhc_post','fla_kda_v2':'kda_chunk'}.get(base,base)
        start=perf_counter_ns();p=api.estimate(op,r['shape'],r['attrs'],chip=sku);elapsed=(perf_counter_ns()-start)/1e6
        prices.append(dict(op=route,chip=sku,shape=r['shape'],wall_ms=elapsed));return p.latency_us,elapsed
    def add(r,old,role):
        k=key(r)
        if k in rows:
            assert rows[k]['shape']==r['shape'] and isclose(rows[k]['ca_us'],r['ca_us'],abs_tol=.005), k
            if r.get('verified_trace_sha256'):
                rows[k].update(requalification_source=r['source'],verified_trace_sha256=r['verified_trace_sha256'])
            return
        row=dict(r);row.update(previous_us=old,previous_version='0.32.0' if role=='archive' else None)
        if role=='archive':row.update(data_role='replay',classification='known qualified window from v0.32',archive_role=r['role'])
        if r['op']=='mhc_post · stream':
            if 'launch_sequence' in r:
                def call_stream(sku):
                    seq=api.OperatorStream(sku);history=r['launch_sequence'];assert history[-1]==r['shape']
                    for sh in history[:-1]:seq.advance('mhc_post',sh)
                    start=perf_counter_ns();val=seq.advance('mhc_post',history[-1]);return val.latency_us,(perf_counter_ns()-start)/1e6
                (us,w),(fu,wf)=call_stream(r['chip']),call_stream(FUT)
            else:
                v=hist[r['identity'][0]];us,fu,w,wf=v['stream_us'],v['future'],v['wall'],v['future_wall']
            for c,t in [(r['chip'],w),(FUT,wf)]:prices.append(dict(op=r['op'],chip=c,shape=r['shape'],wall_ms=t))
        else:(us,w),(fu,wf)=public(r,r['chip']),public(r,FUT)
        if role!='archive' and r['data_role']=='independent' and (us is None or r.get('forecast_before_ca_us') is None or not isclose(us,r['forecast_before_ca_us'],abs_tol=1e-6)):
            row.update(data_role='replay',classification='updated API has no matching frozen price on this geometry')
        row.update(us=us,extrapolated_970dt_us=fu,query_ms=w,query_970_ms=wf,release_origin=role)
        rows[k]=row
    for r in originals:add(r,r['us'],'archive')
    assert len(rows)==1979
    for r in incoming(ROOT,api):add(r,None,'supplement')
    # Every offered price has the installed module as implementation, including imports performed by historical tooling.
    mods={k:str(Path(v.__file__).resolve()) for k,v in sys.modules.items() if (k=='operator_model' or k.startswith('operator_model.')) and getattr(v,'__file__',None)}
    assert all(Path(p).is_relative_to(prefix) for p in mods.values()),mods
    allrows=list(rows.values());save('replay.json',dict(version=api.__version__,units='us',snapshot_utc=datetime.now(timezone.utc).isoformat(),baseline='v0.32',archive_rows=len(originals),supplemental_rows=len(allrows)-len(originals),rows=allrows))
    timed=[p['wall_ms'] for p in prices];measured=sorted(timed);cold=dict(n=len(timed),p50=median(timed),p95=measured[ceil(.95*len(timed))-1],maximum=measured[-1],over_100ms=sum(t>=100 for t in timed),all_under_100ms=max(timed)<100)
    save('wheel-check.json',dict(version=api.__version__,wheel_sha256=man['wheel_sha256'],installed_modules=len(mods),public_calls=cold,slowest=sorted(prices,key=lambda p:-p['wall_ms'])[:10],protocol='In-process uncached estimate or stateful advance invoked per row and on 970DT, no score-level memoization. Config/price tables remain loaded. Python startup, disk audit, CANN and offline CA excluded.'))
    extra=[];reg=json.loads((OUT/'attn/freeze.json').read_bytes())
    for case in reg['jobs']:
        for sku in (DT,PR,FUT):
            start=perf_counter_ns();value=api.estimate('attn_res_fwd',case['shape'],case['attrs'],chip=sku);elapsed=(perf_counter_ns()-start)/1e6
            assert isclose(value.latency_us,case['prices'][sku]['us'],abs_tol=1e-6)
            extra.append(dict(op=AT,chip=sku,shape=case['shape'],wall_ms=elapsed,predicted_us=value.latency_us,qualification='timing of forecast only, no CA implied'))
    save('pending-query-cost.json',dict(version=api.__version__,rows=extra))
    render(ROOT,allrows,AT,prices+extra,save)
    print('archive',len(originals),'supplement',len(allrows)-len(originals),'public calls',cold,flush=True)

if __name__=='__main__':run(sys.argv[1])
