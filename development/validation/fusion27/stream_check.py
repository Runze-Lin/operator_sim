"""Replay ordered dispatch inputs, then compare predicted H with frozen expectations and CA."""
import argparse
import hashlib
import json
from pathlib import Path
import re
from statistics import median
import sys
from time import perf_counter_ns

ROOT=Path(__file__).resolve().parents[3]
BASE=ROOT/'results/v0.26/sink'
OUT=ROOT/'results/v0.27/fusion'
DT,PR,FUT='Ascend950DT_9582','Ascend950PR_957b','Ascend970DT'

def read(path):return json.loads(path.read_bytes())
def digest(path):return hashlib.sha256(path.read_bytes()).hexdigest()

def ordered_inputs(group, sku):
    folder=BASE/'raw'/(group+'-'+sku);file=folder/'batch.txt';log=folder/'run.log'
    batch=[tuple(map(int,line.split()[1:])) for line in file.read_text().splitlines() if line.strip()]
    pattern=r'^BATCH_BEGIN (\d+)\s*$\n(.*?)^BATCH_END \1\s*$'
    events=re.findall(pattern,log.read_text(errors='replace'),flags=re.M|re.S)
    seen=[]
    for idx,text in events:
        [shape]=re.findall(r'^case=mhc_sinkhorn api=aclnnMhcSinkhorn T=(\d+) N=(\d+) iters=(\d+) dtype=FP32$',text,re.M)
        seen.append((int(idx),*map(int,shape)))
    if batch!=seen:raise ValueError('actual run order differs from batch manifest')
    return seen,dict(input=str(file.relative_to(ROOT)),input_sha256=digest(file),hostlog=str(log.relative_to(ROOT)),log_sha256=digest(log))

def metric(rows):
    return dict(distinct_shapes=len({tuple(r['shape']) for r in rows}),windows=len(rows),
                wape_pct=100*sum(abs(r['estimated_us']-r['ca_us'])for r in rows)/sum(r['ca_us'] for r in rows),
                max_pct=100*max(abs(r['estimated_us']/r['ca_us']-1)for r in rows))

def main(installed):
    if not installed:sys.path.insert(0,str(ROOT))
    from operator_model import OperatorStream,estimate
    import operator_model
    import operator_model.stream as stream_module
    if installed and ROOT in Path(operator_model.__file__).resolve().parents:raise ValueError('installed package required')
    if digest(Path(stream_module.__file__)) != digest(ROOT/'operator_model/stream.py'):
        raise ValueError('installed stream module differs from checked source')
    src=BASE/'register.json';reg=read(src);lock={(j['chip'],j['group'],j['index']):j for j in reg['jobs']}
    groups=reg['order'];orders={};observations={};rows=[];wall=[];peak_delta=0.;peak_epoch=0.
    for g in groups:
        for c in (DT,PR):
            order,orig=ordered_inputs(g,c);orders[g,c]=order
            p=BASE/'receipts'/(g+'-'+c+'.json');rec=read(p)
            if rec['register_sha256']!=digest(src):raise ValueError('registration changed')
            observations[g,c]=dict(order=orig,receipt=str(p.relative_to(ROOT)),sha256=digest(p),rows=rec['rows'])
        if orders[g,DT]!=orders[g,PR]:raise ValueError('SKU order mismatch')
        for c in (DT,PR,FUT):
            stream=OperatorStream(c,track_sinkhorn_history=True)
            for idx,t,n,it in orders[g,DT]:
                # Do not pass H, actual active AIV, actual timing, trace or frozen timing.
                start=perf_counter_ns();v=stream.advance('mhc_sinkhorn',(t,n),{'num_iters':it},dtype='fp32');wall.append((perf_counter_ns()-start)/1e6)
                j=lock[c,g,idx];plan=v.projection['target_plan'];e=v.projection['timing_diagnostics']['issue_epochs_us']
                delta=abs(v.latency_us-j['known']['latency_us'])
                peak_delta=max(peak_delta,delta);peak_epoch=max(peak_epoch,*[abs(e[k]-x) for k,x in j['known']['vf_setup_epochs'].items()])
                if (delta>1e-11 or peak_epoch>1e-11 or plan['resident_aiv']!=j['prior_h'] or v.active_aiv!=j['known']['active_aiv']
                        or v!=estimate('mhc_sinkhorn',(t,n),{'num_iters':it,'resident_aiv':j['prior_h']},chip=c)):
                    raise ValueError('incomplete independent sequence equivalence')
                row=dict(sku=c,group=g,index=idx,shape=[t,n],iterations=it,status=v.status,auto_h=plan['resident_aiv'],
                         expected_h=j['prior_h'],active=v.active_aiv,state=plan['history_state'],
                         estimated_us=v.latency_us,frozen_us=j['known']['latency_us'],ca_us=None)
                if c!=FUT:
                    ref=observations[g,c];r=ref['rows'][idx];stage=r['stage']
                    if not r['qualified'] or (r['shape'],r['iters'],r['h'],r['index'])!=([t,n],it,plan['resident_aiv'],idx):
                        raise ValueError('wrong qualifying CA contract')
                    if stage['active_aiv']!=v.active_aiv or set(stage['vf_calls_per_core'].values())!={2*it+2}:
                        raise ValueError('unexpected actual work')
                    row.update(ca_us=stage['ca_us'],ca_source=ref['receipt'],raw_sha256=stage['trace_sha256'])
                rows.append(row)
    comparisons={}
    for c in (DT,PR):
        for name,inside in [('original_domain',True),('large_single_ub',False)]:
            subset=[r for r in rows if r['sku']==c and (r['shape'][0]<=256)==inside]
            comparisons[c+' '+name]=metric(subset)
    expected=45
    if len(rows)!=expected:raise ValueError('lost dispatch')
    benchmarks=[]
    for _ in range(20):
        for c in (DT,PR,FUT):
            for g in groups:
                s=OperatorStream(c,track_sinkhorn_history=True)
                for _,t,n,it in orders[g,DT]:
                    a=perf_counter_ns();s.advance('mhc_sinkhorn',(t,n),{'num_iters':it});benchmarks.append((perf_counter_ns()-a)/1e6)
    wall=wall+benchmarks
    output=dict(scope='Same ordered host dispatches per group; no H was supplied. 970 repeats the 950 inputs on the future target; no future CA.',
                import_route='wheel outside the checkout' if installed else 'source checkout',
                frozen_manifest_sha256=digest(src),stream_source_sha256=digest(ROOT/'operator_model/stream.py'),
                source_orders={g+' '+c:{k:v for k,v in ref.items() if k!='rows'} for (g,c),ref in observations.items()},
                total_sequences=12,dispatches=len(rows),actual_ca_windows=sum(r['ca_us'] is not None for r in rows),
                all_explicit_vs_auto_equal=True,max_frozen_delta_us=peak_delta,max_frozen_epoch_delta_us=peak_epoch,
                timings_ms=dict(n=len(wall),median=median(wall),p99=sorted(wall)[int(.99*len(wall))],max=max(wall)),
                metrics=comparisons,rows=rows)
    fn=OUT/('stream-installed.json' if installed else 'stream.json');fn.write_text(json.dumps(output,indent=2)+'\n')
    print(output['import_route'],output['dispatches'],output['max_frozen_delta_us'],output['timings_ms'])

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--installed',action='store_true');main(parser.parse_args().installed)
