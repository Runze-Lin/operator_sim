"""Fit stages on archived traces and register all prospective ordered streams."""
from collections import defaultdict
from hashlib import sha256
import json,math
from pathlib import Path
import statistics,sys,time

ROOT=Path(__file__).resolve().parents[3];sys.path.insert(0,str(ROOT))
from development.validation.post_candidate import evaluate as e, issue_response as ir
from development.validation.post24.model import predict,milestone_base
from development.validation.sku19.campaign import read,write,DT,PR,now
from operator_model import estimate,load_chip
OUT=ROOT/'results/v0.24/post'; CHIPS=[DT,PR,'Ascend970DT']
SEQS={'channel':[(1,192),(2,448),(8,896)],'wide':[(3,7168),(5,5120),(9,3840)],
      'boundary':[(16,768),(29,320)],'rows':[(69,640),(257,832)]}
FILES=['development/validation/post24/model.py','development/validation/post24/freeze.py',
       'development/validation/post_candidate/issue_response.py','development/validation/post_candidate/evaluate.py',
       'operator_model/post_resource.py','operator_model/post_v12.py','operator_model/wide_v10.py',
       'results/v0.12/post/profile.json','build/vector-v8/MHC','smoke/vector_v8.cpp','smoke/extension_indexer_mhc.cpp',
       'results/post-candidate/phase-evidence.json','results/post-candidate/paired-result.json',
       'results/v0.22/post-stream/replay.json','results/post-candidate/prospective.json']
def sha(path):return sha256(path.read_bytes()).hexdigest()
def known_shapes():
    known=set()
    def visit(x,op=''):
        if isinstance(x,dict):
            op=x.get('op',x.get('operator',op));shape=x.get('shape')
            if op in ('mhc_post','aclnnMhcPost') and isinstance(shape,list) and len(shape)==3 and all(type(a)==int for a in shape):known.add(tuple(shape))
            for v in x.values():visit(v,op)
        elif isinstance(x,list):
            for v in x:visit(v,op)
    for path in (ROOT/'results').glob('v*/evaluation.json'):visit(read(path))
    for name in ['phase-evidence.json','prospective.json','paired-result.json']:visit(read(ROOT/'results/post-candidate'/name),'mhc_post')
    return known


def calibration():
    archive=read(ROOT/'results/post-candidate/phase-evidence.json')
    curves=e.train(archive)
    samples=[]
    for sku,v in read(ROOT/'results/post-candidate/paired-result.json')['chips'].items():
        for r in v['rows']:
            if r['condition']=='reuse':
                samples.append(dict(**v['target_plan'],chip=sku,state='repeat',shape=v['shape'],**r['ca_milestones']))
    params={}
    for route,state,observations in [('post_token','first',archive),('post_channel','repeat',samples)]:
        arr=[row for row in observations if row['state']==state and row['route']==route]
        port=ir.fit_ports(arr)[route][state]
        phases={}
        for q in ['arrival','control','body','retire']:
            levels,slopes=curves[route,q];slopes=slopes.tolist()
            if state=='first':offset=levels[state]
            else:
                # Retain archive-derived width/row/arrival service slopes. Reprice only the
                # measured milestone's fixed response at the repeated-launch boundary.
                vals=[]
                for x in arr:
                    c=load_chip(x['chip']);service=sum(a*b for a,b in zip(slopes,milestone_base(x,c,q)))
                    vals.append((x['phase_us'][q]-service)*c.frequency_ghz/1.65)
                offset=max(0.,statistics.median(vals))
            phases[q]=dict(base_us=offset,slopes=slopes)
        params.setdefault(route,{})[state]=dict(phase=phases,port=port,calibration_windows=len(arr))
    return params


def main():
    path=OUT/'freeze.json'
    if path.exists():raise FileExistsError(path)
    shapes={(t,4,d) for pairs in SEQS.values() for t,d in pairs}
    assert len(shapes)==10 and not shapes&known_shapes()
    params=calibration()
    regs=[]
    for sku in CHIPS:
        c=load_chip(sku)
        for group,pairs in SEQS.items():
            jobs=[]
            for t,d in pairs:
                jobs.extend([('entry',[t,4,d]),('repeat',[t,4,d])])
            if group in ('channel','wide'):jobs.append(('revisit',[pairs[0][0],4,pairs[0][1]]))
            history=0;last=None
            for i,(role,shape) in enumerate(jobs):
                start=time.perf_counter_ns(); pred=predict(shape,c,history,last,params);runtime=(time.perf_counter_ns()-start)/1e6
                pub=estimate('mhc_post',shape,{'resident_aiv':history},chip=c,dtype='bf16')
                unknown=estimate('mhc_post',shape,chip=c,dtype='bf16').latency_us
                assert pub.active_aiv==pred['active']
                regs.append(dict(chip=sku,group=group,i=i,role=role,shape=shape,prior_aiv=history,
                             previous_shape=last,old_known_us=pub.latency_us,old_unknown_us=unknown,
                             candidate_unknown_us=unknown,predict_ms=runtime,**pred))
                history=max(history,pred['active']);last=shape
    conf=FILES+[f'configs/chips/{s}.json' for s in CHIPS]
    write(path,dict(utc=now(),inputs={f:sha(ROOT/f) for f in conf},contract='BF16 deterministic four-stream aclnnMhcPost with hRes; ten entirely new shapes per SKU, the same process and aclrtStream for all ordinals of each group. Exact repeat is an equality check on the observable preceding call, never on a particular shape ID. First, expand, hot, immediate repeats and delayed revisits are scored separately. Never pass the predecessor to the shape-only baseline. 970DT extrapolation only.', measured=CHIPS[:2],groups=list(SEQS),archive_shapes=len(known_shapes()),parameters=params,rows=regs))
if __name__=='__main__':main()
