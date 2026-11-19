"""Qualify an exported all-core device window independently of host process cleanup."""
from collections import Counter
import json
from math import isfinite
from pathlib import Path
import re
from router_validate import DT,OUT,RAW,ROOT,digest,write


def require(value, message):
    if not value:
        raise ValueError(message)


def parse(job,sku,folder):
    log=(folder/'run.log').read_text()
    expected_tag='hbm3_8000M.ini' if sku==DT else 'lpddr5_6400M_bailu.ini'
    require(expected_tag in log and re.findall(r'^runtime_soc=(\S+)',log,re.M)==[sku],'SKU or memory trace')
    t,e,k=job['shape']
    contract=f'case=moe_gating_top_k api=aclnnMoeGatingTopK T={t} E={e} K={k} dtype=FP32 norm_type=1 renorm=0 bias=true'
    require(re.findall(r'^case=.+$',log,re.M)==[contract], 'input contract')
    require(digest(ROOT/job['runner'])==job['runner_sha256'],'runner')
    correct=re.findall(r'^correctness .+$',log,re.M)
    require(correct==[f'correctness ids count={t*k} bad=0',f'correctness weights count={t*k} bad=0'],'golden')
    require(not re.search(r'SMOKE_FAIL|\[ERROR\].*(?:execute_set_flag|execute_wait_flag|deadlock|invalid instruction)',log),'sync failure')
    names=re.findall(r'Start profiling on kernel: (\S+)',log)
    require(len(names)==1 and names[0].startswith('MoeGatingTopK_'),'one kernel')
    traces=list(folder.glob(f'profile/OPPROF_*/device0/{names[0]}/0/simulator/trace.json'))
    if not traces:
        traces=list(folder.glob('profile/OPPROF_*/simulator/trace.json'))
    require(len(traces)==1,'all-core trace unavailable')
    trace=traces[0]
    # Parse the whole JSON document; truncated per-core shards do not qualify.
    data=json.loads(trace.read_bytes())
    windows={}; queues=Counter()
    for row in data['traceEvents']:
        if row['ph']!='X': continue
        core=row['pid']
        require(re.fullmatch(r'core\d+\.veccore\d+',core),'core identity')
        a=row['ts']; b=a+row['dur']
        require(isfinite(a) and isfinite(b) and a<=b,'malformed instruction')
        w=windows.setdefault(core,[a,b,row['name'],row['name']])
        if a < w[0]: w[0],w[2]=a,row['name']
        if b >= w[1]: w[1],w[3]=b,row['name']
        queues[row['tid']]+=1
    listed=re.findall(r'^(core\d+\.veccore\d+)\s+([.\d]+)\s+[.\d]+\s*$',log,re.M)
    require(len(listed)==len(windows)>0 and set(c for c,_ in listed)==set(windows),'all vendor table rows')
    for core,duration in listed:
        a,b,start,end=windows[core]
        require((start,end)==('MOV_XD_SPR','END_LABEL'),'missing first or final instruction')
        require(abs(float(duration)-(b-a))<=0.011,'per-core duration mismatch')
    origin=min(v[0] for v in windows.values()); end=max(v[1] for v in windows.values())
    binary=trace.parents[1]/'dump/aicore_binary.o'
    kernel=dict(kernel=names[0],ca_us=end-origin,active_aic=0,active_aiv=len(windows),
                windows={c:[w[0]-origin,w[1]-origin] for c,w in windows.items()},
                trace=str(trace.relative_to(ROOT)),trace_sha256=digest(trace),binary_sha256=digest(binary),
                queue_events=dict(queues),total_events=sum(queues.values()),terminal_instruction='END_LABEL')
    return dict(shape=job['shape'],sku=sku,correctness=correct,kernel=kernel,predictions=job['predictions'],
                qualified=True,role='same frozen launch, complete all-core trace recovered after host export/cleanup cap',
                physical_launch=False)


def main():
    hold=json.loads((OUT/'router-holdout.json').read_text())
    jobs={tuple(j['shape']):j for j in json.loads((OUT/'router-holdout-freeze.json').read_text())['jobs']}
    controls=[];recovered=[];failed=[]
    for r in hold:
        job=jobs[tuple(r['shape'])]
        try:
            row=parse(job,r['sku'],RAW/(job['id']+'_'+r['sku']))
        except (ValueError,KeyError,OSError) as err:
            if not r['qualified']: failed.append(dict(sku=r['sku'],shape=r['shape'],reason=str(err)))
            continue
        if r['qualified']:
            ref=r['kernel'];new=row['kernel']
            require(ref['trace_sha256']==new['trace_sha256'] and abs(ref['ca_us']-new['ca_us'])<1e-9,'control differs')
            controls.append(dict(shape=r['shape'],sku=r['sku'],ca_us=ref['ca_us'],sha256=new['trace_sha256']))
        else:
            recovered.append(row)
    for row in recovered:
        require(any(r['sku']==row['sku'] for r in controls),'no same-SKU positive control')
    write(OUT/'router-window-recovery.json',dict(controls=controls,recovered=recovered,ineligible=failed))

if __name__=='__main__':
    main()
