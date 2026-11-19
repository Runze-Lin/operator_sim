"""Recover completed all-core inner exports without relaunching CA or modifying source traces."""
from collections import Counter
from pathlib import Path
import math
import re
import sys

ROOT=Path(__file__).resolve().parents[3]
sys.path.insert(0,str(ROOT))
from development.validation.sku19.campaign import OUT, RAW, DT, PR, read, write, digest, now
from development.validation.sku19.collect import make_pair, summarize
from campaign_collect import events


def check(ok,reason):
    if not ok:raise ValueError(reason)


def parse(j,chip,d):
    text=(d/'run.log').read_text(errors='replace')
    check(re.findall(r'^runtime_soc=(\S+)',text,re.M)==[chip] and
          ('hbm3_8000M.ini' if chip==DT else 'lpddr5_6400M_bailu.ini') in text,'SKU/memory')
    check(digest(ROOT/j['runner'])==j['runner_sha256'],'host changed')
    rr,cc=j['shape']
    tag=(f'case=rmsnorm api=aclnnRmsNorm R={rr} C={cc} dtype=BF16 ' if j['op']=='rmsnorm'
         else f'case=dynamic_mx_quant rows={rr} cols={cc} input=BF16 output=E4M3FN ')
    inputs=re.findall(r'^case=.+$',text,re.M)
    check(len(inputs)==1 and inputs[0].startswith(tag),'shape/API contract')
    checks=re.findall(r'^correctness .+$',text,re.M)
    check('SMOKE_PASS' in text and len(checks)==j['nchecks'] and all(re.search(r'\bbad=0\b',s) for s in checks),'golden incomplete')
    check(not re.search(r'SMOKE_FAIL|\[ERROR\].*(?:execute_set_flag|execute_wait_flag|deadlock|invalid instruction)',text),'sync/numeric failure')
    names=re.findall(r'Start profiling on kernel: (\S+)',text)
    check(len(names)==1 and names[0].startswith('RmsNorm' if j['op']=='rmsnorm' else 'DynamicMxQuant'),'kernel dispatch')
    paths=list(d.glob('profile/OPPROF_*/simulator/trace.json')) or list(d.glob(f'profile/OPPROF_*/device0/{names[0]}/0/simulator/trace.json'))
    check(len(paths)==1,'complete inner export missing')
    trace=paths[0]
    with trace.open('rb') as f:
        f.seek(-2,2);check(f.read()==b']}','truncated trace')
    win={};queues=Counter()
    for e in events(trace):
        if e['ph']!='X':continue
        a=e['ts'];b=a+e['dur'];core=e['pid']
        check(math.isfinite(a) and math.isfinite(b) and b>=a and re.fullmatch(r'core\d+\.veccore\d+',core),'bad window')
        w=win.setdefault(core,[a,b,e['name'],e['name']])
        if a<w[0]:w[0],w[2]=a,e['name']
        if b>=w[1]:w[1],w[3]=b,e['name']
        queues[e['tid']]+=1
    listed=re.findall(r'^(core\d+\.veccore\d+)\s+([.\d]+)\s+[.\d]+\s*$',text,re.M)
    check(len(win)==len(listed)>0 and set(win)=={k for k,_ in listed},'vendor duration table')
    for c,dur in listed:
        a,b,first,last=win[c]
        check(first=='MOV_XD_SPR' and last=='END_LABEL' and abs(b-a-float(dur))<=.011,'incomplete core '+c)
    start=min(w[0] for w in win.values());ca=max(w[1] for w in win.values())-start
    tr=str(trace.relative_to(ROOT));sha=digest(trace)
    stage=dict(kernel=names[0],ca_us=ca,active_aic=0,active_aiv=len(win),windows={c:[w[0]-start,w[1]-start] for c,w in win.items()},
               trace=tr,trace_sha256=sha,binary_sha256=digest(trace.parents[1]/'dump/aicore_binary.o'),queue_events=dict(queues),mmad=0,total_events=sum(queues.values()))
    api='aclnnRmsNorm' if j['op']=='rmsnorm' else 'aclnnDynamicMxQuant'
    return dict(op=j['op'],shape=j['shape'],attrs={},chip=chip,ca_us=ca,active_aic=0,active_aiv=len(win),kernel=names[0],api=api,
                correctness=checks,stages=[stage],trace=[tr],trace_sha256=[sha],cost=read(d/'cost.json'),case_id=j['id'],
                backend='official all-core export; recovered before profiler cleanup',dtype='bf16',host_args=j['args'],
                role='same frozen contract',scope='one all-core window',runner_sha256=j['runner_sha256'],input_contract_log=inputs,
                run_log=str((d/'run.log').relative_to(ROOT)))


def recover():
    output=OUT/'window-recovery.json'
    data=read(output) if output.exists() else dict(controls={},recovered={},unresolved={})
    jobs=read(OUT/'freeze.json')['jobs']
    for j in jobs:
        for chip in (DT,PR):
            id_=j['id']+'_'+chip;d=RAW/id_;tag=j['op']+'_'+chip
            if not (d/'outcome.json').exists():continue
            if (d/'qualified.json').exists():
                if tag not in data['controls'] and j['role']=='holdout' and j['shape'][0]<32:
                    official=read(d/'qualified.json')
                    new=parse(j,chip,d)
                    check(official['trace_sha256']==new['trace_sha256'] and abs(official['ca_us']-new['ca_us'])<1e-9,'positive control differs')
                    data['controls'][tag]=dict(id=id_,ca_us=new['ca_us'],trace_sha256=new['trace_sha256'][0])
            elif id_ not in data['recovered'] and id_ not in data['unresolved'] and tag in data['controls']:
                try:
                    record=parse(j,chip,d)
                    data['recovered'][id_]=record;data['unresolved'].pop(id_,None)
                except (IndexError,ValueError,KeyError,OSError) as e:data['unresolved'][id_]=str(e)
    write(output,data)
    print('controls/recovered/unresolved',*[len(data[k]) for k in ('controls','recovered','unresolved')])


def finalize():
    data=read(OUT/'window-recovery.json');jobs=read(OUT/'freeze.json')['jobs']
    records=[];pairs=[];costs=read(OUT/'costs.json')
    for j in jobs:
        both=[]
        for chip in (DT,PR):
            k=j['id']+'_'+chip;d=RAW/k;retry=ROOT/'runs/sku19-retry'/k
            f=d/'qualified.json'
            r=read(f) if f.exists() else data['recovered'].get(k)
            if (retry/'outcome.json').exists():costs.append(read(retry/'outcome.json'))
            if r is None and (retry/'qualified.json').exists():r=read(retry/'qualified.json')
            if r:both.append(r);records.append(r)
        if len(both)==2:
            if j['outputs'][0]['old_dt_us'] is None:
                dt,pr=both;o={k:v for k,v in j['outputs'][0].items() if k!='indices'}
                row=dict(**o,id=j['id'],regime=j['regime'],dt=dt,pr=pr,dt_ca_us=dt['ca_us'],pr_ca_us=pr['ca_us'],
                         speedup_pred=o['dt_us']/o['pr_us'],speedup_ca=dt['ca_us']/pr['ca_us'],speedup_old=None,dtype='bf16')
            else:row=make_pair(j,*both)
            row['sample_role']=j['role'];pairs.append(row)
    write(OUT/'final-pairs.json',pairs);write(OUT/'final-records.json',records)
    write(OUT/'final-costs.json',costs)
    write(OUT/'completion.json',dict(**summarize(costs),qualified_windows=len(records),pairs=len(pairs),
        freeze_commit='e6eb2c7369904f1332356a67829c26bad57703fe',utc=now(),
        files={n:digest(OUT/n) for n in ('registration.json','freeze.json','v018-forecasts.json','v019-forecasts.json','final-pairs.json','final-costs.json','window-recovery.json','final-records.json')}))
    print('paired',len(pairs))

if __name__=='__main__':{'check':recover,'final':finalize}[sys.argv[1]]()
