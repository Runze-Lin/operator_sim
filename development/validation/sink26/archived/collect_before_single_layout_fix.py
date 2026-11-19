"""Four independently restartable ordered streams per SKU, one active acquisition."""
import fcntl
import json
import os
from pathlib import Path
import re
import shutil
import signal
import subprocess
import time
from collections import Counter
from evidence import ROOT,OUT,DT,PR,read,save,sha
from registration import RAW,REG,BINARY,VENDOR
from development.validation.rms_ordered_verify.verify import checked,events,tables,text_bytes
from development.validation.sku17.campaign_collect import nbytes
from development.validation.sku19.campaign import now

def atomic(dst,x):
    dst.parent.mkdir(exist_ok=True,parents=True);tmp=dst.with_suffix('.tmp');save(tmp,x);tmp.replace(dst)
def inputs(jobs):return ''.join(f"mhc_sinkhorn {j['index']} {j['shape'][0]} 4 {j['iters']}\n" for j in jobs)
def capture(jobs,d,limits):
    d.mkdir();fn=d/'batch.txt';fn.write_text(inputs(jobs));chip=jobs[0]['chip']
    cmd=['/usr/bin/time','-f','%U %S','-o',str(d/'cpu.txt'),'bash',str(ROOT/'scripts/cann-run.sh'),'msopprof','simulator','--soc-version='+chip,
         '--dump=false','--launch-count='+str(len(jobs)),'--timeout='+str((limits['wall_s']+59)//60),'--output='+str(d/'profile'),str(BINARY),str(fn)]
    t=time.monotonic();start=now();reason=None;peak=0
    with (d/'run.log').open('x') as log:
        proc=subprocess.Popen(cmd,cwd=d,stdout=log,stderr=subprocess.STDOUT,start_new_session=True,env={**os.environ,'GOMP_SPINCOUNT':'0','OMP_WAIT_POLICY':'PASSIVE'})
        try:
            while proc.poll() is None:
                peak=max(peak,nbytes(d))
                if peak>limits['mib']*2**20 or time.monotonic()-t>limits['wall_s']:
                    reason='raw or wall budget'
                    os.killpg(proc.pid,signal.SIGTERM)
                    try:proc.wait(10)
                    except subprocess.TimeoutExpired:os.killpg(proc.pid,signal.SIGKILL)
                    break
                time.sleep(2)
        finally:
            if proc.poll() is None and not reason:os.killpg(proc.pid,signal.SIGTERM)
            proc.wait()
    cost=dict(started_utc=start,ended_utc=now(),seconds=time.monotonic()-t,exit_code=proc.returncode,stop_reason=reason,raw_bytes=max(peak,nbytes(d)))
    atomic(d/'cost.json',cost);return cost

def measured(path,name,tab,expected_bytes):
    binary=path/'dump/aicore_binary.o';checked(text_bytes(binary)==text_bytes(VENDOR),'released code mismatch')
    p=path/'simulator/trace.json';bounds={};ends=Counter();labels={};size=Counter();vf=Counter()
    for e in events(p):
        c,n=e['pid'],e['name'];b=round(e['ts']*1650);z=round((e['ts']+e['dur'])*1650)
        win=bounds.setdefault(c,[b,z]);win[0]=min(win[0],b);win[1]=max(win[1],z);ends[c]+=n=='END';vf[c]+=n=='VF'
        if n=='END_LABEL':
            checked(c not in labels,'duplicate terminal label');labels[c]=z
        if n.startswith('MOV_SRC_TO_DST'):
            d=json.loads(e['args']['detail'])
            if d['queue_type']=='MTE3' and not (d['is_wait'] or d['is_sync'] or d['is_duration_excluded']):size[c]+=d['processed_bytes']
    checked(set(bounds)==set(tab)==set(size) and set(labels)<=set(bounds),'kernel/terminal/output roster')
    for c,(a,b) in bounds.items():
        checked(re.fullmatch(r'core\d+\.veccore\d+',c) and ends[c]==1 and labels.get(c,b)==b and abs((b-a)/1650-float(tab[c][0]))<.016,'incomplete core '+c)
    checked(sum(size.values())==expected_bytes,'all-output bytes')
    lo=min(x[0] for x in bounds.values());hi=max(x[1] for x in bounds.values())
    return dict(kernel=name,ca_us=(hi-lo)/1650,first_tick=lo,last_tick=hi,active_aiv=len(bounds),active_aic=0,written_bytes=sum(size.values()),
                windows={c:[(a-lo)/1650,(b-lo)/1650] for c,(a,b) in bounds.items()},vf_calls_per_core=dict(vf),
                trace=str(p.relative_to(ROOT)),trace_sha256=sha(p),binary_sha256=sha(binary))

def qualify(jobs,d,cost):
    checked(cost['exit_code']==0 and not cost['stop_reason'],'process/budget')
    chip=jobs[0]['chip'];log=(d/'run.log').read_text(errors='replace');scopes=re.findall(r'^BATCH_BEGIN (\d+)\s*$\n(.*?)^BATCH_END \1\s*$',log,re.M|re.S)
    names=re.findall(r'Start profiling on kernel: (\S+)',log)
    checked(re.findall(r'^runtime_soc=(\S+)',log,re.M)==[chip] and ('lpddr5_6400M_bailu.ini' if chip==PR else 'hbm3_8000M.ini') in log,'target silicon config')
    checked(all(x in log for x in ('SMOKE_PASS','Profiling data parse finished.','All task success.')) and not re.search(r'SMOKE_FAIL|\[ERROR\].*(?:deadlock|invalid instruction|execute_set_flag|execute_wait_flag)',log,re.I),'runtime or parser')
    tabs=tables(log);checked((d/'batch.txt').read_text()==inputs(jobs) and len(names)==len(scopes)==len(tabs)==len(jobs),'host/kernel/ordinal')
    profile,=d.glob('profile/OPPROF_*');seen=Counter();result=[];prev=None
    for j,(idx,lines),name,tab in zip(jobs,scopes,names,tabs):
        t,n=j['shape'];it=j['iters'];vol=t*n*n;path=profile/name/str(seen[name]);seen[name]+=1
        scope=dict(shape=j['shape'],iters=it,index=j['index'],h=j['prior_h'],stage=None)
        try:
            checked(int(idx)==j['index'] and f'case=mhc_sinkhorn api=aclnnMhcSinkhorn T={t} N={n} iters={it} dtype=FP32' in lines,'runtime shape/iterations')
            golden=re.findall(r'^correctness name=output count=(\d+) bad=(\d+)\b',lines,re.M)
            checked(golden==[(str(vol),'0')] and name.startswith('MhcSinkhorn_8997d9c5c22e82688b98bc25d8daa9cd_'),'golden or kernel identity')
            s=measured(path,name,tab,4*vol)
            checked(prev is None or s['first_tick']>prev,'out-of-order launches');prev=s['last_tick']
            scope.update(stage=s,qualified=True,all_output_elements=vol)
        except (OSError,ValueError,KeyError,TypeError,IndexError,AssertionError) as err:scope.update(qualified=False,reason=repr(err))
        result.append(scope)
    return result

def status(state,**fields):atomic(OUT/'state.json',dict(utc=now(),state=state,registration_sha256=sha(REG),**fields))
def run():
    reg=read(REG)
    for f,digest in reg['source_hashes'].items():
        checked(sha(ROOT/f)==digest,'frozen source '+f)
    RAW.mkdir(exist_ok=True);(OUT/'receipts').mkdir(exist_ok=True)
    with (RAW/'lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        for group in reg['order']:
            for chip in (DT,PR):
                ident=group+'-'+chip;d=RAW/ident
                dest=OUT/'receipts'/(ident+'.json')
                if dest.exists():continue
                jobs=[j for j in reg['jobs'] if j['group']==group and j['chip']==chip];lim=reg['budget'][group]
                if not (d/'cost.json').exists():
                    if d.exists():status('interrupted attempt preserved',id=ident);return
                    while True:
                        pro=read(ROOT/'results/v0.23/pro-prefill/state.json')['state'];load=os.getloadavg()[0]
                        ram=int(re.search(r'^MemAvailable:\s+(\d+) kB',Path('/proc/meminfo').read_text(),re.M)[1])*1024/2**30
                        if pro!='queued' and load < .62*len(os.sched_getaffinity(0)) and shutil.disk_usage(ROOT).free>80*2**30 and ram>48:break
                        status('waiting for older Pro or host capacity',id=ident,pro=pro,load=load);time.sleep(45)
                    status('acquiring',id=ident,group_size=len(jobs));capture(jobs,d,lim)
                cost=read(d/'cost.json');status('verifying',id=ident);out=dict(group=group,chip=chip,cost=cost,rows=[],register_sha256=sha(REG))
                try:
                    checked(cost['started_utc']>reg['utc'],'predictions after CA');out['rows']=qualify(jobs,d,cost)
                except (OSError,ValueError,KeyError,TypeError,IndexError,AssertionError) as err:out['reason']=repr(err)
                atomic(dest,out);print(ident,sum(r['qualified'] for r in out['rows']),len(jobs),out.get('reason',''),flush=True)
                from score import score
                score()
        status('finished')
if __name__=='__main__':run()
