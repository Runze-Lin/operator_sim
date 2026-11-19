"""Qualify pre-registered built-in Post streams, including the physical END and output volume."""
from collections import Counter,defaultdict
import fcntl,json,os,re,signal,subprocess,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[3];sys.path.insert(0,str(ROOT))
from development.validation.post24.freeze import OUT,sha
from development.validation.sku19.campaign import read,write,now
from development.validation.sku19.collect import nbytes
from development.validation.rms_ordered_verify.verify import checked, text_bytes, events, tables, digest
from development.validation.post_candidate.extract import extract
RAW=ROOT/'runs/post24';REG=OUT/'freeze.json';PLAN=OUT/'collection.json'
LIB=ROOT/'toolchains/cann/9.2.1/cann-9.2.1/opp/built-in/op_impl/ai_core/tbe/kernel/ascend950/ops_transformer/mhc_post_apt/MhcPost_59130ccee6717ff2c0f15b485e731162.o'
BIN=ROOT/'build/vector-v8/MHC'
def register():
    if PLAN.exists() or RAW.exists():raise FileExistsError(PLAN)
    if not LIB.is_file(): raise FileNotFoundError(LIB)
    files=[REG,LIB,BIN,Path(__file__).resolve(),ROOT/'scripts/cann-run.sh',ROOT/'development/validation/rms_ordered_verify/verify.py',ROOT/'development/validation/post_candidate/extract.py']
    write(PLAN,dict(utc=now(),files={str(f.relative_to(ROOT)):sha(f) for f in files},wall_s=3600,mib=8192,order=['channel','wide','boundary','rows'],
        protocol='Serial, one new host per group and SKU; one ordered actual MHC batch stream; dump=false. Require runtime target, actual model, launch/ordinal/complete host golden, installed ELF .text, official per-core durations, complete aggregate and unique physical END plus END_LABEL on each active AIV, exact bytes written; same-kernel core disagreements with predicted footprint are model findings, not timing failures. No raw receipt replacement or 970 measurement.'))
def inputs(jobs):return ''.join('mhc_post '+str(j['i'])+' '+' '.join(map(str,j['shape']))+'\n' for j in jobs)
def capture(chip,d,jobs,p):
    if d.exists():raise FileExistsError(d)
    d.mkdir();batch=d/'batch.txt';batch.write_text(inputs(jobs))
    args=['/usr/bin/time','-f','%U %S','-o',str(d/'cpu.txt'),'bash',str(ROOT/'scripts/cann-run.sh'),'msopprof','simulator','--soc-version='+chip,'--dump=false','--launch-count='+str(len(jobs)),'--timeout='+str(p['wall_s']//60),'--output='+str(d/'profile'),str(BIN),str(batch)]
    started=now();t=time.monotonic();why=None;peak=0
    with (d/'run.log').open('x') as out:
        child=subprocess.Popen(args,cwd=d,stdout=out,stderr=subprocess.STDOUT,start_new_session=True,env={**os.environ,'OMP_WAIT_POLICY':'PASSIVE','GOMP_SPINCOUNT':'0'})
        try:
            while child.poll() is None:
                peak=max(peak,nbytes(d))
                if peak>p['mib']*2**20 or time.monotonic()-t>p['wall_s']:
                    why='raw cap' if peak>p['mib']*2**20 else 'wall cap';os.killpg(child.pid,signal.SIGTERM)
                    try:child.wait(10)
                    except subprocess.TimeoutExpired:os.killpg(child.pid,signal.SIGKILL)
                    break
                time.sleep(2)
        finally:
            if child.poll() is None and not why:os.killpg(child.pid,signal.SIGTERM)
            child.wait()
    cpu=(d/'cpu.txt').read_text().splitlines();times=cpu[-1].split() if cpu else []
    cost=dict(started_utc=started,finished_utc=now(),seconds=time.monotonic()-t,exit_code=child.returncode,stop_reason=why,raw_bytes=nbytes(d),peak_bytes=max(nbytes(d),peak),cpu_seconds=sum(map(float,times)) if len(times)==2 else None)
    write(d/'cost.json',cost);return cost
def inspect_window(path,table,name,shape):
    tr=path/'simulator/trace.json';binary=path/'dump/aicore_binary.o'
    checked(text_bytes(binary)==text_bytes(LIB),'different installed code')
    bounds={};ends=Counter();label={};writes=Counter();qs=Counter()
    for ev in events(tr):
        k=ev['pid'];nm=ev['name'];a=round(ev['ts']*1650);b=round((ev['ts']+ev['dur'])*1650)
        checked(re.fullmatch(r'core\d+\.veccore\d+',k) and b>=a,'wrong lane or timestamp')
        if k not in bounds:bounds[k]=[a,b]
        else:bounds[k]=[min(a,bounds[k][0]),max(b,bounds[k][1])]
        ends[k]+=nm=='END';qs[ev['tid']]+=1
        if nm=='END_LABEL':
            checked(k not in label,'duplicate terminal label');label[k]=b
        if nm.startswith('MOV_SRC_TO_DST'):
            detail=json.loads(ev['args']['detail'])
            if detail.get('queue_type')=='MTE3' and not any(detail.get(key) for key in ('is_wait','is_sync','is_duration_excluded')):
                writes[k]+=detail['processed_bytes']
    checked(set(table)==set(bounds)==set(label)==set(writes),'official/trace/write/terminal lane rosters disagree')
    for c,(a,b) in bounds.items():
        checked(ends[c]==1 and label[c]==b and abs((b-a)/1650-float(table[c][0]))<.015,'incomplete window or official duration '+c)
    checked(sum(writes.values())==2*shape[0]*shape[1]*shape[2],'shape disagrees with total output bytes')
    a=min(w[0] for w in bounds.values());b=max(w[1] for w in bounds.values());us=(b-a)/1650
    return dict(kernel=name,ca_us=us,first_tick=a,last_tick=b,active_aic=0,active_aiv=len(bounds),writes_bytes=sum(writes.values()),per_core_writes=dict(writes),windows={k:[(x-a)/1650,(y-a)/1650] for k,(x,y) in bounds.items()}, trace=str(tr.relative_to(ROOT)),trace_sha256=digest(tr),binary_sha256=sha(binary),queue_events=dict(qs))
def qualify(chip,d,jobs):
    log=(d/'run.log').read_text();checked(re.findall(r'^runtime_soc=(\S+)',log,re.M)==[chip] and ('lpddr5_6400M_bailu.ini' if 'PR' in chip else 'hbm3_8000M.ini') in log,'wrong SKU/config')
    checked('SMOKE_PASS' in log and 'Profiling data parse finished.' in log and 'All task success.' in log and not re.search(r'SMOKE_FAIL|\[ERROR\].*(?:deadlock|invalid instruction|execute_set_flag|execute_wait_flag)',log,re.I),'host/CA/parser')
    scopes=re.findall(r'^BATCH_BEGIN (\d+)\s*$\n(.*?)^BATCH_END \1\s*$',log,re.M|re.S)
    names=re.findall(r'Start profiling on kernel: (\S+)',log)
    ts=tables(log);checked(len(jobs)==len(scopes)==len(ts)==len(names),'host/trace/roster/ordinal mismatch')
    checked((d/'batch.txt').read_text()==inputs(jobs),'wrong input order')
    result=[];byname=Counter();profile,=d.glob('profile/OPPROF_*')
    for j,(index,scope),tab,name in zip(jobs,scopes,ts,names):
        t,n,w=j['shape'];gold=re.findall(r'^correctness name=output count=(\d+) bad=(\d+)\b',scope,re.M)
        checked(gold==[(str(t*n*w),'0')] and str(j['i'])==index and f'case=mhc_post api=aclnnMhcPost T={t} N={n} D={w} dtype=BF16' in scope,'independent complete tensor check')
        path=profile/name/str(byname[name]);byname[name]+=1
        row=dict(**j,all_output_elements=t*n*w,timing_qualified=False)
        try:
            checked(name.startswith('MhcPost_59130ccee6717ff2c0f15b485e731162_'),'different kernel')
            s=inspect_window(path,tab,name,j['shape']);row.update(stage=s,timing_qualified=True,active_matches=s['active_aiv']==j['active'])
            try:row['ca_milestones']=extract(dict(stage=s,us=s['ca_us']))
            except (ValueError,AssertionError,KeyError,IndexError) as exc:row['milestone_error']=repr(exc)
        except (OSError,ValueError,AssertionError,KeyError,IndexError) as exc:row['reason']=repr(exc)
        result.append(row)
    good=[x['stage'] for x in result if x['timing_qualified']];checked(all(b['first_tick']>a['last_tick'] for a,b in zip(good,good[1:])),'out of sequence absolute timestamps')
    return result
def run():
    src=read(REG);p=read(PLAN);RAW.mkdir(exist_ok=True);(OUT/'receipts').mkdir(exist_ok=True)
    for f,h in {**src['inputs'],**p['files']}.items():checked(sha(ROOT/f)==h,'frozen input changed '+f)
    with (RAW/'lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        for group in p['order']:
            for chip in src['measured']:
                ident=group+'-'+chip;dest=OUT/'receipts'/(ident+'.json');d=RAW/ident
                if dest.exists():continue
                jobs=[r for r in src['rows'] if r['group']==group and r['chip']==chip];write(OUT/'status.json',dict(utc=now(),state='running',id=ident,expected=len(jobs)))
                cost=capture(chip,d,jobs,p);obj=dict(group=group,chip=chip,cost=cost,qualified=False,rows=[],freeze_sha256=sha(REG))
                try:
                    checked(cost['exit_code']==0 and not cost['stop_reason'] and cost['started_utc']>p['utc'],'process or chronology')
                    obj['rows']=qualify(chip,d,jobs);obj['qualified']=all(r['timing_qualified'] for r in obj['rows'])
                except (ValueError,OSError,AssertionError,KeyError,IndexError) as exc:obj['reason']=repr(exc)
                write(dest,obj);print(ident,obj['qualified'],len(obj['rows']),obj.get('reason',''),flush=True)
        write(OUT/'status.json',dict(utc=now(),state='complete',outcomes={f.stem:read(f)['qualified'] for f in sorted((OUT/'receipts').glob('*.json'))}))
if __name__=='__main__':{'register':register,'run':run}[sys.argv[1]]()
