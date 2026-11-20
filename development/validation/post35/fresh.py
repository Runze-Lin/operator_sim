"""New single-device-at-a-time case on the retired post25/26 batch lane."""
from datetime import datetime,timezone
from pathlib import Path
import fcntl,os,re,shutil,sys,time
from registration import ROOT,OUT,HERE,D,P,F,read,save,rel,sha,schedule

sys.path.insert(0,str(ROOT/'development/validation/post33'))
import importlib.util
old_spec=importlib.util.spec_from_file_location("post33_dedup",ROOT/"development/validation/post33/register.py")
old=importlib.util.module_from_spec(old_spec);old_spec.loader.exec_module(old)
from development.validation.post24.collect import BIN,LIB,capture,qualify
from operator_model import estimate
REG=OUT/'fresh-freeze.json';LOCK=ROOT/'runs/post25/queue.lock';RAW=ROOT/'runs/post35'
SEQ=[[69,4,176],[69,4,176],[2,4,368],[2,4,368]]
def utc():return datetime.now(timezone.utc).isoformat()
def freeze():
    if REG.exists() or RAW.exists():raise FileExistsError(REG)
    seen=old.dedup(SEQ)
    assert read(ROOT/'results/v0.25/post/status.json')['finished'] and read(ROOT/'results/v0.26/post/status.json')['state']=='complete'
    rows={}
    for chip in (D,P,F):
        arr=[]
        for i,shape in enumerate(SEQ):
            pred=schedule(shape,chip);pub=estimate('mhc_post',shape,chip=chip)
            arr.append(dict(i=i,active=pub.active_aiv,shape=shape,role=('first','repeat','reuse_after_token','repeat')[i],
                            candidate_us=pred['us'],public_us=pub.latency_us,conditional_us=pred['states'],predicted_route=pred['route']))
        assert arr[0]['candidate_us']==arr[1]['candidate_us'] and arr[2]['candidate_us']==arr[3]['candidate_us']
        rows[chip]=arr
    files=[HERE/'schedule.py',HERE/'fresh.py',OUT/'freeze.json',LIB,BIN,ROOT/'development/validation/post24/collect.py',ROOT/'smoke/vector_v8.cpp',ROOT/'smoke/extension_indexer_mhc.cpp']
    save(REG,dict(utc=utc(),rows=rows,sequence=SEQ,sources={rel(p):sha(p) for p in files},old_model_freeze_sha256=sha(OUT/'freeze.json'),
        novelty=seen,budget=dict(wall_s=2400,mib=2048),slot=rel(LOCK),collection='Real ordered fresh process on each SKU. First and immediate repeat of 35 payload AIVs / two rows; then four-AIV channel switch and immediate repeat. All source-frozen before acquisition. Reuse actual post25/26 released lock, six least-busy physical host cores pinned once, nice 17, passive waits, release slot each SKU. Up to 4 true launches per SKU in a single batch; no change to original queues.'),True)
def mask():
    def counters():
        m={}
        for l in Path('/proc/stat').read_text().splitlines():
            a=l.split()
            if not re.fullmatch(r'cpu\d+',a[0]):continue
            k=int(a[0][3:]);x=list(map(int,a[1:9]));m[k]=sum(x),x[3]+x[4]
        return m
    initial=counters();time.sleep(.7);last=counters();seen=set();chosen=[]
    for cpu in sorted(os.sched_getaffinity(0),key=lambda n:((last[n][1]-initial[n][1])/max(1,last[n][0]-initial[n][0]),-n),reverse=True):
        path=Path(f'/sys/devices/system/cpu/cpu{cpu}/topology/thread_siblings_list')
        physical=path.read_text().strip() if path.is_file() else str(cpu)
        if physical in seen:continue
        seen.add(physical);chosen.append(cpu)
        if len(chosen)==6:break
    assert len(chosen)==6
    return chosen
def run():
    reg=read(REG)
    for f,d in reg['sources'].items():
        if sha(ROOT/f)!=d:raise ValueError(f)
    RAW.mkdir(exist_ok=True);os.nice(17)
    os.environ.update(OMP_NUM_THREADS='6',OMP_THREAD_LIMIT='6',OPENBLAS_NUM_THREADS='1',OMP_WAIT_POLICY='PASSIVE',GOMP_SPINCOUNT='0')
    cpus=mask();os.sched_setaffinity(0,cpus)
    def state(phase,**info):save(OUT/'fresh-state.json',dict(utc=utc(),pid=os.getpid(),phase=phase,cpus=cpus,slot=rel(LOCK),freeze_sha256=sha(REG),**info))
    with (RAW/'owner.lock').open('a') as own:
        fcntl.flock(own,fcntl.LOCK_EX|fcntl.LOCK_NB)
        for chip,tag in ((D,'dt'),(P,'pr')):
            dest=OUT/('fresh-'+tag+'.json');folder=RAW/tag
            if dest.exists():continue
            state('waiting for original retired standalone Post lane',chip=chip)
            jobs=reg['rows'][chip]
            with LOCK.open('r+') as lane:
                fcntl.flock(lane,fcntl.LOCK_EX)
                if folder.exists():state('prior raw retained',chip=chip);return
                if shutil.disk_usage(ROOT).free<30*2**30:state('low disk budget',chip=chip);return
                state('collecting new four-call batch',chip=chip);cost=capture(chip,folder,jobs,reg['budget'])
            state('qualifying after release',chip=chip)
            try:
                if cost['exit_code'] or cost['stop_reason'] or cost['started_utc']<=reg['utc']:raise ValueError('cost, budget or freeze')
                rows=qualify(chip,folder,jobs)
                got=[]
                for j,x in zip(jobs,rows):
                    if not x.get('timing_qualified'):continue
                    s=x['stage'];expected='MhcPost_59130ccee6717ff2c0f15b485e731162'
                    if not re.fullmatch(expected+'_[56]',s['kernel']):raise ValueError('different installed implementation')
                    if [s['active_aic'],s['writes_bytes']] != [0,8*j['shape'][0]*j['shape'][2]]:raise ValueError('wrong traffic or compute type')
                    got.append(x)
                out=dict(rows=rows,qualified_windows=len(got),complete=len(got)==len(jobs))
            except (ValueError,AssertionError,KeyError,OSError,IndexError,StopIteration) as e:
                out=dict(rows=[],qualified_windows=0,complete=False,reason=repr(e))
            save(dest,dict(chip=chip,freeze_sha256=sha(REG),cost=cost,**out),True)
            time.sleep(2)
        state('finished')
if __name__=='__main__':{'freeze':freeze,'run':run}[sys.argv[1]]()
