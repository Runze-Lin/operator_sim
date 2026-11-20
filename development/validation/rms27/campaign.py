"""Pre-register unseen first-or-expanding vs immediate-repeat RMSNorm pairs, then collect serially."""
import fcntl,json,os,sys
from hashlib import sha256
from pathlib import Path
from time import perf_counter_ns
ROOT=Path(__file__).resolve().parents[3];sys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT/'development/validation/sku17'));sys.path.insert(0,str(ROOT/'development/validation/rms_ordered_verify'))
from development.validation.rms27.model import fit,forecast
from development.validation.rms27.landmarks import OLD,OUT
from development.validation.rms_ordered_verify.prefix import host
from development.validation.rms_ordered_verify import verify
from development.validation.rms_dispatch_audit.ordered_ca import capture,RUNNER
from development.validation.sku19.campaign import read,write,now
from operator_model import load_chip,estimate
CHIPS=('Ascend950DT_9582','Ascend950PR_957b','Ascend970DT')
GROUPS={'partition':[(1,736),(4,688),(19,2416),(35,1744)],'siblings':[(2,2816),(3,3520),(5,5680)],'sector':[(31,1552),(129,1376)],'multi':[(521,4032)]}
RAW=ROOT/'runs/rms27';REG=OUT/'freeze.json';PARAMS=OUT/'prices.json'

def register():
    if REG.exists() or RAW.exists():raise FileExistsError(REG)
    held={tuple(s)for group in GROUPS.values() for s in group};assert len(held)==10
    for path in (ROOT/'results/v0.22/evaluation.json',ROOT/'results/v0.24/rms-prefix/freeze.json',OLD/'freeze.json',OLD/'resource-diagnostic.json'):
        used={tuple(x['shape'])for x in read(path)['rows'] if x.get('op','rmsnorm')=='rmsnorm'}
        assert not held&used,(path,held&used)
    prices=fit(read(OUT/'landmarks.json')['rows']);write(PARAMS,prices)
    rows=[]
    for sku in CHIPS:
        chip=load_chip(sku)
        for group,shapes in GROUPS.items():
            sequence=[('prefix',(1,736))] if group=='siblings' else []
            for s in shapes:sequence.extend([('entry',s),('repeat',s)])
            H=0
            for i,(role,shape) in enumerate(sequence):
                t=perf_counter_ns();q=forecast(shape,chip,prices);dt=(perf_counter_ns()-t)/1e6
                a=q['active_aiv'];known=estimate('rmsnorm',shape,{'resident_aiv':H},chip=chip).latency_us
                if role=='entry':assert H<a,(sku,shape,H,a)
                if role=='repeat':assert H>=a
                first=q['conditional']['first_us'];hot=q['conditional']['fully_covered_us']
                rows.append(dict(chip=sku,group=group,i=i,shape=shape,prior_aiv=H,role=role,known_us=known,
                                 cold_hot_us=2*first*hot/(first+hot),runtime_ms=dt,**q))
                H=max(H,a)
    sources=[Path(__file__).resolve(),Path(__file__).with_name('model.py'),Path(__file__).with_name('landmarks.py'),RUNNER,verify.VENDOR,PARAMS,
        OUT/'landmarks.json',ROOT/'scripts/cann-run.sh',ROOT/'development/validation/rms_dispatch_audit/ordered_ca.py',ROOT/'development/validation/rms_ordered_verify/prefix.py',ROOT/'development/validation/rms_ordered_verify/verify.py',ROOT/'operator_model/rms_dispatch.py',ROOT/'operator_model/rms_lane_schedule.py',ROOT/'operator_model/rms_resource.py',ROOT/'operator_model/rms_target_cost.py',ROOT/'operator_model/rms_v12.py',ROOT/'results/v0.22/rms-lane-parameters.json',ROOT/'results/v0.12/rms/profile.json']+[ROOT/'configs/chips'/(c+'.json') for c in CHIPS]
    write(REG,dict(utc=now(),rows=rows,measured=CHIPS[:2],extrapolated=[CHIPS[2]],groups=list(GROUPS),slots=1,wall_per_batch_s=3600,raw_mib_per_batch=12288,
        mechanism='Single-UB lower endpoint from once-resident payload issues, separate descriptor response and sector retirement. Upper endpoint is the longer of first dispatch or opening the second partition. Report harmonic equal-relative-error point. Larger UB loops retain the existing state pipeline.',
        primary='Each 950 needs ten independent matched shapes, full numerical check of y and rstd, active lane/ELF/task checks, all-physical END plus synthetic END_LABEL. WAPE <=25% / max<=50% in both entry and immediate-repeat protocols separately. No replacing failed shapes or hiding larger groups. Frozen comparisons old, cold/hot harmonic, and known H.',
        files={str(p.relative_to(ROOT)):verify.digest(p) for p in sources}))

def run():
    plan=read(REG)
    for path,h in plan['files'].items():assert verify.digest(ROOT/path)==h,path
    RAW.mkdir(exist_ok=True);receipts=OUT/'receipts';receipts.mkdir(exist_ok=True)
    os.environ.update(OMP_WAIT_POLICY='PASSIVE',GOMP_SPINCOUNT='0')
    with (RAW/'.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        for group in plan['groups']:
            for chip in plan['measured']:
                ident=group+'-'+chip;folder=RAW/ident;dest=receipts/(ident+'.json')
                if dest.exists():continue
                jobs=[q for q in plan['rows'] if q['chip']==chip and q['group']==group]
                write(OUT/'status.json',dict(utc=now(),state='running',process=ident,slots=1))
                cost=capture(chip,folder,jobs,plan)
                rec=dict(chip=chip,group=group,freeze_sha256=verify.digest(REG),cost=cost,rows=[])
                try:
                    verify.checked(cost['started_utc']>plan['utc'] and cost['exit_code']==0 and not cost['stop_reason'],'process exit or budget')
                    verify.checked((folder/'inputs.txt').read_text()==''.join(f"rmsnorm {r['i']} {r['shape'][0]} {r['shape'][1]} 0\n" for r in jobs),'observed input order')
                    tasks,tables=host(folder,chip,jobs);profile,=folder.glob('profile/OPPROF_*');pub=profile/verify.KERNEL
                    if not pub.is_dir():pub=profile/'device0'/verify.KERNEL
                    elf=verify.text_bytes(verify.VENDOR);rec['elf_text_sha256']=sha256(elf).hexdigest()
                    verify.checked('Profiling data parse finished.' in (folder/'run.log').read_text() and sorted(int(f.name)for f in pub.iterdir() if f.is_dir())==list(range(len(jobs))),'published ordinals')
                    prev=0
                    for j,task,table in zip(jobs,tasks,tables):
                        try:
                            row=verify.launch(pub/str(j['i']),chip,j,task,table,elf);s=row['stage']
                            verify.checked(row['active_count_matches'] and s['first_tick']>prev,'core roster or ordering');prev=s['last_tick']
                        except (ValueError,OSError,KeyError,IndexError,TypeError) as e:
                            row=dict(**j,timing_qualified=False,reason=repr(e),correctness=task['golden'])
                        rec['rows'].append(row)
                except (ValueError,OSError,KeyError,IndexError,TypeError) as e:rec['reason']=repr(e)
                write(dest,rec);print(ident,sum(r['timing_qualified'] for r in rec['rows']),len(jobs),rec.get('reason',''),flush=True)
                from development.validation.rms27.report import render
                render()
        write(OUT/'status.json',dict(utc=now(),state='complete',slots=1))
        from development.validation.rms27.report import render
        render()

if __name__=='__main__':{'register':register,'run':run}[sys.argv[1]]()
