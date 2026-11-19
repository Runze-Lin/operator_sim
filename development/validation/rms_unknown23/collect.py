"""Serial paired full-window collection for the exact pre-registered input streams."""
from hashlib import sha256
import fcntl,os,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[3];sys.path.insert(0,str(ROOT))
sys.path.insert(0,str(ROOT/'development/validation/sku17'))
sys.path.insert(0,str(ROOT/'development/validation/rms_ordered_verify'))
from development.validation.rms_ordered_verify.prefix import host
from development.validation.rms_ordered_verify import verify
from development.validation.rms_dispatch_audit.ordered_ca import capture
from development.validation.sku19.campaign import read,write,now
from development.validation.rms_unknown23.freeze import OUT
RAW=ROOT/'runs/rms-unknown23'
REG=OUT/'freeze.json'
PLAN=OUT/'collection.json'

def register():
    if PLAN.exists() or RAW.exists():raise FileExistsError(PLAN)
    src=read(REG);fs=[Path(__file__).resolve(),ROOT/'development/validation/rms_dispatch_audit/ordered_ca.py',ROOT/'development/validation/rms_ordered_verify/prefix.py',ROOT/'development/validation/rms_ordered_verify/verify.py',ROOT/'scripts/cann-run.sh',verify.VENDOR,REG]
    write(PLAN,dict(utc=now(),files={str(f.relative_to(ROOT)):verify.digest(f) for f in fs},process_order=['wide','sector','two-rows','multi-UB'],chips=src['measured_skus'],slots=1,wall_per_batch_s=3600,raw_mib_per_batch=12288,implementation='ordered_ca.capture with dump=false and no automatic remeasurement',conditions='Same NORM process+stream, confirmed input order, two all-element goldens per call, official install ELF, full aggregate/shard/timing-table identity, per-core END+END_LABEL and total write volumes. Qualify each separately, compare identical deterministic inputs in pairs.'))


def collect():
    src=read(REG);plan=read(PLAN)
    for path,h in {**src['files'],**plan['files']}.items():verify.checked(verify.digest(ROOT/path)==h,'frozen file changed '+path)
    RAW.mkdir(exist_ok=True);(OUT/'receipts').mkdir(exist_ok=True)
    os.environ.update(OMP_WAIT_POLICY='PASSIVE',GOMP_SPINCOUNT='0')
    with (RAW/'.lock').open('a') as f:
        fcntl.flock(f,fcntl.LOCK_EX|fcntl.LOCK_NB)
        for process in plan['process_order']:
            for chip in plan['chips']:
                folder=RAW/(process+'-'+chip);dest=OUT/'receipts'/(folder.name+'.json');jobs=[r for r in src['rows'] if r['chip']==chip and r['process']==process]
                if dest.exists():continue
                write(OUT/'status.json',dict(utc=now(),state='running',process=process,chip=chip,raw=str(folder.relative_to(ROOT))))
                cost=capture(chip,folder,jobs,plan)
                receipt=dict(chip=chip,process=process,registration_sha256=verify.digest(REG),cost=cost,qualified=False,rows=[])
                try:
                    verify.checked(not cost['stop_reason'] and cost['exit_code']==0 and cost['started_utc']>plan['utc'],'process exit, chronology or cap')
                    expected=''.join(f"rmsnorm {j['i']} {j['shape'][0]} {j['shape'][1]} 0\n" for j in jobs)
                    verify.checked((folder/'inputs.txt').read_text()==expected,'actual host input list differs')
                    tasks,tables=host(folder,chip,jobs);elf=verify.text_bytes(verify.VENDOR)
                    profile,=folder.glob('profile/OPPROF_*/device0/'+verify.KERNEL)
                    receipt.update(host_numeric_groups=len(tasks)*2,elf_text_sha256=sha256(elf).hexdigest(),log_sha256=verify.digest(folder/'run.log'),inputs_sha256=verify.digest(folder/'inputs.txt'))
                    for j,task,tab in zip(jobs,tasks,tables):
                        try:
                            rr=verify.launch(profile/str(j['i']),chip,j,task,tab,elf)
                            verify.checked(rr['active_count_matches'],'observed AIVs differ from frozen history')
                        except (OSError,ValueError,IndexError,TypeError,KeyError) as err:
                            rr=dict(**j,timing_qualified=False,stage=None,reason=repr(err),correctness=task['golden'])
                        receipt['rows'].append(rr)
                    good=[r['stage'] for r in receipt['rows'] if r['timing_qualified']];verify.checked(all(b['first_tick']>a['last_tick'] for a,b in zip(good,good[1:])),'time order across launches')
                    receipt['qualified']=len(good)==len(jobs)
                except (ValueError,OSError,TypeError,IndexError,KeyError) as err:
                    receipt['reason']=repr(err)
                write(dest,receipt)
                print(folder.name,receipt['qualified'],sum(r['timing_qualified'] for r in receipt['rows']),receipt.get('reason',''),flush=True)
        write(OUT/'status.json',dict(utc=now(),state='complete',receipts={f.stem:read(f)['qualified'] for f in sorted((OUT/'receipts').glob('*.json'))}))

if __name__=='__main__':{'register':register,'run':collect}[sys.argv[1]]()
