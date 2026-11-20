"""One audit-triggered rerun on the same SKU, with the exact original prefix and both paired calls."""
import fcntl,sys,os
from collections import defaultdict
from pathlib import Path
ROOT=Path(__file__).resolve().parents[3];sys.path.insert(0,str(ROOT))
from development.validation.rms27.campaign import OUT,RAW,REG,verify,read,write,now,host,capture
REGISTER=OUT/'recovery-freeze.json';OUTCOME=OUT/'recovery'
def register():
    if REGISTER.exists():raise FileExistsError(REGISTER)
    plan=read(REG);prior={f.name:verify.digest(f) for f in (OUT/'receipts').glob('*.json')}
    write(REGISTER,dict(utc=now(),files={f:verify.digest(ROOT/f) for f in (str(REG.relative_to(ROOT)),str(Path(__file__).relative_to(ROOT)))},
         existing_receipts=prior,slots=1,wall_per_batch_s=plan['wall_per_batch_s'],raw_mib_per_batch=plan['raw_mib_per_batch'],
         budget='At most one capture per original unfinished host group, and only if not fully qualified. Prefix ends at the last affected repeat. Publish neither point of a repeated pair unless BOTH launches pass in that same new process with frozen H and all original contracts. Retain original strict and original-only metrics. Never retry valid pairs or switch shapes on account of error.'))
def run():
    plan=read(REG);reg=read(REGISTER);OUTCOME.mkdir(exist_ok=True)
    assert all(verify.digest(ROOT/f)==h for f,h in {**reg['files'],**plan['files']}.items())
    os.environ.update(OMP_WAIT_POLICY='PASSIVE',GOMP_SPINCOUNT='0')
    with (RAW/'.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX)
        assert read(OUT/'status.json')['state']=='complete'
        for orig in sorted((OUT/'receipts').glob('*.json')):
            prior=read(orig);sku=prior['chip'];g=prior['group'];marks=defaultdict(set)
            for v in prior['rows']:
                if v['timing_qualified']:marks[tuple(v['shape'])].add(v['role'])
            jobs=[r for r in plan['rows'] if r['chip']==sku and r['group']==g]
            missing={tuple(r['shape']) for r in jobs if r['role']=='entry' and marks[tuple(r['shape'])]!={'entry','repeat'}}
            if not missing:continue
            limit=max(r['i'] for r in jobs if tuple(r['shape']) in missing and r['role']=='repeat')
            jobs=jobs[:limit+1];id=orig.stem;dest=OUTCOME/(id+'.json');folder=RAW/('recovery-'+id)
            if dest.exists():continue
            write(OUT/'recovery-state.json',dict(utc=now(),state='running',group=g,sku=sku,shapes=sorted(missing)))
            cost=capture(sku,folder,jobs,reg);result=dict(chip=sku,group=g,original_receipt=verify.digest(orig),registration=verify.digest(REGISTER),
                cost=cost,eligible_shapes=sorted(missing),rows=[])
            try:
                verify.checked(cost['started_utc']>reg['utc'] and cost['exit_code']==0 and not cost['stop_reason'],'process/chronology')
                verify.checked((folder/'inputs.txt').read_text()==''.join(f"rmsnorm {r['i']} {r['shape'][0]} {r['shape'][1]} 0\n" for r in jobs),'identical prefix')
                tasks,tables=host(folder,sku,jobs);profile,=folder.glob('profile/OPPROF_*');pub=profile/verify.KERNEL
                if not pub.is_dir():pub=profile/'device0'/verify.KERNEL
                verify.checked('Profiling data parse finished.' in (folder/'run.log').read_text() and sorted(int(f.name)for f in pub.iterdir() if f.is_dir())==list(range(len(jobs))),'published ordinals')
                elf=verify.text_bytes(verify.VENDOR)
                from hashlib import sha256
                result['elf_text_sha256']=sha256(elf).hexdigest();h=0;last=0
                for r,t,tab in zip(jobs,tasks,tables):
                    assert h==r['prior_aiv'];h=max(h,r['active_aiv'])
                    try:
                        v=verify.launch(pub/str(r['i']),sku,r,t,tab,elf);s=v['stage'];verify.checked(v['active_count_matches'] and s['first_tick']>last,'roster/order');last=s['last_tick']
                    except (ValueError,OSError,TypeError,IndexError,KeyError) as err:v=dict(**r,timing_qualified=False,reason=repr(err))
                    result['rows'].append(v)
            except (ValueError,OSError,TypeError,IndexError,KeyError) as err:result['reason']=repr(err)
            write(dest,result)
            print(id,missing,[v['timing_qualified'] for v in result['rows']],flush=True)
        write(OUT/'recovery-state.json',dict(utc=now(),state='complete'))
        from development.validation.rms27.report import render
        render()
if __name__=='__main__':{'register':register,'run':run}[sys.argv[1]]()
