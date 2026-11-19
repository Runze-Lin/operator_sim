"""Recover only v0.21 RMS calls the old smoke host rejected before ACL initialization."""
import fcntl
from pathlib import Path
import os
import re
import sys
from time import sleep

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0,str(ROOT))
from development.validation.rms_host_recovery.build import RUNNER, SOURCE, ORIGINAL, OLD_GATE, NEW_GATE
from development.validation.sku19.campaign import DT, PR, read, write, digest, now
from development.validation.sku19.collect import capture, qualify

OUT = ROOT/'results/v0.21/rms'
REG = OUT/'host-recovery.json'
RETRIES = OUT/'retries'
RAW = ROOT/'runs/rms21-host-recovery'
ORIG_RAW = ROOT/'runs/rms21'
CONTROL = ROOT/'runs/sku20-rms/rmsnorm_32_1280_Ascend950PR_957b/qualified.json'
CONTROL_FILE = RETRIES/'control'/'rmsnorm_32_1280_Ascend950PR_957b.json'


def rel(p):
    return str(p.relative_to(ROOT))


def register():
    if REG.exists() or RAW.exists() or RETRIES.exists():
        raise FileExistsError(REG)
    f=read(OUT/'freeze.json')
    assert all(digest(ROOT/name)==v for name,v in f['refs'].items())
    old_records=read(ROOT/'results/v0.20/rms/records.json')
    families={(r['kernel'],r['stages'][0]['binary_sha256']) for r in old_records}
    assert len(families)==1 and all(r['scope']=='one all-core window' for r in old_records)
    family,elf=families.pop()
    control=read(CONTROL)
    assert control['runner_sha256']==f['jobs'][0]['runner_sha256']
    jobs=[]
    for j in f['jobs']:
        rows,width=j['shape']
        assert rows>0 and 0<width<=8192 and width%16==0
        if width<=7168 and width%64==0:
            continue
        jobs.append(dict(id=j['id'],shape=j['shape'],outputs=j['outputs'],args=j['args'],kernels=j['kernels'],
                         wall_s=j['wall_s'],raw_mib=j['raw_mib'],nchecks=j['nchecks'],sides=[DT,PR]))
    assert len(jobs)==7
    paths=[ORIGINAL,SOURCE,RUNNER,ROOT/f['jobs'][0]['runner'],OUT/'freeze.json',CONTROL,Path(__file__),Path(__file__).with_name('build.py')]
    write(REG,dict(utc=now(),scope='Host-only repair. Original 10 frozen shapes/predictions and 20 endpoints remain intact. Retries for the 14 width-rejected endpoints only, after original receipts prove zero kernel execution. Six endpoints use original host. No model or kernel change.',
         exact_original_contract='aclnnRmsNorm BF16 x alternating +/-1, BF16 gamma=1, ND [R,C], epsilon=1e-6. BF16 y bit exact +/-1, FP32 per-row rstd within 1e-4 of 1/sqrt(1+1e-6). Host checks all R*C outputs and R statistics.',
         change={'removed':OLD_GATE,'replacement':NEW_GATE,'rows_gate_unchanged':[1,262144],'quant_unchanged':True},
         original_freeze=rel(OUT/'freeze.json'), original_utc=f['utc'], files={rel(p):digest(p) for p in paths},
         new_runner=rel(RUNNER),original_runner=f['jobs'][0]['runner'],expected_kernel=family,expected_elf=elf,
         control=dict(chip=PR,shape=[32,1280],prior=rel(CONTROL),role='Historical successful old-host control; not a holdout. Compare numerical checks, exact kernel and AIV count, full all-core time, verify per-core windows.'),
         jobs=jobs,retention='Original and retry receipts kept separately. Retry uses the original per-endpoint wall and byte caps, and only after a pre-ACL rejection. One serial slot released by the completed sku20-rms campaign. Same no-dump one-kernel command, all cores.',
         expected_endpoints=len(jobs)*2))
    RETRIES.mkdir(parents=True)


def one(j,chip,folder,old,registration,path):
    alt=dict(j,runner=registration['new_runner'],runner_sha256=registration['files'][registration['new_runner']])
    assert (j['wall_s'],j['raw_mib'])==(alt['wall_s'],alt['raw_mib'])
    if path.exists():
        return read(path)
    if (folder/'cost.json').exists():
        cost=read(folder/'cost.json')
    else:
        if folder.exists():
            raise RuntimeError(f'Partial attempt exists at {folder}; preserve it for audit.')
        cost=capture(alt,chip,folder)
    assert cost['started_utc']>registration['utc']
    try:
        obs,=qualify(alt,chip,folder)
        stage,=obs['stages']
        if (stage['kernel'],stage['binary_sha256'])!=(registration['expected_kernel'],registration['expected_elf']):
            raise ValueError(f"Observed kernel identity {stage['kernel']} / {stage['binary_sha256']}")
        rows,width=j['shape']
        expected=[f'correctness count={rows*width} bad=0',f'correctness count={rows} bad=0']
        contract=f'case=rmsnorm api=aclnnRmsNorm R={rows} C={width} dtype=BF16 epsilon=1e-6'
        log=(folder/'run.log').read_text(errors='replace')
        if obs['correctness']!=expected or re.findall('^case=.*',log,re.M)!=[contract]:
            raise ValueError('Wrong host/input/golden contract')
        if not(stage['active_aiv']>0 and stage['active_aic']==0 and obs['scope']=='one all-core window'):
            raise ValueError('Unexpected all-core scope or core family')
        obs.update(dtype='bf16',runner_sha256=alt['runner_sha256'],input_contract_log=[contract],host_args=j['args'],
                   run_log=rel(folder/'run.log'))
        out=dict(qualified=True,observation=obs)
        if old.get('is_control'):
            ref=read(ROOT/old['receipt'])
            if (obs['correctness'],obs['active_aiv'],obs['kernel'],stage['binary_sha256'])!=(
                ref['correctness'],ref['active_aiv'],ref['kernel'],ref['stages'][0]['binary_sha256']):
                raise ValueError('Control golden/core/ELF mismatch')
            out.update(comparison=dict(original_us=ref['ca_us'],repeat_us=obs['ca_us'],
                       difference_us=obs['ca_us']-ref['ca_us'],original_aiv=ref['active_aiv'],repeat_aiv=obs['active_aiv'],
                       original_trace_sha256=ref['trace_sha256'],repeat_trace_sha256=obs['trace_sha256'],
                       goldens_identical=True,elf_identical=True))
    except (ValueError,KeyError,IndexError,OSError,AssertionError) as e:
        out=dict(qualified=False,error=repr(e),cost=cost)
    info=dict(id=j['id'],op='rmsnorm',shape=j['shape'],chip=chip,original=old,host_recovery=rel(REG),**out)
    path.parent.mkdir(parents=True,exist_ok=True)
    write(path,info)
    print(path.stem,info['qualified'],info.get('error',''),flush=True)
    return info


def qualify_original_failure(folder,receipt):
    log=folder/'run.log'
    old=read(receipt)
    s=log.read_text(errors='replace')
    before='SMOKE_FAIL: invalid prefill contract'
    if not (old['qualified'] is False and before in s and 'No kernel profiling data was generated' in s and
            not re.search(r'^runtime_soc=|^case=|^correctness |Start profiling on kernel:',s,flags=re.M) and
            not any(folder.glob('profile/**/simulator/trace.json'))):
        raise ValueError('Original failure is not demonstrably pre-ACL; excluding from automatic retry: '+str(folder))
    return dict(receipt=rel(receipt),receipt_sha256=digest(receipt),log=rel(log),log_sha256=digest(log),failure=before,device_kernel_launched=False)


def run():
    registration=read(REG)
    assert all(digest(ROOT/p)==v for p,v in registration['files'].items())
    original=read(OUT/'freeze.json')
    originals={j['id']:j for j in original['jobs']}
    for reg in registration['jobs']:
        j=originals[reg['id']]
        assert all(j[k]==reg[k] for k in ('shape','outputs','args','kernels','nchecks','wall_s','raw_mib'))
    os.environ.update(OMP_WAIT_POLICY='PASSIVE',GOMP_SPINCOUNT='0')
    state_file=RETRIES/'control'/'queue.json'
    state_file.parent.mkdir(parents=True,exist_ok=True)
    # The original v0.21 queue already took a completed dispatch slot. The two
    # old long RMS jobs released more slots after that, one of which we reuse.
    while not all((ROOT/'runs/sku20-rms'/('rmsnorm_32768_1280_'+s)/'outcome.json').is_file() for s in (DT,PR)):
        sleep(10)
    with (RETRIES/'control'/'.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        job=read(ROOT/'results/v0.20/rms/freeze.json')['jobs'][0]
        assert job['shape']==registration['control']['shape']
        checkpoint=lambda **extra:write(state_file,dict(utc=now(),**extra))
        checkpoint(state='control',chip=PR,id=job['id'])
        old=dict(is_control=True,receipt=rel(CONTROL),receipt_sha256=digest(CONTROL))
        ctrl=one(job,PR,RAW/'control'/f"{job['id']}_{PR}",old,registration,CONTROL_FILE)
        if not ctrl['qualified']:
            checkpoint(state='control_failed',error=ctrl['error'])
            return
        for req in registration['jobs']:
            j=originals[req['id']]
            for chip in req['sides']:
                name=j['id']+'_'+chip
                path=RETRIES/(name+'.json')
                if path.is_file():
                    continue
                receipt=OUT/'receipts'/(name+'.json')
                checkpoint(state='await_original_prelaunch_failure',id=j['id'],chip=chip)
                while not receipt.exists():
                    sleep(10)
                old=qualify_original_failure(ORIG_RAW/name,receipt)
                checkpoint(state='collecting',id=j['id'],chip=chip)
                one(j,chip,RAW/name,old,registration,path)
        checkpoint(state='complete',retry_receipts=len(list(RETRIES.glob('*.json'))))


if __name__=='__main__':
    {'register':register,'run':run}[sys.argv[1]]()
