"""Budget one identical-prefix recapture for any registered target missing physical END."""
from datetime import datetime, timezone
import fcntl
import os
from pathlib import Path
import time
from campaign import CHIPS, OUT, PLAN, RAW, capture, host, json, verify, write
from physical import measure

FROZEN=OUT/'recovery-registration.json'


def register():
    if FROZEN.exists():
        raise FileExistsError(FROZEN)
    write(FROZEN,dict(utc=datetime.now(timezone.utc).isoformat(),original=verify.digest(PLAN),
        source_sha256=verify.digest(Path(__file__)),physical_sha256=verify.digest(Path(__file__).with_name('physical.py')),
        groups=['many-rows'],chips=CHIPS[:2],maximum_repetitions_per_target=1,wall_per_batch_s=1800,
        raw_mib_per_batch=4096,
        scope='Only a registered target without a complete physical window may be counted. Replay the identical host prefix through its ordinal, same chip, ABI, memory configuration and both full output goldens. One serial slot after original CA. Earlier rows are controls and never enter scores; retain original disposition.'))


def recover():
    reg=json.loads(FROZEN.read_text());plan=json.loads(PLAN.read_text())
    assert reg['original']==verify.digest(PLAN) and reg['source_sha256']==verify.digest(Path(__file__))
    os.environ.update(OMP_WAIT_POLICY='PASSIVE',GOMP_SPINCOUNT='0')
    with (RAW/'.recovery-lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        while not (OUT/'many-rows-Ascend950PR_957b.json').is_file():
            time.sleep(15)
        for group in reg['groups']:
            for chip in reg['chips']:
                name=group+'-'+chip
                while not (OUT/(name+'-windows.json')).is_file():
                    time.sleep(15)
                original=json.loads((OUT/(name+'-windows.json')).read_text())
                jobs=[r for r in plan['rows'] if (r['chip'],r['group'])==(chip,group)]
                for failure in original.get('failed',[]):
                    i=failure['i']
                    if jobs[i]['role']!='target':
                        continue
                    dest=OUT/(f'recovery-{name}-{i}.json')
                    if dest.exists():
                        continue
                    folder=RAW/f'recovery-{name}-{i}'
                    cost=capture(chip,folder,jobs[:i+1],reg)
                    receipt=dict(original_failure=failure,cost=cost,registered=verify.digest(FROZEN))
                    try:
                        verify.checked(cost['exit_code']==0 and not cost['stop_reason'] and cost['started_utc']>reg['utc'],'process failed')
                        assert (folder/'inputs.txt').read_text().splitlines()==(RAW/name/'inputs.txt').read_text().splitlines()[:i+1]
                        tasks,tables=host(folder,chip,jobs[:i+1])
                        verify.checked(all(len(t)==j['active_aiv'] for t,j in zip(tables,jobs)),'prefix roster')
                        profile,=folder.glob('profile/OPPROF_*/'+verify.KERNEL)
                        elf=verify.text_bytes(verify.VENDOR)
                        target=measure(profile/str(i),chip,jobs[i],tables[i],elf)
                        try:
                            strict=verify.launch(profile/str(i),chip,jobs[i],tasks[i],tables[i],elf)['stage']['ca_us']
                            reason=None
                        except (ValueError,OSError,IndexError,KeyError,TypeError) as err:
                            strict=None;reason=str(err)
                        future,=[r for r in plan['rows'] if (r['chip'],r['group'],r['i'])==(CHIPS[2],group,i)]
                        receipt['target']=dict(**jobs[i],**target,strict_label_us=strict,strict_error=reason,goldens=tasks[i]['golden'],
                                               future_partial_us=future['partial_us'],source='identical prefix recapture')
                    except (ValueError,OSError,IndexError,KeyError,TypeError) as err:
                        receipt['error']=repr(err)
                    write(dest,receipt)

if __name__=='__main__':
    import sys
    {'register':register,'run':recover}[sys.argv[1]]()
