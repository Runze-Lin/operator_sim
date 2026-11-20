"""Freeze RMS issue-graph and both baselines before a single serial matched experiment."""
from hashlib import sha256
import fcntl
import os
from pathlib import Path
import sys
from time import perf_counter_ns

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
from development.validation.rms27 import campaign as prior
from development.validation.rms27.physical import measure
from development.validation.rms28.schedule import forecast, schedule
from operator_model.rms_dispatch import price_rms
from operator_model import load_chip

OUT = ROOT / 'results/v0.28/rms'
RAW = ROOT / 'runs/rms28'
REG = OUT / 'freeze.json'
read, write, verify = prior.read, prior.write, prior.verify
SHAPES = {
    'partition': [(1,1072), (4,784), (21,2336), (37,1808)],
    'siblings': [(2,3088), (3,3504), (6,5776)],
    'sector': [(32,1616), (131,1424)],
    'multi': [(539,4000)],
}
ABLATIONS = ('sector arbitration', 'VF issue overlap', 'partition wake', 'RMS-specific warm', 'shared first payload')

def register():
    if RAW.exists() or REG.exists():
        raise FileExistsError(REG)
    held = {s for group in SHAPES.values() for s in group}
    assert len(held) == 10
    history = [ROOT / p for p in ('results/v0.22/evaluation.json','results/v0.24/rms-prefix/freeze.json',
        'results/rms-unknown23/freeze.json','results/v0.27/rms/freeze.json','results/v0.28/rms/milestones.json')]
    for path in history:
        assert not held & {tuple(v['shape']) for v in read(path)['rows'] if v.get('op','rmsnorm')=='rmsnorm'}, path
    profiles = read(OUT/'prices.json')
    previous = read(prior.PARAMS)
    data = []
    for sku in prior.CHIPS:
        chip = load_chip(sku)
        for group, shapes in SHAPES.items():
            prefix = [('prefix', SHAPES['partition'][0])] if group in ('siblings','multi') else []
            sequence = prefix + [(role,s) for s in shapes for role in ('entry','repeat')]
            h = 0
            for i,(role,s) in enumerate(sequence):
                begin = perf_counter_ns()
                pred = forecast(s,chip,profiles)
                cost_ms = (perf_counter_ns()-begin)/1e6
                old_context,_,a,d = price_rms(s,chip,{'resident_aiv':h})
                assert role != 'entry' or h<a
                assert role != 'repeat' or h>=a
                context = schedule(s,chip,h,role=='repeat',profiles,d)['us']
                data.append(dict(chip=sku, group=group, i=i, role=role, shape=s, prior_aiv=h,
                    known_us=old_context, specific_context_us=context, runtime_ms=cost_ms,
                    prior27_us=prior.forecast(s,chip,previous)['us'],
                    abl={k:forecast(s,chip,profiles,k)['us'] for k in ABLATIONS}, **pred))
                h = max(h,a)
    files = [Path(__file__),Path(__file__).with_name('schedule.py'),Path(__file__).with_name('fit.py'),
        OUT/'prices.json',OUT/'stage-samples.json',prior.PARAMS,prior.RUNNER,verify.VENDOR,
        prior.REG,ROOT/'development/validation/rms_prefix24/physical.py',
        ROOT/'development/validation/rms27/physical.py',ROOT/'scripts/cann-run.sh',
        ROOT/'development/validation/rms_dispatch_audit/ordered_ca.py',
        ROOT/'development/validation/rms_ordered_verify/prefix.py',ROOT/'development/validation/rms_ordered_verify/verify.py',
        ROOT/'operator_model/rms_dispatch.py', ROOT/'operator_model/rms_resource.py',ROOT/'operator_model/rms_lane_schedule.py',
        ROOT/'results/v0.22/rms-lane-parameters.json'] + [ROOT/'configs/chips'/(c+'.json') for c in prior.CHIPS]
    write(REG,dict(utc=prior.now(),rows=data,measured=prior.CHIPS[:2],extrapolated=prior.CHIPS[2:],
        groups=list(SHAPES),slots=1,wall_per_batch_s=3600,raw_mib_per_batch=12288,max_attempts_per_group=1,
        primary='RMS physical all-lane ELF entry/END, per-core official and aggregate windows, golden y and rstd, entire access volumes, same installed ELF, ordered same-stream pair in one process. All ten distinct pairs on each SKU. Score physical instruction duration only. A synthetic END_LABEL is not required. No substitution/retry. Exclude unpaired launches and prefix controls.',
        gate='Entry and repeat separately on both DT/PR: WAPE <= 25% and max <= 50%, ten distinct paired shapes per real target. Report sparse A<=4 and 2+ UB separately. 970DT projection has no timing gold.',
        files={str(p.relative_to(ROOT)):verify.digest(p) for p in files}))

def collect(folder, chip, jobs, cost, reg):
    verify.checked(cost['started_utc']>reg['utc'] and cost['exit_code']==0 and not cost['stop_reason'],'process exit or budget')
    expected = ''.join(f"rmsnorm {r['i']} {r['shape'][0]} {r['shape'][1]} 0\n" for r in jobs)
    verify.checked((folder/'inputs.txt').read_text()==expected,'observed inputs')
    tasks,tables = prior.host(folder,chip,jobs)
    profile, = folder.glob('profile/OPPROF_*')
    pub = profile/verify.KERNEL
    if not pub.is_dir(): pub = profile/'device0'/verify.KERNEL
    verify.checked('Profiling data parse finished.' in (folder/'run.log').read_text()
        and sorted(int(f.name) for f in pub.iterdir() if f.is_dir())==list(range(len(jobs))),'published ordinals')
    elf = verify.text_bytes(verify.VENDOR)
    rows = []
    prev = 0
    for j,task,table in zip(jobs,tasks,tables):
        ident = {k:j[k] for k in ('chip','group','role','shape','i','active_aiv','prior_aiv')}
        try:
            actual = measure(pub/str(j['i']),chip,j,table,elf)
            verify.checked(actual['first_tick']>prev,'physical device order')
            prev = actual['last_tick']
            rows.append(dict(**ident,timing_qualified=True,correctness=task['golden'],
                task={k:v for k,v in task.items() if k!='golden'},elf_text_sha256=sha256(elf).hexdigest(),**actual))
        except (ValueError,IndexError,TypeError,KeyError,OSError) as e:
            rows.append(dict(**ident,timing_qualified=False,reason=repr(e)))
    return rows

def run():
    reg = read(REG)
    for fname,sha in reg['files'].items():
        verify.checked(verify.digest(ROOT/fname)==sha,fname)
    os.environ.update(OMP_WAIT_POLICY='PASSIVE',GOMP_SPINCOUNT='0')
    RAW.mkdir(exist_ok=True)
    receipts = OUT/'receipts'
    receipts.mkdir(exist_ok=True)
    with (RAW/'serial.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        for group in reg['groups']:
            for chip in reg['measured']:
                name = group+'-'+chip
                receipt = receipts/(name+'.json')
                if receipt.exists(): continue
                rows = [v for v in reg['rows'] if (v['chip'],v['group'])==(chip,group)]
                write(OUT/'status.json',dict(state='running',group=group,chip=chip,utc=prior.now()))
                folder = RAW/name
                cost = prior.capture(chip,folder,rows,reg)
                result = dict(chip=chip,group=group,cost=cost,freeze_sha256=verify.digest(REG))
                try: result['rows'] = collect(folder,chip,rows,cost,reg)
                except (ValueError,IndexError,TypeError,KeyError,OSError) as e: result.update(rows=[],reason=repr(e))
                write(receipt,result)
                print(name,sum(q['timing_qualified'] for q in result['rows']),len(rows),result.get('reason',''),flush=True)
        write(OUT/'status.json',dict(state='complete',utc=prior.now()))

if __name__=='__main__':
    {'register':register,'run':run}[sys.argv[1]]()
