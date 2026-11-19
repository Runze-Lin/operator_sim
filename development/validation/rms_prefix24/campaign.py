"""Freeze and collect partial-history RMSNorm sequences on the two calibrated SKUs."""
import argparse
from collections import defaultdict
from datetime import datetime, timezone
import fcntl
from hashlib import sha256
import json
import os
from pathlib import Path
import sys
from time import perf_counter_ns

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'development/validation/sku17'))
sys.path.insert(0, str(ROOT / 'development/validation/rms_ordered_verify'))
from operator_model import OperatorStream, estimate
from development.validation.rms_ordered_verify import verify
from development.validation.rms_ordered_verify.prefix import host
from development.validation.rms_dispatch_audit.ordered_ca import capture

OUT = ROOT / 'results/v0.24/rms-prefix'
RAW = ROOT / 'runs/rms-prefix24'
PLAN = OUT / 'freeze.json'
CHIPS = ('Ascend950DT_9582', 'Ascend950PR_957b', 'Ascend970DT')
# First launch executes outside the observed prefix, second gives its first lower bound.
GROUPS = {
    'narrow-start': [(1,1200),(2,1888),(3,3008),(9,2272),(19,2032)],
    'wide-start': [(55,1920),(3,1808),(9,1568),(34,1728),(47,2592)],
    'many-rows': [(1,1280),(3,1392),(17,2176),(33,2128),(49,2368),(513,3968)],
}


def write(path, obj):
    tmp = path.with_suffix('.tmp')
    tmp.write_text(json.dumps(obj, indent=2, allow_nan=False) + '\n')
    tmp.replace(path)


def freeze():
    if PLAN.exists():
        raise FileExistsError(PLAN)
    targets = {tuple(s) for sequence in GROUPS.values() for s in sequence[2:]}
    assert len(targets) == 10
    for reference in ('results/rms-unknown23/freeze.json', 'results/v0.22/rms-unknown/audit.json', 'results/v0.22/evaluation.json'):
        known = {tuple(r['shape']) for r in json.loads((ROOT/reference).read_text())['rows'] if r.get('op','rmsnorm') == 'rmsnorm'}
        assert not targets & known, targets & known
    rows = []
    for chip in CHIPS:
        for group, shapes in GROUPS.items():
            stream = OperatorStream(chip)
            stream.forget_history()
            true_high = lower = 0
            for i, shape in enumerate(shapes):
                prior = {'resident_aiv':true_high}
                full = estimate('rmsnorm', shape, prior, chip=chip)
                old = estimate('rmsnorm', shape, {'resident_aiv':lower} if lower >= full.active_aiv else {}, chip=chip)
                tick = perf_counter_ns()
                result = stream.advance('rmsnorm', shape) if i else full
                elapsed = (perf_counter_ns()-tick)/1e6 if i else None
                rows.append(dict(i=i, chip=chip, group=group, shape=shape, role=['hidden','observed_prefix','target'][min(i,2)],
                                 active_aiv=full.active_aiv, true_H=true_high, observed_lower=lower, full_us=full.latency_us,
                                 previous_partial_us=old.latency_us, partial_us=result.latency_us,
                                 scenario=result.projection.get('partial_history'), query_ms=elapsed))
                true_high = max(true_high, full.active_aiv)
                if i:
                    lower = max(lower, full.active_aiv)
    paths = [Path(__file__), ROOT/'build/vector-v8/NORM', verify.VENDOR, ROOT/'development/validation/rms_ordered_verify/verify.py',
             ROOT/'development/validation/rms_ordered_verify/prefix.py', ROOT/'development/validation/rms_dispatch_audit/ordered_ca.py',
             ROOT/'scripts/cann-run.sh', *[ROOT/'operator_model'/p for p in ('stream.py','rms_history.py','backend.py','rms_dispatch.py','rms_lane_schedule.py')]]
    write(PLAN, dict(utc=datetime.now(timezone.utc).isoformat(), files={str(p.relative_to(ROOT)):verify.digest(p) for p in paths},
                     wall_per_batch_s=3600, raw_mib_per_batch=12288, rows=rows, chip_order=CHIPS[:2], groups=list(GROUPS),
                     scope='One bounded serial slot, 10 distinct scored shapes on each 950, observed prefixes only, no refits. 970 predictions only.'))


def collect():
    plan = json.loads(PLAN.read_text())
    for path,h in plan['files'].items():
        verify.checked(verify.digest(ROOT/path) == h, 'changed frozen input '+path)
    RAW.mkdir(exist_ok=True)
    os.environ.update(OMP_WAIT_POLICY='PASSIVE',GOMP_SPINCOUNT='0')
    with (RAW/'.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        for group in plan['groups']:
            for chip in plan['chip_order']:
                name = group+'-'+chip
                dest = OUT/(name+'.json')
                if dest.exists():
                    continue
                jobs = [r for r in plan['rows'] if (r['chip'],r['group']) == (chip,group)]
                folder = RAW/name
                cost = capture(chip,folder,jobs,plan)
                measured = []
                receipt = dict(cost=cost, freeze_sha256=verify.digest(PLAN), group=group, chip=chip, rows=measured)
                try:
                    verify.checked(not cost['stop_reason'] and cost['exit_code']==0 and cost['started_utc']>plan['utc'], 'incomplete registered process')
                    tasks,tables = host(folder,chip,jobs)
                    profile, = folder.glob('profile/OPPROF_*/device0/'+verify.KERNEL)
                    elf = verify.text_bytes(verify.VENDOR)
                    previous_end = 0
                    for j,task,table in zip(jobs,tasks,tables):
                        row = verify.launch(profile/str(j['i']),chip,j,task,table,elf)
                        stage = row['stage']
                        verify.checked(row['active_count_matches'] and stage['first_tick'] > previous_end, 'incorrect core count or launch order')
                        previous_end = stage['last_tick']
                        measured.append(dict(**j,ca_us=stage['ca_us'],stage=stage,goldens=row['correctness']))
                except (ValueError,OSError,TypeError,KeyError,IndexError) as err:
                    receipt['error'] = str(err)
                write(dest,receipt)
                print(name,len(measured),receipt.get('error','OK'),flush=True)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=['freeze','collect'])
    action = parser.parse_args().action
    {'freeze':freeze,'collect':collect}[action]()
