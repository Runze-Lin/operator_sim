"""Verify the no-flag stream against pre-CA forecasts from an installed wheel."""
from hashlib import sha256
from pathlib import Path
import json
import os
import shutil
import statistics
import subprocess
import sys
import tempfile
import time

ROOT = Path(__file__).resolve().parents[3] if len(sys.argv)==1 else Path.cwd()
RESULT = ROOT / 'results/v0.27/post'
POLICY = 'results/v0.25/post/repeat-policy-freeze.json'
REGISTRATION = 'results/v0.26/post/registration.json'


def probe(plan):
    from operator_model import OperatorStream, estimate

    data = json.loads(Path(plan).read_text())
    streams = {}
    maximum = 0.
    costs = []
    rows = []
    for row in data:
        identity = row['version'], row['chip'], row['group']
        if identity not in streams:
            streams[identity] = (OperatorStream(row['chip']),
                                 OperatorStream(row['chip'], experimental_post_repeats=False))
        live, legacy = streams[identity]
        t = time.perf_counter_ns()
        default = live.advance('mhc_post', row['shape'], dtype='bf16')
        elapsed = (time.perf_counter_ns() - t) / 1e6
        old = legacy.advance('mhc_post', row['shape'], dtype='bf16')
        stateless = estimate('mhc_post', row['shape'], chip=row['chip'])
        repeated = default.projection['timing_diagnostics'].get('model') == 'post_token_immediate_repeat'
        assert repeated == row['eligible'], (identity, row['shape'], row['role'])
        drift = max(abs(default.latency_us-row['policy_us']), abs(old.latency_us-row['old_us']),
                    abs(stateless.latency_us-row['unknown_us']))
        maximum = max(maximum, drift)
        assert drift < 1.e-10
        costs.append(elapsed)
        rows.append(dict(chip=row['chip'], shape=row['shape'], group=row['group'], role=row['role'], eligible=repeated,
                         expected=row['policy_us'], actual=default.latency_us,
                         old_us=old.latency_us, stateless_us=stateless.latency_us, ms=elapsed))
    return dict(total=len(rows), repeat=sum(row['eligible'] for row in rows),
                maximum_delta_us=maximum, median_ms=statistics.median(costs),
                p95_ms=sorted(costs)[int(.95*(len(costs)-1))], maximum_ms=max(costs), rows=rows)


def run():
    receipts=json.loads((RESULT/'deployment.json').read_text())
    assert receipts['qualified_for_default'], 'Cannot validate a promoted path before its CA gate passes'
    with tempfile.TemporaryDirectory() as path:
        temp=Path(path)
        project=temp/'wheel-project'; project.mkdir()
        for name in ('README.md','pyproject.toml','setup.py','MANIFEST.in'):
            shutil.copy2(ROOT/name, project/name)
        shutil.copytree(ROOT/'operator_model',project/'operator_model',ignore=shutil.ignore_patterns('__pycache__','*.pyc'))
        for line in (ROOT/'MANIFEST.in').read_text().splitlines():
            if line.startswith(('include results/','include configs/')):
                name=line.split(maxsplit=1)[1];(project/name).parent.mkdir(parents=True,exist_ok=True)
                shutil.copy2(ROOT/name, project/name)
        subprocess.run([sys.executable,'-m','pip','wheel','--no-index','--no-build-isolation','--no-deps',
                        '-w',str(temp),'.'],cwd=project,check=True,stdout=subprocess.DEVNULL)
        wheel,=temp.glob('*.whl');dest=temp/'deployed'
        subprocess.run([sys.executable,'-m','pip','install','--no-index','--no-deps',
                        '--no-compile','--target',str(dest),str(wheel)],cwd=temp,
                       stdout=subprocess.DEVNULL,check=True)
        price='results/v0.25/post/prices.json'
        assert (dest/'operator_model'/price).read_bytes()==(ROOT/price).read_bytes()
        plan=[]
        # Both historic forecasters were frozen while none of the included CA windows existed.
        prior=json.loads((ROOT/POLICY).read_text());prior_plan=json.loads((ROOT/'results/v0.25/post/freeze.json').read_text())
        lookup={(r['chip'],r['group'],r['i']):r for r in prior_plan['rows']}
        for r in prior['rows']:
            v=lookup[r['chip'],r['group'],r['i']]
            plan.append(dict(version='v0.25',chip=r['chip'],group=r['group'],role=v['role'],shape=v['shape'],
                             eligible=v['role']=='repeat' and v['plan']['route']=='post_token',
                             policy_us=r['policy_us'],old_us=v['old_known_us'],unknown_us=v['old_unknown_us']))
        ext=json.loads((ROOT/REGISTRATION).read_text())
        for r in ext['rows']:
            plan.append(dict(version='v0.26',chip=r['chip'],group=r['group'],role=r['role'],shape=r['shape'],
                             eligible=r['role']=='repeat' and r['plan']['route']=='post_token',
                             policy_us=r['us'],old_us=r['old_known_us'],unknown_us=r['old_unknown_us']))
        script=temp/'check.py';shutil.copy2(__file__,script);freeze=temp/'plan.json';freeze.write_text(json.dumps(plan))
        proc=subprocess.run([sys.executable,str(script),str(freeze)],cwd=temp,env={**os.environ,'PYTHONPATH':str(dest)},
                            capture_output=True,text=True)
        if proc.returncode:
            raise RuntimeError(proc.stderr)
        checked=json.loads(proc.stdout)
        assert checked['maximum_ms'] < 100
        checked.update(wheel=wheel.name,price_sha256=sha256((dest/'operator_model'/price).read_bytes()).hexdigest(),
                       registrations=[POLICY,REGISTRATION],scope='No-flag production stream, explicit legacy opt-out and stateless; 970 extrapolated')
        (RESULT/'published.json').write_text(json.dumps(checked,indent=2)+'\n')


if __name__=='__main__':
    if len(sys.argv)==2: print(json.dumps(probe(sys.argv[1])))
    else: run()
