"""Reproduce both preregistrations, including the first model in an isolated checkpoint."""
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0,str(ROOT))
from development.validation.sku19.campaign import CHIPS, OUT, digest, forecast, forecast_old, read, write

CHECKPOINT='e6eb2c7369904f1332356a67829c26bad57703fe'


def check(rows, reference):
    assert rows['version']==reference['version'] and len(rows['rows'])==len(reference['rows'])
    count=0
    for a,b in zip(rows['rows'],reference['rows']):
        assert a['id']==b['id'] and a['targets'].keys()==b['targets'].keys()
        for side,got in a['targets'].items():
            want=b['targets'][side]
            assert {k:v for k,v in got.items() if k!='query_ms'}=={k:v for k,v in want.items() if k!='query_ms'}, (a['id'],side)
            count+=1
    return count


def first_model(jobs, dest):
    source=(Path(__file__).parent/'campaign.py').read_text().split('def forecast(jobs):')[1].split('def forecast_old')[0]
    code="from time import perf_counter_ns\nimport json,sys\nsys.path.insert(0,sys.argv[1])\nCHIPS="+repr(CHIPS)+"\ndef forecast(jobs):"+source+"\nprint(json.dumps(forecast(json.load(sys.stdin))))"
    return json.loads(subprocess.check_output([sys.executable,'-I','-c',code,str(dest)],cwd='/tmp',input=json.dumps(jobs),text=True))


def main():
    frozen=read(OUT/'freeze.json')
    confirm=read(OUT.parent/'rms-confirmation/freeze.json')
    sources={}
    with tempfile.TemporaryDirectory(prefix='operator-sim-first-release-') as scratch:
        target=Path(scratch)
        codepaths=subprocess.check_output(['git','ls-tree','-r','--name-only',CHECKPOINT,'operator_model'],cwd=ROOT,text=True).splitlines()
        wanted=set(codepaths) | {p for p in frozen['refs'] if p.startswith(('results/','configs/'))}
        for name in sorted(wanted | frozen['refs'].keys()):
            current=ROOT/name
            if name in frozen['refs'] and digest(current)==frozen['refs'][name]:
                contents=current.read_bytes() if name in wanted else None
                sources[name]='working file'
            else:
                contents=subprocess.check_output(['git','show',f'{CHECKPOINT}:{name}'],cwd=ROOT)
                if name in frozen['refs']:
                    assert hashlib.sha256(contents).hexdigest()==frozen['refs'][name], name
                    sources[name]=CHECKPOINT
            if name in wanted:
                dest=target/name;dest.parent.mkdir(parents=True,exist_ok=True);dest.write_bytes(contents)
        baseline=first_model(confirm['jobs'],target)
        write(OUT.parent/'rms-confirmation/first-schedule-replay.json',dict(role='Unmodified first-freeze schedule replay on new second cohort',**baseline))
        n_old=check(forecast_old(frozen['jobs']),read(OUT/'v018-forecasts.json'))
        n_first=check(first_model(frozen['jobs'],target),read(OUT/'v019-forecasts.json'))
    n_final=check(forecast(confirm['jobs']),confirm['v019_final'])
    assert all(digest(ROOT/p)==v for p,v in confirm['refs'].items())
    n_previous_final=check(forecast_old(confirm['jobs']),confirm['v018'])
    wheel,=Path('/tmp/operator-sim-v18-wheel').glob('*.whl')
    assert digest(wheel)==frozen['prior_wheel_sha256']
    counts=dict(v018_first=n_old,v019_first=n_first,v018_confirmation=n_previous_final,v019_confirmation=n_final)
    write(OUT/'release-audit.json',dict(checkpoint=CHECKPOINT,files=sources,frozen_contracts_reproduced=counts,
          second_model_hashes=confirm['refs'],prior_wheel_sha256=digest(wheel),checked_fields='latency, status, reason, core counts and tiling. Elapsed query time remeasured.'))
    print(counts, len(sources),'input digests')

if __name__=='__main__':main()
