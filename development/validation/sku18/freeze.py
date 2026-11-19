"""Freeze the independent predictions and artifact hashes before invoking either CA simulator."""
from copy import deepcopy
import json
from pathlib import Path
import subprocess
import sys
from register import OUT, RAW, ROOT, digest, read, utc, write

SCRIPT=Path(__file__).with_name('forecast.py')
OLD_PYTHON=Path('/tmp/operator-sim-v17-deploy/bin/python')


def predict(which,jobs):
    py = OLD_PYTHON if which=='prior' else Path(sys.executable)
    out = subprocess.check_output([str(py),'-I',str(SCRIPT),which],cwd='/tmp',
                                  input=json.dumps(jobs),text=True)
    return json.loads(out)


def prior():
    path=OUT/'v017-forecasts.json'
    if path.exists():return read(path)
    reg=read(OUT/'registration.json')
    forecasts=predict('prior',reg['jobs'])
    wheel,=Path('/tmp/operator-sim-v17-wheel').glob('*0.17.0*.whl')
    forecasts.update(utc=utc(),wheel_sha256=digest(wheel),wheel_name=wheel.name,
                     registration_sha256=digest(OUT/'registration.json'))
    write(path,forecasts)
    return forecasts


def main():
    reg=read(OUT/'registration.json')
    old=prior()
    if sys.argv[1]=='prior':return
    if sys.argv[1]!='freeze':raise ValueError('expected prior or freeze')
    target=OUT/'freeze.json'
    assert not target.exists() and not RAW.exists(), 'Freeze cannot replace an opened campaign'
    new=predict('source',reg['jobs'])
    refs={ROOT/'MANIFEST.in',ROOT/'pyproject.toml',SCRIPT,Path(__file__).resolve(),ROOT/'scripts/cann-run.sh'}
    refs.update((ROOT/'operator_model').rglob('*.py'))
    refs.update(ROOT/line.split(maxsplit=1)[1] for line in (ROOT/'MANIFEST.in').read_text().splitlines() if line.startswith('include '))
    refs.update(ROOT/j['runner'] for j in reg['jobs'])
    refs.update((Path(__file__).with_name(n) for n in ('register.py','collect.py')))
    refs.update((ROOT/'development/validation/sku17'/n for n in ('campaign.py','campaign_collect.py')))
    jobs=deepcopy(reg['jobs']);query_walls=[]
    for j,new_row,old_row in zip(jobs,new['cases'],old['cases']):
        assert j['id']==new_row['id']==old_row['id']
        o=dict(op=j['op'],shape=j['shape'],attrs=j['attrs'],indices=[0]);j['outputs']=[o]
        for alias in ('dt','pr','future'):
            current=new_row['targets'][alias];prev=old_row['targets'][alias]
            for field,key in [('latency_us','us'),('active_aic','aic'),('active_aiv','aiv'),('status','status'),('query_ms','query_ms')]:
                o[f'{alias}_{key}']=current[field];o[f'old_{alias}_{key}']=prev[field]
            query_walls.append(dict(op=j['op'],shape=j['shape'],chip=current['chip'],mode=current['mode'],query_ms=current['query_ms']))
    write(OUT/'v018-forecasts.json',new)
    write(OUT/'query-wall.json',dict(import_ms=new['import_ms'],scope='first prediction for each unique case per chip in one process; includes first uncached call',rows=query_walls))
    write(target,dict(utc=utc(),registration_sha256=digest(OUT/'registration.json'),versions={'old':old['version'],'new':new['version']},
        refs={str(p.relative_to(ROOT)):digest(p) for p in sorted(refs)},old_forecasts_sha256=digest(OUT/'v017-forecasts.json'),
        new_forecasts_sha256=digest(OUT/'v018-forecasts.json'),jobs=jobs))
    print('Frozen',len(jobs),'cases; peak query wall ms',max(r['query_ms']for r in query_walls))

if __name__=='__main__':main()
