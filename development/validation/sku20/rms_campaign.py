"""Freeze new RMS widths and measure the exact installed kernel on both 950 SKUs."""
from concurrent.futures import ThreadPoolExecutor
import inspect
import json
import os
from pathlib import Path
import subprocess
import sys
from threading import Lock

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
from development.validation.sku19.campaign import CHIPS, DT, PR, digest, forecast, now, read, records, write
from development.validation.sku19.collect import capture, qualify, make_pair

OUT = ROOT / 'results/v0.20/rms'
RAW = ROOT / 'runs/sku20-rms'
RUNNER = 'build/smoke/large_vector'
# Token counts span a small wave, realistic chunks, and long repeated input tiles.
SHAPES = [(32,1280),(96,1536),(512,1280),(768,1536),(1024,5120),
          (1536,7168),(2048,1280),(4096,1536),(8192,5120),(32768,1280)]
OLD = '/tmp/operator-sim-v19-deploy/bin/python'


def freeze():
    if (OUT/'freeze.json').exists() or RAW.exists():
        raise RuntimeError('Freeze already exists or capture has begun')
    known = {tuple(s) for s in read(ROOT/'results/v0.18/sku/excluded-history.json')['shapes']['rmsnorm']}
    for ver in ('v0.18','v0.19'):
        for path in (ROOT/'results'/ver).rglob('*.json'):
            known.update(shape for op,shape in records(read(path)) if op == 'rmsnorm')
    assert not known.intersection(SHAPES)
    host_sha = digest(ROOT/RUNNER)
    jobs = []
    for t,h in SHAPES:
        jobs.append(dict(id=f'rmsnorm_{t}_{h}', op='rmsnorm', shape=[t,h], dtype='bf16',
            attrs={}, args=['rmsnorm',str(t),str(h)], runner=RUNNER,runner_sha256=host_sha,
            role='holdout', regime='small' if t<=96 else 'multiwave', batch=None, kernels=1,
            nchecks=2, wall_s=14400,raw_mib=max(1024,int(t*h*1.45/1024))))
    code = ('import json,sys\nfrom time import perf_counter_ns\nCHIPS='+repr(CHIPS)+'\n'+
            inspect.getsource(forecast)+'\nprint(json.dumps(forecast(json.load(sys.stdin))))')
    old = json.loads(subprocess.check_output([OLD,'-I','-c',code],cwd='/tmp',input=json.dumps(jobs),text=True))
    new = forecast(jobs)
    assert (old['version'], new['version']) == ('0.19.0','0.20.0')
    for job,before,after in zip(jobs,old['rows'],new['rows']):
        assert job['id'] == before['id'] == after['id']
        output = dict(op='rmsnorm',shape=job['shape'],attrs={},indices=[0])
        for side in CHIPS:
            for pref,row in [('',after),('old_',before)]:
                val = row['targets'][side]
                assert val['us'] and val['query_ms'] < 100
                for k in ('us','status','aiv','aic','query_ms'):
                    output[pref+side+'_'+k] = val[k]
        job['outputs'] = [output]
    files = [Path(__file__), ROOT/RUNNER,ROOT/'smoke/large_vector.cpp',ROOT/'operator_model/chips.py',
             *(ROOT/'operator_model').glob('rms*.py'),*(ROOT/'configs/chips').glob('*.json')]
    write(OUT/'freeze.json',dict(utc=now(),jobs=jobs,known_shapes=len(known),concurrency=2,
        maximum_combined_file_mib=sum(2*j['raw_mib'] for j in jobs),refs={str(p.relative_to(ROOT)):digest(p) for p in files},
        v019=old,v020=new,policy='All-core msopprof dump=false, independent paired RMSNorm. Maximum 4 h per endpoint. 970DT projection only.'))


def run():
    frozen = read(OUT/'freeze.json')
    assert all(digest(ROOT/k)==v for k,v in frozen['refs'].items())
    os.environ.update(OMP_WAIT_POLICY='PASSIVE',GOMP_SPINCOUNT='0')
    lock = Lock()
    def collect(item):
        j,chip=item
        folder=RAW/(j['id']+'_'+chip)
        if (folder/'outcome.json').exists():
            return
        cost=read(folder/'cost.json') if (folder/'cost.json').exists() else capture(j,chip,folder)
        result=dict(id=j['id'], chip=chip, shape=j['shape'], **cost)
        try:
            rec,=qualify(j,chip,folder)
            assert cost['started_utc']>frozen['utc']
            rec.update(role='v0.20 frozen RMS holdout',dtype='bf16',runner_sha256=j['runner_sha256'])
            write(folder/'qualified.json',rec)
            result.update(eligible=True,reason=None)
        except (ValueError,AssertionError,KeyError,IndexError,OSError) as exc:
            result.update(eligible=False,reason=repr(exc))
        with lock:
            write(folder/'outcome.json',result)
            publish(frozen)
            print(j['id'],chip,result['reason'] or 'qualified',flush=True)
    jobs = frozen['jobs']
    order = [jobs[i] for i in (0,4,1,5,2,8,3,6,7,9)]
    with ThreadPoolExecutor(max_workers=frozen['concurrency']) as pool:
        list(pool.map(collect,((j,c) for j in order for c in (DT,PR))))


def publish(frozen):
    endpoints=[]; pairs=[]; costs=[]
    for j in frozen['jobs']:
        measured=[]
        for chip in (DT,PR):
            folder=RAW/(j['id']+'_'+chip)
            if (folder/'outcome.json').exists():costs.append(read(folder/'outcome.json'))
            if (folder/'qualified.json').exists():
                rec=read(folder/'qualified.json');endpoints.append(rec);measured.append(rec)
        if len(measured)==2:pairs.append(make_pair(j,*measured))
    for name,data in [('records',endpoints),('pairs',pairs),('costs',costs)]:
        write(OUT/(name+'.json'),data)

if __name__=='__main__':
    {'freeze':freeze,'run':run,'publish':lambda:publish(read(OUT/'freeze.json'))}[sys.argv[1]]()
