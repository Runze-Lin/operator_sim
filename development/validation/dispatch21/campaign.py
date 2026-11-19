"""Freeze, then qualify matched one-dispatch Addmm and mHC Post on two 950 SKUs."""
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from time import perf_counter_ns
import json
import os
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
from development.validation.sku19.campaign import CHIPS, digest, now, read, write
from development.validation.sku19.collect import capture, qualify
from operator_model import estimate

OUT = ROOT/'results/v0.21/dispatch'
RAW = ROOT/'runs/dispatch21'
CASES = {
    'addmm': [(17,80,96),(272,480,192),(511,384,192),(1,48,80),(31,144,160),
              (96,176,320),(192,384,192),(257,256,192),(384,480,256),(1024,192,256)],
    'mhc_post': [(5,4,288),(37,4,144),(7,4,1728),(15,4,240),(29,4,640),
                 (55,4,352),(71,4,1280),(129,4,1664),(191,4,2112),(253,4,2048)],
}


def prior_shapes():
    known = {op:set() for op in CASES}
    def collect(data):
        if isinstance(data,dict):
            op,shape = data.get('op'),data.get('shape')
            if isinstance(op,str) and op in known and isinstance(shape,list) and all(type(v) is int for v in shape):
                known[op].add(tuple(shape))
            for item in data.values():
                collect(item)
        elif isinstance(data,list):
            for item in data:
                collect(item)
    for p in (ROOT/'results').glob('v*/evaluation.json'):
        if p.parent.name != 'v0.21':
            collect(read(p))
    for p in (ROOT/'results').glob('v*/**/*freeze.json'):
        if 'v0.21' not in p.parts:
            collect(read(p))
    return known


def freeze():
    path = OUT/'freeze.json'
    if path.exists() or RAW.exists():
        raise RuntimeError('Predictions or captures already exist')
    known = prior_shapes()
    jobs = []
    for idx in range(10):
        for op,shapes in CASES.items():
            shape = shapes[idx]
            if shape in known[op]:
                raise ValueError(f'Previously registered {op} {shape}')
            post = op == 'mhc_post'
            attrs = {'has_h_res':True} if post else {'self_shape':[shape[1]],'alpha':1,'beta':1}
            runner = 'build/smoke/' + ('extension_indexer_mhc' if post else 'extension_norm_addmm')
            job = dict(id=op+'_'+'_'.join(map(str,shape)),op=op,shape=shape,attrs=attrs,runner=runner,
                       runner_sha256=digest(ROOT/runner),args=[op,*map(str,shape),*(() if post else ('bias',))],
                       nchecks=1,kernels=1,wall_s=1800,raw_mib=2200)
            out = dict(op=op,shape=shape,attrs=attrs,indices=[0])
            for side,chip in CHIPS.items():
                start = perf_counter_ns()
                est = estimate(op,shape,attrs,chip=chip)
                ms = (perf_counter_ns()-start)/1e6
                assert est.latency_us and ms < 100,(op,shape,est.reason,ms)
                out[side+'_us'] = est.latency_us
                out[side+'_ms'] = ms
                out[side+'_plan'] = est.projection['target_plan']
                if post:
                    out[side+'_first_us'] = estimate(op,shape,{**attrs,'resident_aiv':0},chip=chip).latency_us
            job['outputs'] = [out]
            jobs.append(job)
    baseline = """import sys,json
from operator_model import estimate,__version__
assert __version__=='0.20.0'
js,cs=json.load(sys.stdin)
print(json.dumps([{s:estimate(j['op'],j['shape'],j['attrs'],chip=c).latency_us for s,c in cs.items()} for j in js]))"""
    old = json.loads(subprocess.check_output(['/tmp/operator-sim-v20-deploy/bin/python','-I','-c',baseline],
                                            cwd='/tmp',text=True,input=json.dumps([jobs,CHIPS])))
    for j,values in zip(jobs,old):
        for side,x in values.items():
            j['outputs'][0]['old_'+side+'_us'] = x
    files = [ROOT/'operator_model'/f for f in ('post_resource.py','post_v12.py','matmul_v15.py','matmul_waves.py','matmul_resource.py')]
    files.extend((ROOT/'configs/chips').glob('*.json'))
    files.extend([ROOT/'results/v0.12/post/profile.json',ROOT/'results/v0.16/matmul.json',Path(__file__)])
    write(path, dict(utc=now(), jobs=jobs, seen={k:len(v) for k,v in known.items()},
                     files={str(p.relative_to(ROOT)):digest(p) for p in files}, workers=2,
                     boundary='Full all-core instruction window. Primary estimate omits residency; separately preregister first-launch scenario.'))


def run():
    frozen = read(OUT/'freeze.json')
    for name,sha in frozen['files'].items():
        if digest(ROOT/name)!=sha:
            raise ValueError(f'Changed frozen dependency: {name}')
    os.environ.update(OMP_WAIT_POLICY='PASSIVE',GOMP_SPINCOUNT='0')
    receipts = OUT/'receipts'
    receipts.mkdir(exist_ok=True)
    def one(item):
        j,side = item
        chip = CHIPS[side]
        folder = RAW/(j['id']+'_'+chip)
        path = receipts/(folder.name+'.json')
        if path.exists():
            return
        assert digest(ROOT/j['runner']) == j['runner_sha256']
        cost = read(folder/'cost.json') if (folder/'cost.json').exists() else capture(j,chip,folder)
        if cost['started_utc'] <= frozen['utc']:
            raise RuntimeError('Run predates freeze')
        try:
            obs, = qualify(j,chip,folder)
            result = dict(qualified=True,observation=obs)
        except (ValueError, OSError, KeyError, IndexError) as exc:
            result = dict(qualified=False,error=str(exc),cost=cost)
        write(path,dict(id=j['id'],op=j['op'],shape=j['shape'],chip=chip,**result))
        print(path.stem,result['qualified'],flush=True)
    with ThreadPoolExecutor(max_workers=frozen['workers']) as pool:
        list(pool.map(one, ((j,s) for j in frozen['jobs'] for s in ('dt','pr'))))


if __name__=='__main__':
    {'freeze':freeze,'run':run}[sys.argv[1]]()
