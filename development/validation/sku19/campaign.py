"""Register unseen contracts, then lock predictions from two independently loaded releases."""
from datetime import datetime, timezone
import json
from pathlib import Path
import subprocess
import sys
from time import perf_counter_ns

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
from development.validation.sku18.register import digest, read, records, write

OUT = ROOT / 'results/v0.19/sku'
RAW = ROOT / 'runs/sku19'
DT, PR, FUTURE = 'Ascend950DT_9582', 'Ascend950PR_957b', 'Ascend970DT'
CHIPS = {'dt': DT, 'pr': PR, 'future': FUTURE}
OLD_PYTHON = Path('/tmp/operator-sim-v18-deploy/bin/python')
DESIGN = {
    'rmsnorm': [(2,5952),(9,7488),(23,6720),(54,1984),(58,2176),(67,2432),
                (111,3264),(209,3712),(408,4416),(507,7168)],
    'quant': [(2,832),(4,1472),(11,3136),(26,704),(53,1984),(59,2880),
              (97,2752),(257,4416),(641,5120),(1023,7168)]}
CHUNKS = [('rmsnorm',2048,5120),('quant',2048,5120)]
HOST = {'quant':'build/smoke/three_ops','rmsnorm':'build/smoke/extension_norm_addmm'}


def now():
    return datetime.now(timezone.utc).isoformat()


def register():
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT/'registration.json'
    if path.exists():
        raise RuntimeError('Registration already exists')
    old = read(ROOT/'results/v0.18/sku/excluded-history.json')['shapes']
    known = {(op,tuple(shape)) for op,shapes in old.items() for shape in shapes}
    for f in (ROOT/'results/v0.18').rglob('*.json'):
        known.update(records(read(f)))
    # The new manifest covers both released shapes and unqualified historical attempts.
    jobs = []
    for op,rows,cols,role in ([(op,r,c,'holdout') for op,pairs in DESIGN.items() for r,c in pairs]
                              + [(op,r,c,'prefill_chunk') for op,r,c in CHUNKS]):
        assert role != 'holdout' or (op,(rows,cols)) not in known, (op,rows,cols)
        n=rows*cols
        group='small' if rows<32 else 'boundary' if rows<=128 else 'large'
        runner=HOST[op]
        jobs.append(dict(id=f'{op}_{rows}_{cols}',op=op,shape=[rows,cols],attrs={},dtype='bf16',regime=group,
                         role=role,runner=runner,runner_sha256=digest(ROOT/runner),args=[op,str(rows),str(cols)],
                         kernels=1,nchecks=2 if op=='rmsnorm' else 1,batch=None,
                         wall_s=1500 if role=='prefill_chunk' else 900 if n>3000000 else 330,
                         raw_mib=16384 if role=='prefill_chunk' else 10240 if n>3000000 else 4096))
    write(path,dict(utc=now(),old_version='0.18.0',new_version='0.19.0',jobs=jobs,
                    history_source='v0.18 excluded-history + all v0.18 released JSON including failed attempts',
                    history_counts={o:sum(k==o for k,_ in known) for o in DESIGN},
                    concurrency=4,max_launches=2*len(jobs),max_retained_gib=72,total_wall_s=7200,
                    policy='Isolated first call without residency hint. Both ends official all-core dump=false. 10 matched holdouts per operator; two extra 2048-token whole-kernel chunks reported separately. 970DT has no CA.'))
    print(len(jobs),'registered')


def forecast(jobs):
    from operator_model import __version__, estimate
    out=[]
    for job in jobs:
        row={}
        for side,chip in CHIPS.items():
            t=perf_counter_ns()
            e=estimate(job['op'],job['shape'],chip=chip,dtype='bf16',timing_mode='resource')
            ms=(perf_counter_ns()-t)/1e6
            row[side]=dict(us=e.latency_us,status=e.status,reason=e.reason,aiv=e.active_aiv,
                           aic=e.active_aic,query_ms=ms,plan=(e.projection or {}).get('target_plan'))
        out.append(dict(id=job['id'],targets=row))
    return dict(version=__version__,rows=out)


def forecast_old(jobs):
    # With -I no project sources or user site are on the old interpreter's import path.
    script=Path(__file__).read_text().split('def forecast(jobs):')[1].split('def forecast_old')[0]
    code="import json,sys\nfrom time import perf_counter_ns\nCHIPS="+repr(CHIPS)+"\ndef forecast(jobs):"+script+"\nprint(json.dumps(forecast(json.load(sys.stdin))))"
    result=json.loads(subprocess.check_output([str(OLD_PYTHON),'-I','-c',code],cwd='/tmp',text=True,input=json.dumps(jobs)))
    assert result['version']=='0.18.0'
    return result


def freeze():
    reg=read(OUT/'registration.json')
    if (OUT/'freeze.json').exists() or RAW.exists():
        raise RuntimeError('Campaign already frozen/open')
    jobs=reg['jobs']
    old=forecast_old(jobs)
    new=forecast(jobs)
    assert new['version']=='0.19.0'
    for j,a,b in zip(jobs,old['rows'],new['rows']):
        assert j['id']==a['id']==b['id']
        o=dict(op=j['op'],shape=j['shape'],attrs={},indices=[0])
        for side in CHIPS:
            for prefix,row in [('',b),('old_',a)]:
                val=row['targets'][side]
                if not (prefix and j['role']!='holdout' and j['op']=='rmsnorm'):
                    assert val['us'] and val['query_ms']<100, (j,side,row)
                for key in ('us','aic','aiv','status','query_ms'):
                    o[prefix+side+'_'+key]=val[key]
        j['outputs']=[o]
    for tag,obj in [('v018',old),('v019',new)]:write(OUT/(tag+'-forecasts.json'),obj)
    refs=[Path(__file__),Path(__file__).with_name('collect.py'),ROOT/'scripts/cann-run.sh',ROOT/'development/validation/sku17/campaign_collect.py',ROOT/'development/validation/sku17/campaign.py',ROOT/'development/validation/sku18/register.py',ROOT/'pyproject.toml']
    refs.extend(ROOT/j['runner'] for j in jobs)
    refs.extend(ROOT/'operator_model'/n for n in ('__init__.py','backend.py','chips.py','predictor.py','workload.py','model.py','target_tiling.py','vector_v8.py','wide_v10.py'))
    for prefix in ('rms','quant'):
        refs.extend((ROOT/'operator_model').glob(prefix+'*.py'))
    refs.extend((ROOT/'configs/chips').glob('*.json'))
    refs.extend(ROOT/l.split(' ',1)[1] for l in (ROOT/'MANIFEST.in').read_text().splitlines() if l.startswith('include ') and 'router' not in l)
    for path in ('v018-forecasts.json','v019-forecasts.json','registration.json'):refs.append(OUT/path)
    wheel,=Path('/tmp/operator-sim-v18-wheel').glob('*.whl')
    write(OUT/'freeze.json',dict(utc=now(),refs={str(p.relative_to(ROOT)):digest(p) for p in sorted(set(refs))},
                                prior_wheel_sha256=digest(wheel),jobs=jobs))
    print('Frozen',len(jobs),'contracts')


if __name__=='__main__':
    {'register':register,'freeze':freeze}[sys.argv[1]]()
