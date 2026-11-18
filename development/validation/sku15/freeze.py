"""Freeze public estimates for every preregistered PR shape before first collection."""
from pathlib import Path
import subprocess
import sys
from time import perf_counter_ns
from campaign import ROOT, OUT, RUNS, PR, DT, load, sha, utc, write
sys.path.insert(0,str(ROOT))
from operator_model import estimate, __version__


def main(prior_python):
    if RUNS.exists() or __version__!='0.15.0':
        raise ValueError('Freeze requires the new release and no PR campaign output')
    budget=load(OUT/'budget.json')
    prior_query="""import json,sys,operator_model
assert operator_model.__version__=='0.14.0' and 'site-packages' in operator_model.__file__
rows=json.load(sys.stdin)
print(json.dumps([operator_model.estimate(c['op'],c['shape'],chip='Ascend950PR_957b').latency_us for c in rows]))"""
    import json
    prior=json.loads(subprocess.check_output([prior_python,'-I','-c',prior_query],input=json.dumps(budget['cases']),text=True,cwd='/tmp'))
    manifest=ROOT/'MANIFEST.in'; files={manifest,OUT/'budget.json',ROOT/'scripts/cann-run.sh'}
    files.update(ROOT/p.removeprefix('include ') for p in manifest.read_text().splitlines() if p.startswith('include '))
    files.update((ROOT/'operator_model').rglob('*.py'))
    files.update(ROOT/r['runner'] for r in budget['cases'])
    files.update([OUT/'preflight.json',OUT/'preflight-gate.json',ROOT/'build/capture/libpr15.so'])
    files.update(Path(__file__).parent/p for p in ('campaign.py','collect.py','qualify.py','freeze.py','native.sh','pr_adapter.cpp'))
    before={str(p.relative_to(ROOT)):sha(p) for p in sorted(files)}; rows=[]
    for c, old in zip(budget['cases'],prior):
        row={k:c[k] for k in ('id','op','shape','role','priority')};row['prior_pr_us']=old
        assert old and old>0
        for key,chip in [('dt',DT),('pr',PR)]:
            t=perf_counter_ns(); r=estimate(c['op'],c['shape'],chip=chip); wall=(perf_counter_ns()-t)/1e6
            assert r.latency_us and r.status==('estimated' if key=='dt' else 'extrapolated') and wall<100
            row.update({key+'_us':r.latency_us,key+'_aic':r.active_aic,key+'_aiv':r.active_aiv,key+'_query_ms':wall})
            if key=='pr': row.update(work=r.work,projection=r.projection,rates=r.rates)
        rows.append(row)
    assert before=={k:sha(ROOT/k) for k in before}
    write(OUT/'freeze.json',dict(utc=utc(),version=__version__,sha256=before,predictions=rows),True)
    print('Frozen',len(rows),'shape contracts',flush=True)

if __name__=='__main__':
    main(sys.argv[1] if len(sys.argv)>1 else '/tmp/operator-sim-v14-deploy/bin/python')
