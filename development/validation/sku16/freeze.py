"""Hash all installed model resources and freeze all four comparisons before CA."""
import subprocess,sys,json
from time import perf_counter_ns
from pathlib import Path
from campaign import ROOT, OUT, RUNS, PR, DT, load, sha, utc, write
sys.path.insert(0,str(ROOT))
from operator_model import estimate, __version__
from dataclasses import replace
LEGACY_QUERY="""import dataclasses,json,operator_model,sys
assert operator_model.__version__=='0.15.0' and 'site-packages' in operator_model.__file__
p=operator_model.load_chip('Ascend950PR_957b')
anon=dataclasses.replace(p,name='resource-only-anonymous-PR')
rows=[]
for c in json.load(sys.stdin):
    record={}
    for name,chip in [('prior_pr',p),('legacy_resource',anon),('dt','Ascend950DT_9582')]:
        r=operator_model.estimate(c['op'],c['shape'],chip=chip)
        assert r.latency_us and r.latency_us>0
        record[name+'_us']=r.latency_us;record[name+'_aic']=r.active_aic;record[name+'_aiv']=r.active_aiv
    rows.append(record)
print(json.dumps(rows))
"""
def main(prior_python):
 if RUNS.exists() or __version__!='0.16.0':raise ValueError('Need v0.16 and no PR launch')
 budget=load(OUT/'budget.json')
 old=json.loads(subprocess.check_output([prior_python,'-I','-c',LEGACY_QUERY],input=json.dumps(budget['cases']),text=True,cwd='/tmp'))
 manifest=ROOT/'MANIFEST.in';files={manifest,OUT/'budget.json',ROOT/'scripts/cann-run.sh',ROOT/'build/smoke/three_ops'}
 files.update(ROOT/p.removeprefix('include ') for p in manifest.read_text().splitlines() if p.startswith('include '))
 files.update((ROOT/'operator_model').rglob('*.py'))
 files.update(Path(__file__).parent/p for p in ('campaign.py','collect.py','qualify.py','freeze.py'))
 before={str(p.relative_to(ROOT)):sha(p) for p in sorted(files)};rows=[]
 for c,previous in zip(budget['cases'],old):
    row={k:c[k] for k in ('id','op','shape','role','priority')};row.update(previous)
    for key,chip,mode in [('dt',DT,'auto'),('pr',PR,'resource'),('auto_pr',PR,'auto'),('calibrated_pr',PR,'calibrated')]:
        t=perf_counter_ns();r=estimate(c['op'],c['shape'],chip=chip,timing_mode=mode);ms=(perf_counter_ns()-t)/1e6
        if r.latency_us is None or r.latency_us<=0 or ms>=100:raise ValueError((key,c['id'],r,ms))
        if key!='pr':
            expected=(row['pr_us'] if key=='auto_pr' else previous['prior_pr_us'] if key=='calibrated_pr' else previous['dt_us'])
            if abs(r.latency_us-expected)>1e-8:raise ValueError(('unexpected auto drift',c['id'],key,r.latency_us,expected))
        row.update({key+'_us':r.latency_us,key+'_aic':r.active_aic,key+'_aiv':r.active_aiv,key+'_query_ms':ms})
        if key=='pr':row.update(work=r.work,projection=r.projection,rates=r.rates)
    rows.append(row)
 assert before=={p:sha(ROOT/p) for p in before}
 write(OUT/'freeze.json',dict(utc=utc(),version=__version__,sha256=before,predictions=rows,scope='pr_us: explicit v0.16 resource; legacy_resource_us: v0.15 renamed but otherwise identical PR; dt_us: DT forecast; prior_pr_us: v0.15 SKU-specific; auto_pr_us: verified same as pr_us; calibrated_pr_us: explicit calibrated equals prior_pr_us'),True)
 print('Frozen',len(rows),'PR cases including sequential reserves',flush=True)
if __name__=='__main__':main(sys.argv[1] if len(sys.argv)>1 else '/tmp/operator-sim-v15-deploy/bin/python')
