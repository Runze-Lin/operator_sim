"""Pre-register three small distinguishable FA geometries and qualify each complete CA."""
import argparse, fcntl, hashlib, json, os, re, sys
from pathlib import Path
HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[2]
OUT=ROOT/'results/v0.25/coverage'
RAW=OUT/'raw'
REG=OUT/'fa-freeze.json'
sys.path.insert(0,str(ROOT))
from operator_model import estimate
sys.path.insert(0,str(HERE.parent/'sku17'))
import campaign_collect as ca

DT,PR,FUT=ca.DT,ca.PR,'Ascend970DT'
SHAPES=((9,1008,64,1,1),(17,895,64,1,2),(33,319,128,1,1))


def register():
    if REG.exists():raise FileExistsError(REG)
    base=next(j for j in ca.read(ROOT/'results/v0.17/sku/core-freeze.json')['jobs'] if j['op']=='fa')
    old_shapes={tuple(r['shape']) for r in ca.read(ROOT/'results/v0.21/coverage/audit.json')['rows'] if r['op']=='fa'}
    jobs=[]
    for shape in SHAPES:
        assert shape not in old_shapes
        for sku in (DT,PR):
            ident='fa-'+sku+'-'+'-'.join(map(str,shape));assert not (RAW/ident).exists()
            job={**base,'id':ident,'chip':sku,'shape':list(shape),'args':['fa',*map(str,shape)],'wall_s':560,'raw_mib':900}
            forecast={}
            for target in (DT,PR,FUT):
                old=estimate('fa',shape,chip=target)
                new=estimate('fa',shape,chip=target,timing_mode='resource')
                forecast[target]=dict(old=old.latency_us,candidate=new.latency_us,active_aic=new.active_aic,active_aiv=new.active_aiv,plan=new.projection['target_plan'])
            job['outputs']=[dict(op='fa',shape=list(shape),attrs={},indices=[0])]
            job['forecasts']=forecast;jobs.append(job)
    names=[Path(__file__),ROOT/'operator_model/fa_resource.py',ROOT/'operator_model/backend.py',ROOT/'results/v0.16/fa.json',
           ROOT/'configs/chips'/f'{DT}.json',ROOT/'configs/chips'/f'{PR}.json',ROOT/'configs/chips'/f'{FUT}.json',
           ROOT/base['runner'],ROOT/'scripts/cann-run.sh',Path(ca.__file__)]
    date=ca.utc()
    reg=dict(utc=date,source_hashes={str(f.relative_to(ROOT)):ca.sha(f) for f in names},jobs=jobs,concurrency=1,
             normalized='aclnnFlashAttentionScoreV2, BF16 dense noncausal SBH Q/KV/D/B/H; one complete full kernel window',
             decision='Compare v0.24 auto and fixed current target-resource path for every shape; no new prices; all prior windows replay.',
             total_ca_wall_s=3000)
    with REG.open('x') as f:json.dump(reg,f,indent=2);f.write('\n')
    return reg


def run():
    r=ca.read(REG)
    # Backend will start routing to the frozen resource function when it passes.
    # Other source and collection hashes must remain byte-identical.
    for n,h in r['source_hashes'].items():
        if n!='operator_model/backend.py' and ca.sha(ROOT/n)!=h:raise ValueError(n)
    RAW.mkdir(exist_ok=True);os.environ.update(OMP_WAIT_POLICY='PASSIVE',GOMP_SPINCOUNT='0')
    with (RAW/'lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        used=0
        for j in r['jobs']:
            d=RAW/j['id'];record=OUT/(j['id']+'.json')
            if record.exists():used+=ca.read(record)['cost']['seconds'];continue
            if used>=r['total_ca_wall_s']:break
            ca.write(OUT/'fa-state.json',dict(state='running',id=j['id'],utc=ca.utc()))
            if not (d/'cost.json').exists():
                if d.exists():raise FileExistsError('Interrupted raw attempt '+str(d))
                ca.capture({**j,'wall_s':min(j['wall_s'],r['total_ca_wall_s']-used)},j['chip'],d,r)
            cost=ca.read(d/'cost.json');used+=cost['seconds']
            try:
                result,=ca.qualify(j,j['chip'],d);s=result['stages'][0]
                assert s['kernel'].startswith('FlashAttentionScore_') and s['mmad']>0
                assert len(result['correctness'])==1 and re.search(r'\belements='+str(j['shape'][0]*j['shape'][2]*j['shape'][3]*j['shape'][4])+r'\b',result['correctness'][0])
                result.update(qualified=True,registration_sha256=ca.sha(REG),role='v0.25 prospective')
            except (AssertionError,ValueError,KeyError,IndexError,TypeError,OSError) as e:
                result=dict(qualified=False,reason=repr(e),shape=j['shape'],chip=j['chip'],cost=cost,registration_sha256=ca.sha(REG))
            ca.write(record,result)
        ca.write(OUT/'fa-state.json',dict(state='finished',cost_s=used,utc=ca.utc()))
if __name__=='__main__':
    {'freeze':register,'run':run}[sys.argv[1]]()
