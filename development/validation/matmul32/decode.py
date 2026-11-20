"""Application decode roster, endpoint deduplication, and three queued new controls."""
import fcntl,gzip,json,os,re,shutil,sys,time
from pathlib import Path
from statistics import median
from freeze import ROOT, OUT, DT, PR, FUT,digest
from forward import LOCK,ca,recover
from operator_model import estimate,load_chip
from operator_model.matmul_dispatch31 import resource_matmul_dispatch as old

TARGET=OUT/'decode';REG=TARGET/'freeze.json';RAW=ROOT/'runs/matmul32/decode'
SETS=[('DeepSeek-V4.1-Flash','indexer.wk','dsa.wq_a',81),('DeepSeek-V4-Pro','indexer.weights_proj','dsa.wq_a',16),('Kimi-K3','mla.q_b_proj','mla.kv_b_proj',9)]
def register():
    regs=ca.read(OUT/'freeze.json')
    assert all(digest(ROOT/s)==h for s,h in regs['sha256'].items())
    with gzip.open(ca.CATALOG,'rt') as src:
        nodes={}
        for line in src:
            r=json.loads(line);v=r['scenario']
            if (v['batch'],v['tp'],v['seq'],v['dp'],v['step'],v['chunk'])!=(8,8,32768,1,0,0):continue
            for c in r['contracts']:
                wt=c['tensors'].get('weight_logical')
                if wt and len(wt)==2:nodes[v['model'],c['op']]=wt
    locs=sorted({str(p.relative_to(ROOT)) for v in ('v0.28/matmul','v0.29/matmul','v0.30/matmul-dispatch/device','v0.31/matmul/device','v0.31/matmul/kimi-prefill','v0.32/matmul') for p in (ROOT/'results'/v).rglob('*freeze.json') if 'raw' not in p.parts})
    def shapes(x):
        if isinstance(x,list):
            for r in x:yield from shapes(r)
        if isinstance(x,dict):
            if isinstance(x.get('shape'),list):yield tuple(x['shape'])
            for v in x.values():yield from shapes(v)
    reserved={v for s in locs for v in shapes(ca.read(ROOT/s))};seen={(tuple(r['shape']),r['chip']) for r in ca.read(OUT/'score.json')['rows']}
    # Also find previously started attempts with no qualified endpoint and no release record.
    for raw in (ROOT/'runs',ROOT/'results/v0.29/matmul/decode/raw'):
        for folder,sub,files in os.walk(raw):
            sub[:]=[d for d in sub if d not in ('simulator','ca_workspace','record','profile','log','log_ca','dump','npusim','sources') and not d.startswith(('OPPROF_','PROF_'))]
            if 'run.log' not in files:continue
            with (Path(folder)/'run.log').open(errors='replace') as f:
                for ln in f:
                    sig=re.search(r'case=matmul api=aclnnMatmul M=(\d+) N=(\d+) K=(\d+)',ln)
                    if sig:reserved.add(tuple(map(int,sig.groups())));break
    rows=[];jobs=[]
    for model,primary,extra,newm in SETS:
        for node in (primary,extra):
            n,k=nodes[model,node]
            for m in sorted({b*q for b in (1,9,11,16) for q in (1,9)}):
                shape=(m,n,k);pred={}
                for chip in (DT,PR,FUT):
                    v=estimate('matmul',shape,chip=chip);p=v.projection['target_plan']
                    pred[chip]=dict(us=v.latency_us,old_us=old(shape,load_chip(chip))[0],aic=v.active_aic,aiv=v.active_aiv,mmads=p['mmads_total'],branch=p['branch'])
                rec=dict(model=model,node=node,shape=list(shape),role='primary' if node==primary else 'alternate real projection',scenarios=[dict(batch=b,Q=q,proposed=q-1) for b in (1,9,11,16) for q in (1,9) if b*q==m],predictions=pred)
                rows.append(rec)
                if node==extra and m==newm:
                    if shape in reserved or any((shape,chip) in seen for chip in (DT,PR)):raise ValueError(f'Duplicate {shape}')
                    jobs.append(dict(rec,wall_s=3600,raw_mib=4096))
    implementation=ca.read(OUT/'kimi-head/freeze.json')
    deps=['development/validation/matmul32/decode.py','development/validation/matmul28/campaign.py','development/validation/matmul28/drain.py','build/smoke/matmul28',str(ca.CATALOG.relative_to(ROOT)),*regs['sha256']]
    TARGET.mkdir(parents=True,exist_ok=True)
    with REG.open('x') as f:json.dump(dict(utc=ca.now(),application_rows=rows,jobs=jobs,installed_matmul_elf=implementation['installed_matmul_elf'],sources=locs,dependencies={s:digest(ROOT/s) for s in deps},protocol='TP8/BF16 contiguous ND, no bias. One retired decode CA slot shared with the larger Kimi head; DT then PR once per new geometry. An already qualified observation is never rerun. Hidden-direction dispatch or quantized vLLM graphs are separate.'),f,ensure_ascii=False,indent=2);f.write('\n')

def run():
    r=ca.read(REG);assert all(digest(ROOT/s)==h for s,h in r['dependencies'].items())
    RAW.mkdir(parents=True,exist_ok=True);os.nice(12);os.environ.update(OMP_WAIT_POLICY='PASSIVE',GOMP_SPINCOUNT='0')
    def state(phase,**kw):ca.save(TARGET/'state.json',dict(utc=ca.now(),pid=os.getpid(),phase=phase,**kw))
    state('queued behind active Kimi head pair',current_holder='results/v0.32/matmul/kimi-head/state.json')
    with LOCK.open('a') as fd:
        fcntl.flock(fd,fcntl.LOCK_EX)
        for j in r['jobs']:
            for sku,tag in ((DT,'dt'),(PR,'pr')):
                stem='-'.join(map(str,j['shape']))+'-'+tag;raw=RAW/stem;dest=TARGET/(stem+'.json')
                if dest.exists():continue
                if raw.exists():state('incomplete run retained, no retry',case=stem);return
                while shutil.disk_usage(ROOT).free<24*2**30:state('waiting for free space',case=stem);time.sleep(30)
                raw.mkdir();state('acquiring',case=stem);ca.measure(j,sku,raw);cost=ca.read(raw/'cost.json')
                try:
                    if cost['exit_code'] or cost['stop_reason']:raise ValueError('budget/process failure')
                    try:d=ca.qualify(j,sku,raw)
                    except (ValueError,KeyError,AssertionError,IndexError,OSError) as exc:
                        p=TARGET/(stem+'-initial.json');d=dict(qualified=False,ca_us=None,shape=j['shape'],chip=sku,case=p.stem,reason=repr(exc));ca.save(p,d);d=recover(j,d,raw,acquisition=p) or d
                    if d['qualified']:
                        s,=d['stages'];assert s['binary_sha256']==r['installed_matmul_elf']
                        p=j['predictions'][sku];d['mechanism_checks']=dict(aic=s['active_aic']==p['aic'],aiv=s['active_aiv']==p['aiv'],mmad=s['mmad']==p['mmads'])
                except (ValueError,KeyError,AssertionError,IndexError,OSError) as exc:d=dict(qualified=False,ca_us=None,reason=repr(exc))
                ca.save(dest,dict(d,shape=j['shape'],chip=sku,cost=cost,freeze_sha256=digest(REG)))
        state('finished')

if __name__=='__main__':{'freeze':register,'run':run}[sys.argv[1]]()
