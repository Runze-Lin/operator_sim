"""Inspect real stage boundaries on the latest-returning AIC and AIV of each archived launch."""
from pathlib import Path
import hashlib
import json

ROOT=Path(__file__).resolve().parents[3]
OUT=ROOT/'results/v0.26/presink'
OPER='mhc_pre_sinkhorn'

def load(path):return json.loads(path.read_bytes())

def archive():
    r=load(ROOT/'results/v0.25/coverage/audit.json')['rows']
    result=[]
    for x in r:
        if x['op']!=OPER:continue
        t,n,d=x['shape'];chip=x['chip'];trace=None
        k=ROOT/'results/v0.22/recovered'/f'{OPER}_{t}_{n}_{d}_{chip}.json'
        if k.exists() and load(k)['qualified']:
            obs=load(k)['observation'];trace=obs['trace'];w={c:[(z-obs['first_tick'])/1650 for z in ab] for c,ab in obs['windows_ticks'].items()}
        else:
            raw=ROOT/'results/v0.21/fusion'
            for folder in ('receipts','retries'):
                path=raw/folder/k.name
                if path.exists() and load(path)['qualified']:
                    k=path;obs=load(k)['kernel'];trace=obs['trace'];w=obs['windows'];break
        if not trace:
            k=ROOT/'results/v0.17/sku/records.json'
            for row in load(k):
                if row.get('stages') and row['op']==OPER and row['shape']==x['shape'] and row['chip']==chip and row['trace_sha256']==x['trace_sha256']:
                    obs=row['stages'][0];trace=obs['trace'];w=obs['windows'];break
        if not trace or obs['trace_sha256'] not in x['trace_sha256']:continue
        roles={}
        for kind in ('cube','vec'):
            c=max((c for c in w if kind in c),key=lambda c:w[c][1]); shard=(ROOT/trace).parent/c/'trace.json'
            events=load(shard)['traceEvents'];run=[e for e in events if e['ph']=='X']
            origin=min(e['ts'] for e in run)-w[c][0]
            phases={}
            for e in run:
                name=e['name']
                if name not in ('VF','FIX_L0C_TO_DST','MMAD','SET_CROSS_CORE','WAIT_FLAG_DEVI','END'):continue
                pc=e['args']['pc_addr'];q=phases.setdefault(name,[])
                a=e['ts']-origin;b=a+e['dur']
                detail=json.loads(e['args'].get('detail','{}'))
                q.append({'pc':pc,'us':[a,b],**{k:detail[k] for k in ('m','k','n','matrix_a_dtype') if k in detail}})
            for z in phases.values():z.sort(key=lambda e:e['us'])
            roles[kind]={'core':c,'window':w[c],'milestones':phases,'shard':str(shard.relative_to(ROOT)),'sha256':hashlib.sha256(shard.read_bytes()).hexdigest()}
        result.append(dict(op=OPER,shape=x['shape'],sku=chip,ca_us=obs['ca_us'],actual_aic=x['active_aic'],actual_aiv=x['active_aiv'],
            model_us=x['current_us'],kernel=x['kernel'],elf_sha256=obs['binary_sha256'],trace_sha256=obs['trace_sha256'],source=str(k.relative_to(ROOT)),**roles))
    (OUT/'archive.json').write_text(json.dumps(result,indent=2)+'\n')
    for z in result:
        ve=z['vec']['milestones']['VF'];mm=z['cube']['milestones']['MMAD'];fp=z['cube']['milestones']['FIX_L0C_TO_DST'];bound=[e for e in ve if e['pc'].endswith('14cb0')]
        print(z['sku'][6:11],z['shape'][0],len(mm),round(mm[0]['us'][0],2), round(mm[-1]['us'][1]-mm[0]['us'][0],2),round(fp[-1]['us'][1]-mm[-1]['us'][1],2),[round(e['us'][0],2)for e in bound],round(z['ca_us'],2))
if __name__=='__main__':archive()
