"""Compare timing, placement and direct DT reuse; keep host-only and device evidence separate."""
from collections import defaultdict
from datetime import datetime, timezone
from hashlib import sha256
import json
from pathlib import Path
import re
import sys
from time import perf_counter_ns

ROOT=Path(__file__).resolve().parents[3];sys.path.insert(0,str(ROOT))
from operator_model.matmul_placement import matmul_placement
from development.validation.matmul28.overview import DT,PR,FUT,OUT as PARENT,receipt

OUT=ROOT/'results/v0.29/matmul'; REG=OUT/'timing-freeze.json'
EXTRA=[3585,32,5120]

def read(path):return json.loads(path.read_bytes())
def digest(path):return sha256(path.read_bytes()).hexdigest()
def save(path,data):
    tmp=path.with_suffix(path.suffix+'.tmp');tmp.write_text(json.dumps(data,indent=2,ensure_ascii=False,allow_nan=False)+'\n');tmp.replace(path)

def forecast(shape,chip):
    start=perf_counter_ns();p=matmul_placement(shape,chip=chip);p['query_ms']=(perf_counter_ns()-start)/1e6
    return p

def register():
    raw=ROOT/'runs/matmul29-boundary';assert not REG.exists() and not raw.exists()
    rows=[]
    for item in read(PARENT/'overview-prices.json')['rows']:
        if item['group'] not in ('multi','narrow'):continue
        # No new execution is needed for these. Their own acquisition and pre-existing forecasts stay intact.
        for path in [ROOT/'runs/matmul28'/item['group'],PARENT/item['group']/'raw']:
            assert not any(path.glob('-'.join(map(str,item['shape']))+'-*'))
        assert all(receipt(item,t)['status']=='pending' for t in ('dt','pr'))
        rows.append(dict(group=item['group'],shape=item['shape'],old={k:q['us'] for k,q in item['predictions'].items()}))
    assert len(rows)==8
    from operator_model import estimate
    rows.append(dict(group='boundary',shape=EXTRA,old={c:estimate('matmul',EXTRA,chip=c).latency_us for c in (DT,PR,FUT)}))
    for item in rows:
        item['predictions']={c:forecast(item['shape'],c) for c in (DT,PR,FUT)}
        assert all(p['latency_us'] is not None and p['query_ms']<100 for p in item['predictions'].values())
    paths=[ROOT/'operator_model'/n for n in ('matmul_placement.py','matmul_multisplit.py','matmul_resource.py','matmul_splitk.py','matmul_waves.py','matmul_v15.py','chips.py')]
    paths += [ROOT/'configs/chips'/(c+'.json') for c in (DT,PR,FUT)] + [ROOT/'results/v0.16/matmul.json',ROOT/'build/smoke/matmul28',PARENT/'multi/freeze.json',PARENT/'narrow/freeze.json']
    ref=read(PARENT/'multi/freeze.json');elf=next(p for p in ref['source_sha256'] if p.endswith('.o'));paths.append(ROOT/elf)
    with REG.open('x') as f:
        json.dump(dict(utc=datetime.now(timezone.utc).isoformat(),rows=rows,sources={str(p.relative_to(ROOT)):digest(p) for p in paths},provider_sha256=digest(ROOT/elf),
            role='8 previously frozen, unstarted geometries reused. 1 new timing geometry (prior phase-one workspace only). Conditional mixes are not device observations.',
            admission='Do not supersede existing two-geometry retiling gate. For the release host selector additionally require both boundary SKU windows, kernel/active match, <=25% WAPE and <=50% maximum per SKU across available forward shapes, no higher WAPE than public, archived same-set results and missing outcomes retained.',
            budget='Boundary only: at most one real BF16 library call per 950 target, serial after original prefill lane, 1500 wall seconds and 2048 MiB per endpoint, original acquisition plus same-trace async-drain verification. No new CA for the other eight geometries or for 970DT.'),f,indent=2);f.write('\n')


def metric(rows,field):
    ys=[(r[field],r['ca_us']) for r in rows if r[field] is not None]
    if not ys:return {'n':0,'wape':None,'maximum':None}
    return dict(n=len(ys),wape=100*sum(abs(p-y) for p,y in ys)/sum(y for _,y in ys),maximum=100*max(abs(p/y-1) for p,y in ys))


def boundary_drain(path, item, chip, register):
    """Recover only a missing label from the first supervisor, retaining its receipts."""
    if read(path).get('reason') != 'KeyError(\'case\')':
        return path
    slug = path.stem
    supplemental = path.with_name(slug+'-drain.json')
    if supplemental.exists():
        return supplemental
    initial = path.with_name(slug+'-initial.json')
    acquisition = read(initial)
    if acquisition['chip'] != chip or acquisition['shape'] != item['shape']:
        raise ValueError('Mismatched acquisition')
    from development.validation.matmul28.drain import recover
    job = dict(shape=item['shape'], predictions={chip: {'us': item['predictions'][chip]['latency_us']}})
    data = recover(job, dict(acquisition,case=slug), ROOT/'runs/matmul29-boundary'/slug, acquisition=initial)
    if data and data.get('qualified'):
        stage, = data['stages']
        if stage['binary_sha256'] != register['provider_sha256']:
            raise ValueError('Unexpected Matmul implementation')
        save(supplemental,dict(data,chip=chip,shape=item['shape'],cost=acquisition['cost'],
                              acquisition_receipt=initial.name,register_sha256=digest(REG)))
        return supplemental
    return path

def scoring():
    reg=read(REG); out=[]; hold=[];seen=set(); old=read(PARENT/'public-score.json')
    def add(role,shape,chip,ca,base,p,pdt,kernel,ids,source):
        if seen.intersection(ids):return
        seen.update(ids)
        observable=('mix_aic' in kernel if kernel.startswith('MatMulV3_') else None)
        out.append(dict(role=role,shape=shape,chip=chip,ca_us=ca,old_us=base,placement_us=p['latency_us'],
            direct_dt_us=pdt['latency_us'],future_us=forecast(shape,FUT)['latency_us'],placement=p,
            observed_mixed=observable,mixed_match=None if observable is None else observable==(p['route']=='split_k'),
            kernel=kernel,raw_sha256=ids,source=source))
    for item in reg['rows']:
        group=item['group'];tup=tuple(item['shape'])
        for chip,tag in ((DT,'dt'),(PR,'pr')):
            path=(OUT if group=='boundary' else PARENT/group)/('-'.join(map(str,tup))+'-'+tag+'.json')
            if not path.exists():hold.append(dict(shape=tup,chip=chip,status='pending'));continue
            if group=='boundary':
                path=boundary_drain(path,item,chip,reg)
            r=read(path)
            if not r.get('qualified'):hold.append(dict(shape=tup,chip=chip,status='unqualified',reason=r.get('reason')));continue
            assert r['cost']['started_utc']>reg['utc'] and r['shape']==list(tup) and r['chip']==chip
            stage,=r.get('stages') or [r['observed']];assert stage['binary_sha256']==reg['provider_sha256']
            p=item['predictions'][chip];add('frozen timing',list(tup),chip,r['ca_us'],item['old'][chip],p,item['predictions'][DT],stage['kernel'],[stage['trace_sha256']],str(path.relative_to(ROOT)))
    for r in old['rows']:
        p=forecast(r['shape'],r['sku']);pdt=forecast(r['shape'],DT)
        add(r['role'],r['shape'],r['sku'],r['ca_us'],r['public_us'],p,pdt,r['kernel'],r['raw_ids'],r['source'])
    # Late original acquisitions can finish after the archived public-score snapshot.
    # Always price the public comparator from its pre-existing frozen wheel output.
    for item in read(PARENT/'overview-prices.json')['rows']:
        for chip,tag in ((DT,'dt'),(PR,'pr')):
            obs=receipt(item,tag)
            if obs['status']!='qualified':
                continue
            path=PARENT/obs['receipt']
            raw=read(path)
            st,=raw.get('stages') or [raw['observed']]
            if st['trace_sha256'] in seen:
                continue
            if any(item['shape']==j['shape'] for j in reg['rows']):
                raise ValueError(f'Forward case missed by frozen scorer: {path}')
            add('v0.28 application replay',item['shape'],chip,obs['ca_us'],
                item['predictions'][chip]['us'],forecast(item['shape'],chip),forecast(item['shape'],DT),
                st['kernel'],[st['trace_sha256']],str(path.relative_to(ROOT)))
    group={}
    for role in ('historical archive replay','v0.28 application replay','frozen timing'):
        group[role]={}
        for c in (DT,PR):
            q=[r for r in out if r['chip']==c and r['role']==role]
            group[role][c]={k:metric(q,k) for k in ('old_us','placement_us','direct_dt_us')}
    pairs=defaultdict(dict)
    for r in out:pairs[r['role'],tuple(r['shape'])].setdefault(r['chip'],[]).append(r)
    from statistics import median
    ratios=[]
    for (role,shape),matched in pairs.items():
        if len(matched)!=2:continue
        a,b=matched[DT],matched[PR];q={'role':role,'shape':shape}
        for f in ['old_us','placement_us','ca_us']:
            q[f]=None if any(x[f] is None for x in a+b) else median(x[f] for x in a)/median(x[f] for x in b)
        ratios.append(q)
    rel={role:{k:metric([r for r in ratios if r['role']==role],k) for k in ('old_us','placement_us')} for role in group}
    save(OUT/'score.json',dict(register_sha256=digest(REG),rows=out,pending=hold,summary=group,ratios=ratios,ratio_summary=rel))

if __name__=='__main__':
    if sys.argv[1]=='freeze':register()
    else:scoring()
