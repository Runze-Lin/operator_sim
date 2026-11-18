#!/usr/bin/env python3
"""Freeze timeline template parameters and holdout predictions before CA collection."""
from pathlib import Path
from collections import Counter,defaultdict
from datetime import datetime,timezone
import json,sys,sqlite3,hashlib
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from operator_model.fusion_v8 import predict,work,PRE,FUSED,IDX,FIRST,SECOND,CHAIN
OUT=ROOT/'results/v0.8/fusion'

def project(k):
    db=sqlite3.connect(Path(k['db_path']).resolve().as_uri()+'?mode=ro',uri=True)
    result=[]
    for kind in {c['core_type_id'] for c in k['cores']}:
        c=max((c for c in k['cores'] if c['core_type_id']==kind),key=lambda c:c['end_ticks'])
        sql="""select ExecInstrTickStart,ExecInstrTickEnd,json_extract(ExtendParams,'$.queue_type') from ExecutedInstructions where CoreTypeId=? and ExecInstrCoreId=? and ExecInstrIsDurationExcluded=0"""
        events=defaultdict(Counter)
        origin=db.execute('select min(ExecInstrTickStart) from ExecutedInstructions').fetchone()[0]
        end=origin+c['end_ticks']
        for a,b,q in db.execute(sql,(kind,c['core_id'])):
            events[a][q]+=1;events[b][q]-=1
        ticks=Counter();active=Counter();last=origin
        for at,change in sorted(events.items()):
            mask='+'.join(sorted(q for q,n in active.items() if n)) or 'IDLE'
            ticks[mask]+=at-last; active.update(change);last=at
        ticks['IDLE']+=end-last
        assert sum(ticks.values())==c['end_ticks']
        result.append({s:t/1650 for s,t in ticks.items() if t})
    db.close();return result

def save(path,data):path.write_text(json.dumps(data,indent=2)+'\n')
if __name__=='__main__':
    if (OUT/'freeze.json').exists():raise SystemExit('Frozen already')
    templates=defaultdict(list);contracts={};seen=defaultdict(set)
    for path in sorted((ROOT/'runs/model-v7').glob('*/result.json')):
        r=json.loads(path.read_text());c=r['case'];op=c['op']
        if r['status']!='ok' or op not in (CHAIN,'mhc',IDX,FUSED,PRE):continue
        shape,attrs=c['shape'],c['attrs'];seen[op].add(tuple(shape))
        contracts[op]={k:v for k,v in attrs.items() if k!='expert_counts'}
        for stage,k in zip((FIRST,SECOND) if op==CHAIN else ((PRE,'split_sink','split_post') if op=='mhc' else (op,)),r['measurement']['kernels']):
            if stage==PRE and op=='mhc':continue
            templates[stage].append(dict(work=work(stage,shape,attrs),cores=project(k),shape=shape,db_sha256=k['db_sha256']))
    contracts.update({FIRST:{},SECOND:{}});model={'templates':templates,'contracts':contracts}
    save(OUT/'model.json',model)
    cases=[]
    samples={IDX:[[1,q,l,8,128] for q,l in [(1,640),(1,896),(2,1024),(4,1024),(8,1536),(1,2048),(2,3072),(4,4096),(9,4096),(1,8192)]],
            FUSED:[[t,4,4096] for t in (3,5,6,7,8,9,12,16,24,32)],
            'mhc':[[t,4,d] for t,d in [(1,64),(2,128),(4,256),(9,256),(16,128),(1,512),(4,1024),(9,1024),(16,512),(32,1024)]],
            CHAIN:[[m,k,f,len(counts)] for m,k,f,counts in [(1,64,64,[1]),(3,64,64,[1,2]),(8,64,128,[8]),(16,128,128,[1,15]),(17,128,64,[8,9]),(8,256,128,[1,7]),(32,128,128,[2,6,8,16]),(16,256,256,[16]),(32,256,128,[1,31]),(64,128,256,[8,8,16,32])]]}
    counts_list=[[1],[1,2],[8],[1,15],[8,9],[1,7],[2,6,8,16],[16],[1,31],[8,8,16,32]]
    for op,shapes in samples.items():
        for i,shape in enumerate(shapes):
            assert tuple(shape) not in seen[op];attrs=contracts[op].copy()
            components=([PRE,'mhc'] if op=='mhc' else ([FIRST,SECOND,CHAIN] if op==CHAIN else [op]))
            if op==CHAIN:attrs['expert_counts']=counts_list[i];args=[op,*shape,','.join(map(str,counts_list[i]))]
            elif op==IDX:args=[op,*shape[1:4]]
            else:args=[op,*shape]
            cases.append(dict(id=f'{op}-{i+1:02}',op=op,shape=shape,attrs=attrs,arguments=args,groups={IDX:1,FUSED:3,'mhc':9,CHAIN:3}[op],
                components=components,predictions={s:predict(s,shape,contracts[PRE] if s==PRE else attrs,model)['predicted_us'] for s in components}))
    paths=['operator_model/fusion_v8.py','scripts/fusion-v8-freeze.py','scripts/fusion-v8-collect.py','operator_model/npusim_v7.py','results/v0.8/fusion/model.json','build/smoke/extension_indexer_mhc','build/smoke/extension_moe','build/capture/libsim_soc_adapter.so']
    save(OUT/'freeze.json',dict(utc=datetime.now(timezone.utc).isoformat(),sha256={p:hashlib.sha256((ROOT/p).read_bytes()).hexdigest() for p in paths},cases=cases,timeout_s=360,case_limit_mib=1000))
