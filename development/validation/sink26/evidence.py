"""Read the actual last AIV in each archived full CA without loading a merged JSON."""
import hashlib
import json
import math
from pathlib import Path
import sqlite3
import sys

ROOT=Path(__file__).resolve().parents[3]
OUT=ROOT/'results/v0.26/sink'
sys.path.insert(0,str(ROOT/'development/validation/sku17'))
from campaign_collect import events,sha
from operator_model import estimate

DT,PR,FUT='Ascend950DT_9582','Ascend950PR_957b','Ascend970DT'
def read(p):return json.loads(Path(p).read_text())
def save(p,x):Path(p).write_text(json.dumps(x,indent=2,allow_nan=False)+'\n')
def mini(name,lo,hi,pc,detail):
    return dict(name=name,begin=round(lo,1),end=round(hi,1),pc=pc,queue=detail.get('queue_type'),size=detail.get('burst_len'))

def shard(s):
    p=ROOT/s['trace'];core=max(s['windows'],key=lambda k:s['windows'][k][1])
    sh=p.parent/core/'trace.json'
    first=float('inf');last=-first;selected=[]
    for e in events(sh):
        if e['ph']!='X':continue
        a=e['ts']*1650;b=(e['ts']+e['dur'])*1650;first=min(first,a);last=max(last,b)
        if e['name'] in ('VF','END','END_LABEL','MOV_SRC_TO_DST_ALIGNv2'):
            selected.append(mini(e['name'],a,b,e['args']['pc_addr'],json.loads(e['args']['detail'])))
    origin=first-1650*s['windows'][core][0]
    return str(sh.relative_to(ROOT)),sha(sh),core,origin,first,last,selected

def sql(p):
    with sqlite3.connect(p.resolve().as_uri()+'?mode=ro',uri=True) as db:
        ranges=db.execute('select ExecInstrCoreId,CoreTypeId,min(ExecInstrTickStart),max(ExecInstrTickEnd) from ExecutedInstructions group by 1,2').fetchall()
        core,lane,first,last=max(ranges,key=lambda x:x[-1]);origin=min(x[2] for x in ranges)
        selected=[mini(n,b,e,hex(pc),json.loads(d)) for n,b,e,pc,d in db.execute(
          """select ExecInstrName,ExecInstrTickStart,ExecInstrTickEnd,SourceInstrAddr,ExtendParams
          from ExecutedInstructions where ExecInstrCoreId=? and CoreTypeId=? and
          ExecInstrName in ('VF','END','END_LABEL','MOV_SRC_TO_DST_ALIGNv2')""",(core,lane))]
    return str(p.relative_to(ROOT)),sha(p),f'{core}.{lane}',origin,first,last,selected

def history():
    receipts={}
    for r in read(ROOT/'results/v0.17/sku/records.json'):
        if r['op']=='mhc_sinkhorn' and r.get('stages'):receipts[r['trace_sha256'][0]]=r['stages'][0]
    for p in (ROOT/'results/v0.21/fusion/receipts').glob('mhc_sinkhorn_*.json'):
        r=read(p)
        if r['qualified']:s=r['kernel'];receipts[s['trace_sha256']]=s
    dbs={}
    for r in read(ROOT/'results/v0.12/evaluation.json')['rows']:
        if r['op']!='mhc_sinkhorn':continue
        p=ROOT/r['source']
        if p.suffix=='.json':p,=p.parent.glob('profile/npusim*/record/*.db')
        dbs[sha(p)]=p
    rows=[]
    for r in read(ROOT/'results/v0.25/coverage/audit.json')['rows']:
        if r['op']!='mhc_sinkhorn':continue
        h,=r['trace_sha256']
        if h in receipts: source,digest,core,origin,first,last,es=shard(receipts[h]); isolated=True
        else:
            source,digest,core,origin,first,last,es=sql(dbs[h])
            isolated='model-v7' in source or Path(source).name.startswith('0000')
        for e in es:e['begin']=round(e['begin']-origin,1);e['end']=round(e['end']-origin,1)
        es.sort(key=lambda e:e['begin'])
        t,n=r['shape'];i=r['attrs']['num_iters'];vfs=[e for e in es if e['name']=='VF'];ds=[e for e in es if e['name']=='MOV_SRC_TO_DST_ALIGNv2']
        assert len(vfs)==2*i+2 and len(ds)==3,(source,t,i,len(vfs),ds)
        assert abs(last-origin-1650*r['ca_us'])<3
        phases=dict(setup=vfs[0]['begin'],first=vfs[1]['begin']-vfs[0]['begin'],softmax=vfs[2]['begin']-vfs[1]['begin'],
                    first_col=vfs[3]['begin']-vfs[2]['begin'],steady=(vfs[-1]['begin']-vfs[3]['begin'])/(i-1),
                    drain=last-origin-vfs[-1]['begin'])
        rows.append(dict(shape=r['shape'],iters=i,chip=r['chip'],active=r['active_aiv'],ca_us=r['ca_us'],v025_us=r['current_us'],
                         core=core,source=source,source_sha256=digest,aggregate_sha256=h,isolated=isolated,
                         earliest_core_ticks=round(first-origin,1),full_ticks=round(last-origin,1),phases=phases,events=es))
    return rows

if __name__=='__main__':
    save(OUT/'trace-evidence.json',history())
