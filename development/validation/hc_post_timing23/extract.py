"""Verify raw trace hash, then retain only physical DMA/VF issue and retirement."""
from collections import defaultdict
import hashlib
import json
from pathlib import Path
import sys

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[2]
OUT=ROOT/'results/v0.23/hc-timing'
sys.path.insert(0,str(HERE.parent/'sku17'))
from campaign_collect import events

def sha(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for block in iter(lambda:f.read(2**20),b''):h.update(block)
    return h.hexdigest()

def read(p):return json.loads(p.read_bytes())
def extract(p):
    rc=read(p)
    if not rc.get('qualified'):raise ValueError(f'Not qualified: {p.name}')
    s=rc['stage'];trace=ROOT/s['trace'];assert sha(trace)==s['trace_sha256']
    start={};end={};first_rv={};resource=defaultdict(list)
    for e in events(trace):
        if e['ph'] not in ('X','B','E'):continue
        core=e['pid'];begin=round(e['ts']*1650)
        finish=round((e['ts']+e.get('dur',0))*1650)
        start[core]=min(start.get(core,begin),begin);end[core]=max(end.get(core,finish),finish)
        if e['ph']=='X' and e['tid'].startswith('RVEC'):
            first_rv[core]=min(first_rv.get(core,begin),begin)
        if e['ph']!='X' or not (e['tid'] in ('MTE2','MTE3') or e['name']=='VF'):continue
        detail=json.loads(e['args'].get('detail') or '{}')
        if e['name']=='VF':
            kind='VF';size=0
        elif (detail.get('src_mem'),detail.get('dst_mem')) in (('OUT','UB'),('UB','OUT')) and detail.get('processed_bytes'):
            kind=e['tid'];size=detail['processed_bytes']
        else:continue
        resource[core].append(dict(kind=kind,begin=begin,end=finish,bytes=size,pc=e['args'].get('pc_addr')))
    origin=min(start.values());stop=max(end.values())
    assert abs((stop-origin)/1650-rc['ca_us'])<.002
    items={}
    for k,ops in resource.items():
        for x in ops:x['begin']-=origin;x['end']-=origin
        xfer=[v for v in ops if v['kind']=='MTE2'];vf=[v for v in ops if v['kind']=='VF'];wr=[v for v in ops if v['kind']=='MTE3']
        for many in (xfer,vf,wr):many.sort(key=lambda x:x['begin'])
        # Tiling parameter transfer (96 bytes) precedes the row coefficients.
        assert xfer[0]['bytes']==96
        tokens,hidden=rc['shape'][1],rc['shape'][3]
        widths=[4096]*(hidden//4096)+([hidden%4096] if hidden%4096 else [])
        it=iter(xfer[1:]);ptr=0;rows=[]
        for post in it:
            comb=next(it);assert post['bytes']==16 and comb['bytes']==64
            row=dict(post=post,comb=comb,tiles=[])
            for width in widths:
                xx,res=next(it),next(it);v,c=vf[2*ptr:2*ptr+2];out=wr[ptr]
                assert (xx['bytes'],res['bytes'],out['bytes'])==(2*width,8*width,8*width)
                row['tiles'].append(dict(width=width,x=xx,res=res,arith=v,cast=c,write=out))
                ptr+=1
            rows.append(row)
        assert len(vf)==2*ptr and len(wr)==ptr
        items[k]=dict(start=start[k]-origin,end=end[k]-origin,first_rvec=first_rv[k]-origin,meta=xfer[0],rows=rows)
    assert sum(len(v['rows']) for v in items.values())==tokens
    return dict(source=str(p.relative_to(ROOT)),role=rc['role'],sku=rc['chip'],shape=rc['shape'],window_ticks=stop-origin,
                trace=rc['stage']['trace'],trace_sha256=s['trace_sha256'],device_text_sha256=rc['text_sha256'],cores=items)
if __name__=='__main__':
    for p in [*sorted((ROOT/'results/v0.23/hc-post').glob('Ascend*.json')),*sorted(OUT.glob('development-Ascend*.json'))]:
        out=OUT/'primitives'/(p.stem+'.json');out.parent.mkdir(exist_ok=True)
        if (not out.exists() or 'first_rvec' not in next(iter(read(out)['cores'].values()))) and read(p).get('qualified'):
            data=extract(p);out.write_text(json.dumps(data,separators=(',',':'))+'\n')
