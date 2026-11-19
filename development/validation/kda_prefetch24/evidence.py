"""Cross-stage descriptor and slot evidence from the physically last v0.23 FwdH core."""
import hashlib
import json
from pathlib import Path
import sys

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[2]
OUT=ROOT/'results/v0.24/kda-prefetch'
OLD=ROOT/'results/kda-v23'
sys.path.insert(0,str(HERE.parent/'sku17'))
from campaign_collect import events

def read(path):return json.loads(Path(path).read_bytes())
def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda:f.read(1024*1024),b''):h.update(chunk)
    return h.hexdigest()


def compact(stage,core):
    p=(ROOT/stage['aggregate']).parent/core/'trace.json'
    assert sha(p)==stage['cores'][core]['shard_sha256']
    groups={}
    for e in events(p):
        if e['ph']!='X':continue
        d=json.loads(e['args']['detail'] or '{}')
        route=f'{d.get("src_mem")}->{d.get("dst_mem")}'
        if e['name'] in ('VF','MMAD'):route=e['name']
        elif not d.get('processed_bytes') or route not in ('OUT->L1','OUT->UB','UB->OUT','L0C->UB'):continue
        t=e['ts'];z=e.get('dur',0)
        groups.setdefault(route,[]).append(dict(begin=round(t*1650)-stage['first_tick'],
            end=round((t+z)*1650)-stage['first_tick'],bytes=d.get('processed_bytes'),src=d.get('src_addr'),dst=d.get('dst_addr'),
            pc=e['args']['pc_addr']))
    for group in groups.values():group.sort(key=lambda i:i['begin'])
    return groups


def extract(p):
    r=read(p);assert r['qualified']
    stage=r['stages'][1];assert stage['family']=='chunk_fwd_h'
    t=r['case']['shape'][1];chunks=(t+63)//64
    last=max(stage['cores'],key=lambda c:stage['cores'][c]['last_tick'])
    cluster=last.split('.')[0]
    cube=compact(stage,cluster+'.cubecore0')
    vec=compact(stage,cluster+'.veccore0')
    creads=cube['OUT->L1'];mm=cube['MMAD'];fix=cube['L0C->UB'];vfs=vec['VF']
    assert len(creads)==4*chunks and len(mm)==len(fix)==2*chunks and len(vfs)==1+2*chunks
    # AIC uses one fixed W slot, one shared H/right slot, one fixed Kg slot.
    slots=[creads[i]['dst'] for i in range(4)]
    assert slots[1]==slots[3] and len(set(slots))==3
    assert [v['dst'] for v in creads]==slots*chunks
    u=[v for v in vec['OUT->UB'] if v['dst']==196608]
    assert len(u)==chunks
    writes=vec['UB->OUT'];assert len(writes)==1+3*chunks
    tiles=[]
    for j in range(chunks):
        w,h,k,right=creads[j*4:j*4+4]
        n=min(64,t-64*j);assert [v['bytes'] for v in (w,h,k,right,u[j])]==[256*n,32768,256*n,256*n,256*n]
        prev_h=writes[3*j]
        val,state=vfs[1+2*j:3+2*j]
        y,wr,hout=writes[1+3*j:4+3*j]
        m0,m2=mm[2*j:2*j+2]
        assert h['src']==prev_h['dst'] and right['src']==wr['dst'] and h['begin']>=prev_h['end'] and right['begin']>=wr['end']
        tiles.append(dict(rows=n,W=w,H=h,K=k,right=right,U=u[j],P=fix[2*j],D=fix[2*j+1],
                    mm0=m0,mm2=m2,corrected=val,state=state,y=y,write_right=wr,write_state=hout))
    for prev,new in zip(tiles,tiles[1:]):
        assert new['W']['begin']<prev['write_state']['end'] and new['U']['begin']<prev['write_state']['end']
    return dict(receipt=str(p.relative_to(ROOT)),aggregate_sha256=stage['aggregate_sha256'],family=stage['family'],
                device_elf_sha256=stage['device_text_sha256'],last_core=last,cluster=cluster,window=stage['last_tick']-stage['first_tick'],
                cube_sha256=stage['cores'][cluster+'.cubecore0']['shard_sha256'],vec_sha256=stage['cores'][cluster+'.veccore0']['shard_sha256'],
                tiles=tiles,initial_state_read=vec['OUT->UB'][0],initial_H_write=writes[0])


if __name__=='__main__':
    results={}
    for p in sorted(OLD.glob('supplement-*t129-h3.json')):
        results[p.stem]=extract(p)
    (OUT/'cross-chunk-evidence.json').write_text(json.dumps(results,indent=2)+'\n')
