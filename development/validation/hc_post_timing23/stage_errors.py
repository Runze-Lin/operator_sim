"""Post-freeze attribution from one physically last core per qualified independent shape."""
from pathlib import Path
import json

from extract import OUT,ROOT,events,read,sha
from model import predict


def phases(file):
    rc=read(file);stage=rc['stage']
    windows=stage['windows'];core=max(windows,key=lambda k:windows[k][1])
    start_global=round(1650*min(v[0] for v in windows.values()))
    shard=(ROOT/stage['trace']).parent/core/'trace.json'
    first=None;last=None;vf=[];rd=[];wr=[]
    for e in events(shard):
        if e['ph'] not in ('X','B','E'):continue
        begin=round(e['ts']*1650);end=round((e['ts']+e.get('dur',0))*1650)
        first=begin if first is None else min(first,begin)
        last=end if last is None else max(last,end)
        if e['ph']!='X':continue
        if e['name']=='VF':vf.append((begin,end))
        else:
            details=json.loads(e['args'].get('detail') or '{}')
            direction=details.get('src_mem'),details.get('dst_mem');n=details.get('processed_bytes',0)
            if n and direction in [('OUT','UB'),('UB','OUT')]:
                (rd if direction==('OUT','UB') else wr).append((begin,end,n))
    if abs((last-first)/1650-(windows[core][1]-windows[core][0]))>.001:raise ValueError('Physical core window incomplete')
    origin=last-round(1650*windows[core][1])+start_global
    vf.sort();rd.sort();wr.sort();assert rd[0][2]==96 and rd[1][2]==16 and rd[2][2]==64
    _,t,_,d=rc['shape'];k=(d+4095)//4096
    actual_rows,rem=divmod(len(wr),k)
    assert not rem and len(vf)==2*len(wr) and len(rd)==1+2*actual_rows+2*len(wr)
    model=predict(t,d,rc['chip'],timeline=True);bars=model['bars']
    def points(v,k,num):return [b[num] for b in v if b[3]==k]
    names=['row0_coeff_issue','row0_first_vf_issue','row0_first_cast_issue','first_write_issue','last_write_issue','last_write_retire','full']
    m=[points(bars,'post',4)[0],points(bars,'arith',4)[0],points(bars,'cast',4)[0],points(bars,'write',4)[0],points(bars,'write',4)[-1],points(bars,'write',5)[-1],model['device_us']*1650]
    a=[z-origin for z in (rd[1][0],vf[0][0],vf[1][0],wr[0][0],wr[-1][0],wr[-1][1],last)]
    assert abs(a[-1]/1650-rc['ca_us'])<.001
    return dict(shape=rc['shape'],sku=rc['chip'],core=core,actual_rows=actual_rows,model_critical_rows=model['critical_rows'],
                shards=str(shard.relative_to(ROOT)),shard_sha256=sha(shard),trace_sha256=stage['trace_sha256'],
                ticks={n:dict(model=x,ca=y,delta=x-y) for n,x,y in zip(names,m,a)})


def publish():
    out=OUT/'independent-stage-errors.json'
    previous=read(out) if out.exists() else {}
    for f in sorted(OUT.glob('holdout-Ascend*.json')):
        if f.name in previous:continue
        if read(f)['qualified']:previous[f.name]=phases(f)
    tmp=out.with_suffix('.tmp');tmp.write_text(json.dumps(previous,indent=2)+'\n');tmp.replace(out)
    return len(previous)


if __name__=='__main__':
    print(publish())
