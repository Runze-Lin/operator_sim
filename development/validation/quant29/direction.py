"""Identify bounded directional service costs from sustained Cast controller windows only."""
from collections import Counter
import hashlib
import json
from pathlib import Path
import re
import sys
import time

ROOT=Path(__file__).resolve().parents[3]
OUT=ROOT/'results/v0.29/quant'
sys.path.insert(0,str(ROOT/'development/validation/kda27'))
from quant_service import HEADER, PANEL

# Controller exports the cumulative name as well; anchor the interval column.
CMDS=re.compile(rb'^(Read cnt|Read with AP|Write|Write with AP|Mask write|Mask write with AP) +: (\d+) +\|',re.M)
Q0=re.compile(rb'^Cycle  \| +(\d+) +\|',re.M)
LABELS=['read','write','periods','read2','read_write','write2','slots_squared','read_slots','write_slots']


def fit(counts):
    r,w,n,rr,rw,ww,yy,ry,wy=counts
    det=rr*ww-rw*rw
    assert det>0 and rr>0 and ww>0
    # Two-variable least squares on the physical cone a>=1,b>=1. At least
    # one 32-byte payload beat is required per DFI clock per channel.
    candidates=[(max(1.,(ry-rw)/rr),1.),(1.,max(1.,(wy-rw)/ww)),(1.,1.)]
    a=(ry*ww-wy*rw)/det;b=(wy*rr-ry*rw)/det
    if min(a,b)>=1:candidates.append((a,b))
    def loss(v):
        a,b=v
        return a*a*rr+2*a*b*rw+b*b*ww-2*a*ry-2*b*wy+yy
    a,b=min(candidates,key=loss)
    return dict(read_DFI_per_beat=a,write_DFI_per_beat=b,design_separation=det/(rr*ww))


def tally(raw):
    arr=[0]*9;hist=Counter();h=hashlib.sha256()
    files=sorted(raw.glob('lpddr_state*.log'));assert len(files)==128
    for p in files:
        data=p.read_bytes()
        h.update(p.name.encode());h.update(hashlib.sha256(data).digest())
        prev=0
        for panel in data.split(PANEL)[1:]:
            end,_,total=map(int,HEADER.search(panel,0,400).groups());dt=end-prev;prev=end
            if dt!=1000:continue
            freqs=panel[panel.index(b'\nCycle  |')+1:].splitlines()[0].split(b'|')[1:]
            counts=[int(x) for x in freqs if x.strip()]
            assert len(counts)==65
            if counts[0]>.05*sum(counts):continue
            a=dict((k,int(v)) for k,v in CMDS.findall(panel));assert len(a)==6
            r=a[b'Read cnt']+a[b'Read with AP'];w=sum(v for k,v in a.items() if b'write' in k.lower())
            assert (r+w)*32==total
            if not r or not w:continue
            for i,n in enumerate([r,w,1,r*r,r*w,w*w,dt*dt,r*dt,w*dt]):arr[i]+=n
            hist[min(9,10*r//(r+w))]+=1
    return dict(sufficient_statistics=dict(zip(LABELS,arr)),read_share_decile_periods=dict(sorted(hist.items())),channel_set_sha256=h.hexdigest())


def main():
    t=time.monotonic()
    reg=json.loads((ROOT/'results/v0.27/quant-transfer/freeze.json').read_bytes())
    rows=[]
    for ref in reg['efficiency_source']:
        d=tally(ROOT/ref['raw'])
        q=d['sufficient_statistics'];assert (q['read'],q['write'],q['periods'])==(ref['read_beats'],ref['write_beats'],ref['accepted_periods'])
        rows.append(dict(shape=ref['shape'],case=ref['op'],source=ref['raw'],**d))
    def aggregate(items):
        return [sum(m['sufficient_statistics'][k] for m in items) for k in LABELS]
    for row in rows:
        train=[q for q in rows if q!=row]
        p=fit(aggregate(train));r,w,n,*_=aggregate([row]);yr,yw,nr,*_=aggregate(train)
        row['left_out']=dict(**p,measured_slots=n*1000,predicted_directional_slots=r*p['read_DFI_per_beat']+w*p['write_DFI_per_beat'],
                             predicted_isotropic_slots=(r+w)*1000*nr/(yr+yw))
    parameters=fit(aggregate(rows))
    result=dict(unit='DFI clocks per 32-byte payload; per 950DT physical channel',observations='All 128 channels of three other operators/shape runs. Each complete 1000-clock bin has >=95% occupied DMC queue and both directions.',
                sources=rows,parameters=parameters,seconds=time.monotonic()-t)
    (OUT/'direction-prices.json').write_text(json.dumps(result,indent=2)+'\n')
    print(parameters,flush=True)


if __name__=='__main__':main()
