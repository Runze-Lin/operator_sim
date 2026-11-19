"""Recheck a previous large Cast without launching a simulator."""
from common import *
from evidence import endpoint

def checks():
    for sku in (DT,PR):
        r=read(LEGACY/('cast_2097152_32_'+sku+'.json'));s=r['stage']
        shards={p.parent.name:p for p in (ROOT/s['trace']).parent.glob('core*/trace.json')}
        ends={c:endpoint(p) for c,p in shards.items()}
        align=[max(m['retire'] for m in v.values())-1650*s['windows'][c][1] for c,v in ends.items()]
        print(sku,len(shards),round(max(align)-min(align),4),flush=True)
        if set(ends)!=set(s['windows']) or max(align)-min(align)>3:raise ValueError('cannot verify historical end frontier')
if __name__=='__main__':checks()
