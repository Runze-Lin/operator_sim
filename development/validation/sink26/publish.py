"""Create an independent same-axis figure, including the two T67/I10 contexts as they retire."""
import json
import sys
import time
from evidence import DT,PR,OUT,ROOT,read,shard
from figure import plot

def ready():
    count=0
    for chip in (DT,PR):
        for group in ('first','grow'):
            p=OUT/'receipts'/(group+'-'+chip+'.json');short=chip[6:11].lower()
            stem=f'independent-{short}-67-{group}'
            if not p.is_file() or (OUT/(stem+'.png')).exists():continue
            available=[r for r in read(p)['rows'] if r['shape']==[67,4] and r['qualified']]
            if not available:continue
            r,=available;s=r['stage']
            path,digest,core,origin,first,last,es=shard(s)
            for e in es:
                e['begin']=round(e['begin']-origin,1);e['end']=round(e['end']-origin,1)
            es.sort(key=lambda e:e['begin'])
            row=dict(shape=r['shape'],iters=r['iters'],chip=chip,core=core,ca_us=s['ca_us'],source=path,
                     source_sha256=digest,aggregate_sha256=s['trace_sha256'],events=es)
            plot(row,r['h'],stem);count+=1
    return count
if __name__=='__main__':
    if '--watch' in sys.argv:
        while True:
            ready()
            if (OUT/'state.json').exists() and read(OUT/'state.json')['state']=='finished':break
            if all((OUT/'receipts'/(g+'-'+c+'.json')).exists() for g in ('first','grow') for c in (DT,PR)):break
            time.sleep(45)
    else:ready()
