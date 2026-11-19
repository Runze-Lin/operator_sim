"""Evaluate frozen-style physical issue schedules against all 54 archived full windows."""
import math
from evidence import DT,PR,FUT,OUT,read,save
from calibrate import annotated
from model import forecast
from operator_model import load_chip

def metrics(rows,k):
    if not rows:return None
    errors=[abs(r[k]-r['ca_us']) for r in rows]
    return dict(n=len(rows),wape_pct=100*sum(errors)/sum(r['ca_us'] for r in rows),
                max_ape_pct=100*max(e/r['ca_us'] for e,r in zip(errors,rows)))
def build():
    records=[]
    for chip in (DT,PR):
        for r,h in annotated([x for x in read(OUT/'trace-evidence.json') if x['chip']==chip]):
            c=load_chip(chip);known=forecast(r['shape'],r['iters'],c,h);unknown=forecast(r['shape'],r['iters'],c)
            if known['active_aiv']!=r['active']:raise ValueError((chip,r['shape'],'roster mismatch'))
            records.append(dict(chip=chip,shape=r['shape'],iters=r['iters'],h=h,state=known['selected'],ca_us=r['ca_us'],
                                v025_us=r['v025_us'],known_us=known['latency_us'],unknown_us=unknown['latency_us'],
                                future_unknown=forecast(r['shape'],r['iters'],load_chip(FUT))['latency_us'],
                                source=r['source'],source_sha256=r['source_sha256']))
    groups={'all':records,'T above 64':[r for r in records if r['shape'][0]>64],
            'I 10 or 12':[r for r in records if r['iters'] in (10,12)]}
    for c in (DT,PR):
        items=[r for r in records if r['chip']==c];groups[c]=items
        for st in ('first','expand','resident'):groups[c+' '+st]=[r for r in items if r['state']==st]
    return dict(role='development replay; all historical windows, 950DT alone used to price individual trace milestones',
                groups={g:{k:metrics(rows,k) for k in ('v025_us','unknown_us','known_us')} for g,rows in groups.items()},rows=records)
if __name__=='__main__':save(OUT/'replay.json',build())
