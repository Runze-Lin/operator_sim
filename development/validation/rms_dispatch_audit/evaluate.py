"""Treat a CA process and all its shapes as a single held-out group."""
from collections import defaultdict
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[3]
sys.path.insert(0,str(ROOT))
from operator_model import estimate, load_chip
from development.validation.sku21.acceptance import OUT as V21, score, read, write, CHIPS
from development.validation.rms_dispatch_audit.extract import OUT
from development.validation.rms_dispatch_audit.schedule import fit,both


def evaluate():
    rows=read(OUT/'core-windows.json')['rows']
    groups=defaultdict(list)
    for r in rows:
        groups[str(Path(r['source']).parent)].append(r)
    records=[]
    for run,held in groups.items():
        shapes={tuple(r['shape']) for r in held}
        calibration=[r for p,batch in groups.items() if p!=run for r in batch if tuple(r['shape']) not in shapes]
        param=fit(calibration)
        for r in held:
            chip=load_chip(CHIPS['dt'])
            old=estimate('rmsnorm',r['shape'],chip=chip)
            oldstate=estimate('rmsnorm',r['shape'],{'resident_aiv':r['prior']},chip=chip)
            known,unknown=both(r['shape'],chip,r['prior'],param,old)
            records.append(dict(shape=r['shape'],H=r['prior'],active=old.active_aiv,ca_us=r['window']/1650,
                                us=unknown,known_us=known,old_us=old.latency_us,old_known_us=oldstate.latency_us,
                                source=r['source'],role=r['role'],chip=chip.name))
    params=fit(rows)
    summary=[]
    old_ref=read(ROOT/'results/v0.19/evaluation.json')['rows']
    newer=read(ROOT/'results/v0.20/rms/records.json')
    ext=[]
    for raw in old_ref:
        if raw['op']!='rmsnorm': continue
        for s in ('dt','pr'):
            ext.append(dict(chip=CHIPS[s],shape=raw['shape'],ca_us=raw[s+'_ca_us'],role='v0.19 archived independent launch',source=str(raw[s+'_observation']['trace'])))
    for r in newer:
        ext.append(dict(chip=r['chip'],shape=r['shape'],ca_us=r['ca_us'],role='v0.20 archived independent launch',source=str(r['trace'])))
    for r in ext:
        chip=load_chip(r['chip']);forecast=estimate('rmsnorm',r['shape'],chip=chip)
        original=forecast.projection['timing_diagnostics']['conditional_us']['first']
        known,unknown=both(r['shape'],chip,0,params,forecast)
        records.append(dict(**r,H=0,active=forecast.active_aiv,us=unknown,known_us=known,old_us=forecast.latency_us,old_known_us=original))
    split=defaultdict(list)
    for r in records:split[r['chip'],r['role']].append(r)
    for (chip,cohort),rr in split.items():
        ss={k:score(rr,k) for k in ('us','known_us','old_us','old_known_us')}
        summary.append(dict(chip=chip,cohort=cohort,scores=ss))
        print(chip,cohort,' '.join(f"{k} {v['wape']:.1%}/{v['max_ape']:.1%}" for k,v in ss.items()))
    write(OUT/'candidate.json',dict(policy='Retrospective group holdout by original process excluding all repeated shapes, secondary archived PR/DT scores. No v0.21 new outputs used for training or model selection.',summary=summary,parameters=params,rows=records))
    print(params)


if __name__=='__main__':
    evaluate()
