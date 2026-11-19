"""After the first two holdouts, lock eight additional pre-observation shapes and run serially."""
from pathlib import Path
import hashlib
import os
import sys
import time

import precision_cases as protocol
import series as base
from sequence import LOCKED, PINS
from series import DT, PR, HERE, OUT, ROOT, RAW, probe, read, save

SHAPES=[(16,1),(17,1),(31,2),(32,1),(33,2),(64,2),(65,2),(129,3)]
PRED=OUT/'supplement-predictions.json'
LOCK=OUT/'precision-supplement-register.json'
SELECT=OUT/'supplement-selection.json'
STATE=OUT/'precision-supplement-state.json'
BUDGET=5400


def choose():
    if SELECT.exists():raise FileExistsError(SELECT)
    save(SELECT,dict(utc=base.when(),model_sha256=LOCKED,plan='supplemental, selected before these shapes were observed and after the first two shapes had been selected separately',
          shape_order=SHAPES,reasons=['end of 16-row subblock','new 16-row subblock','short of 32 with head concurrency','first full 32-row inverse','two diagonal inverse blocks with two heads','double-head full 64 chunk','double-head second chunk','multiple full chunks, recurrent chaining and three heads'],
          per_case_seconds=900,per_case_bytes=3*1024**3,ca_seconds_all=BUDGET,source_sha256=probe.digest(Path(__file__).resolve())))


def serial_register():
    if probe.digest(PINS)!=LOCKED:raise ValueError('original model freeze differs')
    selection=read(SELECT)
    if selection['source_sha256']!=probe.digest(Path(__file__).resolve()):raise ValueError('supplement protocol changed')
    if read(OUT/'precision-holdout-state.json')['state']!='finished':raise ValueError('first holdout did not pass')
    original=read(OUT/'precision-holdout-register.json')
    for c in original['cases']:
        if not read(OUT/(c['id']+'.json'))['qualified']:raise ValueError('primary holdout incomplete')
    dev=read(OUT/'development-register.json')
    known={(c['shape'][1],c['shape'][3]) for c in dev['cases']+original['cases']}
    known.add((2,1))
    if known&set(SHAPES) or len(set(SHAPES))!=8:raise ValueError('shape already enrolled')
    for t,h in SHAPES:
        if any(p.is_dir() and p.name.endswith(f'-t{t}-h{h}') for p in RAW.iterdir()):raise ValueError('prior raw CA '+str((t,h)))
    freeze=read(PINS)
    for name,digest in freeze['sha256'].items():
        if probe.digest(ROOT/name)!=digest:raise ValueError('model dependency changed '+name)
    sys.path.insert(0,str(HERE.parent/'kda_resource23'))
    from model import predict
    predictions=[predict([1,t,h,h,128,128],sku) for t,h in SHAPES for sku in (DT,PR,'Ascend970DT')]
    for prev in freeze['predictions']:
        p=next((q for q in predictions if q['chip']==prev['chip'] and q['shape']==prev['shape']),None)
        if p is not None and p!=prev:raise ValueError('supplement prediction drift')
    if PRED.exists() or LOCK.exists():raise FileExistsError('pre-existing supplemental registration')
    save(PRED,dict(utc=base.when(),role='frozen before any supplemental CA; 970DT projection only',model_freeze_sha256=LOCKED,predictions=predictions))
    protocol.FORMS['supplement']=SHAPES
    pins=[PINS,SELECT,PRED,Path(__file__).resolve(),OUT/'precision-holdout-register.json',*(ROOT/n for n in freeze['sha256'])]
    protocol.register('supplement',[str(p) for p in pins])
    # Add the explicit total wall budget before handing the final contract to any runner.
    registration=read(LOCK)
    registration['total_CA_seconds']=BUDGET
    registration['stop_on_failure']='At first failed full acceptance gate, including numerical, synchronization, capacity, physical SKU or ELF identity.'
    registration['primary_holdouts_selection_utc']=original['utc']
    registration['supplement_selection_utc']=selection['utc']
    save(LOCK,registration)
    print('final-register-sha256',probe.digest(LOCK),flush=True)


def run():
    while True:
        current=read(OUT/'precision-holdout-state.json')
        if current['state']=='finished':break
        if current['state']!='running':
            save(STATE,dict(state='blocked by the first holdout',original=current));return
        time.sleep(10)
    if not LOCK.exists():serial_register()
    captured=base.capture
    def bounded(reg,c,folder,remaining):
        done=[read(p)['cost']['seconds'] for p in OUT.glob('supplement-ascend*.json')]
        budget_left=BUDGET-sum(done)
        if budget_left<=0:raise TimeoutError('supplemental total CA budget exhausted')
        return captured(reg,c,folder,min(remaining,budget_left))
    base.capture=bounded
    try:
        protocol.run('supplement')
    except TimeoutError as exc:
        save(STATE,dict(state='total CA budget',message=str(exc)))


if __name__=='__main__':
    if sys.argv[1:]==['choose']:choose()
    elif sys.argv[1:]==['run']:run()
    else:raise SystemExit('choose | run')
