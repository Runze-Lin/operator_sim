"""One separately registered 64-lane lower boundary; preserve the original masked-tail failure."""
from pathlib import Path
import json
import sys
import time

import series
from model import predict

ROLE='supplement'


def register():
    primary_file=series.OUT/'holdout-register.json'
    primary=series.read(primary_file)
    shape=[1,9,4,4160]
    existing={tuple(c['shape']) for c in primary['cases']}
    assert tuple(shape) not in existing
    cases=[]
    for chip in (series.DT,series.PR):
        name=f'{ROLE}-{chip}-t9-d4160'
        if (series.RAW/name).exists() or (series.OUT/(name+'.json')).exists():raise FileExistsError(name)
        cases.append(dict(shape=shape,chip=chip,id=name,predictions={name:predict(9,4160,name) for name in (series.DT,series.PR,'Ascend970DT')}))
    for path,h in primary['hashes'].items():
        if series.original.digest(series.ROOT/path)!=h:raise ValueError('Changed freeze '+path)
    doc={**primary,'utc':series.utc(),'role':ROLE,'cases':cases,
         'primary_sha256':series.original.digest(primary_file),
         'hashes':{**primary['hashes'],**series.hashes([Path(__file__)])},
         'note':'Registered after T7/D4112 failed BF16 equality on both original endpoints; the failed original shape remains in the intent-to-test audit. T9/D4160 adds the next full 64-lane tail boundary. Model, binary, host, numeric criterion and query inputs stay frozen. Run only after original cases.'}
    with (series.OUT/'supplement-register.json').open('x') as f:json.dump(doc,f,indent=2);f.write('\n')


def run():
    while True:
        state=series.read(series.OUT/'holdout-state.json')
        if state['status']=='complete':break
        if state['status']=='stopped':raise RuntimeError('Resolve original campaign before supplement')
        time.sleep(9)
    if not all((series.OUT/(case['id']+'.json')).exists() for case in series.read(series.OUT/'holdout-register.json')['cases']):
        raise RuntimeError('Original registrations remain unanswered')
    series.run(ROLE)


if __name__=='__main__':
    {'register':register,'run':run}[sys.argv[1]]()
