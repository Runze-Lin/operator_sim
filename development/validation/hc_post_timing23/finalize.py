"""Read-only checks of frozen data, source and qualified windows, followed by publication."""
import sys
import time
from pathlib import Path
import json

from extract import OUT, ROOT, read, sha
from report import publish
from stage_errors import publish as save_stages


def finalize():
    freeze=read(OUT/'holdout-register.json');extra=read(OUT/'supplement-register.json')
    for reg in (freeze,extra):
        for relative,digest in reg['hashes'].items():
            assert sha(ROOT/relative)==digest,relative
        for c in reg['cases']:
            path=OUT/(c['id']+'.json')
            if not path.exists():raise ValueError('Awaiting '+path.name)
    save_stages();v=publish()
    assert v['queue_state']['status']=='complete' and v['supplement_state']['status']=='complete'
    if len(v['failed'])!=2 or v['supplement_failed'] or v['pending'] or v['supplement_pending']:
        raise RuntimeError('Independent verification needs review')
    assert all('t7-d4112' in r['id'] for r in v['failed'])
    for s,summary in v['accepted_independent_summary'].items():
        assert summary['n']==10
        print(s,summary,flush=True)
    for path in ('dt-16448-timeline','dt-4080-independent','dt-7168-independent','dt-4160-supplement'):
        assert (OUT/(path+'.png')).stat().st_size>5000 and (OUT/(path+'.json')).exists()


if __name__=='__main__':
    while '--watch' in sys.argv:
        state=OUT/'supplement-state.json'
        if state.exists() and read(state)['status'] in ('stopped','complete'):break
        time.sleep(10)
    finalize()
