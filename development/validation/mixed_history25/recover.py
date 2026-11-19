"""Retry an original ordered process once when the first export misses physical END."""
from pathlib import Path
import sys
import json

ROOT=Path(__file__).resolve().parents[3]
sys.path.insert(0,str(ROOT))
from development.validation.mixed_history25.campaign import OUT,run,write,now
from development.validation.mixed_history25.verify_windows import publish
from development.validation.rms_ordered_verify.verify import digest


def main():
    f=OUT/'independent-freeze.json'
    plan=json.loads(f.read_text())
    role='recovery-'+sys.argv[1]
    chip=next(c for c in plan['chips'] if sys.argv[1] in c)
    groups=[]
    bad=[]
    for group in plan['groups']:
        receipt=OUT/f'independent-{group}-{chip}-windows.json'
        if not receipt.is_file():
            continue
        data=json.loads(receipt.read_text())
        cutoff=max(r['i'] for r in data['rows'] if r['target'])
        if any(not r['qualified'] for r in data['rows'][:cutoff+1]):
            groups.append(group)
            bad.append(digest(receipt))
    if not groups:
        return
    out=OUT/(role+'-freeze.json')
    if not out.is_file():
        write(out,dict(**{**plan,'groups':groups,'chips':[chip],'role':role,'utc':now()},
                       parent_sha256=digest(f),incomplete=bad,
                       recovery_policy='Repeat exactly the original full ordered prefix and predictions once. Only replace targets excluded due to an incomplete physical window. Keep original data and count costs.'))
    run(role)
    publish(role)


if __name__=='__main__':
    main()
