"""Recover a fully written device trace when the frozen collector omitted its receipt label."""
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[3]
sys.path.insert(0,str(ROOT))
from development.validation.matmul_dispatch30.device import OUT, RAW, REGISTER, ca, recover, save


def qualify_ready():
    jobs={tuple(r['shape']):r for r in ca.read(REGISTER)['rows']}
    for initial in sorted(OUT.glob('*-initial.json')):
        label=initial.stem.removesuffix('-initial')
        src=OUT/(label+'.json')
        dest=OUT/(label+'-retired.json')
        if dest.exists() or not src.exists():continue
        first=ca.read(initial)
        result=ca.read(src)
        if result.get('reason')!="KeyError('case')" or first.get('reason')!="ValueError('end marker disagrees with all-core window')":continue
        if result['registration_sha256']!=ca.digest(REGISTER):
            raise ValueError('Mismatched registration')
        job=jobs[tuple(first['shape'])]
        if result['shape']!=first['shape'] or result['chip']!=first['chip']:raise ValueError('Receipt mismatch')
        supplemented=dict(first,case=initial.stem)
        r=recover(job,supplemented,RAW/label,acquisition=initial)
        if not r or not r['qualified']:raise ValueError('Original drain criteria did not pass')
        save(dest,dict(r,registration_sha256=ca.digest(REGISTER),chip=first['chip'],shape=first['shape'],
                       cost=first['cost'],original_result=str(src.relative_to(ROOT)),
                       scope='Original frozen numerical, ELF, full-roster, official-table and END/drain checks using the same raw. Supplied omitted receipt basename only.'))


if __name__=='__main__':qualify_ready()
