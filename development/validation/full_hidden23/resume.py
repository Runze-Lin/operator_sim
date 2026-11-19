"""Retain the never-executed initial attempts and run the registered wrapper through bash."""
from pathlib import Path
import sys
import campaign as original

BASE=original.OUT
REG=BASE/'registration.json'
ACTIVE=BASE/'v2'
ENTRY=Path(__file__).resolve().with_name('native-entry.sh')

def register():
    if ACTIVE.exists():raise FileExistsError(ACTIVE)
    for name in ('control-original','control-redirected'):
        record=original.read(BASE/(name+'.json'))
        raw=original.ROOT/record['cost']['raw']
        assert 'strace: exec: Permission denied' in (raw/'run.log').read_text() and not (raw/'log_ca').exists()
    ACTIVE.mkdir();prior=original.read(REG);src={**prior['sources'],str(Path(__file__).resolve().relative_to(original.ROOT)):original.sha(Path(__file__)),str(ENTRY.relative_to(original.ROOT)):original.sha(ENTRY)}
    original.write(ACTIVE/'registration.json',dict(**{**prior,'utc':original.now(),'sources':src},original_registration_sha256=original.sha(REG),correction='The stock project shell has no execute bit and is invoked by bash in native.py. Both first attempts were refused by exec before CA, results retained. Registered executable shim calls bash on the same verified script; all shape, outputs, controls, clock, Kimi gate and budgets preserved.'))
    original.write(BASE/'queue-status.json',dict(utc=original.now(),state='superseded-before-ca',active='v2/queue-status.json',reason='Both controls reached exec without launching the vendor process. Corrected invocation matches the original stock runner.'))

def run():
    original.NATIVE=ENTRY;original.OUT=ACTIVE;original.REG=ACTIVE/'registration.json';original.RAW=original.RAW/'v2';original.run()

if __name__=='__main__':{'register':register,'run':run}[sys.argv[1]]()
