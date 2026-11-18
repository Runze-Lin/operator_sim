"""Stop orphaned npusim model processes of completed, wall-limited fusion cases."""
import os,signal,time,sys,json
from pathlib import Path
root=Path(__file__).resolve().parents[1]
base=root/'runs/v8cv'
while True:
    for case in base.glob('*/result.json'):
        if json.loads(case.read_text())['error'] not in ('wall budget','output budget'):continue
        for p in Path('/proc').glob('[0-9]*'):
            try:
                if (p/'cwd').resolve()!=case.parent.resolve():continue
                if (p/'exe').resolve().name not in ('extension_moe','extension_indexer_mhc'):continue
                if int((p/'stat').read_text().split()[3])!=1:continue
                os.kill(int(p.name),signal.SIGKILL);print('reaped orphan',p.name,case.parent,flush=True)
            except (OSError,ValueError):pass
    cases=json.loads((root/'results/v0.8/fusion/freeze.json').read_text())['cases']
    if all((base/c['id']/'result.json').exists() for c in cases):break
    time.sleep(5)
