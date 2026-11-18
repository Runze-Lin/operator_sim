"""Replace a numerically correct launch whose trace mixes outputs from two tasks."""
from datetime import datetime, timezone
import json

from retry import ROOT, OUT, collect, save, sha256

if __name__ == '__main__':
    old = json.loads((OUT / 'freeze.json').read_text())
    for path, digest in old['sha256'].items():
        assert sha256(ROOT / path) == digest
    if (OUT / 'isolate256-freeze.json').exists():
        raise RuntimeError('Already frozen')
    warm = dict(old['cases'][3], role='context_warmup', prior_aiv=0)
    target = old['cases'][-1]
    assert target['shape'] == [256,4,1024] and target['prior_aiv'] == warm['prediction']['predicted_active_aiv'] == 64
    methods = ['development/validation/post11/isolate256.py','development/validation/post11/retry.py',
               'development/validation/post11/recover.py','results/v0.11/post/freeze.json']
    contract = dict(utc=datetime.now(timezone.utc).isoformat(),sha256={p:sha256(ROOT / p) for p in methods},
                    isolated_batches=[[warm,target]],per_batch_seconds=360,per_batch_gib=1.25,
                    cause='initial T256 had 408 MTE3 output transfers of 8192B, 311 unique row addresses vs 256 expected')
    save('isolate256-freeze.json',contract)
    good,failed = collect([warm,target],contract)
    print('isolated T256',len(good),'qualified at helper/golden gates,',len(failed),'failed')
