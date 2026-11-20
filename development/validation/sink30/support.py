"""Benchmark the online entry and document source/directory checks without mutating registrations."""
import json
from datetime import datetime, timezone
from statistics import median
from time import perf_counter_ns
from register import ROOT, OUT, DT, PR, FUT, digest, read
from operator_model import estimate, load_chip


def write():
    results={}
    for sku in (DT,PR,FUT):
        c=load_chip(sku)
        # Full public dispatch, all valid (T,I) pairs distinct, no CA or cached estimate.
        elapsed=[]
        for i in range(400):
            t=2+i%249
            it=2+i//249+(i%29)
            begin=perf_counter_ns()
            e=estimate('mhc_sinkhorn',(t,4),{'num_iters':it},chip=c)
            elapsed.append((perf_counter_ns()-begin)/1e6)
            assert e.latency_us is not None
        elapsed.sort()
        results[sku]=dict(calls=len(elapsed),p50_ms=median(elapsed),p95_ms=elapsed[379],max_ms=elapsed[-1])
    (OUT/'query.json').write_text(json.dumps(results,indent=2)+'\n')
    patch=OUT/'path-check.json'
    if not patch.exists():
        dirbase=ROOT/'runs/sink28-register-group'
        prospective=read(OUT/'freeze.json')
        verified=[]
        for f in prospective['future']:
            if f['cohort']!='v28-register':continue
            actual=dirbase/f['sku']
            path=ROOT/f['receipt']
            assert not actual.exists() and not path.exists()
            verified.append(dict(sku=f['sku'],index=f['index'],actual_raw=str(actual.relative_to(ROOT)),
                                 actual_exists=actual.exists(),receipt_exists=path.exists()))
        status=ROOT/'results/v0.28/fusion/eligibility.state.json'
        with patch.open('x')as w:
            json.dump(dict(checked_utc=datetime.now(timezone.utc).isoformat(),freeze_sha256=digest(OUT/'freeze.json'),
                clarification='v28-register actual raw parent is runs/sink28-register-group/ per eligibility_campaign.py. Both actual SKU paths and receipts were still absent at this later check. Frozen inputs and device predictions unchanged.',
                original_state=read(status),code_sha256=digest(ROOT/'development/validation/fusion28/eligibility_campaign.py'),rows=verified),w,indent=2)
            w.write('\n')


if __name__=='__main__':
    write()
