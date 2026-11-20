"""Freeze three models' decode and full-prefill BF16 projections before querying the release host."""
from collections import defaultdict
from datetime import datetime, timezone
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
from development.validation.matmul29 import host_audit
from operator_model import estimate
from operator_model.matmul_placement import matmul_placement

OUT = ROOT/'results/v0.30/matmul-dispatch'
RAW = ROOT/'runs/matmul_dispatch30'
BIN = ROOT/'build/smoke/matmul-dispatch30'
REG = OUT/'freeze.json'
FUT = 'Ascend970DT'
INDEX = ROOT/'results/v0.28/matmul/overview-prices.json'
HOST = Path(host_audit.__file__)
CONTROLS = set(host_audit.CONTROL)


def register():
    indexed = json.loads(INDEX.read_bytes())['rows']
    # Reuse the exact projection widths already derived from pinned serving code/config.
    scenarios = defaultdict(list)
    for name, node, phase in [('DeepSeek-V4-Pro','indexer.weights_proj','decode'),
                              ('DeepSeek-V4.1-Flash','dsa.wq_a','decode'), ('Kimi-K3','mla.q_b_proj','decode'),
                              ('DeepSeek-V4-Pro','indexer.weights_proj','prefill'),
                              ('DeepSeek-V4.1-Flash','indexer.wk','prefill'), ('Kimi-K3','mla.kv_b_proj','prefill')]:
        geometry, = {(x['shape'][1], x['shape'][2]) for x in indexed if (x['model'], x['node']) == (name,node)}
        inputs = [(b,q) for b in (1,9,11,16) for q in (1,9)] if phase == 'decode' else [(8,s) for s in (32768,131072,262144)]
        for batch,q in inputs:
            scenarios[(batch*q,*geometry)].append(dict(model=name,node=node,phase=phase,batch=batch,tokens_per_request=q))
    rows=[]
    for shape in sorted(set(scenarios) | CONTROLS):
        for chip, cores in host_audit.SOCS.items():
            route=matmul_placement(shape,chip=chip)
            mixed=route['route']=='split_k'
            rows.append(dict(shape=shape,soc=chip,control=shape in CONTROLS,scenarios=scenarios[shape],
                             mixed_expected=mixed,workspace_expected=512+(262144*cores if mixed else 0),
                             public_us=estimate('matmul',shape,chip=chip).latency_us,
                             placement_us=route['latency_us'],future_public_us=estimate('matmul',shape,chip=FUT).latency_us))
    paths=[Path(__file__),Path(__file__).with_name('query.cpp'),BIN,INDEX,HOST,ROOT/'operator_model/matmul_placement.py']
    freeze=dict(utc=datetime.now(timezone.utc).isoformat(),rows=rows,sources={str(p.relative_to(ROOT)):host_audit.sha(p) for p in paths},
                contract='aclnnMatmulGetWorkspaceSize, BF16 ND A[M,K] and B[K,N], no bias, no transpose, exact allocations; timing from other full CA only. M=B*Q, 1 token decode or MTP8 verify Q=9; full prefill has no scheduler chunks.',
                budget='Sequential host phase only, max 180 s per SKU; at most 28.26 GiB individual call tensor allocations. Three positive/negative controls; proper runtime SKU and all queries completed with no kernel launch.')
    with REG.open('x') as f:
        json.dump(freeze,f,indent=2);f.write('\n')


def collect():
    host_audit.FROZEN, host_audit.RAW, host_audit.OUT, host_audit.BIN = REG, RAW, OUT, BIN
    RAW.mkdir(exist_ok=True)
    for soc in host_audit.SOCS:
        if not (OUT/(soc+'-workspace.json')).exists():
            host_audit.collect(soc)


if __name__ == '__main__':
    {'freeze':register,'collect':collect}[sys.argv[1]]()
