"""Freeze the candidate without opening any held-out CA output."""
from datetime import datetime, timezone
from hashlib import sha256
from pathlib import Path
import json
import sys

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
from operator_model.matmul_full_load import estimate_matmul_full_load

OUT = ROOT / 'results/v0.29/matmul/full-load'
CA = ROOT / 'results/v0.28/matmul'
RAW = ROOT / 'runs/matmul28'
CHIPS = ('Ascend950DT_9582', 'Ascend950PR_957b', 'Ascend970DT')


def digest(path):
    return sha256(path.read_bytes()).hexdigest()


def read(path):
    return json.loads(path.read_bytes())


def register():
    paths = [Path(__file__), OUT/'development-traces.json', CA/'freeze.json', CA/'long-prefill/freeze.json',
             ROOT/'development/validation/matmul29/full_load_trace.py',
             ROOT/'third_party/ops-nn/matmul/mat_mul_v3/op_host/op_tiling/arch35/matmul_v3_basic_aswt_tiling.cpp',
             ROOT/'results/v0.16/matmul.json']
    paths += [ROOT/'operator_model'/f'{name}.py' for name in ('matmul_full_load','matmul_v15','matmul_resource','chips')]
    paths += [ROOT/'configs/chips'/f'{name}.json' for name in CHIPS]
    rows = []
    for m in (32768, 262144, 1048576, 2097152):
        long = m > 262144
        source = (CA/'long-prefill' if long else CA)/'freeze.json'
        shape = [m,128,512]
        spec, = [r for r in read(source)['rows'] if r['shape'] == shape]
        for chip, tag in zip(CHIPS, ('dt','pr','future')):
            stem = f'{m//8}-{tag}' if long else f'{m}-128-512-{tag}'
            path = (source.parent/(stem+'.json'))
            raw = (RAW/'long-prefill' if long else RAW)/stem
            if tag == 'future':
                role = 'extrapolation only'
            elif m == 32768 or (m == 262144 and tag == 'dt'):
                role = 'development'
            elif path.exists():
                role = 'completed but unseen at freeze'
            elif raw.exists():
                role = 'in flight, blinded at freeze'
            else:
                role = 'registered before acquisition'
            rows.append(dict(shape=shape,chip=chip,model=spec['model'],node=spec['node'],role=role,
                             original_freeze=str(source.relative_to(ROOT)),
                             acquisition=None if tag=='future' else str(path.relative_to(ROOT)),
                             legacy_us=spec['predictions'][chip]['us'],candidate=estimate_matmul_full_load(shape,chip=chip)))
    rec=dict(utc=datetime.now(timezone.utc).isoformat(),rows=rows,sha256={str(p.relative_to(ROOT)):digest(p) for p in paths},
             contract='No new CA. Three developmental whole-kernel receipts. Five target CA endpoints never opened before freezing. Each needs host golden, official kernel, all active physical core END and drain. Report role separately for already running original experiments. No future-chip CA, no promotion to default.')
    with (OUT/'freeze.json').open('x') as file:
        json.dump(rec,file,indent=2);file.write('\n')


if __name__=='__main__':
    register()
