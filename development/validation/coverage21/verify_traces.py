"""Optional raw-file verification for the eight large archived examples."""
from hashlib import sha256
import json

from audit import ROOT, OUT

CASES = {
    ('quant', (2048, 5120)),
    ('fa', (9, 32768, 64, 1, 1)),
    ('fa', (9, 131072, 64, 1, 1)),
    ('matmul', (2048, 32, 5120)),
    ('matmul', (1280, 2048, 512)),
}


def verify():
    evidence=[]
    for row in json.loads((OUT/'audit.json').read_text())['rows']:
        if (row['op'],tuple(row['shape'])) not in CASES:
            continue
        path=ROOT/row['trace']
        digest=sha256()
        with path.open('rb') as stream:
            for block in iter(lambda:stream.read(2**20),b''):
                digest.update(block)
        if digest.hexdigest()!=row['trace_sha256']:
            raise ValueError(f'Raw trace digest disagrees with the archived receipt: {path}')
        evidence.append(dict(op=row['op'],shape=row['shape'],chip=row['chip'],trace=row['trace'],
                             bytes=path.stat().st_size,sha256=digest.hexdigest()))
    if len(evidence)!=8:
        raise ValueError('Expected 8 complete trace files on this archive host')
    (OUT/'verified-raw.json').write_text(json.dumps(evidence,indent=2)+'\n')


if __name__=='__main__':
    verify()
