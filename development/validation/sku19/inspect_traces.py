"""Reuse the existing reducer and analyze only newly qualified traces."""
import sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[3]
sys.path[:0]=[str(ROOT),str(ROOT/'development/validation/sku18')]
from development.validation.sku18.trace_evidence import reduce
from development.validation.sku19.campaign import OUT, read, write

def main():
    target=OUT/'trace-evidence.json'
    cache={r['trace_sha256']:r for r in read(target)['rows']} if target.exists() else {}
    records = {r['trace_sha256'][0]: r for name in ('records.json', 'final-records.json', '../rms-confirmation/records.json')
               if (OUT/name).exists() for r in read(OUT/name)}
    for r in records.values():
        fingerprint=r['trace_sha256'][0]
        if fingerprint not in cache:
            cache[fingerprint]=reduce(r)
            write(target,dict(scope='Terminal-core queue union and issued descriptors. Processed bytes need not reach HBM.',rows=list(cache.values())))
    print(len(cache),'qualified traces reduced')

if __name__=='__main__':main()
