"""Check all published windows and optionally reprice them with the installed backend."""
from argparse import ArgumentParser
from collections import defaultdict
from math import isclose
import json
from .__main__ import HERE, digest, metrics, predict, require, sha_field


def verify(reprice=False):
    folder=HERE/'reference'
    manifest=json.loads((folder/'manifest.json').read_bytes())
    for name,expected in manifest['export_sha256'].items():
        require(digest((folder/name).read_bytes())==expected, 'Reference fingerprint mismatch: '+name)
    summary=json.loads((folder/'summary.json').read_bytes())
    rows=[json.loads(line) for line in (folder/'windows.jsonl').read_text().splitlines()]
    groups=defaultdict(list)
    ids=set()
    if reprice:
        import operator_model as api
    for r in rows:
        key=r['contract'],r['chip'],tuple(r['identity'])
        require(key not in ids and all(sha_field(i) for i in r['identity']), 'Duplicate or invalid physical identity')
        require(r['ca_us']>0 and r['api'] and r['kernel'] and r['data_role'] in ('replay','independent'), 'Invalid reference record')
        ids.add(key)
        require(isclose(100*(r['predicted_us']/r['ca_us']-1),r['signed_error_pct'],abs_tol=1e-9),'Invalid row error')
        if reprice:
            for chip,field in ((r['chip'],'predicted_us'),('Ascend970DT','extrapolated_970dt_us')):
                case=dict(op=r['op'],shape=r['shape'],attrs=r['attrs'],dtype=None,prefix=r.get('prefix'))
                p=predict(api,case,chip)
                require(isclose(p.latency_us,r[field],rel_tol=1e-10,abs_tol=1e-7),f"Prediction changed: {key}, {chip}: {p.latency_us} vs {r[field]}")
        groups[r['contract'],r['chip']].append(r)
    for contract,by_chip in summary.items():
        for chip,s in by_chip.items():
            whole=groups.pop((contract,chip),[])
            for kind in ('all','replay','independent'):
                bucket=whole if kind=='all' else [r for r in whole if r['data_role']==kind]
                observed=metrics(bucket)
                for calc,stored in [('windows','n'),('shapes','shapes'),('wape_pct','wape'),('max_pct','maximum')]:
                    a,b=observed[calc],s[kind][stored]
                    require(a==b or (a is not None and b is not None and isclose(a,b,abs_tol=1e-9)),f'Summary mismatch: {contract}/{chip}/{kind}')
    require(not groups,'Unknown contract')
    return {'windows':len(rows),'contracts':len(summary),'installed_predictions_verified':2*len(rows) if reprice else 0}


if __name__=='__main__':
    parser=ArgumentParser(description=__doc__)
    parser.add_argument('--reprice',action='store_true')
    print(json.dumps(verify(parser.parse_args().reprice)))
