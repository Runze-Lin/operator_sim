"""Check all published windows and optionally reprice them with the installed backend."""
from argparse import ArgumentParser
from collections import defaultdict
import gzip
from math import isclose
import json
from .__main__ import HERE, digest, metrics, predict, require, sha_field, ingest, unseal
from .__main__ import encoded


def packed_quant_rows(path):
    """Reconstruct the published windows from the pre-acquisition forecast and raw-source receipts."""
    with (gzip.open(path, 'rt') if path.suffix == '.gz' else path.open()) as source:
        archive = json.load(source)
    frozen = unseal(archive['freeze'], 'freeze')
    cases = {case['id']: case for case in frozen['cases']}
    for observed in archive['observations']:
        receipt = observed['receipt']
        result = ingest(frozen, receipt)
        require(result == observed['measurement'] and result['data_role'] == 'prospective',
                'FP4 quant observation differs from its frozen collection')
        case = cases.pop(result['case_id'])
        require(case['contract'] == case['op'] == 'quant_mxfp4' and len(receipt['stages']) == 1,
                'Unexpected FP4 quant contract or launch count')
        stage, = receipt['stages']
        shards = stage['physical_shard_sha256']
        require(set(shards) == set(stage['launched_cores']) and all(sha_field(s) for s in shards.values())
                and digest(encoded(shards)) == stage['trace_sha256'], 'Invalid physical trace identities')
        r, c = case['shape']
        require(case['outputs'] == {'Y': r*c, 'scale': 2*r*((c+63)//64)}
                and sha_field(observed['probe_sha256']), 'Invalid packed output contract')
        row = {k: case[k] for k in ('op','contract','chip','shape','dtype','attrs','prefix','api',
                                    'predicted_us','query_ms','extrapolated_970dt_us')}
        row.update({k: result[k] for k in ('case_id','freeze_sha256','receipt_sha256','ca_us','kernel','provider','identity')})
        row.update(data_role='independent', query_970_ms=observed['query_970_ms'],
                   acquisition_seconds=observed['wall_seconds'],
                   signed_error_pct=100*(row['predicted_us']/row['ca_us']-1))
        yield row
    require(not cases, 'Frozen FP4 quant input missing a qualified receipt')


def verify_mx_receipts(path, rows, replay_only=False):
    """Check native output/retirement evidence and independent forecast equality."""
    archive = json.loads(path.read_bytes())
    from .mx_matmul.compare import validate_evidence
    require(set(archive)=={'development','independent'}, 'Missing historical MX phase')
    historical = {c['id'] for item in archive.values() for c in item['freeze']['cases']}
    published = {r['case_id']: r for r in rows if r.get('case_id') in historical}
    require(sum(len(item['freeze']['cases']) for item in archive.values())==len(historical)
            and set(published)==historical, 'Missing historical MX reference or repeated case id')
    seen_geometry = set()
    for phase, role in (('development', 'replay'), ('independent', 'independent')):
        item = archive[phase]
        frozen = unseal(item['freeze'], 'freeze')
        cases = {c['id']: c for c in frozen['cases']}
        require(set(cases)==set(item['phase_counts']) and len(item['receipts'])==len(cases)
                and {r['case']['id'] for r in item['receipts']}==set(cases),
                'Historical MX freeze, phase counts and receipts disagree')
        for receipt in item['receipts']:
            observation = ingest(frozen, receipt)
            validate_evidence(receipt,item['phase_counts'][observation['case_id']],observation,physical=False)
            row = published.pop(observation['case_id'])
            case = cases[observation['case_id']]
            require(row['data_role']==('replay' if replay_only else role) and observation['data_role']=='prospective'
                    and row['receipt_sha256']==observation['receipt_sha256']
                    and row['identity']==observation['identity']
                    and row['freeze_sha256']==frozen['sha256']
                    and all(row[k]==case[k] for k in ('dtype','op','contract','chip','attrs','shape','api'))
                    and row['kernel']==observation['kernel'] and row['provider']==observation['provider']
                    and abs(item['phase_counts'][case['id']]['full_window_us']-row['ca_us'])<.001
                    and row['ca_us']==observation['ca_us'], 'MX receipt/role mismatch')
            geometry = row['contract'], tuple(row['shape'])
            if phase == 'development' or replay_only:
                seen_geometry.add(geometry)
            else:
                require(geometry not in seen_geometry
                        and row['predicted_us']==case['predicted_us']
                        and row['extrapolated_970dt_us']==case['extrapolated_970dt_us'],
                        'Independent shape or pricing differs from its freeze')
    require(not published, 'Missing MX receipt')
    return historical


def verify_mx_projection_rows(endpoints, rows, historical, version='v1.3', replay=False):
    relevant=[r for r in rows if r['contract'] in ('matmul_mx_native','matmul_mx_mixed')]
    published={r.get('case_id'):r for r in relevant}
    require(len(published)==len(relevant) and not historical.intersection(endpoints)
            and set(published)==historical.union(endpoints),'MX window or role not bound to its campaign')
    historical_geometry={(published[i]['contract'],tuple(published[i]['shape'])) for i in historical}
    for ident,e in endpoints.items():
        row=published[ident]
        case=e['case']
        require((case['contract'],tuple(case['shape'])) not in historical_geometry and row['data_role']==('replay' if replay else 'independent')
                and all(row[k]==case[k] for k in ('op','contract','chip','shape','dtype','attrs','api'))
                and all(row[k]==e['measurement'][k] for k in ('ca_us','provider','identity','kernel','freeze_sha256','receipt_sha256'))
                and (replay or (row['predicted_us']==e[version]['us']
                                and row['extrapolated_970dt_us']==e[version+'_970dt']['us'])),
                'MX projection reference differs from its case, receipt, forecast or role')


def verify(reprice=False):
    folder=HERE/'reference'
    manifest=json.loads((folder/'manifest.json').read_bytes())
    for name,expected in manifest['export_sha256'].items():
        require(digest((folder/name).read_bytes())==expected, 'Reference fingerprint mismatch: '+name)
    summary=json.loads((folder/'summary.json').read_bytes())
    rows=[json.loads(line) for line in (folder/'windows.jsonl').read_text().splitlines()]
    require(len(rows)==manifest['protocol_windows'] and len(summary)==manifest['contracts'], 'Manifest counts differ from reference')
    require(len({(r['chip'], tuple(r['identity'])) for r in rows}) == manifest['distinct_chip_and_stage_signatures'],
            'Physical signature count differs from manifest')
    receipts = [row for name in manifest['export_sha256'] if name.startswith('quant-mxfp4-')
                for row in packed_quant_rows(folder / name)]
    quant = [r for r in rows if r['contract'] == 'quant_mxfp4']
    require(len({(r['chip'], tuple(r['shape'])) for r in receipts}) == len(receipts)
            and sorted(quant,key=lambda r:r['case_id']) == sorted(receipts,key=lambda r:r['case_id']),
            'FP4 quant windows or independent geometries differ from receipts')
    if 'mx-matmul.json' in manifest['export_sha256']:
        from .mx_matmul.compare import load_comparison, compare, version_pair
        campaigns = []
        for name in manifest['export_sha256']:
            if name.startswith('mx-projections') and name.endswith('.json.gz'):
                archive = load_comparison(folder/name)
                latest = version_pair(archive)[-1]
                _, endpoints = compare(archive)
                campaigns.append((tuple(map(int, latest[1:].split('.'))), latest, endpoints))
        campaigns.sort(key=lambda entry: entry[0])
        historical = verify_mx_receipts(folder/'mx-matmul.json', rows, replay_only=bool(campaigns))
        for index, (_, version, endpoints) in enumerate(campaigns):
            scope = historical.union(endpoints)
            selected = [r for r in rows if r.get('case_id') in scope]
            verify_mx_projection_rows(endpoints, selected, historical, version, replay=index+1<len(campaigns))
            historical.update(endpoints)
        verify_mx_projection_rows({}, rows, historical)
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
                case=dict(op=r['op'],shape=r['shape'],attrs=r['attrs'],dtype=r.get('dtype'),prefix=r.get('prefix'))
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
