"""Build/read the portable FP4 calibration campaign and its records for the error table."""
import argparse
import gzip
import json
from pathlib import Path
import re
from time import perf_counter_ns

from calibration.__main__ import encoded, freeze, ingest, metrics, read, require, unseal
from calibration.msopprof import sha
from calibration.quant_update.fit import fit
from operator_model import estimate

HERE=Path(__file__).resolve().parent
DEFAULT=HERE/'calibration.json.gz'


def load(path=DEFAULT):
    return json.loads(gzip.decompress(Path(path).read_bytes()))


def rows(bundle, *, development=False):
    """Yield current prices and historical comparison. All latency fields are in µs."""
    if development:
        new=unseal(bundle['development_replay'],'freeze')
        originals={}
        observations=[]
        source_freezes={}
        for filename in ('quant-mxfp4-smoke.json','quant-mxfp4-followup.json.gz'):
            path=HERE.parent/'reference'/filename
            data=load(path) if filename.endswith('.gz') else read(path)
            signed = unseal(data['freeze'], 'freeze')
            originals.update({c['id']:c for c in signed['cases']})
            source_freezes[signed['sha256']] = signed
            observations.extend(data['observations'])
    else:
        new=unseal(bundle['independent_freeze'],'freeze')
        originals={c['id']:c for c in unseal(bundle['comparison_freeze'],'freeze')['cases']}
        observations=bundle['observations']
    cases={c['id']:c for c in new['cases']}
    for item in observations:
        rec=item['receipt']
        c=cases[rec['case']['id']]
        # Older receipts retain their original prospective binding. They are now
        # calibration examples, so use their original signed measurement as truth
        # while reporting this model's replay against it.
        original_freeze = source_freezes[rec['freeze_sha256']] if development else new
        measured = ingest(original_freeze, rec)
        require(measured == item['measurement'] and measured['data_role'] == 'prospective',
                'CA receipt differs from its original prospective acquisition')
        us=measured['ca_us']
        row={k:c[k] for k in ('op','contract','shape','attrs','chip','api','dtype','prefix',
                              'predicted_us','query_ms','extrapolated_970dt_us')}
        row.update({k:measured[k] for k in ('ca_us','kernel','provider','identity','case_id',
                                             'freeze_sha256','receipt_sha256')})
        row.update(data_role='replay' if development else 'independent',
                   signed_error_pct=100*(row['predicted_us']/us-1),
                   query_970_ms=bundle['query_970_ms'][c['id']],previous_us=originals[c['id']]['predicted_us'],
                   pricing_sha256=new['sha256'],observed_aiv=len(rec['stages'][0]['cores']),
                   acquisition_seconds=item['wall_seconds'])
        yield row


def build(root):
    baseline=unseal(read(root/'old-freeze.json'),'freeze')
    frozen=unseal(read(root/'new-freeze.json'),'freeze')
    require([x['id'] for x in baseline['cases']]==[x['id'] for x in frozen['cases']], 'Baseline cases differ')
    historical={}
    for name in ('quant-mxfp4-smoke.json','quant-mxfp4-followup.json.gz'):
        p=HERE.parent/'reference'/name
        src=load(p) if name.endswith('.gz') else read(p)
        for ob in src['observations']:
            require(ingest(src['freeze'],ob['receipt'])==ob['measurement'],'Historical receipt mismatch')
            c=ob['receipt']['case']
            historical[c['id']]=c
    replay=freeze(dict(schema=1,cases=list(historical.values())))
    records=read(root/'independent/progress.json')
    observations=[]; failures=[]
    for record in records:
        r=root/'independent'/record['id']
        if record['state']!='complete':
            failures.append({**record,'cost':read(r/'cost.json') if (r/'cost.json').exists() else None})
            continue
        rec=read(r/'receipt.json')
        measured=read(r/'measurement.json')
        require(ingest(frozen,rec)==measured,'Independent receipt mismatch')
        cost=read(r/'cost.json')
        observations.append(dict(receipt=rec,measurement=measured,terminal_phases=read(r/'phases.json'),
                                 wall_seconds=cost['wall_seconds'],probe_sha256=cost['probe_sha256']))
    comparison={}
    for c in replay['cases']+frozen['cases']:
        start=perf_counter_ns()
        out=estimate(c['op'],c['shape'],c['attrs'],dtype=c['dtype'],chip='Ascend970DT')
        comparison[c['id']]=(perf_counter_ns()-start)/1e6
        require(abs(out.latency_us-c['extrapolated_970dt_us'])<1e-8,'Target prices changed after freeze')
    phases=read(root/'rows.json')
    # The first layout probe collected msopprof's default single launch.
    # Record its cost and identity without promoting it to a complete window.
    partial = root/'view-dt'
    log = (partial/'run.log').read_text()
    kernels = re.findall(r'Start profiling on kernel: (\S+)', log)
    require(len(kernels)==1 and kernels[0].startswith('Slice_') and 'SMOKE_PASS' in log,
            'Unexpected partial-layout capture')
    cost = read(partial/'cost.json')
    partial_view = dict(reason='default launch count captured Slice only', kernels=kernels,
                        log_sha256=sha(partial/'run.log'), exit_code=cost['exit_code'],
                        wall_seconds=cost['wall_seconds'], started_utc=cost['started_utc'],
                        finished_utc=cost['finished_utc'], ca_us=None)
    return dict(schema=1,independent_freeze=frozen,comparison_freeze=baseline,development_replay=replay,
                phase_fit=fit(phases),development_core_phases=phases,observations=observations,
                system_shape_evidence=read(root/'system-sources.json'),
                public_query_benchmark=read(root/'query-benchmark.json'),
                partial_layout_probe=partial_view,
                failures=failures,query_970_ms=comparison,wall_budget_minutes=32,concurrency=1)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--build',type=Path,help='Root of sealed freezes and raw CA output')
    p.add_argument('--path',type=Path,default=DEFAULT)
    p.add_argument('--rows',choices=['development','independent'])
    args=p.parse_args()
    if args.build:
        doc=build(args.build)
        args.path.write_bytes(gzip.compress(encoded(doc),mtime=0))
    else:
        doc=load(args.path)
    if args.rows:
        for row in rows(doc,development=args.rows=='development'):
            print(json.dumps(row))
    else:
        for role in ('development','independent'):
            data=list(rows(doc,development=role=='development'))
            for chip in sorted({r['chip'] for r in data}):
                sample=[r for r in data if r['chip']==chip]
                print(role,chip,metrics(sample))
