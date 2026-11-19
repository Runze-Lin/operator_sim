"""Regression: old square DT diagram and two prior rectangular/tail PR dispatches."""
import json
from extract import summarize
from freeze import ROOT,OUT,DT

if __name__=='__main__':
    old=json.loads(DT.read_text());geometry=summarize(ROOT/old['stage_timeline_source']['ca'],old['shape'],old['trace_sha256'],old['ca_us'])
    assert len(geometry['ca_tiles'])==len(old['ca_tiles'])
    assert all(a['pages']==b['pages'] and a['write']==b['write'] for a,b in zip(geometry['ca_tiles'],old['ca_tiles']))
    checks=[dict(shape=old['shape'],sku='Ascend950DT_9582',sha256=old['trace_sha256'],source=str(DT.relative_to(ROOT)),matched_previous_plot=True,rectangles=geometry['tile_histogram'],mmads=geometry['mmad'],exact_full_coverage=True)]
    archive='results/v0.16/sku/holdout.json'
    samples=[r for r in json.loads((ROOT/archive).read_text()) if r['op']=='matmul' and r['shape'] in ([1024,2048,32],[1536,2048,16])]
    assert len(samples)==2
    for r in samples:
        d=r['pr'];rec=summarize(ROOT/d['trace'],r['shape'],d['trace_sha256'],d['ca_us'])
        checks.append(dict(shape=r['shape'],sku='Ascend950PR_957b',sha256=d['trace_sha256'],source=archive,
                           rectangles=rec['tile_histogram'],mmads=rec['mmad'],exact_full_coverage=True))
    (OUT/'decoder-controls.json').write_text(json.dumps(checks,indent=2)+'\n')
    print(checks)
