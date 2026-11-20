"""Compare the installed v0.31 wheel with v0.32 on the exact same launch sequences."""
from datetime import datetime, timezone
from collections import defaultdict
from hashlib import sha256
import importlib.util
import json
from math import isclose
from pathlib import Path
from statistics import median
import sys
import tempfile
from time import perf_counter_ns
from zipfile import ZipFile

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
from operator_model import OperatorStream, estimate
from development.validation.post29 import replay

OUT = ROOT/'results/v0.32/post'
DT, PR, FUT = replay.DT, replay.PR, replay.FUTURE
OP = 'mhc_post'
WHEEL, = (ROOT/'build/wheels/v031').glob('*.whl')


def save(name, data):
    path = OUT/name
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False)+'\n')


def old_api(directory):
    with ZipFile(WHEEL) as wheel:
        wheel.extractall(directory)
    package = Path(directory)/'operator_model'
    spec = importlib.util.spec_from_file_location('archived_post_baseline', package/'__init__.py',
                    submodule_search_locations=[str(package)])
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    assert mod.__version__ == '0.31.0'
    return mod


def summarize(rows):
    def errors(key):
        return dict(wape_pct=100*sum(abs(r[key]-r['ca_us']) for r in rows)/sum(r['ca_us'] for r in rows),
                    max_pct=100*max(abs(r[key]/r['ca_us']-1) for r in rows))
    times=sorted(r['query_ms'] for r in rows)
    return dict(windows=len(rows), shapes=len({tuple(r['shape']) for r in rows}),
            v031_shape=errors('v031_shape_us'), v032_shape=errors('shape_us'),
            v031_default_stream=errors('old_us'), v031_visible_stream=errors('ordered_us'),
            v032_default_stream=errors('stream_us'),
            ca_within_shape_range=sum(r['shape_range_us'][0]<=r['ca_us']<=r['shape_range_us'][1] for r in rows),
            full_uncached_stream_query_ms=dict(median=median(times), p95=times[(95*len(times)+99)//100-1], max=times[-1]))


def oracle(rows):
    same=defaultdict(list)
    for r in rows:
        same[tuple(r['shape'])].append(r['ca_us'])
    free, bounded, radius, denom = 0., 0., 0., sum(r['ca_us'] for r in rows)
    feasible=True; details=[]
    for shape, ys in sorted(same.items()):
        lo, hi = min(ys), max(ys)
        err=(hi-lo)/(hi+lo)
        radius=max(radius,err)
        center=2*lo*hi/(lo+hi)
        free+=sum(abs(median(ys)-y) for y in ys)
        a,b=.5*hi,1.5*lo
        best=max(a,min(b,median(ys))) if a<=b else None
        feasible=feasible and best is not None
        if best is not None:
            bounded+=sum(abs(best-y) for y in ys)
        details.append(dict(shape=shape,windows=len(ys),measured_us=ys,minimax_point_us=center,
                            minimax_lower_and_achievable_max_pct=100*err, min_wape_point_us=median(ys),
                            best_wape_point_at_max50_us=best))
    return dict(oracle_min_wape_pct=100*free/denom,
                minimax_lower_and_achievable_max_pct=100*radius,
                min_wape_at_max50_pct=100*bounded/denom if feasible else None,
                joint_wape25_max50_feasible=feasible and bounded/denom<=.25, shapes=details)


def score(archived):
    class Dispatch:
        def __init__(self, sku):
            self.sku=sku
            self.current=OperatorStream(sku)
            self.future=OperatorStream(FUT)
            self.old=archived.OperatorStream(sku)
            self.visible=archived.OperatorStream(sku,track_post_history=True)

        def forget(self):
            for stream in (self.current,self.future,self.old,self.visible):
                stream.forget_history()

        def call(self, shape, op=OP, attrs=None):
            start=perf_counter_ns()
            cur=self.current.advance(op,shape,attrs)
            elapsed=(perf_counter_ns()-start)/1e6
            future=self.future.advance(op,shape,attrs)
            before=self.old.advance(op,shape,attrs)
            previous_visible=self.visible.advance(op,shape,attrs)
            if op != OP:return
            point=estimate(OP,shape,chip=self.sku)
            stateless=point.projection['timing_diagnostics']
            assert stateless['model']=='post_unknown_dispatch' and stateless['context']=='unknown'
            visible=previous_visible.projection['timing_diagnostics']
            return dict(chip=self.sku,shape=shape,old_us=before.latency_us,ordered_us=previous_visible.latency_us,
                        v031_shape_us=archived.estimate(OP,shape,chip=self.sku).latency_us,shape_us=point.latency_us,
                        shape_range_us=stateless['state_range_us'], states_us=stateless['state_scenarios_us'],
                        target_plan=visible['target_plan'], scenario=visible.get('ordered_scenario',visible['context']),
                        stream_us=cur.latency_us,stream_model=cur.projection['timing_diagnostics']['model'],
                        explicit_attrs=attrs,query_ms=elapsed,extrapolated_970_stream_us=future.latency_us,
                        extrapolated_970_shape_us=estimate(OP,shape,chip=FUT).latency_us)

    replay.Dispatch=Dispatch
    replay.mixed_archive()
    replay.registered()
    rows=list(replay.rows.values())
    original=json.loads((ROOT/'results/v0.30/post/score.json').read_bytes())
    previous={r['sha256']:r for r in original['rows']}
    assert set(previous)==set(replay.rows)
    for r in rows:
        old=previous[r['sha256']]
        for new_key,old_key in [('old_us','old_us'),('ordered_us','ordered_us'),('v031_shape_us','stateless_us')]:
            assert isclose(r[new_key],old[old_key],abs_tol=1e-7)
    report=dict(rows=rows, summary={}, oracle={}, v031_wheel_sha256=sha256(WHEEL.read_bytes()).hexdigest(),
                protocols='Main full windows matched by SHA; other physical-only windows separate.',
                physical_only_summary={}, physical_only_rows=list(replay.physical.values()),
                same_contract_oracle={}, original_sources=replay.source_hashes)
    for chip in (DT,PR):
        subset=[r for r in rows if r['chip']==chip]
        report['summary'][chip]=summarize(subset)
        report['oracle'][chip]=oracle(subset)
        report['physical_only_summary'][chip]=summarize([r for r in replay.physical.values() if r['chip']==chip])
        lab=original['bounds'][chip]
        exact=[dict(shape=r['shape'],ca_us=us) for r in lab['by_shape'] for us in r['us']]
        report['same_contract_oracle'][chip]=oracle(exact)
        assert isclose(report['same_contract_oracle'][chip]['oracle_min_wape_pct'],lab['unconstrained_min_WAPE_pct'])
    save('replay.json',report)


def freeze(archived):
    target = OUT/'freeze.json'
    if target.exists():raise FileExistsError(target)
    # Only the workload registration and unclosed status are consulted, never an outcome.
    old_reg=ROOT/'results/v0.31/post/freeze.json'
    split_reg=ROOT/'results/v0.29/post/split-register.json'
    directions=json.loads(split_reg.read_bytes())['rows']
    source_names=['operator_model/post_resource.py','operator_model/post_history.py',
        'operator_model/stream.py','operator_model/post_fragment_pipeline.py','operator_model/post_token_pipeline.py',
        'operator_model/post_ordered.py','operator_model/post_repeat.py','results/v0.30/post/prices.json',
        'results/v0.25/post/prices.json','results/v0.28/post/parameters.json']
    predictions=[]
    for chip in (DT,PR,FUT):
        for group,shapes in [('isolated_large',[[1024,4,5120]]),
                            ('changed_split',[r['shape'] for r in directions[chip]])]:
            stream=OperatorStream(chip)
            old=archived.OperatorStream(chip,track_post_history=True)
            for i,s in enumerate(shapes):
                pred=estimate(OP,s,chip=chip)
                diag=pred.projection['timing_diagnostics']
                assert diag['model']=='post_unknown_dispatch'
                ordered=stream.advance(OP,s)
                predictions.append(dict(group=group,chip=chip,ordinal=i,shape=s,v031_shape_us=archived.estimate(OP,s,chip=chip).latency_us,
                    shape_us=pred.latency_us,range_us=diag['state_range_us'],states_us=diag['state_scenarios_us'],
                    v031_visible_stream_us=old.advance(OP,s).latency_us,ordered_us=ordered.latency_us,
                    ordered_model=ordered.projection['timing_diagnostics']['model'],data_role='prefix' if group=='changed_split' and i==0 else 'target'))
    save(target.name,dict(utc=datetime.now(timezone.utc).isoformat(),predictions=predictions,additional_ca_jobs=0,
        source_workloads={str(p.relative_to(ROOT)):sha256(p.read_bytes()).hexdigest() for p in (old_reg,split_reg)},
        source_sha256={p:sha256((ROOT/p).read_bytes()).hexdigest() for p in source_names},
        closed_measurement_requirement='Original independent collector full host golden, released library and timing checks; no timeout or partial windows.',
        scope='Previously registered shapes; frozen price comparison. 970DT projection only. No fit to returning CA.',
        v031_wheel_sha256=sha256(WHEEL.read_bytes()).hexdigest()))


if __name__=='__main__':
    with tempfile.TemporaryDirectory() as d:
        old=old_api(d)
        {'score': score, 'freeze': freeze}[sys.argv[1]](old)
