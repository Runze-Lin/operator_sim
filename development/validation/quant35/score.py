"""Score the frozen cold-stream cost against disjoint CA contracts."""
import json
from hashlib import sha256
from pathlib import Path
import sys
from time import perf_counter_ns

ROOT = Path(__file__).resolve().parents[3]
OUT = ROOT / 'results/v0.35/quant'
DT, PR, FUT = 'Ascend950DT_9582', 'Ascend950PR_957b', 'Ascend970DT'


def read(path):
    return json.loads(path.read_bytes())


def h(path):
    return sha256(path.read_bytes()).hexdigest()


def main(prefix):
    sys.path.insert(0, str(prefix))
    from operator_model import estimate, load_chip
    from operator_model.quant_stream_memory import scenario
    import operator_model
    assert Path(operator_model.__file__).is_relative_to(prefix) and operator_model.__version__ == '0.34.0'
    sys.path.append(str(ROOT))
    from development.validation.release32.evaluate import summary
    reg = read(OUT/'freeze.json')
    price = read(OUT/'service.json')
    partial = read(OUT/'b4-partial.json')
    assert h(OUT/'service.json') == reg['service_sha256']
    assert h(ROOT/price['source_path']) == price['source_sha256']
    archive = read(ROOT/'results/v0.34/replay.json')['rows']
    small = []
    eta = price['quant_mixed_bandwidth_efficiency']
    for r in archive:
        if r['op'] != 'quant':
            continue
        start = perf_counter_ns()
        chip = load_chip(r['chip'])
        public = estimate('quant', r['shape'], chip=chip).latency_us
        v = scenario(r['shape'], chip, public, mixed_bus_efficiency=eta)
        ms = (perf_counter_ns()-start)/1e6
        assert abs(public-r['us']) < 1e-7
        small.append(dict(shape=r['shape'], chip=r['chip'], kernel=r['kernel'], identity=r['identity'],
                          ca_us=r['ca_us'], public=public, candidate=v['latency_us'], query_ms=ms))
    prev = read(ROOT/'results/v0.20/full-quant/native.json')
    assert prev['qualified_core_envelope'] and not prev['instruction_ca_us']
    reg29 = ROOT/'results/v0.29/quant/freeze.json'
    ends = {}
    for chip in (DT, PR):
        path = reg29.parent/(chip+'.json')
        q = read(path) if path.exists() else None
        ca = None
        if q is not None:
            assert q['freeze_sha256'] == reg['source_freeze_sha256'] and q['sku'] == chip and q['shape'] == reg['shape']
            if q['qualified']:
                assert q['cost']['started_utc'] > reg['utc'] and q['instruction_us'] > 0
                ca = q['instruction_us']
        p = reg['predictions'][chip]
        ends[chip] = dict(shape=reg['shape'], public_us=p['base_us'], candidate_us=p['latency_us'],
                          ca_us=ca, receipt=str(path.relative_to(ROOT)) if path.exists() else None,
                          signed_pct=None if ca is None else 100*(p['latency_us']/ca-1))
    paired = all(r['ca_us'] is not None for r in ends.values())
    speed = {k:ends[DT][k]/ends[PR][k] for k in ('public_us','candidate_us','ca_us')} if paired else None
    b8 = reg['previous_native_dataset']['completed_b8'][DT]
    score = dict(reference='B8 DT controller payload on saturated mixed windows; new chip inherits the dimensionless service fraction',
                 efficiency=eta, matched=small, metrics={c:{k:summary([r for r in small if r['chip']==c], k)
                 for k in ('public','candidate')} for c in (DT,PR)}, max_query_ms=max(v['query_ms'] for v in small),
                 dt_b8_native=dict(shape=prev['shape'], ca_wrapper_us=prev['core_envelope_us'], public_us=b8['base_us'],
                                   candidate_us=b8['latency_us'], instruction_us=None, role='development'),
                 dt_b4_partial=dict(shape=partial['shape'], measured_prefix_us=partial['observed']['common_us'],
                                    physical_payload_duty=partial['observed']['busy_mixed_efficiency'], kernel_us=None),
                 forward=ends, forward_dt_over_pr=speed,
                 freeze_sha256=h(OUT/'freeze.json'))
    tmp = OUT/'score.tmp'
    tmp.write_text(json.dumps(score,indent=2)+'\n')
    tmp.replace(OUT/'score.json')
    print('Small-shape windows:', len(small), score['metrics'], 'maximum ms',score['max_query_ms'])


if __name__=='__main__':
    main(Path(sys.argv[1]).resolve())
