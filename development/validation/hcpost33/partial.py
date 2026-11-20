"""Bound time from the archived partial HcPost trace and cross-check the packed counters."""
from hashlib import sha256
import json
from pathlib import Path
import struct

ROOT = Path(__file__).resolve().parents[3]
SKU = 'Ascend950DT_9582'
SHAPE = [1, 1024, 4, 5120]


def read(path):
    return json.loads(path.read_bytes())


def main():
    audit_path = ROOT/f'results/v0.32/hcpost/{SKU}-audit.json'
    freeze_path = ROOT/'results/v0.32/post/hc-pr-freeze.json'
    audit, freeze = read(audit_path), read(freeze_path)
    forecast, = (r for r in freeze['results'] if r['sku'] == SKU)
    assert audit['chip'] == SKU and forecast['shape'] == SHAPE
    base, = (ROOT/f'runs/specialized31/post/vllm_hc_post-{SKU}').glob(
        'profile/OPPROF_*/device0/HcPost_*/0/simulator')
    with (base/'visualize_data.bin').open('rb') as stream:
        extent, _ = struct.unpack('<QI', stream.read(12))
        assert extent == (base/'trace.json').stat().st_size
        merged = sha256()
        remaining = extent
        while remaining:
            block = stream.read(min(remaining, 4*1024**2))
            if not block:
                raise EOFError('incomplete packed trace')
            merged.update(block)
            remaining -= len(block)
        assert merged.hexdigest() == audit['trace_sha256']
        size, _ = struct.unpack('<QI', stream.read(12))
        secondary = stream.read(size)
        assert len(secondary) == size
    perf = json.loads(secondary.rstrip(b'\0\n'))
    cores = perf['Cores']
    assert set(cores) == set(audit['terminal_counts']) and len(cores) == 64
    count = {}
    for name in ('END', 'END_LABEL'):
        row, = (r for r in perf['Instructions'] if r['Source'].partition(' ')[0] == name)
        assert len(cores) == len(row['Instructions Executed'])
        count[name] = dict(zip(cores, row['Instructions Executed']))
    assert all(count['END'][c] == count['END_LABEL'][c] == shard == combined
               for c, (shard, combined) in audit['terminal_counts'].items())
    lo, hi = audit['elapsed_bounds_ticks']
    rate = 1000 * read(ROOT/f'configs/chips/{SKU}.json')['frequency_ghz']
    # The archive rounded the two endpoints to whole device cycles.
    observed = (hi-lo)/rate
    lower = (hi-lo-1)/rate
    candidate = forecast['candidate_us']
    assert 0 < candidate < lower
    out = dict(chip=SKU, shape=SHAPE, status='partial instruction window', ca_us=None,
               cycles_per_us=rate, observed_ticks=[lo,hi], observed_span_us=observed,
               conservative_lower_us=lower, missing_end=[c for c in cores if not count['END'][c]],
               missing_output_bytes=2*audit['output_elements']-audit['transferred_bytes'],
               packed_trace_sha256=merged.hexdigest(), packed_counters_sha256=sha256(secondary).hexdigest(),
               forecast_freeze_sha256=sha256(freeze_path.read_bytes()).hexdigest(),
               public_us=forecast['public_baseline_us'], candidate_us=candidate,
               candidate_under_by_at_least_us=lower-candidate,
               candidate_absolute_error_at_least_pct=100*(1-candidate/lower))
    (ROOT/'results/v0.33/hcpost/partial.json').write_text(json.dumps(out,indent=2)+'\n')


if __name__ == '__main__':
    main()
