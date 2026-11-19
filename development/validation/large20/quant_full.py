"""One low-volume real full-hidden Kimi activation quant, on the existing DT CA slot."""
from datetime import datetime
import gzip
import json
from pathlib import Path
import sys
import time

from dram import counters
from large_cast import CHIPS, ROOT, digest, estimate, now, read, write
from native import capture_vector

DIR = ROOT / 'results/v0.20/full-quant'
RUN = ROOT / 'runs/kimi-full-quant20'
HOST = ROOT / 'build/smoke/large_vector'
PREVIOUS = ROOT / 'results/v0.20/large/cast_262144_5120_native.json'
SHAPE = [8*32768, 7168]


def freeze():
    with gzip.open(ROOT/'results/v0.20/workloads/contracts.jsonl.gz', 'rt') as f:
        s = next(x for line in f if (x := json.loads(line))['scenario'] == dict(
            model='Kimi-K3', seq=32768, batch=8, tp=8, dp=1, ep=8, chunk=0, step=0, ep_rank=0))
    node = next(n for n in s['contracts'] if n['op'] == 'kda.mixed_input_dynamic_mx_quant')
    call = node['standalone_call']
    assert call['shape'] == SHAPE and call['op'] == 'quant'
    outputs = {}
    for alias, chip in CHIPS.items():
        est = estimate(**call, chip=chip, timing_mode='resource')
        outputs[alias] = dict(chip=chip, us=est.latency_us, aiv=est.active_aiv,
                              per_core=est.projection['target_plan'], status=est.status)
    files = [HOST, ROOT/'smoke/large_vector.cpp', ROOT/'build/capture/libsim_soc_adapter.so',
             ROOT/'operator_model/quant_resource.py', ROOT/'configs/chips/Ascend950DT_9582.json',
             Path(__file__), Path(__file__).with_name('native.py')]
    n = SHAPE[0]*SHAPE[1]
    job = dict(utc=now(), shape=SHAPE, wall_s=12*3600, raw_cap_gib=24, api='aclnnDynamicMxQuant',
               scenario=s['scenario'], dispatch_condition=node['condition'], outputs=outputs,
               items_to_verify=[n, n//32], logical_bytes=dict(input=2*n, y=n, scale=n//32),
               acquisition='950DT single unchunked dispatch. Check full FP8 y and E8M0 scales; measure native per-core wrapper start→done. Instruction window and cross-SKU CA unavailable.',
               queued_after=str(PREVIOUS.relative_to(ROOT)), binary_and_source={str(p.relative_to(ROOT)):digest(p) for p in files})
    DIR.mkdir(parents=True,exist_ok=True)
    with (DIR/'freeze.json').open('x') as f:
        json.dump(job, f, ensure_ascii=False, indent=2)
        f.write('\n')


def run():
    j = read(DIR/'freeze.json')
    if (DIR/'native.json').exists():
        return
    # The active Cast and this launch share one native CA slot. The preceding
    # job writes a receipt on completion or budget expiry. Do not submit twice.
    deadline = datetime.fromisoformat(j['utc']).timestamp() + 6*3600
    while not PREVIOUS.is_file():
        if time.time() >= deadline:
            raise TimeoutError('The preceding native CA did not produce a terminal receipt')
        time.sleep(45)
    registration = DIR/'capture-registration.json'
    refs = dict(j['binary_and_source'])
    if registration.is_file():
        receipt = read(registration)
        if digest(DIR/'freeze.json') != receipt['original_freeze_sha256']:
            raise ValueError('Original model and launch registration changed')
        allowed = {'development/validation/large20/quant_full.py','development/validation/large20/native.py'}
        if set(receipt['capture_sources']) != allowed:
            raise ValueError('Only the two host collection helpers can be replaced')
        refs.update(receipt['capture_sources'])
    for name, expected in refs.items():
        if digest(ROOT/name) != expected:
            raise ValueError(f'Frozen input changed: {name}')
    result = capture_vector(j, RUN, HOST, ['quant', *map(str,SHAPE)], j['items_to_verify'])
    result['op'] = 'quant'
    write(DIR/'native.json', result)
    if result['qualified_core_envelope']:
        physical = counters(RUN)
        if physical:
            write(DIR/'controllers.json',physical)


if __name__=='__main__':
    {'freeze':freeze,'run':run}[sys.argv[1]]()
