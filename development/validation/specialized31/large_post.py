"""Large standalone Post: price two distinct providers before a paired serial acquisition."""
import fcntl
import json
import math
import os
from pathlib import Path
import shutil
import sys

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
from development.validation.large29.fusion import fusion_campaign as base

OUT = ROOT / 'results/v0.31/post'
RAW = ROOT / 'runs/specialized31/post'
REG = OUT / 'freeze.json'


def freeze():
    previous = base.read(base.REG)
    tokens, channels = 1024, 5120
    jobs = []
    for op, shape, runner in (('vllm_hc_post', [1,tokens,4,channels], base.HC),
                              ('mhc_post', [tokens,4,channels], base.BUILTIN)):
        options = base.attrs[op]
        if any(j['op'] == op and j['shape'] == shape for j in previous['jobs']):
            raise ValueError('already registered')
        assert all(tuple(row['shape']) != tuple(shape) for row in
                   base.read(ROOT / 'results/v0.30/replay.json')['rows'] if row['op'] == op)
        prices = {c: base.obs(op, shape, c, options) for c in base.SKUS}
        if op == 'mhc_post':
            for c in base.SKUS:
                prices[c]['known_first'] = base.obs(op, shape, c, dict(options, resident_aiv=0))
        jobs.append(dict(id=op, op=op, shape=shape, index=base.arguments(op,shape),
                         runner=base.file_id(runner), outputs=[dict(op=op)], attrs=options,
                         predictions=prices, expected_numerics={'output': tokens*4*channels},
                         wall_s=3*3600, raw_mib=24*1024))
    files = [Path(__file__), Path(base.__file__), base.REG, base.HC, base.BUILTIN,
             *[ROOT / 'configs/chips' / (s + '.json') for s in base.SKUS]]
    item = dict(utc=base.now(), scenario='Flash width D=5120, four residual streams, 1024 per-rank '
                'scheduled tokens (e.g. 8 requests x 128 tokens per chunk). Each provider has its own '
                'one-call process and host numerical test. Standalone elapsed times do not validate a fused serving dispatch.',
                jobs=jobs, dependencies={base.file_id(p):base.sha(p) for p in files},
                procedure='Serial, niced, full eligible host CPU set, same published host, package, whole-output golden, '
                'SKU, installed .text, physical END and per-lane official duration. Each endpoint gets one 3h/24GiB attempt; '
                'reserve 64GiB above raw cap. No 970 CA.')
    with REG.open('x') as f:
        json.dump(item,f,indent=2,allow_nan=False); f.write('\n')


def collect():
    reg = base.read(REG)
    for file, sha in reg['dependencies'].items():
        if base.sha(ROOT/file) != sha:
            raise ValueError(f'Frozen dependency changed: {file}')
    providers = base.read(base.REG)
    RAW.mkdir(parents=True,exist_ok=True)
    os.nice(15)
    os.environ.update(OMP_WAIT_POLICY='PASSIVE', GOMP_SPINCOUNT='0')
    def state(phase, **kwargs):
        base.save(OUT/'state.json',dict(utc=base.now(),pid=os.getpid(),phase=phase,**kwargs))
    with (RAW/'lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX|fcntl.LOCK_NB)
        for j in reg['jobs']:
            for chip in base.DEV:
                prefix=j['op']+'-'+chip
                dest=OUT/(prefix+'.json')
                if dest.exists():
                    continue
                raw=RAW/prefix
                if not (raw/'cost.json').exists():
                    if raw.exists():
                        raise RuntimeError(f'Incomplete raw exists at {raw}')
                    if shutil.disk_usage(ROOT).free < (64+j['raw_mib']/1024)*2**30:
                        raise OSError('Disk reserve unavailable')
                    state('acquiring',endpoint=prefix)
                    base.launch(j,chip,raw)
                state('qualifying',endpoint=prefix)
                cost=base.read(raw/'cost.json')
                try:
                    if cost['started_utc']<=reg['utc'] or cost['stop_reason'] or cost['exit_code']!=0:
                        raise ValueError('incomplete acquisition or budget')
                    receipt=base.verify(j,chip,raw,providers)
                    stage,=receipt['stages']
                    assert stage['active_aiv']>0 and stage['physical_end_all_lanes']
                except (ValueError,OSError,AssertionError,RuntimeError,StopIteration,KeyError,IndexError) as e:
                    receipt=dict(qualified=False,reason=repr(e),ca_us={})
                ref=dict(op=j['op'],chip=chip,shape=j['shape'],cost=cost,freeze_sha256=base.sha(REG),**receipt)
                base.save(dest,ref)
        state('finished')


if __name__ == '__main__':
    {'freeze': freeze, 'run': collect}[sys.argv[1]]()
