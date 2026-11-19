"""One supplementary four-chunk KDA pair and an offline audit of the five-chunk failure."""
import argparse
import fcntl
import json
import os
from pathlib import Path
import subprocess
import sys
import time

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sys.path.insert(0, str(HERE.parent/'kda_prefetch24'))
from capture import ca, qualify, RAW, HOST, GOLD, LEGACY, DT, PR
from evidence import sha, read, OUT as PREVIOUS
from frozen import predict
sys.path.insert(0, str(HERE.parent/'kda_prefetch24'))
from supplement import capacity

OUT = ROOT/'results/v0.26/kda'
REGISTER = OUT/'register.json'
T, H = 248, 1
PARENT = PREVIOUS/'supplement/register.json'


def diagnostic():
    import numpy as np
    from reference import prepare
    observed = {}
    t, h = 257, 1
    reference, mathematical = prepare(t,h)
    cpu = np.fromfile(GOLD/'257-1/aqk.bin', dtype='<f8').reshape(h,t,64)
    np.testing.assert_array_equal(reference[0],cpu)
    for sku in (DT, PR):
        case = f'supplement-{sku}-{t}-{h}'
        receipt = read(PREVIOUS/'supplement'/(case+'.json'))
        actual = RAW/case/'aqk.bf16'
        assert sha(actual) == receipt['actual_aux_bf16']['aqk']['sha256'] and not receipt['qualified']
        arr = (np.fromfile(actual, dtype='<u2').astype('<u4') << 16).view('<f4').reshape(h,t,64)
        tolerance = 2e-6+abs(cpu)/64
        bad = np.argwhere(abs(arr-cpu)>tolerance)
        assert len(bad)==1
        key = tuple(bad[0])
        observed[sku] = dict(zero_based_h_row_column=list(map(int,key)),actual=float(arr[key]),
                             official_algorithm_reference=float(cpu[key]),fp64_math=float(mathematical[0][key]),
                             abs_error=float(abs(arr[key]-cpu[key])),allowed=float(tolerance[key]),
                             bf16_sha256=sha(actual),all_output_counts=receipt['output_groups'])
    assert observed[DT]==observed[PR]
    ca.save(OUT/'failed-257.json', observed)


def freeze():
    if REGISTER.exists():
        raise FileExistsError(REGISTER)
    parent = read(PARENT)
    shape = [1,T,H,H,128,128]
    for marker in (PREVIOUS/'register.json', PARENT, ROOT/'results/kda-v23'):
        files = [marker] if marker.is_file() else marker.glob('*.json')
        for path in files:
            d=read(path)
            if isinstance(d,dict):
                cases=d.get('cases',[])+[d.get('case',{})]
                if any(isinstance(c,dict) and c.get('shape')==shape for c in cases):
                    raise ValueError('Already captured or registered')
    pins={k:v for k,v in parent['source_hashes'].items() if '/raw/golden/' not in k}
    for k,v in pins.items():
        if sha(ROOT/k)!=v:
            raise ValueError(k)
    golden=GOLD/f'{T}-{H}'
    if golden.exists():
        raise FileExistsError(golden)
    subprocess.run([str(ROOT/'toolchains/envs/cann-9.2.1/bin/python'),str(LEGACY/'reference.py'),str(T),str(H),str(golden)],
                   env={**os.environ,'OPENBLAS_NUM_THREADS':'1'},check=True)
    for p in [Path(__file__), PARENT, *golden.iterdir()]:
        pins[str(p.relative_to(ROOT))]=sha(p)
    pred={sku:predict(shape,sku) for sku in (DT,PR,'Ascend970DT')}
    reg={k:parent[k] for k in ('published_device_text','budget_per_capture_seconds','budget_per_capture_bytes',
                                'min_free_bytes','min_available_ram_bytes','max_load_per_online_cpu')}
    reg.update(utc=ca.when(),parent_sha256=sha(PARENT),source_hashes=pins,budget_total_capture_seconds=1440,
               registration_reason='T=248/H=1 retains row 243, column 44 that failed in T=257. Last 16-row group truncates to 8. This targeted follow-up was chosen after the numerical failure, before acquiring either new endpoint. No model, input signal or tolerance change.',
               cases=[dict(id=f'prospective26-{sku}-{T}-{H}',sku=sku,shape=shape,physical_memory=ca.MEMORY[sku],
                           oracle=dict(attention=T*H*128,aqk=T*H*64,akk=T*H*64,state_vk=H*128*128),
                           predictions=pred) for sku in (DT,PR)])
    with REGISTER.open('x') as f:
        json.dump(reg,f,indent=2);f.write('\n')


def run():
    r=read(REGISTER)
    for name,h in r['source_hashes'].items():
        if sha(ROOT/name)!=h:
            raise ValueError(name)
    ca.BIN=HOST
    with (RAW/'lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        for c in r['cases']:
            folder=RAW/c['id']
            receipt=OUT/(c['id']+'.json')
            if receipt.exists():
                continue
            while not capacity(r)[1]:
                time.sleep(45)
            if not (folder/'cost.json').exists():
                folder.mkdir()
            ca.save(OUT/'state.json',dict(state='validating endpoint',case=c['id'],utc=ca.when()))
            cost=ca.capture(r,c,folder,r['budget_per_capture_seconds'])
            v=qualify.verify(r,c,folder,cost)
            files={k:folder/(k+'.bf16') for k in ('aqk','akk')}
            v['actual_aux_bf16']={k:dict(bytes=f.stat().st_size,sha256=sha(f)) for k,f in files.items() if f.is_file()}
            v['qualified']=v['qualified'] and all(f.is_file() and f.stat().st_size==2*c['oracle'][k] for k,f in files.items())
            ca.save(receipt,dict(**v,case={k:c[k] for k in ('id','shape','sku','oracle')},cost=cost,registration_sha256=sha(REGISTER)))
        ca.save(OUT/'state.json',dict(state='complete',utc=ca.when()))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('action',choices=('freeze','run','diagnostic'))
    args=p.parse_args()
    {'freeze':freeze,'run':run,'diagnostic':diagnostic}[args.action]()
