"""Freeze cache-aware Cast estimates and mark any acquisition already started."""
from time import perf_counter
from large_cast import CHIPS, DT, PR, OUT, RAW, ROOT, MODULE, digest, estimate, now, read, write

FROZEN=OUT/'cast-v20-freeze.json'

def freeze():
    if FROZEN.exists():raise FileExistsError(FROZEN)
    stamp=now()
    jobs=read(OUT/'registration.json')['jobs']
    jobs=jobs+[dict(id='cast_262144_5120',shape=[262144,5120])]
    rows=[]
    for j in jobs:
        t,h=j['shape']
        shape=j['shape']
        sources={}
        for sku in (DT,PR):
            path = RAW/(j['id']+'_'+sku)
            if h==5120 and t==262144:
                sources[sku]='already running, native wrapper only' if sku==DT else 'unmeasured'
            elif t==262144:
                sources[sku]='development: CA timing observed before model freeze'
            else:
                sources[sku]='parallel replay: CA started before new model freeze' if (path/'run.log').exists() else 'independent: CA started after new model freeze'
        targets={}
        for name in CHIPS.values():
            start=perf_counter()
            e=estimate('cast_bf16_fp32',shape,chip=name,timing_mode='resource')
            elapsed=(perf_counter()-start)*1000
            assert e.latency_us and e.projection['timing_diagnostics']['modeled_external_bytes']>=2*t*h
            targets[name]=dict(us=e.latency_us,query_ms=elapsed,diagnostics=e.projection['timing_diagnostics'])
        rows.append(dict(id=j['id'],shape=shape,roles=sources,targets=targets))
    write(FROZEN,dict(utc=stamp,resources_sha256=digest(MODULE),chip_sha256={s:digest(ROOT/'configs/chips'/f'{s}.json') for s in CHIPS.values()},
          old_freeze='registration.json',development_evidence=['cast_262144_32_native-dram.json','cast_32_4096_native-dram.json'],
          cold_inputs_and_writeback='Assume read-once input and noncompeting L2, reserve both input buffers per active AIV. Capability extrapolation; target CA not required.',
          jobs=rows))

if __name__=='__main__':freeze()
