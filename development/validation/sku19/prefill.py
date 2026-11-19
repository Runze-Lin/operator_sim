"""Price actual per-card shapes and run bounded, single-launch unchunked CA pilots."""
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import json
import sys
ROOT=Path(__file__).resolve().parents[3]
sys.path.insert(0,str(ROOT))
from development.validation.sku19.campaign import CHIPS, DT, OUT, digest, forecast, forecast_old, now, read, write
from development.validation.sku19.collect import capture, qualify

TARGET=OUT.parent/'prefill'
RAW=ROOT/'runs/sku19-unchunked'
SCENARIOS=[('Flash-32k-b8',8,32768,5120),('Kimi-Pro-32k-b8',8,32768,7168),
           ('Flash-128k-b1',1,131072,5120),('Kimi-Pro-512k-b1',1,524288,7168)]
BINARY='build/smoke/large_vector'


def plan():
    TARGET.mkdir(exist_ok=True)
    if (TARGET/'registration.json').exists():raise RuntimeError('prefill already registered')
    jobs=[];by_id={}
    for name,b,s,width in SCENARIOS:
        total=b*s
        for limit in (512,2048,total):
            for op in ('rmsnorm','quant'):
                ident=f'{op}_{limit}_{width}'
                by_id[ident]=dict(op=op,shape=[limit,width],id=ident)
                jobs.append(dict(name=name,op=op,chip_batch=b,input_per_request=s,width=width,
                                 total_input_tokens=total,rank_token_budget=limit,kernel_rows=limit,
                                 chunks=total//limit,id=ident))
    new=forecast(list(by_id.values()))
    old=forecast_old(list(by_id.values()))
    write(TARGET/'forecasts.json',dict(utc=now(),scenarios=jobs,v019=new,v018=old,
          contract='One per-token operator per chunk * chunk count. Isolated launches, full hidden dimension, no padding/sequence parallelism or other graph work. Quant is conditional on an activation quant boundary.'))
    src=ROOT/'smoke/large_vector.cpp'
    runners=[]
    for op in ('rmsnorm','quant'):
        shape=[262144,5120]
        runners.append(dict(id=f'{op}_262144_5120',op=op,shape=shape,attrs={},runner=BINARY,runner_sha256=digest(ROOT/BINARY),
                            args=[op,*map(str,shape)],wall_s=900,raw_mib=3072,kernels=1,batch=None,nchecks=2,
                            outputs=[dict(op=op,shape=shape,attrs={},indices=[0])]))
    write(TARGET/'registration.json',dict(utc=now(),scope='Unchunked DT only, bounded pilot; exact dispatch and all outputs when completed. Partial windows and failed golden are ineligible.',
          runner_source_sha256=digest(src),forecast_sha256=digest(TARGET/'forecasts.json'),
          main_freeze_sha256=digest(OUT/'freeze.json'),jobs=runners))


def run():
    reg=read(TARGET/'registration.json')
    assert digest(TARGET/'forecasts.json')==reg['forecast_sha256']
    assert digest(OUT/'freeze.json')==reg['main_freeze_sha256']
    def one(j):
        d=RAW/j['id'];out=TARGET/(j['id']+'.json')
        if out.exists():return
        assert digest(ROOT/BINARY)==j['runner_sha256']
        cost=capture(j,DT,d)
        result=dict(op=j['op'],shape=j['shape'],chip=DT,cost=cost,reference_us=None,eligible=False)
        try:
            record,=qualify(j,DT,d)
            result.update(record=record,eligible=True,reference_us=record['ca_us'])
        except (ValueError,AssertionError,OSError,IndexError,KeyError) as exc:
            result['reason']=repr(exc)
        text=(d/'run.log').read_text(errors='replace')
        result['kernel_dispatch_seen']='Start profiling on kernel:' in text
        result['correctness_pass']='SMOKE_PASS' in text
        result['signals']=[s for s in text.splitlines() if any(k in s for k in ('case=','SMOKE_FAIL','[ERROR]','[WARN]'))][-12:]
        write(out,result)
        print(j['id'],result['eligible'],cost['seconds'],flush=True)
    # Keep the number of simultaneous heavyweight pilots bounded to one.
    for job in reg['jobs']:one(job)

if __name__=='__main__':{'plan':plan,'run':run}[sys.argv[1]]()
