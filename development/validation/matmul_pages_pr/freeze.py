"""Seal 950PR MatMul page timings and tile counts before any observation."""
from datetime import datetime,timezone
from hashlib import sha256
from math import isclose
from pathlib import Path
import json
import sys

ROOT=Path(__file__).resolve().parents[3]
OUT=ROOT/'results/v0.21/matmul-pages-pr'
sys.path.insert(0,str(ROOT))
from operator_model import estimate,load_chip

SHAPE=[1280,2048,512]
PR='Ascend950PR_957b'
RUNNER=ROOT/'runs/dt-matmul-two-tiles-two-k-pages-runner/three_ops'
DT=ROOT/'results/v0.20/examples/matmul-tile-pages.json'


def digest(path):return sha256(path.read_bytes()).hexdigest()


def schedule(sku):
    y=estimate('matmul',SHAPE,chip=sku,timing_mode='resource')
    d=y.projection['timing_diagnostics'];plan=d['target_plan'];f=1000*load_chip(sku).frequency_ghz
    launch=d['stages_us']['launch'];p=d['pages_per_output_tile'];cost=d['wave_cost_cycles']
    assert p>0 and plan['waves']==2
    a,b=cost['full'],cost['tail']
    read1=max(a['Ri'],(a['R']+a['L'])/2)
    cube0=a['R']+a['L'];write0=cube0+a['P']+a['C']
    cube1=max(read1+b['R']+b['L'],cube0+a['Ci'])
    write1=max(cube1+b['P']+b['C'],write0+a['Wi'])
    tiles=[]
    rects=[plan['tile'][:2],max(plan['tail_wave_histogram'],key=lambda a:a[0]*a[1])[:2]]
    def us(x):return launch+x/f
    for rect,rd,core,wr,v in zip(rects,[0,read1],[cube0,cube1],[write0,write1],[a,b]):
        pages=[];step=v['P']/(p-1) if p>1 else 0
        for i in range(p):
            # A settled page is shifted by the modeled II; per-opcode starts
            # inside the stage are not individually resolved by the cost model.
            r=rd+i*step;c=core+i*step
            pages.append(dict(MTE2=[us(r),us(r+v['R'])],
                MTE1=[us(r+v['R']),us(r+v['R']+v['L'])],MMAD=[us(c),us(c+v['C'])]))
        tiles.append(dict(rect=rect,pages=pages,write=[us(wr),us(wr+v['Wi']),us(wr+v['W'])]))
    assert isclose(max(t['write'][-1] for t in tiles)+d['stages_us']['shared_link_floor'],y.latency_us,abs_tol=1e-8)
    return dict(latency_us=y.latency_us,launch_us=launch,model_tiles=tiles,diagnostics=d)


if __name__=='__main__':
    dt=json.loads(DT.read_text())
    old=schedule('Ascend950DT_9582');assert isclose(old['latency_us'],dt['model_us'],abs_tol=1e-8)
    # Reproduce all 30 DT plotting coordinates. PR is derived with the same equations.
    for a,b in zip(old['model_tiles'],dt['model_tiles']):
        for x,y in zip(a['write'],b['write']):assert abs(x-y)<1e-8
        for ap,bp in zip(a['pages'],b['pages']):
            for k in ('MTE2','MTE1','MMAD'):
                for x,y in zip(ap[k],bp[k]):assert abs(x-y)<1e-8
    orig=dt['frozen']['case']
    assert digest(RUNNER)==orig['runner_sha256']
    j={**orig,'wall_s':3600,'raw_mib':6144,'runner':str(RUNNER.relative_to(ROOT))}
    src=[RUNNER,RUNNER.with_suffix('.cpp'),DT,Path(__file__),ROOT/'configs/chips'/f'{PR}.json',
         *(ROOT/f'operator_model/{n}.py' for n in ('matmul_resource','matmul_v15','matmul_waves','chips')),
         ROOT/'scripts/cann-run.sh',ROOT/'development/validation/sku17/campaign_collect.py']
    reg=dict(utc=datetime.now(timezone.utc).isoformat(),sku=PR,shape=SHAPE,attrs={},dtype='bf16',
        kernel_contract='aclnnMatmul contiguous BF16 ND, ordinary non-transposed narrow-precision output',
        job=j,resource_model=schedule(PR),sources={str(p.relative_to(ROOT)):digest(p) for p in src},
        admission='One new PR capture after completed KDA compact slot; no retries. 60 min, 6 GiB raw maximum.',
        original_dt_trace_sha256=dt['trace_sha256'],future_us=estimate('matmul',SHAPE,chip='Ascend970DT').latency_us,
        scope='Full all-core instruction makespan; compulsory output count and numeric golden, runtime SKU, target memory configuration, completed official parse, all-core duration check.')
    with (OUT/'freeze.json').open('x') as fp:fp.write(json.dumps(reg,indent=2)+'\n')
    print(reg['utc'],reg['resource_model']['latency_us'])
