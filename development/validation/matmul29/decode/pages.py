"""Compare same K pages of a single-wave Kimi projection against the terminal CA core."""
import argparse
import json
from pathlib import Path
import sys
import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt
from matplotlib.patches import Patch
import campaign as bench
from development.validation.matmul_pages_pr import extract,freeze
from development.validation.matmul_pages_pr.extract import summarize
from unittest.mock import patch
# The original renderer is executable as a sibling script. Bind its two sibling
# imports only while loading it; the acquisition module has its own freeze.py.
with patch.dict(sys.modules, {'freeze':freeze, 'extract':extract}):
    from development.validation.matmul_pages_pr.publish import draw_panel,PALETTE
from operator_model import estimate,load_chip


def make(m,sku):
    reg=bench.read(bench.REG);job,=(r for r in reg['targets'] if r['m']==m)
    obs,_,cite=bench.source_record(job,sku)
    if not obs.get('qualified'):raise ValueError('Complete CA not yet qualified')
    st,=obs['stages']
    physical=summarize(bench.ROOT/st['trace'],job['shape'],st['trace_sha256'],st['ca_us'])
    d=estimate('matmul',job['shape'],chip=sku,dtype='bf16',timing_mode='resource').projection['timing_diagnostics']
    plan=d['target_plan'];rp=job['predictions'][sku];assert plan['waves']==1 and rp['placement']['route']=='ordinary'
    clock=1000*load_chip(sku).frequency_ghz;launch=d['stages_us']['launch']
    v=d['wave_cost_cycles']['tail'];pages=d['pages_per_output_tile'];ii=v['P']/max(1,pages-1)
    us=lambda c:launch+c/clock
    grid=[]
    for i in range(pages):
        lo=ii*i
        grid.append(dict(MTE2=[us(lo),us(lo+v['R'])],MTE1=[us(lo+v['R']),us(lo+v['R']+v['L'])],
                         MMAD=[us(lo+v['R']+v['L']),us(lo+v['R']+v['L']+v['C'])]))
    rect=max(plan['tail_wave_histogram'],key=lambda x:x[0]*x[1])[:2]
    w=v['P']+v['R']+v['L']+v['C'];wt=[us(w),us(w+v['Wi']),us(w+v['W'])]
    assert abs(wt[-1]+d['stages_us']['shared_link_floor']-rp['us'])<1e-7
    model=[dict(rect=rect,pages=grid,write=wt)]
    target=physical['ca_tiles'];chip=sku.split('_')[0];data=dict(shape=job['shape'],chip=sku,
        source=job['role'],receipt=cite,trace=st['trace'],trace_sha256=st['trace_sha256'],
        first='earliest physical core instruction in complete window',ca_us=st['ca_us'],model_us=rp['us'],
        model_plan=plan,model_tiles=model,ca_core=physical['core'],ca_tiles=target,ca_geometry=physical['tile_histogram'],
        ca_mmad=physical['mmad'],model_launch_us=launch,logical_fma_ops=2*m*bench.N*bench.K)
    n=max(sum(1+len(z['pages']) for z in tiles) for tiles in (model,target));xmax=max(st['ca_us'],rp['us'])
    fig,axes=plt.subplots(2,1,sharex=True,figsize=(12,6.9+.41*n))
    fig.subplots_adjust(left=.20,top=.82,right=.95,bottom=.12,hspace=.36)
    fig.text(.20,.96,f'{chip}  |  Kimi MLA query projection  ·  {m} × 2304 × 1536',weight='bold',fontsize=18)
    fig.text(.20,.917,f"TP8  ·  BF16  ·  {physical['core_count']} AICs in CA  ·  {physical['mmad']} MMAD issues across the chip",fontsize=12)
    fig.legend(handles=[Patch(color=PALETTE[k],label=s)for k,s in [('MTE2','MTE2 · GM → L1'),('MTE1','MTE1 · L1 → L0'),
        ('MMAD','Cube · MMAD'),('FIXP','FIXP · output')]],ncol=4,loc='upper left',bbox_to_anchor=(.19,.906),frameon=False)
    for ax,tiles,total,measured in ((axes[0],model,rp['us'],False),(axes[1],target,st['ca_us'],True)):
        draw_panel(ax,tiles,total,measured=measured,launch=launch,core=physical['core'],maxrows=n,limit=xmax)
    axes[-1].set_xlabel('Device time from earliest instruction (µs)')
    fig.text(.20,.054,'Upper panel: frozen single-wave page II and service envelopes. Lower: CA descriptor spans and individual MMAD issue ticks.',size=9)
    fig.text(.20,.030,'Dashed lines: full all-core windows. FIXP descriptor lifetime can include dependency waits. Pale model bars follow the page II.',size=9)
    prefix=f"pages-{m}-{bench.TAGS[sku]}";fig.savefig(bench.D/(prefix+'.png'),dpi=145);plt.close(fig)
    bench.save(bench.D/(prefix+'.json'),data)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('m',type=int);p.add_argument('sku',choices=['dt','pr'])
    args=p.parse_args();make(args.m,bench.DT if args.sku=='dt' else bench.PR)
