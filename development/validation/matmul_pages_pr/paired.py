"""Plot the archived two-wave Matmul on both SKUs using identical time scales."""
import json

import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt
from matplotlib.patches import Patch

from freeze import DT, OUT, ROOT
from publish import PALETTE, draw_panel


def main():
    dt=json.loads(DT.read_text())
    pr=json.loads((OUT/'page-audit.json').read_text())
    registered=json.loads((OUT/'freeze.json').read_text())
    assert dt['shape']==pr['shape']==registered['shape']
    dt_plan=dt['frozen']['resource_model']['target_plan']
    dt_rects=[dt_plan['tile'][:2],dt_plan['tail_wave_histogram'][0][:2]]
    dt_forecast=[dict(tile,rect=rect) for tile,rect in zip(dt['model_tiles'],dt_rects)]
    dt_measured=[dict(tile,rect=[geom['size']]*2)
                 for tile,geom in zip(dt['ca_tiles'],dt['per_core_tiles'][dt['core']])]
    model=registered['resource_model']
    columns=[('950DT',dt_forecast,dt_measured,dt['model_us'],dt['ca_us'],dt['launch_us'],dt['core'],
              dt_plan['active_aic'],dt['counts']['CUBE']),
             ('950PR',model['model_tiles'],pr['ca_tiles'],model['latency_us'],pr['ca_us'],model['launch_us'],
              pr['core'],pr['core_count'],pr['mmad'])]
    limit=max(max(column[3:5]) for column in columns)
    fig,axes=plt.subplots(2,2,figsize=(16,8),sharex=True)
    fig.subplots_adjust(left=.11,right=.982,top=.76,bottom=.14,hspace=.42,wspace=.34)
    fig.text(.11,.944,'MatMulV3  |  1280 × 2048 × 512',fontsize=25,weight='bold',color='#192e3b')
    fig.legend([Patch(color=v) for v in PALETTE.values()],['GM → L1','L1 → L0','Cube / MMAD','FIXP'],
               loc='upper right',bbox_to_anchor=(.985,.976),frameon=False,ncol=4)
    for i,(sku,pred,actual,mp,ca,launch,core,cores,mmads) in enumerate(columns):
        x=axes[0,i].get_position().x0
        fig.text(x,.849,f'{sku}   ·   {cores} AICs   ·   {mmads} MMADs',fontsize=16,weight='bold',color='#344f5e')
        for row,(tiles,duration) in enumerate(((pred,mp),(actual,ca))):
            draw_panel(axes[row,i],tiles,duration,measured=bool(row),launch=launch,core=core.split('.')[0],
                       maxrows=6,limit=limit)
        axes[-1,i].set_xlabel('Device time from first instruction on any core (µs)',labelpad=8)
    fig.text(.11,.044,'Fixed axes across both SKUs. Ticks mark actual MMAD issues. Faded model reads are inferred from page spacing; CA FIXP windows include waits.',
             color='#425866',fontsize=10.5)
    fig.savefig(ROOT/'docs/examples/figures/matmul-tile-pages-dt-pr.png',dpi=150,facecolor='white')
    plt.close(fig)


if __name__=='__main__':
    main()
