"""Same-scale comparisons. The measured panel uses archived descriptor timestamps."""
from math import ceil
import os
from pathlib import Path
import sys
os.environ.setdefault('MPLCONFIGDIR','/tmp/operator-sim-mpl')
import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt
from matplotlib.patches import Patch
from matplotlib.ticker import MaxNLocator

ROOT=Path(__file__).resolve().parents[3]
sys.path.insert(0,str(ROOT))
from development.validation.vector21.campaign import assert_resource_revision, S,C,DT,PR,OUT,read,write
from operator_model import estimate
from operator_model.rms_resource import READ_PER_KIB, WRITE_PER_KIB

V20=OUT.parents[1]/'v0.20'
PIPES=('Read','Vector','Write')
COLORS=('#127a83','#7352a1','#b34b2c')


def sigmoid(shape,chip,d=None,frozen=None):
    if d is None:
        d=read(V20/'newop'/('timeline-prefill128k'+('-pr' if chip==PR else '')+'.json'))
    assert d['shape']==shape and d['chip']==chip
    if frozen is None:
        e=estimate(S,shape,chip=chip)
        frozen=dict(us=e.latency_us,diagnostics=e.projection['timing_diagnostics'])
    diag=frozen['diagnostics']
    plan=diag['target_plan']
    sch=diag['schedule_us']
    model=[]
    for i in range(plan['critical_core_tiles']):
        if i==plan['critical_core_tiles']-1:
            phases=[(sch[f'last_{p}_at'],sch[f'last_{p}']) for p in ('read','vector','write')]
        else:
            a=sch['first_read_at']+sch['interval']*i
            b=a+sch['first_read']
            w=b+sch['first_vector']+sch['write_dependency']
            phases=[(a,sch['first_read']),(b,sch['first_vector']),(w,sch['first_write'])]
        model.append({p:[a,a+w] for p,(a,w) in zip(PIPES,phases)})
    return dict(model=model,observed=d['ca_tiles'],ca_us=d['ca_us'],core=d['terminal_core'],
                trace_sha256=d['trace_sha256'],model_us=frozen['us'],plan=plan,
                first=sch['first_read_at'],phase='Double-buffer schedule')


def cast(shape,chip,d=None,frozen=None):
    path=V20/'large'/f'flash-256ki-tiles-{chip}-plot.json'
    if d is None:
        d=read(path)
    if frozen is None:
        e=estimate(C,shape,chip=chip)
        frozen=dict(us=e.latency_us,diagnostics=e.projection['timing_diagnostics'])
    diag=frozen['diagnostics']
    s=diag['stages_us']
    p=diag['target_plan']
    plan=d['dma'];vec=[(a,w) for a,w in d['conversion_union'] if w>0]
    assert len(vec)==len(plan['MTE2'])==len(plan['MTE3'])==p['busy_core_waves']
    obs=[]
    for rd,(start,w),wr in zip(plan['MTE2'],vec,plan['MTE3']):
        assert rd['bytes']*2==wr['bytes']
        obs.append(dict(Read=[rd['start'],rd['start']+rd['length']],Vector=[start,start+w],
                        Write=[wr['start'],wr['start']+wr['length']]))
    cap=p['ub_tile_elements'];local=p['busy_core_elements']
    tiled_start=s['arrival']+s['control_to_read']
    first_end=tiled_start+s['read']+s['vector']+s['dependency_to_write']+s['write']
    follow=frozen['us']-first_end
    full_ii=follow*cap/(local-cap)
    fx=1000*diag['external_floor_cycles']*diag['effective_external_gbs']/diag['modeled_external_bytes']
    model=[]
    for i in range(p['busy_core_waves']):
        count=min(cap,local-i*cap)
        # The API sets the total max(local stages, aggregate service). This
        # offline interpolation assigns the exposed following time by payload.
        end=first_end+min(local-cap,i*cap)*full_ii/cap
        rd_time=((965-8*READ_PER_KIB)+count*2/1024*diag['read_cycles_per_kib'])/fx
        vec=s['vector']*ceil(count/64)/ceil(cap/64)
        write=((455-16*WRITE_PER_KIB)+count*4/1024*diag['write_cycles_per_kib'])/fx
        v_end=end-write-s['dependency_to_write']
        model.append(dict(Read=[v_end-vec-rd_time,v_end-vec],Vector=[v_end-vec,v_end],Write=[end-write,end]))
    assert abs(model[-1]['Write'][-1]-frozen['us'])<1e-8
    return dict(model=model,observed=obs,ca_us=d['ca_us'],core=d['terminal'],
                trace_sha256=d['trace_sha256'],model_us=frozen['us'],plan=p,
                first=tiled_start,phase='Rate-equivalent tile schedule')


def render(op,shape,chip,source=None,slug=None,frozen=None):
    d=(sigmoid if op==S else cast)(shape,chip,source,frozen)
    n=len(d['observed']);assert n==len(d['model'])
    indices=sorted({0,1,n//2,n-2,n-1})
    tag='dt' if chip==DT else 'pr'
    slug=slug or ('sigmoid-128ki' if op==S else 'cast-256ki')+'-'+tag
    write(OUT/(slug+'.json'),dict(op=op,shape=shape,chip=chip,origin='earliest core instruction',**d))
    fig,ax=plt.subplots(2,1,figsize=(12,7.6),sharex=True)
    fig.subplots_adjust(left=.15,right=.967,top=.786,bottom=.125,hspace=.53)
    title=('FP32 Sigmoid' if op==S else 'BF16 → FP32 Cast')
    fig.text(.15,.957,f'950{tag.upper()}  {title}   |   {shape[0]:,} × {shape[1]}',fontsize=23,weight='bold',color='#172e3b')
    fig.text(.15,.913,f'Model and CA  ·  {d["plan"]["active_aiv"]} AIVs  ·  {n} UB tiles on the terminal core  ·  first, middle and last tiles shown',fontsize=12,color='#435b6a')
    fig.legend(handles=[Patch(color=c,label=s) for c,s in zip(COLORS,('MTE2 / GM → UB','Vector / SIMD','MTE3 / UB → GM'))],loc='upper left',bbox_to_anchor=(.14,.89),ncol=3,frameon=False)
    maximum=1.05*max(d['model_us'],d['ca_us'])
    for ax0,series,caption,end in zip(ax,('model','observed'),
        ('Resource model  ·  '+d['phase'],f'CA trace  ·  {d["core"]} (last to finish)'),(d['model_us'],d['ca_us'])):
        ax0.set_title(caption,loc='left',fontsize=14,fontweight='bold',color='#233b48',pad=9)
        ax0.set_title(f'{end:.3f} µs',loc='right',fontsize=15,fontweight='bold',color='#233b48',pad=9)
        ax0.set_xlim(0,maximum); ax0.set_ylim(len(indices)-.45,-.55)
        for j,idx in enumerate(indices):
            for k,(pipe,color) in enumerate(zip(PIPES,COLORS)):
                a,b=d[series][idx][pipe]; y=j+(k-1)*.18
                ax0.barh(y,b-a,left=a,height=.17,color=color,edgecolor=color,lw=.2)
                ax0.vlines(a,y-.1,y+.1,lw=.8,color=color)
            if j and idx-indices[j-1]>1:
                ax0.axhline(j-.5,color='#d6dfe5',lw=.8)
        if series=='model':
            ax0.barh(-.20,d['first'],height=.16,color='#bac9d0')
        ax0.axvline(end,color='#476270',ls='--',lw=1.2)
        ax0.set_yticks(list(range(len(indices))),[f'UB tile {i+1}'+(' · tail' if i==n-1 else '') for i in indices])
        ax0.tick_params(axis='both',length=0,labelcolor='#345063')
        ax0.grid(axis='x',color='#dee6eb');ax0.set_axisbelow(True);ax0.spines[:].set_visible(False)
    ax[1].xaxis.set_major_locator(MaxNLocator(nbins=12))
    ax[1].set_xlabel('Device time from first instruction on any core (µs)',labelpad=11)
    foot=('Upper panel distributes exposed shared service by following-tile bytes; its internal slots are rate equivalents.'
          if op==C else 'Upper panel schedules two UB buffers. Lower Vector spans run from the first RV_VLD to the last RV_VST in each tile.')
    fig.text(.15,.025,foot,fontsize=9,color='#435b6a')
    fig.savefig(ROOT/'docs/examples/figures'/('v021-'+slug+'.png'),dpi=146)
    plt.close(fig)
if __name__=='__main__':
    assert_resource_revision()
    for op,shape in ((S,[1048576,12]),(C,[2097152,32])):
        for chip in (DT,PR):
            render(op,shape,chip)
