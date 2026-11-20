"""Same-row resource chain and verified CA intervals for H=1→4 versus H=4→4 on 950PR."""
from pathlib import Path
import sys,json
ROOT=Path(__file__).resolve().parents[3];sys.path.insert(0,str(ROOT))
from development.validation.rms_ordered_verify.verify import digest,events
from development.validation.rms27.landmarks import OUT
from development.validation.sku19.campaign import read,write
from operator_model import estimate
from operator_model.rms_dispatch import price_rms
from operator_model import load_chip

def measured(row):
    s=row['stage'];base=s['first_tick'];folder=(ROOT/s['trace']).parent;pc0=int((folder.parent/'dump/pc_start_addr.txt').read_text(),16)
    cells=[]
    for core,ref in s['cores'].items():
        shard=folder/core/'trace.json';assert digest(shard)==ref['shard_sha256'];inst={}
        for ev in events(shard):
            if ev['name']!='VF' and not ev['name'].startswith('MOV_SRC_TO_DST'):continue
            pc=int(ev['args']['pc_addr'],16)-pc0;detail=json.loads(ev['args']['detail'])
            kind={0x3e9c:'tiling',0x4878:'gamma',0x4a48:'x',0x4e64:'rstd VF',0x5008:'normalize',0x4f38:'rstd',0x5128:'y'}.get(pc,'reduce')
            assert kind not in inst
            inst[kind]=dict(b=(round(ev['ts']*1650)-base)/1650,e=(round((ev['ts']+ev['dur'])*1650)-base)/1650,
                            dst=detail.get('dst_addr'),src=detail.get('src_addr'),bytes=detail.get('processed_bytes'),pc=hex(pc))
        assert len(inst)==8 and inst['rstd']['bytes']==4 and inst['y']['bytes']==2*row['shape'][1]
        cells.append(dict(core=core,arrival=(ref['first_tick']-base)/1650,end=(ref['last_tick']-base)/1650,ops=inst))
    cells.sort(key=lambda x:x['ops']['rstd']['dst'])
    y0=cells[0]['ops']['y']['dst'];r0=cells[0]['ops']['rstd']['dst'];w=row['shape'][1]
    assert all(c['ops']['rstd']['dst']==r0+4*i and c['ops']['y']['dst']==y0+2*w*i for i,c in enumerate(cells))
    return cells

def build():
    receipt=read(OUT/'receipts/partition-Ascend950PR_957b.json');a,b=[r for r in receipt['rows'] if r['shape']==[4,688]]
    assert (a['prior_aiv'],b['prior_aiv'])==(1,4) and a['timing_qualified'] and b['timing_qualified']
    warm=measured(b);added=measured(a);idx=max(range(4),key=lambda i:added[i]['end']);sku=b['chip'];array=[]
    for r,cells in [(a,added),(b,warm)]:
        chip=load_chip(r['chip'])
        if r['role']=='entry':
            us,_,_,diag=price_rms(r['shape'],chip,{'resident_aiv':1});assert abs(us-r['upper_us'])<1.e-12
            stages=[];t=0
            for kind,value in diag['stages_us'].items():
                name='front' if kind in ('arrival','frontend_to_rstd_issue','drain') else 'read' if kind=='MTE2' else 'vector' if kind.startswith('SIMD') else 'write'
                stages.append(dict(name=name,b=t,e=t+value));t+=value
            assert abs(t-us)<1.e-8
        else:
            # Frozen model prices one sector's issue-order statistics, without assigning a rank to a physical AIV.
            # Draw the median-row issue and actual modeled last completion on each port.
            import math
            from operator_model.rms_resource import READ_PER_KIB,WRITE_PER_KIB
            _,_,_,diag=price_rms(r['shape'],chip,{})
            p=read(OUT/'prices.json');z=r['detail'];need=z['gamma_issue_cycles'];w=r['shape'][1];size=2*w/1024
            def service(k,x,scale=1.):
                base,slope=p['curves'][k];return base+x*slope*scale
            rp=diag['read_cycles_per_kib'];wp=diag['write_cycles_per_kib']
            rscale=rp/READ_PER_KIB if chip.details.get('mte2_bandwidth_gbs') else 1
            vscale=diag['vector_price_cycles_per_reference_cycle']
            gamma=need+service('gamma_response',size,rscale);x=need+p['x_lead'];x_end=x+service('x_response',size,rscale)
            ready=max(gamma,x_end,need+2*size*rp);stat=ready+service('reduce',math.ceil(w/128),vscale)
            yi=stat+service('normalize',math.ceil(w/128),vscale)
            def part(name,b,e,dy=0.):return dict(name=name,b=b/1650,e=e/1650,dy=dy)
            us=r['covered_us']
            stages=[part('front',0,need),part('read',need,gamma,-.14),part('read',x,x_end,.14),
                    part('vector',ready,stat,-.14),part('vector',stat,yi,.14),
                    part('write',stat,z['final_rstd_cycles'],-.14),part('write',yi,z['final_y_cycles'],.14),
                    part('front',max(z['final_rstd_cycles'],z['final_y_cycles']),us*1650)]
        array.append(dict(role=r['role'],H=r['prior_aiv'],conditional_us=us,unknown_us=r['us'],stages=stages,
                          lane=cells[idx],device_us=r['stage']['ca_us'],terminal=max(cells,key=lambda c:c['end'])['core'],
                          trace=r['stage']['trace'],sha256=r['stage']['trace_sha256'],hot_resource=r['detail'] if r['role']=='repeat' else None))
    for operand,key in [('gamma','src'),('x','src'),('rstd','dst'),('y','dst')]:
        assert added[idx]['ops'][operand][key]==warm[idx]['ops'][operand][key]
    return dict(shape=a['shape'],sku=sku,token=idx,active=4,origin='All-core DEVICE first instruction, identical µs axis throughout',entries=array)

def draw(d):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch
    from matplotlib.lines import Line2D
    colors=dict(front='#a4b8c2',read='#137b86',vector='#7455aa',write='#b34f30',resource='#59738a')
    fig,axes=plt.subplots(2,2,figsize=(13.2,6.8),sharex=True);fig.subplots_adjust(left=.14,right=.97,top=.75,bottom=.15,wspace=.20,hspace=.42)
    fig.suptitle('950PR RMSNorm  |  4 × 688',x=.14,ha='left',y=.975,fontsize=21,fontweight='bold',color='#15303d')
    fig.text(.14,.89,f"Four active AIVs  ·  same BF16 x, γ and output row {d['token']}  ·  first seen after one active AIV, then repeated",fontsize=11)
    keys=[('front','Dispatch / scalar'),('read','MTE2'),('vector','Vector / VF'),('write','MTE3')]
    fig.legend(handles=[Patch(color=colors[a],label=b)for a,b in keys]+[Line2D([],[],ls=':',color='#334855',label='Unknown-history estimate')],loc='upper left',bbox_to_anchor=(.135,.875),ncol=6,frameon=False,fontsize=9)
    max_x=max(m['device_us']for m in d['entries'])*1.10; names=['Dispatch','MTE2 reads','Vector','MTE3 writes']
    lanes={'front':0,'read':1,'vector':2,'write':3,'resource':4}
    for col,m in enumerate(d['entries']):
        top,bottom=axes[:,col];lane=m['lane'];tit='Second partition starts' if not col else 'Immediate full reuse'
        top.set_title('Resource model  |  '+tit,loc='left',fontsize=10,weight='bold');bottom.set_title('CA  |  '+lane['core']+(' · last' if lane['core']==m['terminal'] else ' · matched output row'),loc='left',fontsize=10,weight='bold')
        def bar(ax,y,a,b,k):ax.broken_barh([(a,b-a)],(y-.12,.24),facecolor=colors[k])
        for s in m['stages']:bar(top,lanes[s['name']]+s.get('dy',0.),s['b'],s['e'],s['name'])
        if col:
            for label,q in [('rstd',m['hot_resource']['final_rstd_cycles']),('y',m['hot_resource']['final_y_cycles'])]:
                x=q/1650;top.plot(x,3,marker='|' if label=='rstd' else 'o',ms=8 if label=='rstd' else 3,color=colors['write'])
        bottom.plot(lane['arrival'],0,'|',ms=11,color='#314956');offset={'tiling':-.2,'gamma':0,'x':.2,'reduce':-.21,'rstd VF':0,'normalize':.21,'rstd':-.15,'y':.15}
        for kind,v in lane['ops'].items():
            k='read' if kind in ('tiling','gamma','x') else 'write' if kind in ('rstd','y') else 'vector'
            bar(bottom,lanes[k]+offset[kind],v['b'],v['e'],k)
        for ax,us in [(top,m['conditional_us']),(bottom,m['device_us'])]:
            ax.axvline(m['unknown_us'],ls=':',lw=1.3,color='#334855');ax.axvline(us,ls='--',lw=1,color='#233946')
            ax.set_title(f'{us:.3f} µs',loc='right',fontsize=11,weight='bold');ax.set_ylim(3.55,-.55);ax.set_yticks(range(4),names if col==0 else ['']*4);ax.set_xlim(0,max_x);ax.grid(axis='x',color='#dbe4e8',zorder=0);ax.set_axisbelow(True);ax.spines[['top','left','right']].set_visible(False);ax.tick_params(axis='y',length=0)
        bottom.set_xlabel('Time from first device instruction (µs)')
    fig.text(.14,.065,'Dotted: same shape-only point for both launches. Dashed: conditional modeled finish above, complete device CA window below.',fontsize=9)
    fig.text(.14,.035,'Warm model: median-row issues; write spans reach the last modeled y and rstd. Physical lane issue rank is not resolved.',fontsize=9)
    fig.savefig(OUT/'pr-second-partition.png',dpi=155);plt.close(fig)
if __name__=='__main__':
    d=build();draw(d);write(OUT/'pr-second-partition.json',d)
