"""Compare the same four logical output rows under ordinary reuse and an immediately identical launch."""
from pathlib import Path
import json,sys
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
ROOT=Path(__file__).resolve().parents[3];sys.path.insert(0,str(ROOT))
from development.validation.post24.collect import events
from development.validation.post24.figure import union
from development.validation.post27.issue import OUT,P25,predict,vector_work
from development.validation.sku19.campaign import read,write
from operator_model import OperatorStream,load_chip
from operator_model.post_resource import resource_post
from operator_model.post_token_pipeline import token_pipeline
SKU='Ascend950DT_9582';TARGET=[196,4,1536]
def collect(r,wanted=None):
    s=r['stage'];p=ROOT/s['trace'];t,n,d=r['shape'];origin=s['first_tick'];stores=[];clk=1650
    for ev in events(p):
        if not ev['name'].startswith('MOV_SRC_TO_DST'):continue
        z=json.loads(ev['args']['detail'])
        if z['queue_type']!='MTE3' or any(z.get(k) for k in ('is_sync','is_wait','is_duration_excluded')):continue
        assert z['dst_mem']=='OUT' and z['burst_number']==4 and z['dst_stride']==z['burst_len']==d*2
        stores.append((z['dst_addr'],z['processed_bytes'],ev['pid']))
    stores.sort();base=stores[0][0];stride=2*n*d;assert len(stores)==t and all(adr==base+i*stride and sz==stride for i,(adr,sz,k) in enumerate(stores))
    terminal=max(s['windows'],key=lambda k:s['windows'][k][1])
    if wanted is None:
        by_lane={}
        for i,(_,_,pid) in enumerate(stores):by_lane.setdefault(pid,[]).append(i)
        longest=max(map(len,by_lane.values()));assert longest==r['plan']['rows_per_core']
        chosen=max((pid for pid,v in by_lane.items() if len(v)==longest),key=lambda pid:s['windows'][pid][1])
        wanted=by_lane[chosen]
    lanes={stores[u][2] for u in wanted};assert len(lanes)==1
    lane=lanes.pop();reads=[];writes=[];vectors=[]
    for ev in events(p.parent/lane/'trace.json'):
        z=json.loads(ev['args']['detail']);a=(round(ev['ts']*clk)-origin)/clk;b=(round((ev['ts']+ev['dur'])*clk)-origin)/clk
        if b<=a or any(z.get(k) for k in ('is_sync','is_wait','is_duration_excluded')):continue
        if z['instr_type'].startswith(('RVEC','SIMD')):vectors.append((a,b))
        if ev['name'].startswith('MOV_SRC_TO_DST'):
            v=dict(s=a,e=b,src=z['src_addr'],dst=z['dst_addr'],size=z['processed_bytes'])
            if z['queue_type']=='MTE2':reads.append(v)
            if z['queue_type']=='MTE3':writes.append(v)
    reads.sort(key=lambda v:v['s']);writes.sort(key=lambda v:v['s'])
    assert len(writes)==len(wanted) and len(reads)==4*len(wanted)+1 and reads[0]['size']==96
    rows=[]
    for i,w in enumerate(writes):
        token=(w['dst']-base)//stride;assert token==wanted[i]
        rr=reads[1+4*i:5+4*i];assert [z['size']for z in rr]==[64,16,2*d,8*d]
        if rows:assert [v['src']-o['src']for v,o in zip(rr,rows[-1]['reads'])]==[64,16,2*d,8*d]
        rows.append(dict(token=token,reads=rr,write=w))
    return dict(core=lane,terminal=terminal,critical_lane=(lane==terminal),tokens=wanted,device_us=s['ca_us'],core_window_us=s['windows'][lane],rows=rows,metadata=reads[0],simd=union(vectors),gm_base=base,checked_tokens=t,sha256=s['trace_sha256'],trace=s['trace'])
def schedule(r, prospective=False):
    c=load_chip(SKU);prices=read(P25/'prices.json');newparams=read(OUT/'parameters.json');x=range(r['plan']['rows_per_core'])
    baseline,_,_,diag=resource_post(r['shape'],c,{'resident_aiv':r['prior_aiv']})
    opt=OperatorStream(SKU,experimental_post_repeats=True)
    if r['role']=='repeat':
        opt.advance('mhc_post',r['shape']);public=opt.preview('mhc_post',r['shape'])
        spec=prices;key='repeat';frozen=public.latency_us
    else:
        q=predict(r['shape'],c,r['prior_aiv'],r['previous_shape'],newparams,prices);key=q['cohort'];stage=dict(prices['by_state'][r['state']]);c0=newparams['control'][key]
        if c0['processes']>=2:stage['control_to_operand']=c0['median_cycles']
        work=vector_work(r['shape'][2],c,prices)
        for scope in (key,'shared'):
            for k,v in newparams['curves'].get(scope,{}).items():
                if v:stage[k]=max(v['minimum'],work+v['fixed_compute_tail'])
        spec={**prices,'by_state':{r['state']:stage}};key=r['state'];frozen=q['us']
    pipe=token_pipeline(r['shape'],c,diag['target_plan'],key,spec,diag['shared_floor_us'],x)
    assert abs(pipe['us']-frozen)<1e-10
    return dict(**pipe,old_public_us=baseline,previous=r['previous_shape'],prior_aiv=r['prior_aiv'],role=r['role'],scope=('v0.27 pre-CA changed shape' if prospective else 'v0.27 ordinary development replay') if r['role']=='entry' else 'pre-CA exact repeat')
def build(prospective=False):
    target=[157,4,1088] if prospective else TARGET
    if prospective:
        from development.validation.post26.freeze import OUT as V26
        f=V26/'receipts'/('multi-'+SKU+'.json')
    else:
        f=P25/'receipts'/('steady-'+SKU+'.json')
    rs=[r for r in read(f)['rows'] if r['shape']==target]
    assert [r['role']for r in rs]==['entry','repeat'] and all(r['timing_qualified']for r in rs)
    repeat=collect(rs[1]);warm=collect(rs[0],repeat['tokens'])
    if prospective:
        ext=read(V26/'registration.json');shadow=read(OUT/'registration.json')
        assert read(f)['cost']['started_utc']>max(ext['utc'],shadow['utc'])
        q,=[x for x in shadow['rows'] if x['chip']==SKU and x['shape']==target]
        rep,=[x for x in ext['rows'] if x['chip']==SKU and x['shape']==target and x['role']=='repeat']
        assert abs(schedule(rs[0],True)['us']-q['us'])<1.e-10 and abs(schedule(rs[1],True)['us']-rep['us'])<1.e-10
    return dict(shape=target,sku=SKU,prospective=prospective,cohort='Same GM logical token indices in both numerical host checks. Select the last-finishing maximally assigned core of repeat and use these rows on the other launch too; device windows span all launched lanes.',entries=[dict(model=schedule(r,prospective),ca=v) for r,v in zip(rs,[warm,repeat])])
def render(data,slug='same-token-history'):
    t,n,d=data['shape'];nr=len(data['entries'][0]['ca']['tokens']);lanes=data['entries'][0]['model']['active_aiv']
    cs={'MTE2':'#177c87','MTE3':'#b34f2d','ready':'#b39bc8','SIMD':'#7655a5','control':'#b2c3c9'}
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'text.color':'#203946','axes.spines.top':False,'axes.spines.right':False,'axes.spines.left':False})
    fig,axes=plt.subplots(2,2,figsize=(14.8,8.7),sharex=True,sharey='row')
    fig.subplots_adjust(left=.125,right=.972,top=.78,bottom=.14,wspace=.15,hspace=.38)
    fig.suptitle(f'950DT mHC Post  |  {t} × {n} × {d}',x=.125,y=.964,ha='left',fontweight='bold',fontsize=22)
    fig.text(.125,.905,f'Same {nr} output tokens in both launches  ·  {lanes} AIVs  ·  BF16 inputs and four residual streams',fontsize=12)
    fig.legend(handles=[Patch(color=cs[k],label=l) for k,l in [('MTE2','MTE2 · four operand reads'),('ready','Operand-ready to write issue'),('MTE3','MTE3 · row write'),('SIMD','Measured AIV vector activity')]],loc='upper left',bbox_to_anchor=(.12,.892),ncol=4,frameon=False,fontsize=10)
    xmax=1.06*max(max(z['ca']['device_us'],z['model']['us']) for z in data['entries'])
    for j,e in enumerate(data['entries']):
        m,c=e['model'],e['ca'];title=('Changed shape · v0.27 shadow','Immediate repeat · frozen option')[j]
        for panel,ax in enumerate(axes[:,j]):
            us=m['us'] if panel==0 else c['device_us']
            ax.set_title(('Model  |  '+title) if panel==0 else 'CA  |  '+c['core']+(' · last' if c['critical_lane'] else ' · matched rows'),loc='left',fontsize=11,weight='bold')
            ax.set_title(f'{us:.3f} µs',loc='right',fontsize=12,weight='bold')
            def bar(y,lo,hi,k,h=.10):ax.broken_barh([(lo,hi-lo)],(y-h/2,h),facecolor=cs[k])
            if panel==0:bar(0,0,m['stages_us']['arrival']+m['stages_us']['control_to_first_read'],'control',.23)
            else:bar(0,c['metadata']['s'],c['metadata']['e'],'MTE2',.13)
            for i in range(nr):
                y=i+1
                if panel==0:
                    row=m['timeline'][i];rr=row['reads_us'];ws,we=row['write_us']
                    bar(y+.13,max(z[1] for z in rr),ws,'ready',.07)
                else:
                    row=c['rows'][i];rr=[(v['s'],v['e']) for v in row['reads']];ws,we=row['write']['s'],row['write']['e']
                for k,(lo,hi) in enumerate(rr):bar(y-.27+k*.115,lo,hi,'MTE2',.085)
                bar(y+.245,ws,we,'MTE3',.13)
            if panel:
                for a,b in c['simd']:bar(nr+1,a,b,'SIMD',.26)
            else:ax.text(.008,nr+1,'Per-instruction timing unavailable',transform=ax.get_yaxis_transform(),va='center',fontsize=8,color='#637681')
            labels=(['Dispatch to first read'] if not panel else ['Metadata · 96 B'])+[f'Token {k}' for k in c['tokens']]+['AIV vector activity']
            ax.set_yticks(range(nr+2),labels);ax.set_ylim(nr+1.55,-.55);ax.set_xlim(0,xmax);ax.axvline(us,ls='--',lw=1,color='#324d59')
            ax.set_axisbelow(True);ax.grid(axis='x',color='#d5e0e6',lw=.8);ax.tick_params(axis='y',length=0,pad=6)
            for l in range(nr+1):ax.axhline(l+.5,color='#e4eaf0',lw=.65,zorder=0)
        axes[-1,j].set_xlabel('Device time (µs)')
    fig.text(.125,.064,'All panels use the same clock and microsecond scale. Dashed lines mark the full device window, including other cores.',fontsize=9)
    note=('Both forecasts were frozen before CA. The changed-shape schedule is prospective; exact repeat uses its separately registered resource prices.'
          if data['prospective'] else 'The changed-shape case was used for method development. Exact-repeat time was frozen before this CA measurement.')
    fig.text(.125,.040,note,fontsize=9)
    fig.savefig(OUT/(slug+'.png'),dpi=155);plt.close(fig)
if __name__=='__main__':
    prospective='prospective' in sys.argv;slug='unseen-same-token' if prospective else 'same-token-history'
    d=build(prospective);render(d,slug);write(OUT/(slug+'.json'),d)
