"""Generate a common-clock DT FA overlay for a prospective 8-page launch."""
from collections import defaultdict
from pathlib import Path
import json,sys
from math import isclose
from audit import ROOT, OUT, read
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'sku17'))
from campaign_collect import events, sha
from operator_model import estimate, load_chip


def data():
    f=OUT/'fa-Ascend950DT_9582-9-1008-64-1-1.json';r=read(f);shape=r['shape'];sku=r['chip']
    s=r['stages'][0];assert r['qualified'] and len(r['stages'])==1 and s['active_aic']==1
    cube,= [c for c in s['windows'] if 'cube' in c]
    aggregate=ROOT/s['trace'];shard=aggregate.parent/cube/'trace.json'
    assert sha(aggregate)==s['trace_sha256']
    early=float('inf');end=-early
    clusters=defaultdict(list);loads=defaultdict(list)
    for e in events(shard):
        if e['ph']!='X':continue
        early=min(early,e['ts']);end=max(end,e['ts']+e['dur'])
        d=json.loads(e['args'].get('detail') or '{}')
        if (d.get('src_mem'),d.get('dst_mem'))==('OUT','L1'):
            loads[e['args']['pc_addr']].append(dict(absolute_us=[e['ts'],e['ts']+e['dur']],size=d['processed_bytes'],address=d['src_addr']))
        if e['name']!='MMAD':continue
        dims=[d[k] for k in ('m','n','k')]
        clusters[e['args']['pc_addr']].append(dict(dims=dims,absolute_us=[e['ts'],e['ts']+e['dur']]))
    origin=early-s['windows'][cube][0];assert isclose(end-origin,s['windows'][cube][1],abs_tol=.001)
    pred=estimate('fa',shape,chip=sku,timing_mode='resource')
    plan=pred.projection['target_plan'];d=pred.projection['timing_diagnostics']
    w,=d['wave_types'];clock=1000*load_chip(sku).frequency_ghz
    q,kv,depth,b,h=shape;count=plan['iterations'];page=plan['tile'][1]
    assert len(clusters)==2 and s['mmad']==2*count and page<=128
    ordered=sorted(clusters.items(),key=lambda pair:pair[1][0]['absolute_us'][0])
    ca={};model={};pc={}
    for key,(address,evs) in zip(('QK','PV'),ordered):
        pc[key]=address;assert len(evs)==count
        ca[key]=[];model[key]=[];cursor=0
        for i,e in enumerate(evs):
            n=min(page,kv-i*page)
            assert e['dims']==([q,n,depth] if key=='QK' else [q,depth,n])
            ca[key].append([x-origin for x in e['absolute_us']])
            part=w['last'] if i==count-1 else w['first']
            if i:cursor+=part['ii_cycles']
            x=pred.projection['stages_us']['launch']+(w[key.lower()+'_start_cycles']+cursor)/clock
            model[key].append([x,x+part[key.lower()]/clock])
    # A single Q payload and two disjoint monotonically increasing BF16 KV
    # tensors. Source addresses and tail bytes distinguish pages and streams.
    (_,q_load),= [(k,ls[0]) for k,ls in loads.items() if len(ls)==1]
    assert q_load['size']==2*q*depth
    streams=sorted(((k,ls) for k,ls in loads.items() if len(ls)==count),key=lambda pair:pair[1][0]['absolute_us'][0])
    assert len(streams)==2 and streams[0][1][-1]['address']+streams[0][1][-1]['size']<=streams[1][1][0]['address']
    costs=read('results/v0.16/fa.json')['routes']['short'];start=pred.projection['stages_us']['launch']
    q_window=[v-origin for v in q_load['absolute_us']]
    for key,(address,ls) in zip(('K','V'),streams):
        pc[key]=address;ca[key]=[];model[key]=[];cursor=0
        for i,entry in enumerate(ls):
            n=min(page,kv-i*page);assert entry['size']==2*n*depth
            assert entry['address']==ls[0]['address']+i*page*2*depth
            ca[key].append([v-origin for v in entry['absolute_us']])
            assert ca[key][-1][0]<ca['QK' if key=='K' else 'PV'][i][0]
            part=w['last'] if i==count-1 else w['first']
            if i:cursor+=part['ii_cycles']
            begin=start+(cursor+costs['k_lag']+(costs['v_lag'] if key=='V' else 0))/clock
            model[key].append([begin,begin+(costs['read_intercept']+entry['size']/w['read_bytes_per_cycle'])/clock])
    first_q_model=[start,start+(costs['read_intercept']+2*q*depth/w['read_bytes_per_cycle'])/clock]
    old=next(c['forecasts'][sku]['old'] for c in read(OUT/'fa-freeze.json')['jobs'] if c['id']==r['case_id'])
    last=max(s['windows'],key=lambda key:s['windows'][key][1]);ca_us=s['ca_us']
    assert isclose(s['windows'][last][1],ca_us,abs_tol=.001)
    return dict(sku=sku,shape=shape,page_size=page,page_rows=[min(page,kv-page*i) for i in range(count)],
                old_auto_us=old,current_model_us=pred.latency_us,ca_us=ca_us,model=model,ca=ca,
                model_stages_us=pred.projection['stages_us'],first_issue_error_us={k:model[k][0][0]-ca[k][0][0] for k in ca},
                q_read_us={'model':first_q_model,'ca':q_window},descriptor_pc={k:pc[k] for k in ('K','V')},
                origin='earliest physical instruction of any launched core',cube=cube,last_core=last,core_windows=s['windows'],
                kernel=s['kernel'],elf_sha256=s['binary_sha256'],receipt=str(f.relative_to(ROOT)),receipt_sha256=sha(f),
                aggregate=str(aggregate.relative_to(ROOT)),aggregate_sha256=sha(aggregate),shard=str(shard.relative_to(ROOT)),
                shard_sha256=sha(shard),mmad_pc={k:pc[k] for k in ('QK','PV')},register_sha256=r['registration_sha256'],
                evidence='CA: MMAD dimensions, K/V GM and L1 descriptor PC/bytes/GM stride. Model: later pages reuse resource II for expected K/V and MMAD issue; unshown vector, FIX, scalar and flags remain in full window')


def draw(j):
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch
    from math import ceil
    fg='#193340';colors={'K':'#5187ae','V':'#b75e35','QK':'#7459a7','PV':'#107c80'}
    fig,(up,down)=plt.subplots(2,1,figsize=(12.2,7.7),sharex=True)
    fig.subplots_adjust(left=.19,right=.95,top=.80,bottom=.13,hspace=.41)
    fig.text(.19,.94,'950DT  FlashAttentionScoreV2   |   Q = 9, KV = 1,008, D = 64',fontsize=20,fontweight='bold',color=fg)
    fig.text(.19,.894,'One head · one AIC and two AIVs · 7 full KV pages + 112-token tail',fontsize=12,color='#3b596a')
    fig.legend([Patch(color=v) for v in colors.values()]+[Line2D([],[],marker='|',markersize=10,ls='',color=fg)],
               ['K · MTE2','V · MTE2','QK · score MMAD','PV · output MMAD','issue'],loc='center left',bbox_to_anchor=(.185,.852),frameon=False,ncol=5)
    for ax,limits,name,title in ((up,j['model'],j['current_model_us'],'Target resources · page initiation schedule'),
                                  (down,j['ca'],j['ca_us'],'CA · '+j['cube']+'  |  final exit '+j['last_core'])):
        q1,q2=j['q_read_us']['model' if ax is up else 'ca']
        ax.barh(-1,q1,left=0,height=.40,color='#e4ecf0');ax.text(q1/2,-1,'Before first Q read',ha='center',va='center',size=8.2,color='#496477')
        ax.barh(-1,q2-q1,left=q1,height=.20,color='#5187ae')
        for idx,n in enumerate(j['page_rows']):
            for key,dy in [('K',-.30),('QK',-.10),('V',.10),('PV',.30)]:
                a,b=limits[key][idx];y=idx+dy
                ax.barh(y,b-a,left=a,height=.14,linewidth=0,color=colors[key],zorder=3)
                ax.vlines(a,y-.09,y+.09,color=colors[key],lw=1.0,zorder=4)
        ax.axvline(name,lw=1.25,ls=(0,(3,3)),color='#4c6473')
        ax.set_yticks(range(-1,len(j['page_rows'])),['Q · first read']+[f'KV {i*128+1:,}–{i*128+n:,}' for i,n in enumerate(j['page_rows'])]);ax.invert_yaxis()
        ax.tick_params(length=0,labelsize=9,pad=5);ax.spines[:].set_visible(False)
        ax.grid(axis='x',color='#d9e2e7');ax.set_axisbelow(True)
        ax.set_title(title,loc='left',fontsize=13,weight='bold',color=fg)
        ax.set_title(f'{name:.3f} µs',loc='right',fontsize=15,weight='bold',color=fg)
    end=ceil(max(j['ca_us'],j['current_model_us']));down.set_xlim(0,end+.15);down.set_xticks(range(end+1))
    down.set_xlabel('Time from earliest instruction on any core (µs)',fontsize=11,labelpad=8)
    a=j['first_issue_error_us']
    fig.text(.19,.058,f"First issue, model − CA:   QK {a['QK']:+.3f} µs     PV {a['PV']:+.3f} µs.      Dashed line: whole-kernel completion (all cores).",fontsize=10,color='#345464')
    fig.text(.19,.029,'K/V and MMAD windows after the first page advance by modeled II. CA ticks mark true issue; softmax and control remain in the full window.',fontsize=9.2,color='#345464')
    fig.savefig(OUT/'fa-dt-1008-pages.png',dpi=150,facecolor='white');plt.close(fig)


if __name__=='__main__':
    path=OUT/'fa-dt-1008-pages.json'
    if '--snapshot' in sys.argv:j=read(path)
    else:
        j=data();path.write_text(json.dumps(j,indent=2)+'\n')
    draw(j)
