"""Two-row worked example. Shared device origin and final timestamps from complete archive."""
import os
os.environ.setdefault('MPLCONFIGDIR','/tmp/gating35-mpl')
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from campaign import ROOT,OUT,PR,load_chip,read,rel,sha,write
from evidence import selected,merge_ops
from candidate import predict,profile


def run():
    sh=[65,128,8];r=next(x for x in read(ROOT/'results/v0.19/workloads/router-qualified.json') if x['shape']==sh and x['sku']==PR);st=r['kernel'];core=max(st['windows'],key=lambda c:st['windows'][c][1]);path=(ROOT/st['trace']).parent/core/'trace.json';events=[x for x in read(path)['traceEvents'] if x.get('ph')=='X'];origin=min(x['ts'] for x in events)-st['windows'][core][0];assert abs(max(x['ts']+x['dur'] for x in events)-origin-st['ca_us'])<.0001
    hw=load_chip(PR);p=predict(sh,hw);a=sh[1];qty=1000*hw.frequency_ghz;par=profile();rank=p['rank_body_engine_cycles'];data=[];model=[];fvs=[]
    def mark(array,row,color,lo,hi):array.append(dict(row=row,color=color,begin_us=lo,end_us=hi))
    for x in selected(path):
        op,lo,hi,detail=x;lo=lo/1650-origin;hi=hi/1650-origin
        if op=='MOV_SRC_TO_DST_ALIGNv2':
            if detail.get('src_mem')=='OUT' and detail.get('burst_len')==4*a:mark(data,0,'read',lo,hi)
            elif detail.get('dst_mem')=='OUT':mark(data,3,'write',lo,hi)
        if op=='VF':fvs.append((lo,hi))
    assert len(fvs)==7
    for i in range(2):
        t0=fvs[1+3*i][0];tf=fvs[3+3*i][1];mark(data,i+1,'vector',t0,fvs[1+3*i][1]);mark(data,i+1,'vector',fvs[2+3*i][0],fvs[2+3*i][1]);mark(data,i+1,'vector',fvs[3+3*i][0],tf)
        for op,lo,hi,d in selected(path):
            c={'VBS32':'sort','VMS4V2':'merge','MOV_UB_TO_UB':'ub'}.get(op);u=lo/1650-origin
            if c and t0<=u<=tf:mark(data,i+1,c,u,hi/1650-origin)
    stages=p['stages_us'];q=stages['launch']+stages['first_batch'];mark(model,0,'control',0,stages['launch']);mark(model,0,'read',stages['launch'],q)
    vec_scale=par['reference_fp32_ops_per_cycle']/(hw.vector_tops['32']*1000/(hw.vector_cores*hw.frequency_ghz));norm=(par['first_sigmoid_bias_to_sort_cycles'][0]+a*par['first_sigmoid_bias_to_sort_cycles'][1])*vec_scale
    merge,_=merge_ops(a);assert merge==[(128,1)];copy=par['ub_copy_cycles'][0]+a*par['ub_copy_cycles'][1];bs=par['sort32_cycles'][0]+a*par['sort32_cycles'][1];m=par['merge_cycles_fixed_per_element_extra_repeat'];duration=m[0]+m[1]*128
    t=q+par['row_begin_cycles']/qty
    for i in range(2):
        for c,d in [('vector',norm+i*vec_scale*par['repeated_sigmoid_extra_cycles']),('sort',bs),('ub',copy),('merge',duration),('ub',copy)]:
            mark(model,i+1,c,t,t+d/qty);t+=d/qty
        sel=t;end=t+par['select_and_normalize_cycles']*vec_scale/qty;mark(model,i+1,'vector',t,end);t=sel+par['next_row_issue_after_select_cycles']/qty
    mark(model,3,'write',end,p['us']);assert abs(end+stages['tail_write']-p['us'])<1e-7
    colors={'control':'#b7c3cd','read':'#007876','vector':'#3479bb','sort':'#a84814','merge':'#805aaa','ub':'#a2ac2a','write':'#bd6c1a'}
    fig,axes=plt.subplots(2,1,figsize=(11,5.6),sharex=True);labels=['Bias / logits DMA','Token 1','Token 2','Weights / IDs DMA'];height=.49
    for ax,rows,cap,last in [(axes[0],model,'Resource candidate · critical path',p['us']),(axes[1],data,'CA · '+core+' (last to finish)',st['ca_us'])]:
        for bar in rows:
            y=bar['row'];shift=0 if bar['color'] not in ('sort','merge','ub') else -.11;ax.broken_barh([(bar['begin_us'],bar['end_us']-bar['begin_us'])],(y-height/2+shift,height if bar['color'] not in ('sort','merge','ub') else .25),facecolors=colors[bar['color']]);
        ax.set_yticks(range(4),labels=labels);ax.set_ylim(3.65,-.65);ax.set_xlim(0,1.04*max(st['ca_us'],p['us']));ax.grid(axis='x',alpha=.2);ax.axvline(last,ls='--',lw=1.1,color='#344951');ax.set_title(f'{cap}    {last:.3f} µs',loc='left',weight='bold',fontsize=11);ax.spines[['right','top']].set_visible(False)
    fig.suptitle('950PR MoE Gating TopK  ·  65 tokens × 128 experts, top 8',fontsize=15,weight='bold',x=.12,ha='left');axes[-1].set_xlabel('Device time from first core instruction (µs)');fig.legend([Patch(facecolor=c) for c in colors.values()],['Control','GM → UB','VF','VBS32','VMS4V2','UB copy','UB → GM'],loc='upper center',ncol=7,bbox_to_anchor=(.54,.934),frameon=False,fontsize=9)
    fig.subplots_adjust(left=.18,right=.98,bottom=.14,top=.80,hspace=.55);fig.savefig(OUT/'two-rows-pr.png',dpi=165);plt.close(fig)
    write(OUT/'two-rows-pr.json',dict(shape=sh,chip=PR,trace=st['trace'],trace_sha256=st['trace_sha256'],shard=rel(path),shard_sha256=sha(path),ca_us=st['ca_us'],candidate_us=p['us'],last_core=core,model=model,ca=data,units='device us from earliest original instruction; archive retains END_LABEL; per-row VF shown as descriptor window'))
if __name__=='__main__':run()
