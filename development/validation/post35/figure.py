"""Two-source last-lane comparison, with calibrated conditional launch and no invented CA rows."""
import sys
from registration import ROOT,OUT,D,read,save
from operator_model import load_chip
from operator_model.post_resource import resource_post
from operator_model.post_fragment_pipeline import covered_fragment_schedule
from operator_model.post_repeat import channel_tiles
from schedule import schedule

def main():
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch
    target=load_chip(D);sh=[65,4,5120];cfg=schedule(sh,D)
    _,_,_,d=resource_post(sh,target,{})
    rowpipe=covered_fragment_schedule(target,d,channel_tiles(sh[2],target),inspect_rows=(0,1))
    rows=[r for r in read(OUT/'physical.json')['rows'] if r['shape']==sh and r['chip']==D]
    q=cfg['prices'];fig,ax=plt.subplots(2,2,figsize=(10,5.15),sharex=True,sharey=True)
    fig.subplots_adjust(left=.15,right=.98,hspace=.55,wspace=.12,bottom=.18,top=.75)
    stages=dict(read='#16858b',write='#b5512e',issue='#6777a6',point='#243948')
    fig.suptitle('950DT   mHC Post   |   65 × 4 × 5120',x=.145,y=.97,ha='left',fontsize=18,weight='bold')
    fig.text(.15,.875,'33 AIVs  ·  two UB fragments per row  ·  same shape, different stream prefixes',fontsize=10,color='#394d56')
    fig.legend([Patch(color=stages['read']),Patch(color=stages['write']),Patch(color=stages['issue'])],['CA input DMA','Output DMA response','Modeled write issue'],loc='upper left',bbox_to_anchor=(.14,.87),frameon=False,ncol=3,fontsize=9)
    data=[]
    for col,actual in enumerate(rows):
        state=actual['roles']['prior'];parts=q[state];v=cfg['states'][state];label='Changed shape' if state=='reuse' else 'Immediate repeat'
        shifted=[p['write_issue_us']-rowpipe['timeline'][0]['write_issue_us']+parts['first_write'] for p in rowpipe['timeline']]
        assert abs(shifted[-1]+parts['drain']+parts['exit']-v)<1e-7
        for n,issue in enumerate(shifted):ax[0,col].plot(issue,n,marker='|',ms=13,mew=2,color=stages['issue'])
        ax[0,col].barh(3,parts['drain'],left=shifted[-1],height=.19,color=stages['write'])
        ax[0,col].set_title(f"{label}  ·  {v:.3f} µs",loc='left',fontsize=11,fontweight='bold')
        ax[0,col].axvline(v,color='#233445',dashes=(2,2),lw=1)
        ax[0,col].axvline(cfg['us'],color='#6752ab',dashes=(5,2),lw=1)
        for n,item in enumerate(actual['fragments']):
            for k,dy in [('h',-.17),('x',-.02),('write',.17)]:
                z=item[k];ax[1,col].barh(n+dy,(z['b']-z['a'])/1650,left=z['a']/1650,height=.13,color=stages['write' if k=='write' else 'read'])
        ax[1,col].axvline(actual['ca_us'],color='#233445',dashes=(2,2),lw=1)
        ax[1,col].set_title(f"CA {actual['core']}  ·  {actual['ca_us']:.3f} µs",loc='left',fontsize=10,fontweight='bold')
        data.append(dict(state=state,shape_only_us=cfg['us'],conditional_us=v,modeled_write_issues_us=shifted,last_ack_after_us=parts['drain'],ca_us=actual['ca_us'],core=actual['core'],source=actual['shard'],shard_sha256=actual['shard_sha256'],fragments=actual['fragments']))
    for row in ax:
        for a in row:
            a.set_axisbelow(True);a.grid(axis='x',color='#d9e2e8',lw=.65);a.set_ylim(3.5,-.5);a.set_xlim(0,6.1)
            a.set_yticks(range(4),['row 1 · tile 1','row 1 · tile 2','row 2 · tile 1','row 2 · tile 2']);a.tick_params(axis='y',length=0,labelsize=8)
            for spine in ('top','right','left'):a.spines[spine].set_visible(False)
    fig.text(.57,.098,'Device time from earliest AIV instruction (µs)',ha='center')
    fig.text(.15,.041,'Upper: conditional write issues, final response and total (black); purple marks the same shape-only point. Lower: physical descriptors.',fontsize=8,color='#50646d')
    fig.savefig(OUT/'two-slot-dt.png',dpi=155);plt.close(fig)
    save(OUT/'two-slot-dt.json',dict(rows=data,scope='Conditional model milestones, same shape-only point for both priors. Physical CA input/output descriptors and global device endpoint.'))
if __name__=='__main__':main()
