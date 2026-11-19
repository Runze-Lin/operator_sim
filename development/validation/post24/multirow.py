"""Recreate the published queue attribution and the terminal AIV's five real Post items."""
from collections import Counter,defaultdict
import json
from pathlib import Path
import statistics,sys
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
ROOT=Path(__file__).resolve().parents[3];sys.path.insert(0,str(ROOT))
from development.validation.post24.figure import union
from development.validation.rms_ordered_verify.verify import events
from development.validation.sku19.campaign import read,write
OUT=ROOT/'results/v0.24/post'
PRIORITY=('SIMD','MTE3','MTE2','SCALAR')
ORDER=('arrival','SCALAR','MTE2','SIMD','MTE3','other')
COL={'arrival':'#aebbc3','SCALAR':'#527289','MTE2':'#137e83','SIMD':'#7555a4','MTE3':'#b85131','other':'#c7cdd2'}
def measure(r, identify):
    s=r['stage'];tr=ROOT/s['trace'];k=max(s['windows'],key=lambda c:s['windows'][c][1]);local=tr.parent/k/'trace.json';origin=s['first_tick'];end=s['last_tick']
    counts=defaultdict(Counter);first=None;raw=defaultdict(list);reads=[];writes=[]
    for v in events(local):
        d=json.loads(v['args']['detail']);a=round(v['ts']*1650);b=round((v['ts']+v['dur'])*1650)
        first=a if first is None else min(first,a);q=d['queue_type']
        if not d.get('is_duration_excluded'):
            counts[a][q]+=1;counts[b][q]-=1
        if d.get('is_sync') or d.get('is_wait') or d.get('is_duration_excluded') or b<=a:continue
        lo,hi=(a-origin)/1650,(b-origin)/1650
        if q=='SIMD':raw[q].append((lo,hi))
        if q in ['MTE2','MTE3'] and v['name'].startswith('MOV_SRC_TO_DST'):
            row=dict(start=lo,done=hi,n=d['processed_bytes'],src=d['src_addr'],dst=d['dst_addr'])
            if q=='MTE2':reads.append(row)
            else:writes.append(row)
    counts[first];counts[end];last=first;busy=Counter();acc=Counter()
    for tick,delta in sorted(counts.items()):
        q=next((q for q in PRIORITY if busy[q]>0),'other');acc[q]+=tick-last;busy.update(delta);last=tick
    acc['arrival']=first-origin;act={k:acc[k]/1650 for k in ORDER};assert abs(sum(act.values())-s['ca_us'])<1e-8
    model={k:r['stages'].get(k,0) for k in ORDER};assert abs(sum(model.values())-r['us'])<1e-8
    reads.sort(key=lambda x:x['start']); writes.sort(key=lambda x:x['start']); t,n,d=r['shape']
    assert len(writes)==5 and len(reads)==21 and reads[0]['n']==96
    items=[]
    for i,w in enumerate(writes):
        group=reads[1+4*i:5+4*i]
        assert [g['n'] for g in group]==[64,16,2*d,2*n*d] and w['n']==n*d*2
        items.append(dict(input=group,output=w))
    if identify:
        gm=[]
        for ev in events(tr):
            if ev['name'].startswith('MOV_SRC_TO_DST'):
                g=json.loads(ev['args']['detail'])
                if g['queue_type']=='MTE3' and not any(g.get(f) for f in ('is_wait','is_sync','is_duration_excluded')):
                    gm.append((g['dst_addr'],g['processed_bytes']))
        gm.sort();assert len(gm)==t and [k for k,_ in gm]==[gm[0][0]+2*i*n*d for i in range(t)] and all(x==2*d*n for _,x in gm)
        for item in items:item['token']=(item['output']['dst']-gm[0][0])//(2*n*d)
        for a,b in zip(items,items[1:]):
            step=b['token']-a['token'];assert step==1 and [rb['src']-ra['src'] for ra,rb in zip(a['input'],b['input'])]==[step*x for x in [64,16,2*d,8*d]]
    return dict(shape=r['shape'],chip=r['chip'],role=r['role'],critical=k,active=s['active_aiv'],rows=len(items),model_us=r['us'],ca_us=s['ca_us'],model_stages_us=model,
          ca_exposed_us=act,errors_us={q:model[q]-act[q] for q in ORDER},milestones_us=r['ca_milestones']['phase_us'],simd=union(raw['SIMD']),metadata=reads[0],items=items,trace=s['trace'])
def main():
    p=OUT/'receipts/rows-Ascend950DT_9582.json'
    rows=[r for r in read(p)['rows'] if r['shape']==[257,4,832] and r['timing_qualified']]
    data=[measure(r,r['role']=='repeat') for r in rows];warm=next(x for x in data if x['role']=='repeat')
    write(OUT/'multirow.json',dict(priority=PRIORITY,source_method='scripts/wide-v10-trace.py::inspect, copied sweep rule on the complete msOpProf shard rather than an npusim DB; skip duration-excluded only; high-priority label wins when queues overlap.',data=data))
    fig,ax=plt.subplots(2,1,figsize=(11,7),sharex=True,gridspec_kw={'height_ratios':[1,2.5]});fig.subplots_adjust(left=.23,right=.96,top=.8,bottom=.16,hspace=.5)
    fig.suptitle('950DT mHC Post  |  [257, 4, 832]  ·  immediate repeat',x=.23,y=.975,ha='left',fontweight='bold',fontsize=17)
    fig.text(.23,.905,f"{warm['active']} AIVs  ·  {warm['critical']} finishes last  ·  5 complete token items on this AIV",fontsize=12)
    fig.legend(handles=[Patch(color=COL[k],label=k) for k in ORDER if k!='other'],loc='upper left',bbox_to_anchor=(.225,.895),ncol=5,frameon=False)
    a,b=ax;a.set_title('Published resource budgets and measured exposed queue time',loc='left',weight='bold',fontsize=12)
    for y,key,total in [(1,'model_stages_us',warm['model_us']),(0,'ca_exposed_us',warm['ca_us'])]:
        z=0
        for k in ORDER:
            u=warm[key][k];a.barh(y,u,left=z,height=.34,color=COL[k]);z+=u
        a.text(z+.04,y,f'{total:.3f} µs',va='center',weight='bold')
    a.set_yticks([1,0],['Resource model','CA exposed time']);a.set_ylim(-.48,1.55)
    b.set_title('CA descriptors and vector activity  |  same microsecond axis',loc='left',weight='bold',fontsize=12)
    def bars(y,itvs,h,color):b.broken_barh([(u,v-u) for u,v in itvs],(y-h/2,h),facecolor=color)
    m=warm['metadata'];bars(0,[(m['start'],m['done'])],.35,COL['MTE2'])
    bars(1,warm['simd'],.38,COL['SIMD']);labels=['96 B metadata','Vector on this AIV']
    for i,row in enumerate(warm['items']):
        y=i+2;labels.append(f"Token {row['token']} · four outputs")
        for j,r in enumerate(row['input']):bars(y-.26+j*.12,[(r['start'],r['done'])],.09,COL['MTE2'])
        w=row['output'];bars(y+.26,[(w['start'],w['done'])],.13,COL['MTE3'])
    b.set_yticks(range(len(labels)),labels);b.set_ylim(len(labels)-.45,-.6)
    for x in ax:
        x.set_axisbelow(True);x.grid(axis='x',alpha=.25);x.tick_params(axis='y',length=0)
        x.spines[['top','right','left']].set_visible(False);x.set_xlim(0,warm['model_us']*1.13)
    b.axvline(warm['ca_us'],color='#253b47',ls='--',lw=1);b.set_xlabel('Device time (µs)')
    fig.text(.23,.075,'Upper bars sum disjoint queue contributions. Their order is for accounting. Lower bars use observed instruction times.',fontsize=9)
    fig.text(.23,.051,'Each token row has four reads and one 6,656-byte write. Vector instructions are labeled only at AIV granularity.',fontsize=9)
    fig.savefig(OUT/'post-multirow-repeat.png',dpi=155);plt.close(fig)
    md=['# 五轮 token 核的重复调用误差','', '950DT 上 [257,4,832] 的热态仍使用公开 v0.23 token 资源模型。下表按其校准时的原始规则给末核每个实际周期分配唯一的队列标签。多个队列同时在忙时依次优先记到 SIMD、MTE3、MTE2、SCALAR，未覆盖的时段归 other；最早设备指令到末核加入单独计 arrival。该分账适合精确核对预测贡献，不表示低优先级队列完全停顿。', '',
        '| 贡献 µs | 模型 | CA 相同分账 | 模型 − CA |','|---|---:|---:|---:|']
    for k in ORDER:md.append(f"| {k} | {warm['model_stages_us'][k]:.3f} | {warm['ca_exposed_us'][k]:.3f} | {warm['errors_us'][k]:+.3f} |")
    md+= [f"| 全窗 | {warm['model_us']:.3f} | {warm['ca_us']:.3f} | {warm['model_us']-warm['ca_us']:+.3f} |", '', '![五个 token 的四笔读与合并写](post-multirow-repeat.png)','',
        '| 末核真实轮次 / token | 首条系数读发起 µs | 本轮最后一笔操作数完成 µs | 写回发起 / 完成 µs | 写回字节 |','|---|---:|---:|---:|---:|']
    for i,z in enumerate(warm['items']):
        w=z['output'];md.append(f"| {i+1} / {z['token']} | {z['input'][0]['start']:.3f} | {max(x['done'] for x in z['input']):.3f} | {w['start']:.3f} / {w['done']:.3f} | {w['n']} |")
    starts=[v['input'][0]['start'] for v in warm['items']]; ws=[v['output']['start'] for v in warm['items']]
    read_gaps=[b-a for a,b in zip(starts,starts[1:])];wg=[b-a for a,b in zip(ws,ws[1:])]
    md+=['',f"全芯片的 257 笔写入对应完整连续输出，每笔 6,656 B。根据 GM 目的地址，最后完成的 {warm['critical']} 写 token {warm['items'][0]['token']}–{warm['items'][-1]['token']}，每个 token 的四条输入地址分别按 64、16、1,664、6,656 B 前进，独立确认了 5 轮。首次到第二轮输入发起相隔 {read_gaps[0]:.3f} µs，后续三段发起间隔中位 {statistics.median(read_gaps[1:]):.3f} µs。连续写回发起间隔中位 {statistics.median(wg):.3f} µs。第二轮首笔读 {starts[1]:.3f} µs，早于第一轮写回完成 {warm['items'][0]['output']['done']:.3f} µs。流水实际重叠，直接把每轮响应时间相加会重复计费。", '', 'SIMD 指令仅能可靠归到该 AIV，图中单列。该证据能定位到固定开销、队列暴露时间和跨轮重叠，无法把软件调度、缓存或硬件发射中的单一因素确定为唯一根因。', '', '[逐阶段、逐描述符与原始 trace 路径](multirow.json) · [v0.12 使用的分账实现](../../../scripts/wide-v10-trace.py)']
    (OUT/'multirow.md').write_text('\n'.join(md)+'\n')
if __name__=='__main__':main()
