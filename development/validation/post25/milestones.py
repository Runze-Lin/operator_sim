"""Compare candidate and CA on the same terminal row count, preserving start, issue and drain."""
import statistics,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[3];sys.path.insert(0,str(ROOT))
from development.validation.post25.calibrate import predict,OUT
from development.validation.sku19.campaign import read,write
from operator_model import load_chip
def main():
    prices=read(OUT/'prices.json');data=[];skips=[]
    for f in sorted((OUT/'receipts').glob('*.json')):
        for r in read(f)['rows']:
            key=dict(sku=r['chip'],shape=r['shape'],role=r['role'],mode=r['mode'],group=r['group'],ordinal=r['i'])
            if not r['timing_qualified'] or 'ca_milestones' not in r:
                skips.append(dict(**key,reason=r.get('reason') or r.get('milestone_error') or 'unqualified'));continue
            c=r['ca_milestones'];rr=c['dma_read'];ww=c['dma_write'];d=r['shape'][-1];count=len(ww)
            if count!=r['plan']['rows_per_core'] or len(rr)!=1+4*count or any(w['bytes']!=8*d for w in ww):
                skips.append(dict(**key,reason='terminal core row count or descriptor geometry',measured_count=count,plan_count=r['plan']['rows_per_core'],operand_reads=len(rr)-1));continue
            grp=[rr[1+4*i:5+4*i] for i in range(count)];assert all([a['bytes'] for a in g]==[64,16,2*d,8*d] for g in grp)
            z=r['pipeline'];mod=dict(z['stages_us']);act=dict(arrival=c['phase_us']['arrival'],control_to_first_read=c['phase_us']['control'])
            act['first_read_to_ready']=max(a['end_us'] for a in grp[0])-grp[0][0]['start_us']
            act['ready_to_first_write']=ww[0]['start_us']-max(a['end_us'] for a in grp[0])
            act['overlapping_rows']=ww[-1]['start_us']-ww[0]['start_us']
            act['last_write_response']=ww[-1]['end_us']-ww[-1]['start_us']
            act['exit_after_write']=r['stage']['ca_us']-ww[-1]['end_us']
            act['shared_link_floor']=0
            assert abs(sum(act.values())-r['stage']['ca_us'])<2e-5
            tl=predict(r['shape'],load_chip(r['chip']),r['prior_aiv'],prices,range(count),r['previous_shape'])['pipeline']['timeline']
            def issues(reads,writes):
                def one(vals):
                    diffs=[b-a for a,b in zip(vals,vals[1:])]
                    return dict(first=diffs[0] if diffs else None,steady_median=statistics.median(diffs[1:]) if len(diffs)>1 else None,gaps_us=diffs)
                return dict(read=one(reads),write=one(writes))
            sim=issues([a['reads_us'][0][0] for a in tl],[a['write_us'][0] for a in tl])
            obs=issues([g[0]['start_us'] for g in grp],[w['start_us'] for w in ww])
            data.append(dict(**key,rows=count,model_us=r['us'],ca_us=r['stage']['ca_us'],model=mod,measured=act,gap_us={q:mod[q]-act[q] for q in mod},modeled_intervals=sim,measured_intervals=obs))
    write(OUT/'milestones.json',dict(matched=data,skipped=skips,scope='CA phases are computed on the all-core last finisher at exact DMA boundaries. Roster/descriptor disagreements are not collapsed into duration calibration. Shared floor is a lower bound and measured timings combine all resource contention.'))
    md=['# 逐行阶段在独立样本上的偏差','',f'{len(data)} 个末核的实际轮数、四类读和写描述符都与模型结构一致。另有 {len(skips)} 个整窗无法用相同粒度归因，原因单独写入 JSON。单位均为 µs。', '',
    '| 芯片 | T×4×D | 状态 | 末核轮数 | 总差 模型−CA | 最大阶段差及符号 | 首次读间隔 模型/CA | 后续读间隔中位 模型/CA | 首次写间隔 模型/CA | 后续写间隔中位 模型/CA |','|---|---|---|---:|---:|---|---:|---:|---:|---:|']
    for x in data:
        def show(kind,name):
            u=x['modeled_intervals'][kind][name];v=x['measured_intervals'][kind][name];return f'{u:.3f}/{v:.3f}' if u is not None else '—'
        q=max(x['gap_us'],key=lambda k:abs(x['gap_us'][k]));md.append(f"| {x['sku']} | {'×'.join(map(str,x['shape']))} | {x['mode']} | {x['rows']} | {x['model_us']-x['ca_us']:+.3f} | {q} {x['gap_us'][q]:+.3f} | "+' | '.join(show(k,v) for k in ('read','write') for v in ('first','steady_median'))+' |')
    md+=['','首行就绪取四笔输入 DMA 的最晚返回。读写稳态发起间隔取第二次转移之后的观测差值中位。末轮写响应取最后一个完整输出的 MTE3 完成时间减去发起时间。同一时间区间中可有向量与内存并行，按实际依赖时刻比较可避免把重叠重复加总。[完整逐段阶段与原始间隔](milestones.json)','']
    (OUT/'milestones.md').write_text('\n'.join(md))
if __name__=='__main__':main()
