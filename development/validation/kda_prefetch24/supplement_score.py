"""Publish the two initial and eight supplementary independent shapes, by receipt."""
import json
from evidence import OUT, ROOT, read, sha
from frozen import predict
from report import measures

FOLDER=OUT/'supplement'
ARTIFACTS=((OUT/'register.json',OUT),(FOLDER/'register.json',FOLDER))


def publish():
    rows=[];pending=[];failed=[];seen=set()
    for path,folder in ARTIFACTS:
        reg=read(path)
        register_sha=sha(path)
        for c in reg['cases']:
            f=folder/(c['id']+'.json')
            if not f.exists():
                pending.append(c['id']);continue
            r=read(f)
            if r['registration_sha256']!=register_sha:
                raise ValueError('Receipt has a different registration: '+str(f))
            if not r['qualified']:
                failed.append(dict(id=c['id'],source=str(f.relative_to(ROOT)),reason=r.get('reason'),cost=r.get('cost')))
                continue
            stages=[s for s in r['stages'] if s['family']]
            key=tuple(s['aggregate_sha256'] for s in stages)
            if key in seen or len(stages)!=3:raise ValueError('Duplicate or incomplete evidence '+str(f))
            seen.add(key)
            ca=[s['ca_us'] for s in stages]+[r['main_stages_us']]
            target=c['predictions'][c['sku']]
            assert target==predict(c['shape'],c['sku'])
            assert target['active_aic']==[s['active_aic'] for s in stages]
            rows.append(dict(id=c['id'],shape=c['shape'],sku=c['sku'],method_register=sha(OUT/'register.json'),
                             source=str(f.relative_to(ROOT)),source_sha256=sha(f),stage_sha256=key,
                             cost_s=r['cost']['seconds'],peak_bytes=r['cost']['peak_bytes'],
                             actual=ca,v023=target['stages_old']+[target['v023_us']],candidate=target['stages_us']+[target['us']],
                             extrapolated_970_us=c['predictions']['Ascend970DT']['us']))
    summary={};pairs=[]
    title=['# KDA 跨块预取的累计前向验证','',
           '同一冻结预测接受了两批独立验证。第一批两种 shape 的结果在选取后八种形状时已经可见，补充采集仍使用原参数、正式发行 ELF、四项数值参考与三个完整设备窗。','',
           '| SKU | 已合格 shape | FwdH WAPE / 最大 | 三段 WAPE / 最大 | v0.23 三段 WAPE / 最大 |',
           '|---|---:|---:|---:|---:|']
    for sku in ('Ascend950DT_9582','Ascend950PR_957b'):
        observed=[r for r in rows if r['sku']==sku]
        summary[sku]=st={n:{k:measures(observed,i,k) for k in ('candidate','v023')} for i,n in enumerate(('prepare','fwdh','finalize','total'))}
        def fmt(stage,m):
            measure=st[stage][m]
            return f"{measure['wape']:.2f}% / {measure['maximum']:.2f}%" if measure else 'N/A'
        title.append(f'| {sku} | {len(observed)} / 10 | {fmt("fwdh","candidate")} | {fmt("total","candidate")} | {fmt("total","v023")} |')
    title+=['',f'待回执 {len(pending)} 端，未通过 {len(failed)} 端。时间单位为 µs。','',
            '| T / H | SKU | FwdH 新 / CA | 旧合计 / 新合计 / CA | 970DT 合计外推 | CA 墙钟 s |',
            '|---|---|---:|---:|---:|---:|']
    values={}
    for r in rows:
        sk=r['shape'][1],r['shape'][2]
        n,o,a=r['candidate'],r['v023'],r['actual']
        values.setdefault(sk,{})[r['sku'][-4:]]=r
        title.append(f"| {sk[0]} / {sk[1]} | {r['sku']} | {n[1]:.3f} / {a[1]:.3f} | {o[3]:.3f} / {n[3]:.3f} / {a[3]:.3f} | {r['extrapolated_970_us']:.3f} | {r['cost_s']:.1f} |")
    title+=['','| T / H | 三段 DT/PR 旧 / 新 / CA | 方向正确 |','|---|---:|---|']
    for shape,ends in values.items():
        if len(ends)!=2:continue
        ratio={k:ends['9582'][k][3]/ends['957b'][k][3] for k in ('v023','candidate','actual')}
        right=(ratio['candidate']>1)==(ratio['actual']>1)
        pairs.append(dict(shape=list(shape),**ratio,direction_correct=right))
        title.append(f"| {shape[0]} / {shape[1]} | {ratio['v023']:.4f} / {ratio['candidate']:.4f} / {ratio['actual']:.4f} | {'是' if right else '否'} |")
    title+=['','[两批样本与完整逐窗口数值](comparison.json) · [第二批资源门槛、运行和恢复](queue.md) · [首批单独报告](../report.md)']
    payload=dict(independent_target_shapes_per_sku=10,first_register_sha256=sha(ARTIFACTS[0][0]),
                 supplement_register_sha256=sha(ARTIFACTS[1][0]),missing=pending,failed=failed,
                 metrics=summary,rows=rows,pairs=pairs)
    for name,data in [('comparison.json',json.dumps(payload,indent=2)+'\n'),('report.md','\n'.join(title)+'\n')]:
        dest=FOLDER/name;tmp=dest.with_suffix('.tmp');tmp.write_text(data);tmp.replace(dest)


if __name__=='__main__':
    publish()
