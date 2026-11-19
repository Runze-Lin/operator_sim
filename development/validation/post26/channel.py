"""Read pre-existing cold/repeat measurements for the separate channel-split kernel."""
import sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[3];sys.path.insert(0,str(ROOT))
from development.validation.sku19.campaign import read,write
from development.validation.post26.freeze import OUT

def main():
    rows=[]
    def add(sku,shape,first,repeat,first_phase,repeat_phase,source):
        lo,hi=sorted((first,repeat))
        rows.append(dict(sku=sku,shape=shape,first_us=first,repeat_us=repeat,
                         min_possible_max_pct=100*(hi-lo)/(hi+lo),point_us=2*hi*lo/(hi+lo),
                         first_phases_us=first_phase,repeat_phases_us=repeat_phase,
                         phase_differences_us={k:first_phase[k]-repeat_phase[k] for k in first_phase},source=source))
    pair=ROOT/'results/post-candidate/paired-result.json'
    for s,v in read(pair)['chips'].items():
        first,repeat=v['rows'];add(s,v['shape'],first['ca_us'],repeat['ca_us'],
          first['ca_milestones']['phase_us'],repeat['ca_milestones']['phase_us'],str(pair.relative_to(ROOT)))
    for p in sorted((ROOT/'results/v0.24/post/receipts').glob('wide-*.json')):
        v=read(p);first,repeat=v['rows'][:2];assert first['state']=='first' and repeat['role']=='repeat' and first['plan']['route']=='post_channel'
        if first['timing_qualified'] and repeat['timing_qualified']:
            add(v['chip'],first['shape'],first['stage']['ca_us'],repeat['stage']['ca_us'],
                first['ca_milestones']['phase_us'],repeat['ca_milestones']['phase_us'],str(p.relative_to(ROOT)))
    write(OUT/'channel-observability.json',dict(scope='historical exact shapes; builtin key 5 channel route, not post26 prospective token results',rows=rows))
    md=['# 小 token、宽通道的独立机制','', '| 芯片 | T×4×D | 首次 / 相同重复 µs | 任一 shape-only 点的最小可能最大相对误差 | 首次减重复的最大实际阶段差 µs |','|---|---|---:|---:|---:|']
    for r in rows:
        q=max(r['phase_differences_us'],key=lambda k:abs(r['phase_differences_us'][k]))
        md.append(f"| {r['sku']} | {'×'.join(map(str,r['shape']))} | {r['first_us']:.3f} / {r['repeat_us']:.3f} | {r['min_possible_max_pct']:.2f}% | {q} {r['phase_differences_us'][q]:+.3f} |")
    md+=['',r'同一 SKU 与数值输入得到两个完整设备窗口 $a<b$ 时，未知历史的最优单点为 $2ab/(a+b)$，其两端的相对误差都等于 $(b-a)/(b+a)$。这给单组两窗的最大误差一个精确下界；整套算子的 WAPE 仍取决于所有形状和频数。','',
         '这些窗口按通道切分，内置库使用不同 ELF 入口。小 token 可将一个 token 的通道片分到多个 AIV，写回四路输出的片段；本轮 token 路线按完整 token 行写出，端口结构不同。两者只共享硬件价格，并不能照搬逐行四笔读、单笔写的排程。','',
         '调用方能够知道的状态包括该流是否首次发起、上次成功派发的具体算子/合同、已知至少驻留的 AIV 数，以及是否刚执行了相同内置 Post。裸 shape 无法区分首次和相同重复。即使知道 AIV 数，也不能仅凭总数确认物理地址、缓存保留、编译器调度或跨算子影响。上述阶段差来自计时标记，不能唯一归因于缓存或互连。','',
         '[原始两端配对与每阶段数据](channel-observability.json) · [真实四路输出分片示例](../../v0.24/post/report.md)']
    (OUT/'channel-observability.md').write_text('\n'.join(md).rstrip()+'\n')
if __name__=='__main__':main()
