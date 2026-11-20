"""Update public per-endpoint rows when the new/previous owners publish a final receipt."""
import argparse
import fcntl
import os
from pathlib import Path
import time
from acquire import ROOT, OUT, OLD, DT, PR, FUT, REG, read, now, rel, score, save
from certificate import sha


def fmt(value): return 'N/A' if value is None else f'{value:.3f}'


def refresh():
    score(); reg=read(REG); data=read(OUT/'score.json'); archive=reg['archive_inventory']; state=read(OUT/'state.json') if (OUT/'state.json').exists() else {}; old_state=read(OLD/'state.json'); current=reg['forecasts']
    table=['| 数据角色与 M（N=128、K=512） | 芯片 | 预测 µs | 完整 CA µs | 有符号误差 % | CA 墙钟 s |','|---|---|---:|---:|---:|---:|']
    for row in data['rows']:
        table.append(f"| {('新冻结分块' if row['role']=='independent_chunk' else '既有整次 prefill 预约')} · {row['shape'][0]:,} | {row['chip']} | {fmt(row['prediction_us'])} | {fmt(row['ca_us'])} | {fmt(row['signed_error_pct'])} | {fmt(row['capture_s'])} |")
    pairs='；'.join(f"{x['role']} 预测 {x['predicted_dt_over_pr']:.3f}、实测 {fmt(x['measured_dt_over_pr'])}" for x in data['pairs'])
    state_line=f"新采集的最近状态为 `{state.get('phase','queued')}`，芯片为 `{state.get('chip','—')}`。原整次 prefill 队列记录为 `{old_state.get('phase')}`、`{old_state.get('endpoint','—')}`。"
    lines=['# BF16 Addmm 大规模双端验证','',f'已取得 {sum(r["qualified"] for r in data["rows"])} / 4 个合格大形窗口。{state_line} 每个端点在完整数值、身份和时序检查结束后计入表。','',*table,'',f'同 shape 的耗时比定义为 DT/PR。{pairs}。970DT 对 65,536 和 262,144 行的预测分别为 {current[FUT]["prediction_us"]:.3f}、{reg["existing_full"]["prices"][FUT]:.3f} µs，均为外推。','',
    '## 覆盖范围和实际调用','',f"冻结前扫描 {archive['raw_logs']:,} 份 run.log，识别 {archive['native_host_case_lines']} 条原生 `aclnnAddmm` 调用记录。核对标准回放后，DT 有 34 窗 / 33 个形状，PR 有 20 窗 / 20 个形状；最大 M 为 1,024，最大 MNK 为 [512,512,256]。发现的 {len(archive['large_bias_free_examples'])} 份 M≥32768、N128、K512 的 `aclnnMatmul` 原生记录均为无偏置合同。扫描路径和重用排除见冻结清单。",'',
    '新形 [65536,128,512] 采用 Flash 的 M=8×8192、N=index_head_dim=128、K=head_dim=512，等分四个 chunk 覆盖 B8×32Ki 的 token 行。选用拥有 K 的 C1 来源层，每个当前 token 产生一行。实际 vLLM `wk` 在该源码修订显式设 `bias=False`。本验证取其矩阵尺寸，另构造列偏置 `self[128]`。完整 262,144 行实验保留原双端预约。实际应用的分块取决于调度器的总 token 预算。[模型配置](../../../configs/models/DeepSeek-V4.1-Flash.json) · [锁定的 vLLM Indexer 源码](https://github.com/vllm-project/vllm-ascend/blob/c09d3e20b9c45c80b34ae3781440d90904a44023/vllm_ascend/models/deepseek_v41/indexer.py#L74-L80) · [C1 token 来源](https://github.com/vllm-project/vllm-ascend/blob/c09d3e20b9c45c80b34ae3781440d90904a44023/vllm_ascend/attention/dsa_v41.py#L510-L521)','',
    '列偏置实验显式调用 CANN 9.2.1 `aclnnAddmm`。A、B、self 和 Y 是 ND BF16，α=β=1，cubeMathType=0。初始化 Y 为 NaN。CPU 在 FP64 累加 XW+b，复用确定性输入沿 K 的 17 元素周期加速金标，逐元素检查绝对加相对容差。相同输入若仅计算 BF16(XW)，8,388,608 个结果里有 5,582,129 个会失败。大形有 8,589,934,592 个 Cube FLOP 和 8,388,608 个标量偏置加法，比旧档最大样例的 Cube 工作量大 64 倍。逻辑 A/B/bias/Y 分别为 64 MiB、128 KiB、256 B、16 MiB。','',
    '计时从最早核心首条指令到最晚流水退休。资格器核验实际 SKU、DRAM、唯一 STARS 任务、实际调用、输出、正式库 ELF、完整的逐核物理 END 和设备表。用原生窄路径读取完整 X 指令窗并保留每核 SHA，不开启可选原始 dump，保留完整内核流水。指令级偏置工作由 `MOV_L1_TO_BT` 交叉核验，部分指令无字节字段，记录为 N/A。终止或部分轨迹仅用于诊断。','',
    '## 可区分的两种多波排程','',
    '| 芯片 | 默认 tile / 波次 / MMAD | B 常驻候选 tile / 波次 / MMAD | 默认 µs | 候选 µs（搬运重叠 / 两项字节服务串行） |','|---|---|---|---:|---:|']
    for chip,p in current.items():
        alt=p['b_resident_scenario']; lines.append(f"| {chip} | {p['tile']} / {p['waves']} / {p['mmads']} | {alt['tile']} / {alt['waves']} / {alt['mmads']} | {fmt(p['prediction_us'])} | {fmt(alt['overlapped_us'])} / {fmt(alt['serialized_service_us'])} |")
    lines+=['','候选来自相同矩阵尾维的无偏置 MatMul 真实体内流水，先验条件为 B 在每个活动 AIC 的 L1 只读一份、每个输出块执行偏置搬运，且新分块的 MMAD 数与观测吻合。候选的两个点分别假设偏置完全重叠，或另外支付 cold bias 扇出与关键核 BT 字节服务；后者取 `2·N·C/(1000·BW_HBM)+2·N·W/(1000·BW_MTE1)` 微秒，C 为活动核数、W 为关键核波数、带宽用 GB/s。描述符发起等待尚不可辨识，这两个点不能当整窗上下界。[独立 MatMul 档案 DT](../../v0.32/matmul/128ki-dt-geometry.json) · [PR](../../v0.32/matmul/128ki-pr-geometry.json)','',
    '## 现有流水的资格器对照','',
    '| 芯片 | [M,N,K] | CA µs | AIC | MMAD | L1→BT | 全量输出 / 漏偏置会失败 |','|---|---|---:|---:|---:|---:|---:|']
    ctrl=OUT/'controls.json'
    if ctrl.exists():
        for x in read(ctrl)['archived']:lines.append(f"| {x['sku']} | {x['shape']} | {x['ca_us']:.3f} | {x['active_aic']} | {x['mmads']} | {x['bt_moves']} | {x['output_checked']:,} / {x['rejected_without_bias']:,} |")
    for row in data['rows']:
        if row['role']=='independent_chunk' and row['receipt']:
            r=read(ROOT/row['receipt']); v=r['verdict']
            if v.get('qualified'):
                lines+=['',f"**{row['chip']} 实际大形**。{v['active_aic']} AIC，{v['mmads']:,} 条 MMAD、{v['fixp_descriptors']:,} 次 FIXP 写片、{v['bias_bt_moves']:,} 次 L1→BT；完整 B 描述符计数 {v['full_rhs_descriptor_address_counts']}。指令几何为 `{v['mmad_geometry']}`。公开预测偏差 {row['signed_error_pct']:+.2f}%。[完整端点回执]({Path(row['receipt']).name})"]
            else:lines+=['',f"**{row['chip']} 大形未通过**。{v.get('reason')}。已保留原始回执。"]
    lines+=['','同一物理互斥槽每次运行一个端点，结束即释放给其他队列。每端预算 4 小时和 16 GiB；相同形状只采一次，所有中断保留。当前三颗芯片的公开模型查询墙钟为 '+', '.join(f"{s} {p['query_ms']:.2f} ms" for s,p in current.items())+'。','', '[逐端价格与完整回执](score.json) · [采前冻结](freeze.json) · [三项离线对照](controls.json) · [旧整次 prefill 预约](../../v0.29/large/cube/freeze.json)']
    temp=OUT/f'.report.{os.getpid()}.md';temp.write_text('\n'.join(lines)+'\n');temp.replace(OUT/'report.md')


def watch():
    raw=ROOT/'runs/addmm34'; fix=[OUT/f'{s}.json' for s in ('dt','pr')]+[OLD/f'addmm-{s}.json' for s in ('dt','pr')];seen=None
    with (raw/'report.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        while True:
            sig=tuple(p.stat().st_mtime_ns if p.exists() else None for p in fix+[OUT/'state.json',OLD/'state.json',OUT/'controls.json'])
            if sig!=seen:refresh();seen=sig
            if all(p.exists() for p in fix):return
            time.sleep(30)

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('action',choices=['refresh','watch']);args=parser.parse_args();{'refresh':refresh,'watch':watch}[args.action]()
