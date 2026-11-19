"""Render the accepted and unobservable per-launch windows without replacing original receipts."""
from pathlib import Path
import json

ROOT=Path(__file__).resolve().parents[3]
OUT=ROOT/'results/rms-dispatch-audit'

def main():
    evidence={}
    for chip in ('Ascend950DT_9582','Ascend950PR_957b'):
        f=OUT/f'verified-ordered-{chip}.json'
        if f.exists():evidence[chip]=json.loads(f.read_bytes())
    lines=['# RMSNorm 同一 ACL stream 的逐次设备窗复核','',
        '这次复核只读取已经结束的采集。宿主在循环外创建一个 ACL stream，十次调用逐次同步并核对 Y 和 RSTD。每个合格时延都来自本次调用全部活跃 AIV 的最早指令到最晚端点。一次命令的真实退出码与单个 launch 的可观测性分别列出。','',
        '| SKU | 单进程退出 | 原始 profiler 回执 | 实际墙钟 s | 合格 / 注册 | 数值检查通过组 |','|---|---:|---|---:|---:|---:|']
    for chip,record in evidence.items():
        original=record['original_process'];lines.append(f"| {chip} | {original['exit_code']} | {original['stop_reason'] or '正常'} | {original['seconds']:.1f} | {record['qualified_launches']}/10 | {record['host_goldens']} |")
    lines+=['','### 按冻结顺序复核的设备时延','',
            '| 索引 | `[token, channel]` | 950DT 全核窗 µs | 950DT 实际 AIV | 950PR 全核窗 µs | 950PR 实际 AIV |','|---:|---|---:|---:|---:|---:|']
    for i in range(10):
        row=[str(i)]; shape=next(iter(evidence.values()))['rows'][i]['shape'];row.append(str(shape))
        for chip in ('Ascend950DT_9582','Ascend950PR_957b'):
            if chip not in evidence:row+=['待复核','待复核'];continue
            r=evidence[chip]['rows'][i];st=r['stage'];row+=([f"{st['ca_us']:.4f}",str(st['active_aiv'])] if st else ['N/A',str(r.get('incomplete_evidence',{}).get('official_active_cores','N/A'))]);
        lines.append('| '+' | '.join(row)+' |')
    lines+=['', '### 时延、顺序和完整性的证据','',
        '1. 冻结文件、宿主 `inputs.txt`、实际 `case=`、`BATCH_BEGIN/END`、同段两个数值检查和单个库名按序一一对应。核实 `runtime_soc`、物理内存模拟器配置、同一 task 的 `TASK_BEGIN/DONE`、全批 `SMOKE_PASS` 和 profiler 的设备任务成功标志。CA 内部 `full_task_id` 对应 launch 序号；该内部日志的整数 `stream_id` 与宿主 ACL 流句柄属于不同接口。',
        '2. 使用权威的 `OPPROF_*/device0/RmsNorm_normal_all_bf16_high_performance_5000/<索引>/`。其中索引 0 的设备文件保存完整，即使 profiler 未将它复制到 `device0` 之外。每一行均验证捕获 ELF `.text` 与已安装 CANN 9.2.1 ops-nn 同名二进制相同，符号表给出设备入口 `.text+0x3e00`。',
        '3. 十份官方逐核 duration 表须与各 launch 的逐核 JSON 名单完全对应。所有 JSON 都解析到文件末。每个活跃 AIV 都必须包含入口 PC 的 `MOV_XD_SPR`，ELF `.text+0x51d0` 的实际 `END` 和末端 `END_LABEL`。逐核起止时长与官方两位小数 duration 表逐行比较（绝对差不超过 0.00501 µs）。聚合 trace 和所有核切片还逐核比较首末、真实 END、指令数和全部非同步指令 `(PC, opcode, start_cycle, end_cycle)` 的可重算多重集指纹。',
        '4. 额外用设备输出验证是哪一个 shape。`.text+0x4f38` 写 FP32 RSTD，`.text+0x5128` 写 BF16 Y。每个 Y 描述符的 burst 长度必须等于 `2×channel`，所有核累计 Y 行数为 `token`，Y 写入字节数为 `2×token×channel`，RSTD 为 `4×token`。这些观测独立于保存目录编号及预测的核数。',
        '5. 与已有 CA 报告一致，`stage.ca_us=(全核最晚 END_LABEL−全核最早设备指令)/1650`，1650 的单位为周期/µs。`physical_ca_us` 同时给出去掉合成标签后的真实指令窗，`cores` 保留每一核的官方 duration、首末 cycle、真实 END 位置和切片哈希。多核存在到达间隔，聚合时不能简单取最长的单核 duration。','']
    for chip,e in evidence.items():
        for r in e['rows']:
            if not r['timing_qualified']:
                diag=r.get('incomplete_evidence',{});s=diag.get('shard',{}); a=diag.get('aggregate',{})
                lines.append(f"**{chip} 索引 {r['i']}，{r['shape']}。** {r['reason']}。" + (f"官方表列出 {diag['official_active_cores']} 个核。{diag['core']} 官方时长为 {diag['official_duration_us']} µs，核切片最后观察到 cycle {s['last_seen_tick']}。该核的切片和合并文件分别记录了 {s['terminal_opcode_counts'].get('END',0)}、{a['terminal_opcode_counts'].get('END',0)} 条 `END`。保留两项数值检查，设备时间记 N/A。" if s and a else '本次缺少全部活跃核的可复核指令窗。'))
                lines.append('')
    for chip,e in evidence.items():lines.append(f"[{chip} 逐 launch 核对及哈希](verified-ordered-{chip}.json) · [原始回执](ordered-{chip}.json)")
    lines+=['','```bash','python development/validation/rms_ordered_verify/verify.py Ascend950DT_9582 Ascend950PR_957b','python development/validation/rms_ordered_verify/report.py','```','']
    outfile=OUT/'scope.md';tmp=outfile.with_suffix('.tmp');tmp.write_text('\n'.join(lines));tmp.replace(outfile)

if __name__=='__main__':main()
