"""Render one matched-pair WAPE and timing table from qualified receipts only."""
import json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[3];OUT=ROOT/'results/rms-unknown23'

def main():
    j=json.loads((OUT/'independent.json').read_text());lines=['# 冻结后同形双前序 CA 验证','',f'已核验 {j["completed_processes"]}/8 个宿主进程，{len(j["pairs"])}/20 个 shape/SKU 配对，{len(j["rows"])}/40 个有效目标窗口。每条流的重复前缀单独留档。','', '| SKU | 协议 | 合格窗 | v0.22 仅 shape WAPE / 最大 | 候选仅 shape WAPE / 最大 | 提供 H，WAPE / 最大 |','|---|---|---:|---:|---:|---:|']
    for s in j['groups']:
        if s['group'] not in ('all','entry','immediate_repeat'):continue
        text={'all':'合并','entry':'每形第一次','immediate_repeat':'紧接同形复跑'}[s['group']];values=[s['scores'][k] for k in ('previous_us','candidate_us','known_us')];lines.append('| '+' | '.join([s['chip'][6:11],text,str(s['n']),*[f'{r["wape"]:.2%} / {r["max_ape"]:.2%}' for r in values]])+' |')
    lines+=['','同芯片的两次实测共享相同 shape、dtype、epsilon、输入生成、宿主逻辑流、kernel ELF 和设备计时口径。H 是同一进程前序派发使用的 AIV 数量高水位。下表列完整配对，首次列包含独立首发和扩核。','', '| SKU | `[T,D]` | H 前→后 | CA 前 / 后 µs | v0.22 → 候选，仅 shape µs | 已知 H 前 / 后 µs | 同点最大误差下界 |','|---|---|---:|---:|---:|---:|---:|']
    for p in j['pairs']:
        rs=[r for r in j['rows'] if r['chip']==p['chip'] and r['shape']==p['shape']];r,s=sorted(rs,key=lambda i:i['i']);lines.append(f'| {r["chip"][6:11]} | `{r["shape"]}` | {p["entry_H"]} → {p["repeat_H"]} | {p["entry_us"]:.3f} / {p["repeat_us"]:.3f} | {r["previous_us"]:.3f} → {r["candidate_us"]:.3f} | {r["known_us"]:.3f} / {s["known_us"]:.3f} | {p["minimax"]:.2%} |')
    lines+=['','对每组实测时延 E、R，只用相同可见输入给出同一个点时，最大相对误差的下界为 `|E−R|/(E+R)`。两个协议各自 WAPE 的最优最坏下界为 `Σ|E−R|/Σ(E+R)`，其中求和只取本表该 SKU 已完成配对；追加样本会改变该 WAPE 下界。任何已完成配对的 40% 可行区间互不相交，就能排除该 SKU 在包括这对样本的完整数据集上达到 40% 最大误差。','', '| SKU | 完整对 | 最大相对误差的不可约下界 | 两协议 WAPE 的不可约下界 | 双协议同时 WAPE≤20%、最大≤40% |','|---|---:|---:|---:|---|']
    for r in j['bounds']:
        lines.append(f'| {r["chip"][6:11]} | {r["n"]} | {r["minimax_max"]:.2%} | {r["both_protocol_minimax_wape_without_max"]:.2%} | '+('有可行点' if r['max40_and_each_wape20'] else '无可行点')+' |')
    total=sum(r['ca_wall_s'] for r in j['receipts'].values());lines+=['',f'已完成进程的累计采集墙钟为 {total:.1f} s。验收逐次核对 BF16 y 与 FP32 rstd 的全部元素、原始输入序号和 CA task，逐核核对正式 ELF 的入口和物理 END、最终 END_LABEL、两个输出写回字节、官方 duration 及合并与分核 trace。CA 的 SQE `stream_id` 在本仿真器的每个发射上递增；同一 ACL stream 由已冻结 NORM 宿主在循环外的单次创建、逐发同步与全程任务顺序确认。','', '[逐形结果与约束](independent.json) · [逐核核验](verified/) · [采集预算和可执行文件](collection.json)']
    (OUT/'independent.md').write_text('\n'.join(lines)+'\n')
if __name__=='__main__':main()
