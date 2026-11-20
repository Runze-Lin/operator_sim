"""Keep host-scheduler controls separate from operator latency evidence."""
import json
from pathlib import Path

OUT=Path(__file__).resolve().parents[3]/'results/v0.30/host'


def read(p):return json.loads(p.read_bytes())


def report():
    reg=read(OUT/'freeze.json')
    lines=['# 小形 Matmul 的 CA 宿主并发对照','',
           '以真实 Kimi 投影 [1,2304,1536] 为同形控制。原收据均含正确性与全部活动核的 END/退休。试验为宿主分配四个物理 CPU，配置 esl_top_thread_cnt=1，另设 CAMODEL_OMP_THREADS=2。两款芯片串行、每端最多 480 秒。','',
           '| 芯片 | 已归档设备 µs / 采集墙钟 s | 控制墙钟 s | 日志顶层线程 / 内层报告并发 | 控制完整设备 µs |',
           '|---|---:|---:|---:|---:|']
    for chip,tag in [('Ascend950DT_9582','dt'),('Ascend950PR_957b','pr')]:
        ref=reg['controls'][chip]; p=OUT/(tag+'.json')
        if not p.exists(): wall,workers,t='待回收','—','—'
        else:
            r=read(p); wall=f"{r['cost']['seconds']:.1f}"; workers='/'.join((','.join(r['esl_workers']) or '?',','.join(r['reported_inner_concurrency']) or '?'))
            t=f"{r['ca_us']:.3f}" if r.get('kernel_qualified') else '—'
        lines.append(f"| {chip} | {ref['baseline_us']:.3f} / {ref['reference_wall_s']:.1f} | {wall} | {workers} | {t} |")
    extra=OUT/'pr-retirement.json'
    if extra.exists():
        detail=read(extra)
        lines+=['',f"PR 在截止前已写出 18 个核的 END 与全部退休事件，2,304 个 BF16 输出数值正确，MMAD 总数为 216。离线按首条指令至末核退休重建窗为 {detail['physical_event_window_us']:.3f} µs，比原正式窗高 {detail['relative_delta_pct']:.2f}%。宿主进程未在预算内完整退出，原冻结验收仍失败。这个差值表明在本次控制中设备窗也发生了变化。[逐核退出及原始指纹](pr-retirement.json)"]
    lines+=['','本次环境变量未使内层日志报告的并发降至 2。正式长形继续使用原先的 CA 配置，新增解码队列单槽运行。宿主墙钟和设备 µs 分别计量。','',
            '[冻结与对照指纹](freeze.json) · [采集进度](state.json)','']
    (OUT/'report.md').write_text('\n'.join(lines))


if __name__=='__main__':report()
