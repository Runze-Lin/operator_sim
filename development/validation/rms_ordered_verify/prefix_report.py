"""Report the prefix targets and controls without adding the controls to the sample count."""
import sys
import time
import prefix
import prefix_published
import verify


def us(value):
    return 'N/A' if value is None else f'{value:.4f}'


def yes(value):
    return '通过' if value else '缺失'


def main():
    queue = prefix.read(prefix.PLAN)['queue']
    sections = []
    statuses = {}
    terminal = True
    for chip, spec in queue.items():
        source = prefix.OUT / (chip + '.json')
        if not source.exists():
            terminal = False
            statuses[chip] = {'state': 'pending', 'scored_count': 0}
            sections.append(f'### {chip}\n\n目标 {spec["scored_shape"]}，等待终态。')
            continue
        try:
            source = prefix_published.inspect(chip)
        except (OSError, KeyError, TypeError, ValueError) as exc:
            statuses[chip] = {'state': 'failed verification', 'scored_count': 0, 'reason': str(exc)}
            sections.append(f'### {chip}\n\n完成采集，计时待核验。{exc}')
            continue
        data = prefix.read(source)
        statuses[chip] = {'state': 'verified' if data['scored_count'] else 'incomplete',
                          'scored_count': data['scored_count'], 'receipt': verify.rel(source),
                          'receipt_sha256': verify.digest(source)}
        cost = data['process']
        table = [f'### {chip}', '',
                 f"采集 {cost['seconds']:.1f} s，峰值 {cost['peak_bytes']/2**20:.1f} MiB，"
                 f"进程退出码 {cost['exit_code']}，停止条件 {cost['stop_reason'] or '自然结束'}。"
                 f"后续新增有效计时 {data['scored_count']} 条。[逐核和指令证据]({source.name})", '']
        if 'controls' not in data:
            table.append(f"该采集未通过核验，原因为 {data['reason']}。")
        else:
            rows = [*data['controls'], data['target']]
            stages = [r['stage'] for r in rows if r['timing_qualified']]
            assert all(lhs['last_tick'] < rhs['first_tick'] for lhs, rhs in zip(stages, stages[1:]))
            table.extend([f"输入前缀 SHA256 `{data['input_sha256']}`。"
                          f"完成 {data['complete_host_golden_groups']} 组数值核对。", '',
                          '| 序号 | 角色 | shape | 原 CA µs | 新 CA µs | 新−原 µs | 原/新分核与 ELF 一致 | 新全核真实 END 与完整窗 |',
                          '|---:|---|---|---:|---:|---:|---|---|'])
            for r in rows:
                target = r['i'] == spec['target_i']
                stage = r['stage'] or {}
                table.append(f"| {r['i']} | {'计分目标' if target else '控制'} | {r['shape']} | "
                             f"{us(r['original_ca_us'])} | {us(stage.get('ca_us'))} | "
                             f"{us(r['whole_window_delta_us'])} | "
                             f"{yes(r['same_actual_core_roster'] and r['same_published_elf_text'])} | "
                             f"{yes(r['timing_qualified'])} |")
                if not r['timing_qualified']:
                    table[-1] = table[-1][:-1] + ' · ' + r['reason'] + ' |'
        sections.append('\n'.join(table))
    intro = """# RMSNorm 原调度顺序的前缀补采

本页与[核验状态](verification-status.json)使用 profiler 发布后的正式目录或封顶时保留的 device0 目录。采集控制器的 status.json 记录初次读取工作目录的结果。

本次复核只覆盖原十次连续调用中缺少物理 END 的两个目标。DT 重放序号 0–6，PR 重放序号 0–7，各端沿用原冻结预测、同一宿主程序、单一 ACL 流、顺序、张量形状、BF16 和 profiler 设置。宿主在读取序列前创建一次 ACL 流，对全部调用复用该句柄，末尾销毁。CA 内部 TASK_BEGIN 的整数 stream_id 随 launch 编号。序号从零开始，每端最后一个 shape 可补一条记录，前面的调用用于复核执行环境。原 10 次采集的进程退出码及每条计时资格保持在[原始核验](../scope.md)。

[采前登记](register.json)包含整段输入、预测、宿主 ELF、库 ELF 和验证脚本的哈希。每端仅启动一轮，60 分钟和 3072 MiB 封顶，双方串行。所有可计时行均核对两组数值、真实指令流入口、每颗活动核真实 END 与终点标签、逐核官方 duration、聚合及分核指令内容、写回总行数与字节。表中的 CA 是所有活动核的最早指令到最后 CA 终点，单位 µs。原采集中的缺失窗口标 N/A。"""
    content = intro + '\n\n' + '\n\n'.join(sections) + '\n'
    temp = prefix.OUT / 'report.md.tmp'
    temp.write_text(content)
    temp.replace(prefix.OUT / 'report.md')
    prefix.put(prefix.OUT / 'verification-status.json', {
        'state': 'finished' if terminal else 'partial',
        'scored_count': sum(q['scored_count'] for q in statuses.values()),
        'chips': statuses,
        'registration_sha256': verify.digest(prefix.PLAN)
    })
    print('terminal' if terminal else 'partial')


if __name__ == '__main__':
    if sys.argv[1:] == ['--watch']:
        # A fresh session can resume offline validation after the collector
        # finishes. Poll only status here to avoid rereading large trace files.
        while True:
            status = prefix.OUT / 'status.json'
            if status.exists() and prefix.read(status).get('state') == 'finished':
                break
            time.sleep(30)
    main()
