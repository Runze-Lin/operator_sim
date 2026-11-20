"""Render a reversible terminal/ongoing report without modifying either receipt."""
import argparse
import json
from pathlib import Path
import re
import time

ROOT=Path(__file__).resolve().parents[3];BASE=ROOT/'results/v0.27/kda';DEST=BASE/'host-control';RAW=BASE/'raw/host-control'
def read(p):return json.loads(p.read_bytes())
def fmt(v):return 'N/A' if v is None else f'{v:.3f}'
def report():
    reg=read(DEST/'register.json');o=read(BASE/'v1-Ascend950DT_9582.json');prev=read(DEST/'original-native-evidence.json');near=read(BASE/'host-scheduling.json')
    state=read(DEST/'state.json') if (DEST/'state.json').exists() else {};counts={};latest=[];terminal=(DEST/'dt.json').exists();right=read(DEST/'dt.json') if terminal else {}
    if (RAW/'run.log').exists():
        with (RAW/'run.log').open(errors='replace') as fp:
            for line in fp:
                match=re.search(r'Start profiling on kernel: (\S+)',line)
                if match:latest.append(match.group(1))
                if 'already has same set_flag' in line:counts['repeat_set']=counts.get('repeat_set',0)+1
    drawn=sum(x.startswith('ChunkKdaFwd_') for x in latest);active=state.get('headroom',reg['initial_headroom'])
    cost=right.get('cost',{}); real=[]
    for arr in ('attention','aqk','akk','state_vk'):
        if arr in right.get('output_groups',{}):
            v=right['output_groups'][arr];real.append(f'{arr} {v["bad"]}/{v["count"]}')
    rows=['# V1 的 DT 宿主对照','',
          '原样调用 FLA A5 26.9.1 `aclnnChunkKdaFwd`，输入 `[1,2,1,1,128,128]`，两条请求各一 token，保留四个逐元素金标。这个实验检验同一正式库的四次阶段发起能否越过辅助转置。', '',
          '|记录|原 DT|单次 DT 宿主对照|950PR|','|---|---|---|---|',
          f'|采集进度|超时|{("完成" if terminal else state.get("state","排队"))}|未采集|',
          f'|墙钟 s|{o["cost"]["seconds"]:.1f}|{fmt(cost.get("seconds"))}|—|',
          f'|可观察的 V1 主核发起|0 / 4|{drawn} / 4|—|',
          f'|完成且可核验的 V1 指令窗|0 / 4|{sum(s.get("ca_us") is not None for s in right.get("stages",[]) if s["kernel"].startswith("ChunkKdaFwd_"))} / 4|—|',
          f'|四组逐项超差/元素数|N/A|{", ".join(real) if real else "N/A"}|—|',
          f'|重复置旗|目标尚未发起|{counts.get("repeat_set",0) if drawn else "目标尚未发起"}{"（部分日志）" if drawn and not terminal else ""}|—|',
          f'|合格 V1 主核总时间 µs|N/A|{fmt(right.get("formal_v1_kernel_us"))}|N/A|','',
          '原 DT 只捕获第一个 FP32 Transpose 的 `.text`。STARS tick 490 是任务开始，500 是 AIV0 的 block start，542 是发往 AIV0 的 kick。原生 task 文件只有 BEGIN，无 END，官方逐核表为零。超时退出时整机日志虽报 cycle count 10000，无法把它归属 KDA。该次测到的是前置辅助核进入设备，无法判断一次 V1 重复置旗是否发生。','',
          f'本次沿用原 host、ELF、每个输入/金标 SHA、标准 msopprof 及资格代码，与原队列使用同一互斥锁。原 8 CPU 条件曾临时增加 16 个 CPU 约 50 秒，恢复时整机剩余空闲量为 {near.get("idle_cpu_equivalents")} 个等值 CPU；旧控制器同时检查物理核冲突和余量，原日志没有保存触发时逐线程位置。此次在连续四个 2 秒窗口检查空闲物理核，只使用 {len(active["actual_selected_cpus"])} 个，每个实测忙碌率 ≤20%，采前整机空闲 {active["idle_equivalents"]:.1f} CPU、1 分钟负载 {active["load"][0]:.2f}。给其它作业预留至少 96 个空闲等值 CPU。墙钟限 1,440 s，目录限 8 GiB，仅 DT 一次。采前宿主环境与原运行时段不同，实际机时差不能全归于一个数值参数。', '',
          '[上次原生日志指纹和首个真实 block](original-native-evidence.json) · [冻结及预留资源](register.json) · [对照状态](state.json)']
    if terminal:
        rows+=['','## 实际窗口','', '|设备核、ordinal|参与核数|诊断窗 µs|正式组合资格|','|---|---:|---:|---|']
        for s in right.get('stages',[]):
            n=len(s.get('cores',{})) if isinstance(s.get('cores'),dict) else s.get('active_cores',s.get('cores'))
            if n is None and isinstance(s.get('per_core'),dict): n=len(s['per_core'])
            rows.append(f'|`{s["kernel"]}` #{s["ordinal"]}|{n if n is not None else "见回执"}|{fmt(s.get("ca_us"))}|{"通过" if right["qualified"] else "N/A"}|')
        if right['qualified']:rows+=['','四份 V1 主核与全部辅助核的指标均达到冻结判据，此记录是两条独立 1-token 序列的 DT 开发样本。PR 与其他全数据量仍缺实测。']
        elif drawn and counts.get('repeat_set',0):rows+=['','该宿主配置到达了正式 V1 主核，实际回调重现重复 SET。此结果排除了“所有失败都发生在辅助启动阶段”的解释。逐段窗口只用于定位，不能用来标定正常硬件耗时。']
        elif drawn:rows+=['','V1 已发起，但原始验收条件未同时满足；排障窗口只保留作诊断。']
        else:rows+=['','在本次边界内仍没有 V1 主核发起证据。四阶段同步表现保持未知。']
        rows+=['','[实测、ELF 和全核数值/时序回执](dt.json) · [原生任务与回调归属依据](native-evidence.json)']
    else:rows+=['','采集中的零条回调不代表同步通过。此表由原始程序状态自动更新。']
    faults_file=DEST/'fault-sites.json'
    if faults_file.exists():
        phases=read(faults_file)['phases']
        rows+=['', '## 将错误缩小到官方 V1 的 Prepare 和 Finalize', '',
               '|同族物理 ordinal · 宿主阶段|重复 SET|不同故障 PC|与单序列 FULL 故障 PC 相同数|首个匹配 WAIT 之前已二次 SET 的 AIC / 全部 AIC|',
               '|---|---:|---:|---:|---:|']
        for z in phases:
            rows.append(f"|{z['physical_launch']} · {z['source_stage_order']}|{z['duplicate_callbacks']}|{z['unique_offsets']}|{z['common_sites_with_single_sequence']}|{z['aic_with_any_two_sets_before_first_matching_wait']} / {z['aic_count']}|")
        rows+=['','132 次回调的 PC 均在相应的实际载入 `.text` 中反汇编为 `SET_FLAG.M.MTE1`，四个 stage 与单序列控制的融合 `.text` 相同。原单序列 FULL 的故障 PC 与切成四段后的故障 PC 没有重合；同一二进制实际走了不同的代码段。对每个阶段重新从各 AIC 窗口起点计数，在 Prepare 和 Finalize 内各找到两个物理 AIC、各三个事件 ID 的严格前缀证据。同一阶段内第一次匹配 WAIT 发起前已有两次 SET，跨阶段重置事件无法消除这类错误。不能把整阶段的 132 次回调都唯一定位到单行 C++。', '',
               '例如 Prepare 的 `core0.cubecore0` 对 ID 0 在 tick 42,398 / 56,413 两次置旗，对应相对加载基址 `0x11316c` / `0x141568`，后一个位置正是模拟器报告的错误 PC。供应商可以分别检查 phase 0 和 phase 3 的 TPipe 与 Catlass 本地资源生命周期，然后复用本页同一公开调用和四组原始数组验证修复。两个中间阶段的零回调仅描述这次执行，不能推出其它输入都同步正确。[逐 PC 指令、每核首 WAIT 前证据和原始 trace SHA](fault-sites.json)']
    target=DEST/'report.md';tmp=target.with_suffix('.tmp');tmp.write_text('\n'.join(rows)+'\n');tmp.replace(target);return terminal
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--follow',action='store_true');a=p.parse_args();
    while not report() and a.follow:time.sleep(25)
