"""Render qualified device windows only after the frozen worker finishes."""
import argparse
import json
from pathlib import Path
import re,time

ROOT=Path(__file__).resolve().parents[3];OUT=ROOT/'results/v0.27/kda'

def dump(path,s):
    tmp=path.with_suffix('.tmp');tmp.write_text(s);tmp.replace(path)

def write():
    state=json.loads((OUT/'v1-state.json').read_bytes())
    complete=state.get('state')=='finished';rows=[]
    for label,soc in [('950DT','Ascend950DT_9582'),('950PR','Ascend950PR_957b')]:
        p=OUT/('v1-'+soc+'.json');log=OUT/'raw'/soc/'run.log'
        if p.exists():
            x=json.loads(p.read_bytes());body=f'[{p.name}]({p.name})'
            if x.get('skipped'):
                rows.append([label,'DT 全合同条件未触发 PR','—','—','—','N/A',body]);continue
            groups=x.get('output_groups',{});numeric=', '.join(str(groups.get(k,{}).get('bad','—')) for k in ['attention','aqk','akk','state_vk'])
            n=sum(s['kernel'].startswith('ChunkKdaFwd_') for s in x.get('stages',[]));us=x.get('formal_v1_kernel_us')
            rows.append([label,'合格' if x['qualified'] else '未合格',str(n)+'/4',numeric,(str(x.get('sync_errors','—')) if n else '目标未发起'),f'{us:.4f}' if us is not None else 'N/A',body]);continue
        if log.exists():
            matches=0;errors=0
            with log.open(errors='replace') as f:
                for line in f:
                    matches+=bool(re.search(r'Start profiling on kernel: ChunkKdaFwd_',line));errors+=('already has same set_flag' in line)
            rows.append([label,'采集中',str(matches)+'/4 已见开始标志','待结束',f'{errors}（截至此刻的重复SET）','N/A','[实时状态](v1-state.json)'])
        else:rows.append([label,'按 DT 资格条件排队','—','—','—','N/A','[冻结判据](v1-register.json)'])
    lines=['# Packed V1 四段原生执行','',
           '两条各一 token，总 T=2、H=1、K=V=128。原始四组输出的逐元素超差数顺序为 O / Aqk / Akk / 最终 VK。正式 µs 是同一公开 V1 申请的全部四个主设备 kernel 窗之和。辅助转置、提交间隔及 Kimi 外部 Q/K 归一化和状态路由另计。','',
           '|芯片|状态|物理 V1 主核|四组数值超差元素|同步检查|正式主核 µs|依据|','|---|---|---:|---|---:|---:|---|']
    lines.extend('|'+ '|'.join(r)+'|' for r in rows)
    lines+=['', '只有两组公开符号绑定、四组数值、未改初态、四份正式融合设备代码、全部主辅全核窗、时间顺序、正常退出与零同步错误同时成立才有正式设备时间。采中的零个错误须等程序结束才能判定。上游 host 中同一 stage 参数依次取 0、1、2、3；实际 ELF ordinal、原始窗口、官表与错误由逐卡回执留存。[同一 ABI 的发布版代码](https://github.com/flashserve/flash-linear-attention-npu/blob/5cc699069c7f91a35aabd583032c8720a0b626fa/fla/ops/ascendc/kda/chunk_kda_fwd/op_host/op_api/aclnn_chunk_kda_fwd.cpp#L879-L898) · [采集成本与宿主调度](host-scheduling.json)']
    if (OUT/'host-control/report.md').exists():
        lines+=['', '同一组两请求输入的独立宿主对照在原运行结束后登记，仍由同一 DT 正式库和四组金标核验。两次运行各有独立回执。[对照结果](host-control/report.md)']
    dump(OUT/'v1-outcome.md','\n'.join(lines)+'\n');return complete
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--follow',action='store_true');a=p.parse_args()
    while not write() and a.follow:time.sleep(45)
