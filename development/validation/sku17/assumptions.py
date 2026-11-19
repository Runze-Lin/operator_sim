"""Change one 970DT hardware or kernel assumption at a time, without target labels."""
from dataclasses import replace
from campaign import ROOT,FUT,write
from operator_model import estimate,load_chip
from operator_model.quant_resource import _costs

def alternatives(op,hw):
    freq=hw.frequency_ghz
    options={}
    if op=='quant':
        rate=_costs()['write_bytes_per_cycle']*freq
        options={'每核写端口 0.75×':{'mte3_bandwidth_gbs':.75*rate},
                 '每核写端口 1.25×':{'mte3_bandwidth_gbs':1.25*rate},
                 '每核 UB 192 KiB':{'ub_bytes':192*1024},'每核 UB 384 KiB':{'ub_bytes':384*1024}}
    elif op=='matmul':
        options={'输出在窗内全写回片外':{'matmul_output_retained_fraction':0},
                 '可用 L2 / HBM 带宽均为标称 75%':{'matmul_l2_efficiency':.75,'matmul_hbm_efficiency':.75}}
    else:
        options={'允许每个忙 AIC 借用第二个 AIV':{'fa_aiv_per_aic':2}}
    return ({label:replace(hw,details={**hw.details,**changes}) for label,changes in options.items()}
            |{'整片 HBM 带宽 0.75×':replace(hw,hbm_bandwidth_gbs=hw.hbm_bandwidth_gbs*.75)})

def main():
    hw=load_chip(FUT)
    cases={'quant':[(1,256),(576,7168),(4096,8192)],'matmul':[(3,64,64),(1536,1792,16),(4096,4096,4096)],
           'fa':[(9,32768,64,1,1),(17,32768,64,4,8),(9,32768,64,128,8)]}
    rows=[]
    for op,shapes in cases.items():
        for shape in shapes:
            for scenario,chip in ({'配置默认':hw}|alternatives(op,hw)).items():
                result=estimate(op,shape,chip=chip,timing_mode='resource')
                z=result.projection['timing_diagnostics']
                rows.append(dict(op=op,shape=shape,scenario=scenario,us=result.latency_us,
                    tile=z['target_plan']['tile'],cores=[result.active_aic,result.active_aiv],
                    logical_io_gbs=result.rates['logical_io_gbs']))
    dest=ROOT/'results/v0.17';write(dest/'sensitivity.json',dict(role='970DT scenario only',rows=rows))
    lines=['# 970DT 未知参数对估时的影响','','以下均为同一未来芯片的条件外推。每次只调整指定假设，保持模型参数、shape 和其余配置不变。选取的 0.75× / 1.25× 与 UB 大小用于观察灵敏度，不代表芯片规格区间或误差置信区间。','']
    for op,shapes in cases.items():
        lines.extend([f'## {op}','','| 情景 | '+' | '.join(str(list(s))+' (µs)' for s in shapes)+' |',
                      '|---|'+ '---:|'*len(shapes)])
        for scenario in ['配置默认',*alternatives(op,hw)]:
            group=[r for r in rows if r['op']==op and r['scenario']==scenario]
            lines.append('| '+scenario+' | '+' | '.join(f'{r["us"]:.3f}' for r in group)+' |')
        lines.append('')
    lines+=['Quant 的默认每核完成速率和发起速率分别定价；本表只覆盖写端口，不同时重设读完成与读发起。UB 大小时，能容纳的行/列会影响 tile；未跨过分块阈值的 shape 可保持相同时间。Matmul 的窗内回写比例控制需在算子完成前计入 HBM 的字节，仍要为全部输出的本地 FIX/L2 服务付费。FA 的 AIV 借用会改变每个 AIC 分配到的 Vector 工作量；在所有 AIV 均忙时，主机实际可借的核数有限。','',
            '[逐点分块、核数及预测逻辑带宽](sensitivity.json) · [970DT 默认参数来源](../../configs/chips/Ascend970DT.json)']
    (dest/'sensitivity.md').write_text('\n'.join(lines)+'\n')
if __name__=='__main__':
    main()
