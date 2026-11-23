"""Complete CA measurements with no corresponding model price, stored outside WAPE."""
from math import isclose, prod

from .__main__ import HERE, digest, encoded, read, require, sha_field, stage_window, unseal
from .system_reference import archive


def _row(op,contract,chip,shape,attrs,api,provider,dtype,stages,timing,cost,correctness,source):
    require(cost>0 and all(v['bad']==0 and v['checked']>0 for v in correctness.values()),
            'Invalid numerical evidence or cost')
    for s in stages:
        hashes=s['physical_shard_sha256']
        require(set(s['launched_cores'])==set(hashes) and all(sha_field(h) for h in hashes.values())
                and digest(encoded(hashes))==s['trace_sha256'],'Invalid physical shard identities')
        stage_window(s)
    clock=stages[0]['ticks_per_us']
    cores=[v for s in stages for v in s['cores'].values()]
    require(all(s['ticks_per_us']==clock for s in stages) and
            isclose(timing,(max(c['retire_tick'] for c in cores)-min(c['first_tick'] for c in cores))/clock),
            'Device duration does not match common clock')
    return dict(op=op,contract=contract,chip=chip,shape=shape,attrs=attrs,dtype=dtype,api=api,provider=provider,
                kernel=[s['kernel'] for s in stages],identity=[s['trace_sha256'] for s in stages],
                active_cores=[len(s['cores']) for s in stages],data_role='measurement_only',ca_us=timing,
                predicted_us=None,signed_error_pct=None,extrapolated_970dt_us=None,acquisition_seconds=cost,
                correctness=correctness,source=source)


def rows():
    source='cache_load/fp8-smoke.json.gz'
    for r in archive(HERE/source):
        require(r['correctness']=={'Y':{'bad':0,'checked':prod(r['shape'])}} and r['scale_dtype']=='fp32'
                and r['predicted_us'] is None,'FP8 cache-load contract changed')
        yield _row('cache_load','cache_load_fp8_antiquant',r['chip'],r['shape'],
                   {'implementation':r['api'],'scale_shape':r['scale_shape'],'sqrt_mode':False},r['api'],r['provider'],
                   'E4M3FN / FP32 → BF16',r['stages'],r['ca_us'],r['wall_seconds'],r['correctness'],source)
    source='elementwise/reference.json.gz'
    for r in archive(HERE/source):
        require(r['correctness']=={'Y':{'bad':0,'checked':prod(r['output'])}},'Missing elementwise output')
        yield _row(r['op'],r['op'],r['chip'],r['lhs'],{'rhs_shape':r['rhs'],**r['attrs']},r['api'],r['provider'],
                   r['dtype'],[r['stage']],r['ca_us'],r['wall_seconds'],r['correctness'],source)
    for alias in ('dt','pr'):
        source=f'quant_update/strided-950{alias}.json'
        r=unseal(read(HERE/source),'layout-observation')
        c=unseal(r['contract'],'layout-capability')
        require(r['correctness']=={k:{'checked':n,'bad':0} for k,n in c['outputs'].items()},'Invalid strided outputs')
        stages=[dict(kernel=s['kernel'],kernel_binary_sha256=s['object_sha256'],trace_sha256=s['trace_sha256'],
                     physical_shard_sha256=s['shard_sha256'],ticks_per_us=r['ticks_per_us'],
                     launched_cores=sorted(s['cores']),cores=s['cores']) for s in r['stages']]
        require(isclose(sum(map(stage_window,stages)),r['sum_kernel_windows_us']) and
                isclose(r['sum_kernel_windows_us']+r['interkernel_idle_us'],r['device_span_us']),'Invalid multi-kernel gap')
        yield _row('quant_mxfp4','quant_mxfp4_strided',c['chip'],c['shape'],{'input_stride':c['stride'],
                   'storage_offset':c['storage_offset']},c['api'],r['provider'],'BF16 → E2M1 + E8M0',
                   stages,r['device_span_us'],r['wall_seconds'],r['correctness'],source)


def render():
    from .report_pages import table
    records=list(rows())
    body=[]
    for r in records:
        shape='×'.join(map(str,r['shape']))
        impl=' → '.join(f"`{k}`" for k in r['kernel'])
        body.append([r['op'],r['dtype'],shape,r['chip'],f"`{r['api']}`",impl,' / '.join(map(str,r['active_cores'])),
                     f"{r['ca_us']:.4f}",f"{r['acquisition_seconds']:.2f}",f"[收据](../../calibration/{r['source']})"])
    return '\n\n'.join(['# 已测量、尚无通用价格的调用','[总误差分析](../error-analysis.md) · [系统捕获的全部 80 个名字](../system-workload-coverage.md)',
            '每一行均已核对有效输出、实际安装 ELF 和每个物理核的退休。模型时间、预测误差、等效 Roofline 利用率和 970DT 外推均为 N/A。两个端点各一例只能证明这些输入能够运行。单 kernel 采用全芯片首发至末退，多 kernel 使用同一个设备时钟并计入发射之间的间隔。',
            table(['调用','精度','主输入','芯片','ACLNN','实际 kernel','每阶段 AIV','CA µs','离线秒','证据'],body),
            'Add 的两输入都是连续 [9,7168]，alpha=1；Mul 的另一路为 [9,1]，沿末轴广播。FP8 缓存的每行配一个 FP32 scale，按 [128,1] 传入，输出 BF16。FP4 量化输入是 stride=[6288,1]、起始偏移 6144 个 BF16 元素的 Kimi 视图。单独运行时的 Slice 可能由系统的上游物化或融合消掉。输入变化应重新核对派发。',
            '从仓根运行 `python -m calibration.lookup cache_load --contract cache_load_fp8_antiquant --shape 1 128 512` 可读取 CA、源收据、数值与布局。'])+'\n'
