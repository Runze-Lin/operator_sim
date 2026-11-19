"""Publish authentic release identities, observed windows, and the repair boundary."""
from pathlib import Path
from collections import Counter
import subprocess,sys
from packages import HERE,ROOT,OUT,RAW,read,save,probe,when
sys.path.insert(0,str(HERE.parent/'kda_v1_24'))
from inspect_sync import code_map

def main():
    new=read(OUT/'dt-t2-h1-varlen.json');oldroot=ROOT/'results/v0.24/kda-v1';old=read(oldroot/'dt-t2-h1-varlen.json')
    main=lambda r:next(i for i in r['stages'] if i['kernel'].startswith('ChunkKdaFwd_'))
    rec=main(new);root=ROOT/rec['raw_path'];bin=root/'dump/aicore_binary.o'
    base=int((bin.parent/'pc_start_addr.txt').read_text(),16)
    offs={int(pc,16)-base:n for pc,n in new['sync_fault_pc_counts'].items()}
    decoded=code_map(bin,offs)
    old_arrays=read(oldroot/'dt-t2-h1-varlen-arrays.json')['outputs']
    new_arrays={k:probe.digest(RAW/'dt-t2-h1-varlen'/(k+('.f32' if k=='state_vk' else '.bf16'))) for k in old_arrays}
    same_arrays=all(x['actual_sha256']==new_arrays[k] for k,x in old_arrays.items())
    coeff=dict(utc=when(),release='26.10.0-release',device_text_old=main(old)['device_text_sha256'],device_text_new=rec['device_text_sha256'],
        dynamic_symbol_library=probe.rel(RAW/'release/fla_npu/opp/vendors/fla_npu_transformer/op_api/lib/libcust_opapi.so'),
        same_device_arrays_as_pinned_baseline=same_arrays,actual_array_sha256=new_arrays,
        complete=new['full_windows'],correct=new['golden'],sync_errors=new['sync_errors'],opcode_counts=dict(Counter(x['mnemonic'] for x in decoded.values())),
        duplicate_set_instructions=[dict(offset=hex(k),occurrences=offs[k],**v) for k,v in sorted(decoded.items())],
        confirmed_full_observation_us=rec.get('ca_us'),calibration_time_us=new['main_ca_us'],
        release_binary=probe.rel(bin),release_binary_sha256=probe.digest(bin),acquisition_cost=new['cost'])
    save(OUT/'release-comparison.json',coeff)
    checks=read(OUT/'m-mte1-retirement.json')['cases'];control=read(OUT/'compiler-controls.json')
    newr=next(r for r in checks if r['label']=='release-dt-t2-h1-varlen')
    count=newr['kernel_init_prefix_certificates'];ncores=newr['cores_with_all_3_initial_double_sets']
    x=newr['cores']['core0.cubecore0']['initial_double_set_before_any_wait']
    examples='，'.join(f"ID {flag} 在 tick {f['first_set_tick']} 与 {f['second_set_tick']}" for flag,f in x.items())
    text=f"""# 官方 FLA 26.10 的融合 KDA V1 复核

**950DT，B=1、T=2、单本地头、K=V=128。** 以 Kimi 的显式变长元数据 `cu=[0,2]` 走公开 `aclnnChunkKdaFwd`。目标发行包为独立复核的 26.10.0；已固定的 Kimi 镜像依赖 26.9.1。

|发布与芯片|四组逐元素金标|已核对的 launch|M→MTE1 重复 SET|主核原始观察窗|可校准耗时|
|---|---|---:|---:|---:|---:|
|Kimi 固定的 26.9.1 / DT|全通过|4|{old['sync_errors']}|{main(old)['ca_us']:.4f} µs|N/A|
|新正式发布的 26.10.0 / DT|全通过|{new['official_tables']}|{new['sync_errors']}|{rec['ca_us']:.4f} µs|N/A|
|新正式发布的 26.10.0 / PR|未触发冻结条件|—|—|—|N/A|

26.10.0 动态链接到该 wheel 的两个 FLA 公开符号。主核捕获的 `.text` 为 `{rec['device_text_sha256']}`，与新发行对象完全一致。核内共 {rec['mmads']} 条 MMAD，参加主核的有 {rec['active_aic']} 个 AIC、{rec['active_aiv']} 个 AIV。全调用含前后 Transpose，每个 launch 都逐核核对了官方 duration、指令签名、物理 END 与合并 B/E 事件。O、Aqk、Akk、最终 VK 状态四组实际数组与旧发布逐字节相同，初始状态仍然完整。CA 墙钟 {new['cost']['seconds']:.1f} 秒，监测到的采集目录峰值 {new['cost']['peak_bytes']/2**30:.3f} GiB。重复置旗使这条调用继续缺少正常耗时参考。PR 按登记的 DT 前置条件跳过，本轮新增 CA 共 1 次。[采前清单](registration.json) · [整调用逐核回执](dt-t2-h1-varlen.json) · [设备代码、实测数组及故障指令](release-comparison.json)

## 哪些新版本经过了核查

- GitHub Releases 的 A5/aarch64 稳定发布于 11 月 15 日，feature 发布于 11 月 17 日。两份 wheel 都通过发布方的文件长度和 SHA256 校验，各自提供 8 个 V1 对象、4 种可区分的机器代码。两个新包之间的 4 种代码全部相同，相比 26.9.1 的 4 种全部改变。feature 的说明是放宽 pack 内子序列数量上限，重复采它不能验证新的融合设备实现。26.10 的 OPP 标注编译器为 9.2.1，旧版为 9.1.0。[正式 stable release](https://github.com/flashserve/flash-linear-attention-npu/releases/tag/v26.10.0-release) · [feature release](https://github.com/flashserve/flash-linear-attention-npu/releases/tag/v26.10.0-feature) · [两版 SHA、ELF 与库文件](packages.json)
- 比较 wheel 中对应的 12 个融合设备源文件，26.9.1 与 26.10 的逻辑行内容一致，存在换行编码差异。当前主线 `dc1113e2` 和 v26.11 分支 `7b0b1b7c` 在融合 op_kernel 下的 Git blob 全数相同，相对 26.10 的 6 个改动文件全部切换编译架构判据（310 宏切换到 3510 宏）。截至此次 GitHub API 可见快照，v26.11 没有带 A5/aarch64 wheel 的正式 Release。主线其他模块、源码包的完整系统运行无法由这次 V1 源码比较推定。[逐文件固定 Git blob 与源码差分](upstream-source.json)

## 可以直接修的前置条件

CANN 9.2.1 的 DAV_3510 `TPipe::Init` 在每个 AIC 上为 M→MTE1 ID 0、1、2 各置一次旗；`DestroyWithoutPipeAll` 逐一等待并释放这三项。FLA 的 V1 根函数构造外层 TPipe，多个 KDA Prepare 子例程又局部构造 `Catlass::Arch::Resource<Ascend950>`。该随包发布的资源类按值持有另一个 TPipe，并在构造函数体内立即调用 `pipe.Destroy()`。C++ 在进入构造函数体前已经构造了成员，因此内层的三个 SET 可能先于外层的 WAIT。消除这种重复初始化需要让拥有相同核事件的资源生命周期串行，或为子例程提供共享既有 pipe 的非拥有资源视图。直接删除一个 WAIT 或盲目销毁仍有在途搬运的外层对象均不能保证正确性。

在 26.10 实际 trace 上，{ncores} 个 AIC 的三个事件全部能找到**第一次 WAIT 发起之前已有两次 SET**的证据，共 {count} 个“核 × 事件”对；芯片第一工作核的对应记录为{examples}。完整窗口另外还有其他时段的错误，回调把它们归到 {len(offs)} 个 ELF 偏移，反汇编全部映射到真实的 M→MTE1 SET。跨管线的等待开始和退休时刻不同，后续完整窗口不能通过一个简单布尔回放复造所有回调计数；此处只用首 WAIT 之前的强证据定位。粗粒度入口和 ELF 缺少 C++ 行表，无法把每个内联站点唯一归到源码行。[逐核动态事件、PC、源 trace 指纹](m-mte1-retirement.json)

我们使用实际 C310 Cube 编译器把真实 wheel 头文件的 `Resource<Ascend950>` 放到一段隔离诊断源码中。单一 TPipe 或重用它时，其反汇编在首个可能 WAIT 之前有 3 条 SET；当外层 TPipe 仍活跃时创建第二个普通 TPipe 或实际 wheel 的 Resource，则为 6 条。隔离实验在创建第二个对象前显式结束第一个的生命周期后，前缀恢复为 3 条。这里比较机器码顺序，条件 WAIT 的实际执行由上面的原生 CA 窗口验证。此编译对照没有改供应商代码，也没有新增 CA 或生成生产延迟。[编译器、头文件哈希和六组反汇编](compiler-controls.json) · [诊断源码](../../../development/validation/kda_v1_25/pipe_control.cpp)

建议供应商从 `chunk_kda_fwd.cpp` 的外层 pipe 和 `arch35/chunk_kda_fwd_prepare.h` 中使用 Resource 的分支检查所有权，再验证通用后端、大于 AIC 数的状态头和 dense 专用分支。验收同样要求每核无重复旗、全部原生数值和完整窗。独立构建的修改只能登记为另一种实现，现有基准不变。
"""
    if not {'real_wheel_catlass_resource','release_before_wheel_resource'}<=set(r['source_case'] for r in control['rows']):raise ValueError('compiler controls incomplete')
    p=OUT/'README.md';tmp=p.with_suffix('.tmp');tmp.write_text(text);tmp.replace(p)

if __name__=='__main__':main()
