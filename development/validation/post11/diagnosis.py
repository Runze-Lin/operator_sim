"""Render measured errors, state dependence and failed trace attribution after publication gates."""
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
from operator_model.post_v11 import predict
OUT = ROOT / 'results/v0.11/post'
read = lambda name: json.loads((OUT/name).read_text())
summary = read('summary.json')['mhc_post']['validation']
cost = read('costs.json')['mhc_post']
rows = [r for r in read('rows.json') if r['role']=='validation']
history = read('development-output-audit.json')
warmups=[]
for path in sorted(OUT.glob('retry-v11_mhc_post_*.json')):
    warm, = [r for r in read(path.name)['completed'] if r['role']=='context_warmup']
    assert warm['prior_aiv']==0 and warm['active_aiv']==64
    cold=predict('mhc_post',warm['shape'],{**warm['attrs'],'resident_aiv':0})
    warmups.append(dict(shape=warm['shape'],prior_aiv=0,active_aiv=warm['active_aiv'],
                        source=warm['source'],db_sha256=warm['db_sha256'],reference_us=warm['reference_us'],
                        explicit_first_use_us=cold['predicted_us'],role='context_warmup'))
(OUT/'warmup-controls.json').write_text(json.dumps(warmups,indent=2)+'\n')
warm_times='、'.join(f"{r['reference_us']:.3f}" for r in warmups)
lines = ['# mHC Post 独立验证与误差来源', '',
         '| 独立组 | 个数 | v0.10 默认 WAPE / 最大 | v0.11 默认 WAPE / 最大 | 已知核覆盖 WAPE / 最大 | 固定中位数 WAPE / 最大 |',
         '|---|---:|---:|---:|---:|---:|']
def metrics(s):
    return f"{s['wape']:.2%} / {s['max_ape']:.2%}"
for group,name in [('all','全部'),('few_wide','少 token 宽通道'),('controls','普通'),('fixed_width_ladder','其中 D=1024 逐 token')]:
    g=summary[group]
    lines.append('| '+name+' | '+str(g['predicted_us']['n'])+' | '+' | '.join(metrics(g[k]) for k in ('previous_us','predicted_us','sequence_us','fixed_us'))+' |')
lines += ['', '比较使用同十个新 shape 和同一组已通过 CPU golden、helper 时序、AIV 及输出字节归属检查的 CA 参考。前 33 个 shape 只作拟合与开发回放。小宽组在采集前定义为 T≤4 且 D≥2048。固定基线使用 33 个开发时延的中位数。两版默认均不传 resident_aiv。已知核覆盖仅根据采前顺序预测。', '',
          '| T,N,D | CA µs | v0.10 | v0.11 | 已知 H | H | 实际入口 / AIV | 关键核 SCALAR / MTE2 / MTE3 / SIMD 记录 | 主输出字节 |',
          '|---|---:|---:|---:|---:|---:|---|---|---:|']
for r in rows:
    counts=' / '.join(str(r['counts'].get(q,0)) for q in ('SCALAR','MTE2','MTE3','SIMD'))
    lines.append(f"| {r['shape']} | {r['reference_us']:.3f} | {r['previous_us']:.3f} | {r['predicted_us']:.3f} | {r['sequence_us']:.3f} | {r['prior_aiv']} | {r['kernel']} / {r['active_aiv']} | {counts} | {r['output_audit']['union_bytes']} |")
lines+=['', '### 少 token 宽通道为何仍难估', '',
        '在首次 [3,4,4352]，CA 5.288 µs，默认按切通道组的复用状态给出 2.325 µs；明确 H=0 后为 4.749 µs。在扩核 [4,4,6144]，CA 4.095 µs，默认 2.347 µs，H=48 时为 4.270 µs。另三个小宽复用点 CA 2.298–2.573 µs，默认 2.280–2.302 µs。旧版的混合均值能分摊首次与热态的误差；新方法更偏向开发批内出现次数较多的热态。本组新样本含一次首次与一次扩核，热态收益不足以抵消这两次低估。', '',
        f"三个隔离补测的预热还有一个可重复对照。相同 [4,4,6144] 作为新进程首算，CA 分别为 {warm_times} µs，与原批第四次扩核的 4.095 µs 使用相同入口和 64 个 AIV。其区别直接说明同形耗时受调用历史影响；具体的缓存、地址映射、发射竞争各占多少尚未辨识。原始辅助数据含按状态估计的六段时间，供系统层跟踪 H 时复核。[独立列示的实际预热 H=0 与 CA](warmup-controls.json)", '',
        '### 跨发射记录污染及重采', '',
        '首次单流发射 10 次，均通过全输出 CPU golden。T128 与 T192 的 helper 在结束阶段检测到先后时刻倒置的 RV 记录，分别 invalid=1144/1163、写入 0 行。T256 尽管 SQLite 完整且 helper 返回 ok，却出现 408 笔 8192 B 写入、311 个不同目的行址。输出只需 256 行，整库写 3,342,336 B、去重覆盖 2,547,712 B、地址跨度 2,744,320 B，应为 2,097,152 B。其地址序列含前一个 T192 的三行分核步长以及本次四行步长，原始 5.695 µs 不具备单发射归属。三者均排除在计时误差外，合格的前七条沿用原批。', '',
        '三例分别重新在独立进程内先以 [4,4,6144] 覆盖 64 核，再发射对应未获合格计时的目标。沿用原冻结预测；换进程后同为 H=64，指令缓存、分配器与连续原流状态仍可能不同。对每个候选 DB 展开每个 OUT 写的 burst_number、burst_len、dst_stride 和目的地址；合格条件是总写字节、去重区间并集、首末跨度全部等于唯一逻辑 BF16 输出 2TND 字节。完整时序还必须满足 DB 起讫与 helper 频率、任务归属、CPU 数值门禁。所有 10 个独立点实测入口及参与 AIV 与分核预测吻合。', '',
        f"对用于拟合的 33 个历史 DB 事后复核同一输出字节门禁，通过 {sum(r['qualified'] for r in history)}/{len(history)}。[原批资格及失败详情](original-attempt.json) · [本轮输出审计](output-audit.json) · [开发集输出审计](development-output-audit.json)", '',
        '### 离线采集与在线成本', '',
        '| 进程 | 发射数（含预热） | CA 墙钟 s | DB MiB |','|---|---:|---:|---:|']
for b in cost['batches']:
    lines.append(f"| {b['name']} | {b['launches']} | {b['wall_s']:.1f} | {b['db_bytes']/2**20:.1f} |")
lines += ['', f"全部 {cost['launches']} 次离线发射（10 个独立 shape，含 3 个失败/不合格计时和 3 次重复的预热），进程墙钟相加 {cost['batch_wall_s']:.1f} s；两次重采并行。DB 共 {cost['db_bytes']/2**20:.1f} MiB。每个新 shape 在已加载 profile 的进程内重算 100 次，中位形状的查询中位时延 {cost['query_median_us']:.2f} µs、1000 次中最大 {cost['query_max_us']:.2f} µs。设备 µs 与 Python 查询墙钟分别测量。", '', '[完整逐 shape 结果](rows.json) · [预测、参数与预算冻结](freeze.json) · [两次空数据库重采合同](retry-freeze.json) · [混入记录重采合同](isolate256-freeze.json)']
(OUT/'diagnosis.md').write_text('\n'.join(lines)+'\n')
