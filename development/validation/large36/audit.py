"""Offline proof for the 8192-key indexer; never launches the simulator."""
from collections import Counter
from datetime import datetime,timezone
from hashlib import sha256
import importlib.util
import json
import os
from pathlib import Path
import re
import sqlite3
import sys

ROOT=Path(__file__).resolve().parents[3]
OUT=ROOT/'results/v0.36/large'
BASE=ROOT/'results/v0.34/indexer'
DT,PR,FUT='Ascend950DT_9582','Ascend950PR_957b','Ascend970DT'
SHAPE=[1,1,8192,8,128]
ATTRS=dict(dtype='bf16',layout='BSND',sparse_count=512,sparse_mode=3,return_values=False)
LOG_KEY=b'case=lightning_indexer api=aclnnLightningIndexer B=1 Q=1 L=8192 G=8 D=128 dtype=BF16 topk=512 sparse_mode=3 return_values=false'


def read(p):
    return json.loads(p.read_bytes())


def rel(p):
    return str(p.relative_to(ROOT))


def digest(p):
    with p.open('rb') as f:
        h=sha256()
        for block in iter(lambda:f.read(2**20),b''):
            h.update(block)
    return h.hexdigest()


def prove():
    physical,= (r for r in read(BASE/'physical.json') if r['shape']==SHAPE and r['chip']==DT)
    identity=read(BASE/'identity.json')
    ref=identity['official_receipts'][0]
    db=ROOT/physical['source']
    root=ROOT/'runs/v8cv/lightning_indexer-10'
    assert digest(db)==physical['sha']
    verdict=read(root/'result.json')
    log=(root/'run.log').read_bytes()
    syslog,=root.glob('profile/*/npusim.log')
    assert LOG_KEY in log and b'runtime_soc=Ascend950DT_9582' in log
    assert b'core' in syslog.read_bytes() and b'hbm3_8000M.ini' in syslog.read_bytes()
    assert log.count(b'correctness name=topk_set count=512 bad=0 exact_integer_set=true all_indices_checked=true')==1
    assert log.count(b'SMOKE_PASS')==1 and not verdict['error'] and verdict['case']['attrs']==ATTRS
    assert len(list(root.rglob('*.db')))==1
    with sqlite3.connect(f'file:{db}?mode=ro&immutable=1',uri=True) as c:
        assert c.execute('pragma integrity_check').fetchall()==[('ok',)]
        info=dict(c.execute('select * from KernelInfo'))
        cores=c.execute("""select CoreTypeId,ExecInstrCoreId,count(*),min(ExecInstrTickStart),
                      max(ExecInstrTickEnd),sum(ExecInstrName='END'),max(case when ExecInstrName='END' then ExecInstrTickEnd end),
                      sum(ExecInstrName='MMAD'),sum(ExecInstrName='RV_CHISTv2' and ExecInstrTickEnd>ExecInstrTickStart)
                      from ExecutedInstructions group by 1,2 order by 1,2""").fetchall()
        assert c.execute('select count(*) from ExecutedInstructions where ExecInstrTickEnd<ExecInstrTickStart').fetchone()[0]==0
    assert {(a,b) for a,b,*_ in cores}=={(a,b) for a in [1,2,3] for b in range(32)}
    assert all(n>0 and ends==1 and lo<=end<=hi for _,_,n,lo,hi,ends,end,mm,ch in cores)
    lo=min(row[3] for row in cores); hi=max(row[4] for row in cores)
    assert hi-lo==info['exec_time_ticks']==physical['duration_ticks'] and len(cores)==info['ai_core_active']
    assert sum(x[2] for x in cores)==info['instrs_exec_num'] and (info['frequency_ghz'],info['block_dim'],info['launch_id'])==(1.65,32,0)
    ca=(hi-lo)/(1000*info['frequency_ghz'])
    assert abs(ca*1000-info['exec_time_ns'])<1e-7 and abs(ca-verdict['measurement']['reference_us'])<1e-8
    assert [sum(x[i] for x in cores) for i in (7,8)]==[64,256]
    engine=ROOT/'development/validation/indexer34/identity.py'
    spec=importlib.util.spec_from_file_location('image_identity',engine)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    image=ROOT/ref['elf']
    assert digest(image)==identity['official_elf_sha256']
    match=module.inspect(db,physical,module.read_image(image),identity['official_receipts'])
    assert match['exact_observed_mixed_program'] and match['common_text_origin_candidates_hex']==['0xff30']
    return dict(qualified=True,protocol='all-core native executed instruction window',role='previously used DT stage-development case, independently rechecked evidence only',
        original_elapsed_s=verdict['wall_seconds'],ca_us=ca,physical_sku=DT,physical_memory='hbm3_8000M.ini',
        first_tick=lo,last_tick=hi,ticks=hi-lo,frequency_ghz=info['frequency_ghz'],all_records=sum(x[2] for x in cores),
        per_engine_cores=dict(Counter(r[0] for r in cores)),golden_sorted_topk_indices=512,golden_scope='exact membership; order within top512 not checked',
        per_core_fields=['engine','physical_core','records','first_issue_tick','last_retire_tick','physical_ends','physical_end_tick','mmad','positive_duration_chist'],
        cores=cores,expected_input_layout=dict(q=[1,1,8,128],k=[1,8192,1,128],weight=[1,1,8],out=[1,1,1,512]),logical_key_bytes=8192*128*2,
        ca_output_indices_bytes=512*4,mmad=64,positive_duration_histograms=256,
        warning_summary_count=log.count(b'cfg find path STAT.su_perf_summary_log_enable is unqualified'),
        match=match,identity_scope='all executed positions across 3 engines share the same origin; the native DB does not include the whole object',
        raw_fingerprints={rel(p):digest(p) for p in [db,root/'run.log',syslog,root/'result.json',image,engine,BASE/'identity.json']})


def duplicates():
    targets=[];count=0
    # Prune bulk data folders; dispatch contracts appear in run.log, not instruction dumps.
    for base in [ROOT/'runs',ROOT/'results']:
        for folder,dirs,files in os.walk(base):
            dirs[:]=[s for s in dirs if s not in ('profile','simulator','log','log_ca','dump','build','record','sources','ca_workspace','__pycache__') and not s.startswith(('OPPROF_','npusim_'))]
            if 'run.log' not in files:continue
            p=Path(folder)/'run.log';count+=1
            with p.open('rb') as f:
                if any(LOG_KEY in line for line in f):targets.append(rel(p))
    regs={}
    for path,field in [('v0.29/large/fusion/freeze','jobs'),('v0.34/indexer/freeze','job')]:
        p=ROOT/'results'/(path+'.json');data=read(p)[field]
        regs[rel(p)]=[j['shape'] for j in (data if isinstance(data,list) else [data]) if j['op']=='lightning_indexer']
    return dict(scanned_logs=count,matching_geometry_logs=sorted(targets),registrations=regs,
        pr_result_found=any('Ascend950PR_957b' in (ROOT/p).read_text(errors='replace') for p in targets))


def progress():
    paths={
        'Pro MoE 4096 rows, GMM1 / GMM2':'runs/large29/cube/moe-dt',
        'Flash built-in MhcPost 1024 rows':'runs/specialized31/post/mhc_post-Ascend950DT_9582',
        'Flash Addmm 65536 rows':'runs/addmm34/dt',
        'Kimi chunk FLA V2 [1024,12]':'runs/large29/cube/kda-v2-dt',
        'Flash split mHC [512,4,5120]':'results/v0.29/large/fusion/raw/flash-chunk512-Ascend950DT_9582'}
    report={}
    for name,folder in paths.items():
        src=ROOT/folder
        log=src/'run.log';info=dict(raw=folder,physical_launch=log.exists(),cost_receipt=(src/'cost.json').is_file())
        if log.exists():
            text=log.read_text(errors='replace')
            counts=re.findall(r'Thread \d+ is running\. Cycle Count: (\d+)',text)
            info.update(last_progress_counter=int(counts[-1]) if counts else None,
                actual_kernels=re.findall(r'Start profiling on kernel: (\S+)',text),
                numerical_success='SMOKE_PASS' in text or bool(re.search(r'^correctness .*bad=0',text,re.M)),
                task_done='[TASK_DONE]' in text,per_launch_json=len(list(src.glob('profile/OPPROF_*/*/*/simulator/trace.json'))),
                log_updated_utc=datetime.fromtimestamp(log.stat().st_mtime,timezone.utc).isoformat())
        report[name]=info
    return report


def main():
    assert sys.flags.isolated
    prefix=Path(sys.argv[1]).resolve();sys.path.insert(0,str(prefix))
    import operator_model as api
    assert Path(api.__file__).resolve().is_relative_to(prefix)
    dt=prove();dup=duplicates();pro=progress()
    prices={c:api.estimate('lightning_indexer',SHAPE,ATTRS,chip=c).latency_us for c in (DT,PR,FUT)}
    payload=dict(utc=datetime.now(timezone.utc).isoformat(),public_version=api.__version__,shape=SHAPE,attrs=ATTRS,
        forecasts_us=prices,dt=dt,dt_signed_pct=100*(prices[DT]/dt['ca_us']-1),pr_complete_ca_us=None,
        target_970dt_role='extrapolated',duplicate_check=dup,observed_large_jobs=pro,new_physical_acquisitions=0)
    OUT.mkdir(exist_ok=True,parents=True)
    (OUT/'audit.json').write_text(json.dumps(payload,ensure_ascii=False,indent=2)+'\n')
    lines=['# LightningIndexer 的 8192-key 解码与大形缺口','',
        '在 950DT 既有原始 CA 数据库上重新得到 **22.709 µs**。本例一个 query 面对 8192 个可用 key，单个 BF16 key 张量是 2 MiB；它能检验单 query 长 KV 排程。此样本此前已用于 DT 阶段开发，本次复核不增加独立样本数，也不增加双端已合格路线数。','',
        '| 本地 shape [B,Q,L,G,D] | 950DT 公开 / 实测 µs | 950PR 公开 / 实测 µs | 970DT 外推 µs |',
        '|---|---:|---:|---:|',
        f"| {SHAPE} | {prices[DT]:.3f} / {dt['ca_us']:.3f} ({payload['dt_signed_pct']:+.2f}%) | {prices[PR]:.3f} / N/A | {prices[FUT]:.3f} |",'',
        '安装包对应 aclnnLightningIndexer 的稠密 BSND 接口，BF16、top512、sparse_mode=3。查询、key、权重张量分别为 [1,1,8,128]、[1,8192,1,128]、[1,1,8]。CPU 按实际 BF16 输入重建所有候选分数，核对返回的 512 个索引集合。集合内的名次顺序不在此合同中。','',
        f"SQLite 完整性检查通过。32 个 AIC 与 64 个 AIV 各有一条物理 END，逐核扫描 {dt['all_records']:,} 条记录。全片首发 tick {dt['first_tick']}，全片最后退休 tick {dt['last_tick']}，二者差 {dt['ticks']} tick 除以 1650 tick/µs。设备日志确认 Ascend950DT_9582 和 hbm3_8000M；一个 AIC 发出 64 次 MMAD，一个 AIV 有 256 次正时长直方图指令。完整复算与原数据库时长和原数值回执一致。原采集墙钟 {dt['original_elapsed_s']:.1f} s。", '',
        '对 AIC 与两类 AIV 已执行的每个 PC/机器码，用一个共同基址匹配官方发布的 LightningIndexer 混合内核。匹配从 .text+0xff30 开始，AIV 入口位于 .text+0x58100。原生库未提供完整加载对象路径；核对覆盖实际执行字节。原生日志报告可选 STAT 汇总项缺失，以上时长直接来自完整的逐核指令记录。[原始计时与正确性回执](../../../runs/v8cv/lightning_indexer-10/result.json) · [原始运行日志](../../../runs/v8cv/lightning_indexer-10/run.log) · [官方内核身份对照](../../v0.34/indexer/identity.json)','',
        '8192 个稠密候选位置可在容量上与 32Ki 的 4:1 压缩比较。Pro 的配置含 4:1 层但 top-k 为 1024，Flash 的 top-k 为 512；该库合同不能证明任何一款模型在 32Ki 时实际走相同的稠密 top512 路径。已登记的 Q9×8192 指令样本、Q9×32768 稠密样本均与本例不同。未采集 Q1×8192 的 PR，保留公开预测。[Flash 配置](../../../configs/models/DeepSeek-V4.1-Flash.json) · [Pro 配置](../../../configs/models/DeepSeek-V4-Pro.json) · [Q9 原登记](../../v0.34/indexer/freeze.json)','',
        '## 尚未得到完整回执的既有大形','',
        '| 已登记的 DT 调用 | 日志实际启动 | 已见 kernel 数 | 最新模拟器进度计数 | 整次运行成功回执 |','|---|---|---:|---:|---|']
    for name,x in pro.items():
        lines.append('| '+name+' | '+('是' if x['physical_launch'] else '未见')+' | '+str(len(x.get('actual_kernels',[])))+' | '+str(x.get('last_progress_counter') or 'N/A')+' | '+('已落盘，须另核验' if x['cost_receipt'] else '未见')+' |')
    lines+=['',f"审计时间 {payload['utc']}。进度计数只用于确认采集进展；没有全核退休和数值回读就不记设备耗时。检查 {dup['scanned_logs']} 份原始 run.log，在精确 Q1 合同上命中 {len(dup['matching_geometry_logs'])} 份，均已写入[机读记录](audit.json)。此次没有发起新 CA。", '',
        '已有轻量手段各有可用边界。npusim 的 compact SQLite 在这个 DT 同形保留了每条指令的首发与退休；不生成可视报告也可复核。无指令的原生 wrapper 在已归档的纯 AIV Quant/Cast 上可以重建全核起止，但缺少同一混合设备路径上的指令端点偏差验证，因此未将其用作上述混合算子的全核指令耗时。[轻量计时原始约束](../../v0.29/large/vector/report.md) · [采集成本与设备窗差别](../../v0.33/matmul/report.md) · [复核脚本](../../../development/validation/large36/audit.py)','']
    (OUT/'report.md').write_text('\n'.join(lines))


if __name__=='__main__':
    main()
