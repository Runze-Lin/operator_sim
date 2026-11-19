"""Durable single-slot CA and source-qualified report, with same-attempt native fallback."""
import fcntl
import math
import os
import re
import shutil
import threading
import time
from common import *
from evidence import capture,finish,plot_ready

KDA=ROOT/'results/kda-v23/precision-supplement-state.json'
KIMI=ROOT/'results/v0.20/full-quant/native.json'
KIMI_LOG=ROOT/'runs/kimi-full-quant20/run.log'
SUMMARY=re.compile(r'core: (\d+), subcore: (\d+), start: (\d+), tick: (\d+), kernelId: (\d+), blkID: (\d+).*?done')


def free_slot(j):
    ok_since=None
    while True:
        kda_state=read(KDA)['state']
        remain=shutil.disk_usage(OUT).free
        slots=len(os.sched_getaffinity(0));load=os.getloadavg()[0]
        with open('/proc/meminfo') as f:ram=int(next(l for l in f if l.startswith('MemAvailable:')).split()[1])*1024
        other=not KIMI.exists() and KIMI_LOG.exists()
        can_start=kda_state!='running' and remain>j['raw_mib']*2**20+5*2**30 and ram>2*j['raw_mib']*2**20 and (not other or load<.5*slots)
        if can_start:
            if ok_since is None:ok_since=time.monotonic()
            if not other or time.monotonic()-ok_since>=120:break
        else:ok_since=None
        state('queued',case=j['id'],kda=kda_state,kimi_native_still_open=other,host_cpu_count=slots,load_1m=load,
              available_ram_GiB=round(ram/2**30,1),free_disk_GiB=round(remain/2**30,1),headroom_seconds=0 if ok_since is None else round(time.monotonic()-ok_since))
        time.sleep(15)


def mirror_native(folder,stop):
    """Capture authentic native summary lines if this vendor launch exposes them."""
    seen={};target=folder/'native-snapshot.json'
    while True:
        before=len(seen)
        for p in (folder/'log_ca').glob('core*_summary_log'):
            try:
                for c,s,a,b,k,blk in SUMMARY.findall(p.read_text(errors='replace')):
                    key=c+'.'+s
                    if key not in seen and int(b)>int(a):seen[key]=dict(first=int(a),last=int(b),kernel=int(k),block=int(blk))
            except FileNotFoundError:pass
        if len(seen)!=before:save(target,seen)
        if stop.wait(5):break


def fallback(j,d):
    p=d/'native-snapshot.json';text=(d/'run.log').read_text(errors='replace');t,h=j['shape'];sku=j['sku']
    required=[f'runtime_soc={sku}',('hbm3_8000M.ini' if sku==DT else 'lpddr5_6400M_bailu.ini'),'SMOKE_PASS',f'correctness count={t*h} bad=0',f'api=aclnnCast R={t} D={h}']
    if not p.exists() or not all(x in text for x in required) or 'SMOKE_FAIL' in text:return None
    rows=read(p);started=re.findall(r'\[block_start\]\s+: AIV, task_id=0, core_id=(\d+),',text)
    ended=re.findall(r'\[block_end\]\s+: AIV, task_id=0, core_id=(\d+),',text)
    ids={str(2*int(c)+int(s)-1) for c,s in (k.split('.') for k in rows)}
    if len(rows)!=len(ids) or not ids or ids!=set(started) or ids!=set(ended) or len(ended)!=len(ids) or len(set(v['kernel'] for v in rows.values()))!=1:return None
    if re.findall(r'Start profiling on kernel: (\S+)',text)!=[j['kernel']]:return None
    actual=list(d.glob('profile/OPPROF_*/dump/aicore_binary.o'))
    if len(actual)!=1 or sha(actual[0])!=j['kernel_binary_sha256']:return None
    return dict(complete=True,scope='native per-core wrapper start-to-done envelope only, no instruction timing or tile trace',
                us=(max(v['last'] for v in rows.values())-min(v['first'] for v in rows.values()))/1650,all_cores=rows,numeric_elements=t*h,
                snapshot_sha256=sha(p),kernel=j['kernel'],elf_sha256=j['kernel_binary_sha256'])


def refresh(reg):
    headers=['# DeepSeek V4 Pro 大 prefill 的 indexer Cast','',
      '每个 rank 一次处理 8 条未切 chunk 的序列，每行是该模型的 64 个 indexer 系数。全元素检查将 BF16 原始位模式扩展成 FP32 并逐项比较。[采集前冻结与归档范围](freeze.json) · [后台状态](state.json)','',
      '| 实际行数 × 64 | 芯片 | 冻结预测 µs | 完整 CA µs | 相对误差 | 970DT 外推 µs | 设备端证据 |','|---|---|---:|---:|---:|---:|---|']
    for j in reg['jobs']:
        p=OUT/(j['id']+'.json');r=read(p) if p.exists() else {}
        b=OUT/(j['id']+'-blocks.json')
        evidence=('['+str(len(read(b)['ca_tiles']))+' 个块]('+b.name+') · [预测与 CA 同轴]('+j['id']+'-plot.png)') if b.exists() else ('[回执]('+p.name+')' if p.exists() else '排队或采集中')
        headers.append('| '+ ' | '.join([str(j['shape'][0]),j['sku'][6:11],f"{j['forecasts'][j['sku']]['us']:.3f}",
              'N/A' if not r.get('qualified') else f"{r['ca_us']:.3f}",'N/A' if not r.get('qualified') else f"{r['signed_pct']:+.2f}%",
              f"{j['forecasts'][FUTURE]['us']:.3f}",evidence])+' |')
    headers+=['','128k 的展平元素数等于 Flash [2097152,32] 的 67,108,864 个。它可以检验同形状类别下的行宽等价性，不能算作未见数据量。256k 含 134,217,728 个元素。970DT 的列按芯片参数外推。原生 wrapper 窗如能保留，另列于回执的 native 字段；只有完整指令窗才计入上表。']
    tmp=OUT/'README.tmp';tmp.write_text('\n'.join(headers)+'\n');tmp.replace(OUT/'README.md')


def run():
    reg=read(REG);RAW.mkdir(exist_ok=True)
    with (RAW/'queue.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        # Device bytes and scoring helpers are frozen. Forecasts stored in the manifest are authoritative.
        for path,expected in reg['hashes'].items():
            if sha(ROOT/path)!=expected:raise ValueError('frozen source differs: '+path)
        os.environ.update(OMP_WAIT_POLICY='PASSIVE',GOMP_SPINCOUNT='0')
        for j in reg['jobs']:
            out=OUT/(j['id']+'.json');d=RAW/j['id']
            if out.exists():continue
            d.mkdir(exist_ok=True)
            if not (d/'cost.json').exists():
                if (d/'run.log').exists():
                    state('interrupted prior unscored attempt; preserved',case=j['id']);return
                free_slot(j)
                state('acquiring',case=j['id'],start=when(),budget_seconds=j['wall_s'],budget_MiB=j['raw_mib'])
                stop=threading.Event();watcher=threading.Thread(target=mirror_native,args=(d,stop),daemon=True);watcher.start()
                try:capture(j,j['sku'],d)
                finally:stop.set();watcher.join()
            state('scoring existing full trace',case=j['id'])
            try:
                result=finish(j,d)
            except (ValueError,OSError,KeyError,IndexError,AssertionError,TypeError) as exc:
                result=dict(case=j['id'],shape=j['shape'],chip=j['sku'],qualified=False,ca_us=None,cost=read(d/'cost.json'),reason=repr(exc),native=fallback(j,d))
            save(out,result)
            if result['qualified']:
                try:plot_ready(j,result)
                except (ValueError,OSError,KeyError,IndexError,AssertionError,TypeError) as exc:
                    result['figure_extraction_error']=repr(exc);save(out,result)
            refresh(reg)
        state('finished',cases=[j['id'] for j in reg['jobs']]);refresh(reg)

if __name__=='__main__':run()
