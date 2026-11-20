"""Single-slot full-width Pro and Kimi supplement after the RMS acquisition closes."""
import fcntl
import gzip
import json
import os
from pathlib import Path
import shutil
import sys
import time

ROOT=Path(__file__).resolve().parents[3]
sys.path.insert(0,str(ROOT))
from development.validation.matmul28 import campaign as ca
from development.validation.matmul28.drain import recover
from development.validation.kda27.quant_transfer import headroom

OUT=ca.OUT/'wide-prefill'; RAW=ca.RAW/'wide-prefill'
ROUTES={ca.PRO:'indexer.weights_proj',ca.KIMI:'mla.kv_b_proj'}

def freeze():
    rows=[]
    with gzip.open(ca.CATALOG,'rt') as f:
        for line in f:
            doc=json.loads(line); s=doc['scenario']
            if s['model'] not in ROUTES or (s['batch'],s['seq'],s['tp'],s['dp'],s['chunk'],s['step'])!=(8,32768,8,1,0,0):continue
            node=ROUTES[s['model']]
            layers=[r for r in doc['contracts'] if r['op']==node]
            (n,k),={tuple(r['tensors']['weight_logical']) for r in layers}
            m=s['batch']*s['seq']; shape=[m,n,k]
            assert all(r['tensors']['x']==[m,k] and r['tensors']['y']==[m,n] for r in layers)
            forecasts={}
            for chip in [ca.DT,ca.PR,ca.FUT]:
                est=ca.estimate('matmul',shape,chip=chip,dtype='bf16')
                forecasts[chip]={'us':est.latency_us,'active_aic':est.active_aic,'active_aiv':est.active_aiv}
            rows.append(dict(model=s['model'],node=node,shape=shape,predictions=forecasts,wall_s=86400,raw_mib=98304))
    assert len(rows)==len(ROUTES)
    OUT.mkdir(exist_ok=True)
    sources=[Path(__file__),ca.CATALOG,ca.BIN,Path(__file__).with_name('campaign.py'),Path(__file__).with_name('drain.py')]
    with (OUT/'freeze.json').open('x') as f:
        json.dump(dict(utc=ca.now(),rows=rows,source_sha256={str(p.relative_to(ROOT)):ca.digest(p) for p in sources},
                       protocol='BF16 ND logical projection, installed ACLNN single Matmul. Same all-core, golden and optional asynchronous-drain check as primary suite. One physical acquisition slot released by RMS, one attempt per SKU. 970 extrapolated.'),f,indent=2);f.write('\n')

def report(reg):
    lines=['# 单卡 B8×32Ki 的 Pro 和 Kimi 全批 Matmul','',
           '两个投影都在一个 rank 上一次提交全部 262144 行，N、K 取 TP8 后真实权重维度。计时合同为 BF16 ND 的独立库调用；量化部署须另核对。每个目标双端各一次，970DT 只有外推，单位为 µs。','',
           '| 模型 · 投影 | [M,N,K] | 输入 / 输出 GiB | DT 预测 / CA | PR 预测 / CA | 970DT 外推 |','|---|---|---:|---:|---:|---:|']
    for r in reg['rows']:
        vals=[]; m,n,k=r['shape']
        for chip,tag in [(ca.DT,'dt'),(ca.PR,'pr')]:
            path=OUT/(r['model']+'-'+tag+'.json');d=ca.read(path) if path.exists() else {}
            meas=f'{d["ca_us"]:.3f}' if d.get('qualified') else ('未合格' if d else '待回收')
            pair=f'{r["predictions"][chip]["us"]:.3f} / {meas}'
            vals.append(f'[{pair}]({path.name})' if d else pair)
        lines.append(f'| {r["model"]} · {r["node"]} | {r["shape"]} | {2*m*k/2**30:.2f} / {2*m*n/2**30:.2f} | {vals[0]} | {vals[1]} | {r["predictions"][ca.FUT]["us"]:.3f} |')
    lines+=['','复用 RMSNorm 已释放的单槽。每次上限为 24 小时、96 GiB 原始文件，数值、活动核名册和退出均合格后才填 CA。[冻结参数](freeze.json) · [实时状态](state.json) · [Flash 长上下文](../long-prefill/report.md)','']
    tmp=OUT/'report.tmp';tmp.write_text('\n'.join(lines));tmp.replace(OUT/'report.md')

def run():
    reg=ca.read(OUT/'freeze.json')
    for p,h in reg['source_sha256'].items():
        if ca.digest(ROOT/p)!=h:raise ValueError(p)
    def state(phase,**kwargs):ca.save(OUT/'state.json',dict(utc=ca.now(),pid=os.getpid(),phase=phase,**kwargs))
    RAW.mkdir(exist_ok=True); report(reg)
    env=os.sched_getaffinity(0)
    with (ROOT/'runs/rms28/serial.lock').open('a') as slot:
        state('waiting for RMS slot');fcntl.flock(slot,fcntl.LOCK_EX)
        os.nice(8);os.environ.update(OMP_WAIT_POLICY='PASSIVE',GOMP_SPINCOUNT='0')
        for chip,tag in [(ca.DT,'dt'),(ca.PR,'pr')]:
            for r in reg['rows']:
                name=r['model']+'-'+tag; receipt=OUT/(name+'.json'); folder=RAW/name
                if receipt.exists():continue
                if not (folder/'cost.json').exists():
                    if folder.exists():raise FileExistsError(folder)
                    os.sched_setaffinity(0,env)
                    while True:
                        info=headroom()
                        if info['eligible'] and shutil.disk_usage(ROOT).free>2*r['raw_mib']*2**20+16*2**30:break
                        state('waiting for spare resources',case=name,host=info);time.sleep(30)
                    os.sched_setaffinity(0,info['chosen_cpus'][:64]);state('acquiring',case=name,host=info)
                    folder.mkdir();ca.measure(r,chip,folder)
                try:
                    d=ca.qualify(r,chip,folder)
                except (ValueError,KeyError,AssertionError,OSError,IndexError) as error:
                    d=dict(qualified=False,reason=repr(error),ca_us=None)
                raw=OUT/(name+'-initial.json')
                ca.save(raw,dict(**d,case=name,chip=chip))
                if 'end marker disagrees' in d.get('reason',''):
                    try:d=recover(r,ca.read(raw),folder,acquisition=raw)
                    except (ValueError,KeyError,AssertionError) as exc:d=dict(qualified=False,reason=repr(exc),ca_us=None)
                ca.save(receipt,dict(**d,shape=r['shape'],chip=chip,cost=ca.read(folder/'cost.json'),initial=raw.name,freeze_sha256=ca.digest(OUT/'freeze.json')))
                report(reg)
        state('finished')

if __name__=='__main__':{'freeze':freeze,'run':run}[sys.argv[1]]()
