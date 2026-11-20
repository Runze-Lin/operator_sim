"""Reuse the decode acquisition slot for full-rank B8 128Ki/256Ki Flash projections."""
from datetime import datetime, timezone
from hashlib import sha256
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

OUT=ca.OUT/'long-prefill'
RAW=ca.RAW/'long-prefill'
CATALOG=ca.CATALOG

def register():
    rows=[]
    with gzip.open(CATALOG,'rt') as f:
        for line in f:
            row=json.loads(line);s=row['scenario']
            if (s['model'],s['batch'],s['tp'],s['dp'],s['chunk'],s['step'])!=(ca.FLASH,8,8,1,0,0) or s['seq'] not in (131072,262144):
                continue
            layers=[x for x in row['contracts'] if x['op']=='indexer.wk']
            (n,k),={tuple(x['tensors']['weight_logical']) for x in layers}
            assert all(x['tensors']['x']==[8*s['seq'],k] for x in layers)
            shape=[s['batch']*s['seq'],n,k]
            rows.append(dict(model=s['model'],node='indexer.wk',seq=s['seq'],requests=8,shape=shape,
                             wall_s=43200 if s['seq']==131072 else 86400,raw_mib=32768 if s['seq']==131072 else 65536,
                             predictions={c:dict(us=ca.estimate('matmul',shape,dtype='bf16',chip=c).latency_us) for c in (ca.DT,ca.PR,ca.FUT)}))
    assert len(rows)==2 and len({tuple(r['shape']) for r in rows})==2
    files=[Path(__file__),ROOT/'development/validation/matmul28/campaign.py',ROOT/'development/validation/matmul28/drain.py',ca.BIN,CATALOG,ca.OUT/'freeze.json']
    reg=dict(utc=ca.now(),rows=sorted(rows,key=lambda r:r['seq']),source_sha256={str(p.relative_to(ROOT)):ca.digest(p) for p in files},scope='All launches; full numerical golden and physical END plus engine drain on every core. The decode slot is reused after its last job. 970DT has no measurement.')
    OUT.mkdir(exist_ok=True)
    with (OUT/'freeze.json').open('x') as out: json.dump(reg,out,indent=2);out.write('\n')

def report(reg):
    lines=['# 单卡 B8 整批 128Ki、256Ki 的 Matmul','',
           'DeepSeek V4.1 Flash 的 indexer.wk 在每个请求上投影 512 维输入到 128 维。此实验使用固定模型配置、TP8、BF16 连续 ND 单次 aclnnMatmul。每个新几何按全部 B×S 行分配张量并核验每个输出。设备时间为 µs。','',
           '| 每请求长度 | 每卡 [M,N,K] | 950DT 预测 / CA | 950PR 预测 / CA | 970DT 外推 |','|---:|---|---:|---:|---:|']
    for j in reg['rows']:
        row=[]
        for chip,tag in [(ca.DT,'dt'),(ca.PR,'pr')]:
            score=OUT/(f'{j["seq"]}-{tag}.json')
            d=ca.read(score) if score.exists() else {}
            v=f'{d["ca_us"]:.3f}' if d.get('qualified') else ('未合格' if d else '待回收')
            val=f'{j["predictions"][chip]["us"]:.3f} / {v}'
            row.append(f'[{val}]({score.name})' if d else val)
        lines.append(f'| {j["seq"]//1024} Ki | {j["shape"]} | {row[0]} | {row[1]} | {j["predictions"][ca.FUT]["us"]:.3f} |')
    lines+=['','复用 decode 结束后释放的槽。每端 128Ki、256Ki 的墙钟预算分别为 12 和 24 小时，原始数据预算为 32 和 64 GiB；超出预算保留记录和已通过的检查。[预测和采集预算](freeze.json) · [队列状态](state.json) · [32Ki 和 decode 逐形表](../report.md)','']
    tmp=OUT/'report.tmp';tmp.write_text('\n'.join(lines));tmp.replace(OUT/'report.md')

def run():
    reg=ca.read(OUT/'freeze.json')
    for p,h in reg['source_sha256'].items():
        if ca.digest(ROOT/p)!=h: raise ValueError(p)
    RAW.mkdir(exist_ok=True)
    os.nice(9)
    os.environ.update(OMP_WAIT_POLICY='PASSIVE',GOMP_SPINCOUNT='0')
    allowed=sorted(os.sched_getaffinity(0));os.sched_setaffinity(0,allowed[:max(1,min(96,len(allowed)//2))])
    with (RAW/'lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        ca.save(OUT/'state.json',dict(utc=ca.now(),phase='waiting for decode slot',pid=os.getpid()))
        report(reg)
        while ca.read(ca.OUT/'decode-status.json')['phase']!='finished': time.sleep(30)
        with (ca.RAW/'decode.lock').open('a') as slot:
            fcntl.flock(slot,fcntl.LOCK_EX)
            for j in reg['rows']:
                for chip,tag in ((ca.DT,'dt'),(ca.PR,'pr')):
                    name=f'{j["seq"]}-{tag}'; receipt=OUT/(name+'.json');folder=RAW/name
                    if receipt.exists():continue
                    if not (folder/'cost.json').exists():
                        if folder.exists():raise ValueError('unfinished attempt: '+name)
                        while shutil.disk_usage(ROOT).free<2*j['raw_mib']*2**20+16*2**30:
                            ca.save(OUT/'state.json',dict(utc=ca.now(),phase='waiting for disk',case=name));time.sleep(30)
                        ca.save(OUT/'state.json',dict(utc=ca.now(),phase='acquiring',case=name,pid=os.getpid()))
                        folder.mkdir();ca.measure(j,chip,folder)
                    try:
                        outcome=ca.qualify(j,chip,folder)
                    except (ValueError,KeyError,AssertionError,OSError,IndexError) as exc:
                        outcome=dict(qualified=False,ca_us=None,reason=repr(exc))
                    value=dict(**outcome,chip=chip,case=name,cost=ca.read(folder/'cost.json'),shape=j['shape'],freeze_sha256=ca.digest(OUT/'freeze.json'))
                    ca.save(receipt,value)
                    if 'end marker disagrees' in value.get('reason',''):
                        try:
                            aux=recover(j,value,folder,acquisition=receipt)
                        except (ValueError,KeyError,AssertionError) as exc:
                            aux=dict(qualified=False,ca_us=None,reason=repr(exc))
                        aux['initial_check']=value
                        ca.save(receipt,dict(**aux,shape=j['shape'],chip=chip,cost=value['cost'],freeze_sha256=value['freeze_sha256']))
                    report(reg)
        ca.save(OUT/'state.json',dict(utc=ca.now(),phase='finished'))

if __name__=='__main__':{'freeze':register,'run':run}[sys.argv[1]]()
