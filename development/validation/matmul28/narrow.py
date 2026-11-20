"""One serial, prospective six-shape validation of a model-derived narrow projection."""
from collections import Counter
import fcntl
import gzip
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import time

ROOT=Path(__file__).resolve().parents[3]
sys.path.insert(0,str(ROOT))
from development.validation.matmul28.campaign import BIN,CATALOG,DT,PR,FUT,FLASH,PRO,measure,qualify,now,save,read,digest
from development.validation.matmul28.drain import recover
from development.validation.matmul28.streamk_audit import shards,PROVIDER
from operator_model.chips import load_chip
from operator_model.matmul_resource import resource_matmul
from operator_model.matmul_splitk import estimate_matmul_splitk

OUT=ROOT/'results/v0.28/matmul/narrow'
RAW=OUT/'raw'
POST=ROOT/'results/v0.28/post/state.json'
FREEZE=OUT/'freeze.json'
SOURCE_DIMS={FLASH:(32,5120),PRO:(64,7168)}
ROWS={FLASH:(1,4,9,16),PRO:(4,8)}


def geometry():
    result={}
    with gzip.open(CATALOG,'rt') as f:
        for line in f:
            row=json.loads(line);s=row['scenario']; name=s['model']
            if name not in ROWS or (s['batch'],s['seq'],s['tp'],s['dp'],s['chunk'],s['step'])!=(8,32768,8,1,0,0):continue
            node,=[c for c in row['contracts'] if c['op']=='indexer.weights_proj']
            assert tuple(node['tensors']['weight_logical'])==SOURCE_DIMS[name]
            assert node['placement']=='replicated' and name not in result
            result[name]=dict(scenario=s,node=node)
    assert set(result)==set(ROWS)
    return result


def historical(shapes):
    pat={s:re.compile(rb'\[\s*'+rb'\s*,\s*'.join(str(v).encode() for v in s)+rb'\s*\]') for s in shapes}
    names={s:re.compile(rb'(?:\bcase=matmul\b[^\n]{0,80})?\bM='+str(s[0]).encode()+rb' N='+str(s[1]).encode()+rb' K='+str(s[2]).encode()+rb'\b') for s in shapes}
    manifests=logs=0; seen=[]
    for root,dirs,files in os.walk(ROOT/'results'):
        dirs[:]=[d for d in dirs if d not in ('raw','profile','workloads','narrow','log','log_ca')]
        for f in files:
            if not f.endswith('.json') or f=='trace.json':continue
            p=Path(root)/f; data=p.read_bytes();manifests+=1
            for s,pattern in pat.items():
                if pattern.search(data):seen.append((s,str(p.relative_to(ROOT))))
    for root,dirs,files in os.walk(ROOT/'runs'):
        dirs[:]=[d for d in dirs if d not in ('profile','log','log_ca','dump','simulator','device0') and not d.startswith('OPPROF_')]
        for fn in ('run.log','host.log','batch.txt'):
            if fn not in files:continue
            p=Path(root)/fn;logs+=1
            with p.open('rb') as f:
                tail=b''
                while block:=f.read(2**20):
                    data=tail+block;tail=data[-150:]
                    for s,pattern in names.items():
                        if pattern.search(data):seen.append((s,str(p.relative_to(ROOT))))
    return dict(manifests=manifests,operational_logs_or_batches=logs,matching_records=seen,
                scope='Local measured result manifests and historical acquisition host logs. Geometry catalogue and development source are not prior hardware observations.')


def freeze():
    OUT.mkdir(parents=True,exist_ok=True)
    if FREEZE.exists() or RAW.exists():raise FileExistsError('acquisition already registered')
    shapes=[(m,*SOURCE_DIMS[name]) for name,ms in ROWS.items() for m in ms]
    evidence=historical(shapes)
    if evidence['matching_records']:raise ValueError(evidence['matching_records'])
    geom=geometry();jobs=[]
    for name,ms in ROWS.items():
        for m in ms:
            n,k=SOURCE_DIMS[name];shape=[m,n,k];forecast={}
            for sku in (DT,PR,FUT):
                target=load_chip(sku);start=time.perf_counter_ns()
                old,aic,aiv,d=resource_matmul(shape,target)
                ordinary_ms=(time.perf_counter_ns()-start)/1e6;start=time.perf_counter_ns()
                new=estimate_matmul_splitk(shape,chip=target)
                candidate_ms=(time.perf_counter_ns()-start)/1e6
                if max(ordinary_ms,candidate_ms)>=100:raise ValueError('model query budget exceeded')
                forecast[sku]=dict(us=new['latency_us'],ordinary_us=old,ordinary_aic=aic,ordinary_aiv=aiv,
                                   ordinary_plan=d['target_plan'],candidate=new,
                                   ordinary_query_ms=ordinary_ms,candidate_query_ms=candidate_ms,
                                   target_extrapolated=sku==FUT)
            jobs.append(dict(shape=shape,model=name,node='indexer.weights_proj',predictions=forecast,
                             runtime_example={'batch':1 if m==9 else m,'query_per_request':9 if m==9 else 1},
                             wall_s=900,raw_mib=2048))
    here=Path(__file__).resolve().parent
    files=[here/f for f in ('narrow.py','campaign.py','drain.py','runner.cpp','streamk_audit.py')]
    files += [BIN,CATALOG,PROVIDER.with_suffix('.json'),PROVIDER.with_suffix('.o'),ROOT/'smoke/large_vector.cpp',ROOT/'scripts/cann-run.sh',
              ROOT/'results/v0.16/matmul.json',ROOT/'development/validation/sku17/campaign_collect.py']
    files += [ROOT/'operator_model'/f for f in ('matmul_resource.py','matmul_splitk.py','matmul_waves.py','matmul_v15.py','chips.py')]
    files += [ROOT/'configs/chips'/(sku+'.json') for sku in (DT,PR,FUT)]
    files += [ROOT/'configs/models'/(name+'.json') for name in ROWS]
    freeze=dict(utc=now(),source_geometry=geom,rows=jobs,novelty=evidence,
                source_sha256={str(p.relative_to(ROOT)):digest(p) for p in files},
                baseline='resource_matmul(shape, chip) ordinary route explicitly; never the evolving public default.',
                candidate='estimate_matmul_splitk(shape, chip=..., dtype=bf16,layout=ND_ND,deterministic_level=0). No fitted new coefficients.',
                scope='Contiguous, non-transposed BF16 A[M,K] @ B[K,N], no bias. Catalogue confirms per-rank geometry, not live vLLM dispatch.',
                acceptance='Per-target full reference tensor, exact SDK ELF and runtime SKU, complete all-core window and physical ends. Original verdict saved before existing asynchronous-drain recovery. Count MMAD and actual branch independently. Never use unqualified endpoint in errors.',
                budget='12 distinct endpoint attempts, serial, 900 wall seconds and 2048 MiB raw each, no retries. Wait for six original Post outcomes. Lower host priority than the larger Matmul lanes. No CA on 970DT.')
    save(FREEZE,freeze)
    (OUT/'.gitignore').write_text('raw/\nsupervisor.log\nworker.lock\n')
    report()


def report():
    reg=read(FREEZE);items=[];pairs=[]
    for j in reg['rows']:
        endpoints={};shape=j['shape'];pred=j['predictions']
        for chip,tag in ((DT,'dt'),(PR,'pr')):
            stem='-'.join(map(str,shape))+'-'+tag;path=OUT/(stem+'.json')
            r=read(path) if path.exists() else {}
            p=pred[chip];dt_reference=pred[DT]['us']
            item=dict(model=j['model'],shape=shape,chip=chip,qualified=r.get('qualified',False),status=r.get('status','pending'),
                      projected_970_us=pred[FUT]['us'],candidate_us=p['us'],ordinary_us=p['ordinary_us'],reuse_dt_us=dt_reference,
                      prediction_route=p['candidate']['route'],predicted_aic=p['candidate']['active_aic'],predicted_aiv=p['candidate']['active_aiv'],
                      ca_us=r.get('ca_us'),cost=r.get('cost'),reason=r.get('reason'),observed=r.get('observed'),
                      receipt=path.name if path.exists() else None)
            if r.get('qualified'):
                ca=r['ca_us']; endpoints[chip]=item
                item['signed_ape_pct']={k:100*(item[k]/ca-1) for k in ('candidate_us','ordinary_us','reuse_dt_us')}
            items.append(item)
        if len(endpoints)==2:
            a,b=endpoints[DT],endpoints[PR]
            pairs.append(dict(shape=shape,**{k:a[k]/b[k] for k in ('ca_us','candidate_us','ordinary_us','reuse_dt_us')}))
    groups={}
    for sku in (DT,PR):
        rows=[r for r in items if r['chip']==sku and r['qualified']]
        metrics={}
        for f in ('candidate_us','ordinary_us','reuse_dt_us'):
            metrics[f]=dict(wape_pct=100*sum(abs(r[f]-r['ca_us']) for r in rows)/sum(r['ca_us'] for r in rows) if rows else None,
                            max_ape_pct=max((abs(r['signed_ape_pct'][f]) for r in rows),default=None))
        groups[sku]=dict(n=len(rows),metrics=metrics)
    ratio_metrics={f:(100*sum(abs(p[f]-p['ca_us']) for p in pairs)/sum(p['ca_us'] for p in pairs) if pairs else None) for f in ('candidate_us','ordinary_us','reuse_dt_us')}
    save(OUT/'score.json',dict(utc=now(),freeze_sha256=digest(FREEZE),rows=items,groups=groups,dt_over_pr=pairs,ratio_wape_pct=ratio_metrics))
    def num(x):return 'N/A' if x is None else f'{x:.3f}'
    text=['# DeepSeek 窄投影 BF16 MatMulV3 独立验证','',
          'Flash 的每卡投影权重为 [32,5120]，Pro 为 [64,7168]。M 表示本次调用的 token 总行数，M=9 可以来自单请求校验 9 个 token。每次采集覆盖一个完整的连续 BF16 乘法，实际服务的量化或融合派发要另行核实。','',
          '| SKU | 合格/登记 | 候选 WAPE / 最大 % | 普通 WAPE / 最大 % | 直接 DT WAPE / 最大 % |',
          '|---|---:|---:|---:|---:|']
    for sku,g in groups.items():
        s=' | '.join(f"{num(g['metrics'][f]['wape_pct'])} / {num(g['metrics'][f]['max_ape_pct'])}" for f in ('candidate_us','ordinary_us','reuse_dt_us'))
        text.append(f"| {sku} | {g['n']}/6 | {s} |")
    text+=['','| [M,N,K] | SKU | 普通 µs | 候选 µs | 直接 DT µs | CA µs | 候选偏差 % | 实测 key / AIC / AIV / MMAD | 离线秒 / 峰值 MiB | 970DT 外推 µs | 资格 |',
           '|---|---|---:|---:|---:|---:|---:|---|---:|---:|---|']
    for r in items:
        o=r.get('observed') or {};c=r.get('cost') or {}
        sig='/'.join(str(o.get(k,'—')) for k in ('tiling_key','active_aic','active_aiv','mmad'))
        st=r['status']+((' '+r['reason']) if r['reason'] else '')
        text.append(f"| {r['shape']} | {r['chip']} | {num(r['ordinary_us'])} | {num(r['candidate_us'])} | {num(r['reuse_dt_us'])} | {num(r['ca_us'])} | {num(r.get('signed_ape_pct',{}).get('candidate_us'))} | {sig} | {num(c.get('seconds'))} / {num(c['peak_bytes']/2**20 if 'peak_bytes' in c else None)} | {num(r['projected_970_us'])} | {st} |")
    text+=['','| [M,N,K] | CA DT/PR | 候选 DT/PR | 普通 DT/PR | 直接复用 DT |','|---|---:|---:|---:|---:|']
    for p in pairs:
        text.append(f"| {p['shape']} | {p['ca_us']:.4f} | {p['candidate_us']:.4f} | {p['ordinary_us']:.4f} | 1 |")
    if not pairs:text.append('| 暂无双端合格配对 | N/A | N/A | N/A | N/A |')
    times=[p['candidate_query_ms'] for j in reg['rows'] for p in j['predictions'].values()]
    text+=['',f'18 次冻结候选查询中最长为 {max(times):.3f} ms。CA 结果只取两个 950 SKU。完整 golden、官方 kernel 和 ELF、每核首末、MMAD 数与分支从各端回执追溯。[冻结合同](freeze.json) · [逐端及双端数值](score.json) · [当前运行状态](state.json)']
    tmp=OUT/'report.md.tmp';tmp.write_text('\n'.join(text)+'\n');tmp.replace(OUT/'report.md')


def cpu_stats():
    result={}
    for line in Path('/proc/stat').read_text().splitlines():
        if line.startswith('cpu') and line[3:4].isdigit():
            name,*v=line.split();result[int(name[3:])]=(sum(map(int,v[:8])),int(v[3])+int(v[4]))
    return result


def host_mask(allowed):
    large=set()
    for path in ('prefill-status.json','long-prefill/state.json','wide-prefill/state.json'):
        p=OUT.parent/path
        if not p.exists():continue
        j=read(p)
        if j.get('phase')!='acquiring':continue
        try:large.update(os.sched_getaffinity(j['pid']))
        except (KeyError,ProcessLookupError):pass
    before=cpu_stats();time.sleep(1.5);after=cpu_stats()
    busy={c:1-(after[c][1]-before[c][1])/max(1,after[c][0]-before[c][0]) for c in allowed}
    # First use physically distinct CPUs outside reserved, large-case masks. An
    # idle CPU in such a mask is usable at lower priority if all alternatives
    # are occupied. The large processes keep their own affinity and priority.
    preferred=sorted(allowed,key=lambda c:(0 if c not in large and busy[c]<.45 else 1 if busy[c]<.15 else 2 if c not in large else 3,busy[c],c))
    chosen=[];physical=set()
    for cpu in preferred:
        topo=(Path('/sys/devices/system/cpu')/f'cpu{cpu}/topology/thread_siblings_list').read_text().strip()
        if topo in physical:continue
        physical.add(topo);chosen.append(cpu)
        if len(chosen)==64:break
    return chosen,dict(physical_selected=len(chosen),outside_large_masks=len(set(chosen)-large),
                       host_cpus_below_45pct=sum(v<.45 for v in busy.values()),process_nice=os.nice(0),selected_cpus=chosen)


def verify(j,sku,slug,folder):
    path=OUT/(slug+'-acquisition.json')
    cost=read(folder/'cost.json');reg=read(FREEZE)
    try:
        if cost['started_utc']<=reg['utc'] or cost['seconds']>j['wall_s'] or cost['peak_bytes']>j['raw_mib']*2**20:
            raise ValueError('frozen time, storage or chronology limit')
        initial=qualify(j,sku,folder)
    except (ValueError,KeyError,OSError,AssertionError,IndexError) as e:
        initial=dict(qualified=False,ca_us=None,reason=repr(e))
    receipt=dict(case=slug,chip=sku,shape=j['shape'],cost=cost,freeze_sha256=digest(FREEZE),**initial)
    if path.exists():
        assert read(path)==receipt
    else:save(path,receipt)
    chosen=initial;basis='original'
    if not initial['qualified']:
        try:supplement=recover(j,receipt,folder,acquisition=path)
        except (ValueError,KeyError,OSError,AssertionError,IndexError,StopIteration) as e:
            supplement=None;chosen={**initial,'reason':str(initial.get('reason'))+'; drain: '+repr(e)}
        if supplement:
            save(OUT/(slug+'-drain.json'),supplement);chosen=supplement;basis='physical END and drain'
    result=dict(case=slug,chip=sku,shape=j['shape'],cost=cost,freeze_sha256=digest(FREEZE),
                acquisition_sha256=digest(path),timing_evidence=basis,
                qualified=chosen['qualified'],ca_us=chosen['ca_us'],reason=chosen.get('reason'))
    if chosen['qualified']:
        st,=chosen['stages'];key=re.search(r'_high_performance_(\d+)',st['kernel'])
        observed={k:st[k] for k in ('kernel','ca_us','active_aic','active_aiv','mmad','trace','trace_sha256','binary_sha256')}
        family_hash=reg['source_sha256'][str(PROVIDER.with_suffix('.o').relative_to(ROOT))]
        observed.update(tiling_key=int(key[1]) if key else None,provider_elf_match=st['binary_sha256']==family_hash)
        result['observed']=observed;result['correctness']=chosen['correctness']
        if not observed['provider_elf_match']:
            result.update(qualified=False,ca_us=None,reason='different library ELF; elapsed separately retained for scope audit')
        elif observed['tiling_key']==4098:
            try:observed['physical_k_partition']=shards(st,j['shape'])
            except (OSError,ValueError,AssertionError,KeyError,IndexError,StopIteration) as e:
                observed['partition_mapping_unresolved']=repr(e)
    result['status']='qualified' if result['qualified'] else 'unqualified'
    return result


def state(phase,**extra):
    save(OUT/'state.json',dict(utc=now(),phase=phase,pid=os.getpid(),freeze_sha256=digest(FREEZE),**extra))


def run():
    reg=read(FREEZE)
    for p,h in reg['source_sha256'].items():
        if digest(ROOT/p)!=h:raise ValueError('frozen source changed '+p)
    with (OUT/'worker.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        allowed=os.sched_getaffinity(0);os.nice(14)
        os.environ.update(OMP_WAIT_POLICY='PASSIVE',GOMP_SPINCOUNT='0')
        while not POST.exists() or read(POST).get('phase')!='complete':
            state('waiting for original six Post receipts',predecessor=str(POST.relative_to(ROOT)));time.sleep(20)
        assert len(list((POST.parent/'receipts').glob('*.json')))==6
        RAW.mkdir(exist_ok=True)
        completed=0
        for j in reg['rows']:
            for sku,tag in ((DT,'dt'),(PR,'pr')):
                slug='-'.join(map(str,j['shape']))+'-'+tag;folder=RAW/slug;result=OUT/(slug+'.json')
                if result.exists():completed+=1;continue
                if not folder.exists():
                    while shutil.disk_usage(ROOT).free<(20*2**30):
                        state('waiting for disk',case=slug,completed=completed);time.sleep(30)
                    chosen,host=host_mask(allowed);os.sched_setaffinity(0,chosen)
                    state('acquiring',case=slug,completed=completed,host=host)
                    folder.mkdir();measure(j,sku,folder)
                    os.sched_setaffinity(0,allowed)
                state('verifying',case=slug,completed=completed)
                if (folder/'cost.json').exists():
                    answer=verify(j,sku,slug,folder)
                else:
                    answer=dict(shape=j['shape'],chip=sku,qualified=False,ca_us=None,status='unqualified',
                                reason='incomplete previous acquisition; registered attempt will not be repeated')
                save(result,answer);completed+=1;report()
        assert completed==12
        report();state('finished',completed=completed,qualified=sum(r['qualified'] for r in read(OUT/'score.json')['rows']))


def start():
    pth=OUT/'supervisor.log'
    with pth.open('a') as log:
        proc=subprocess.Popen([sys.executable,str(Path(__file__).resolve()),'run'],cwd=ROOT,
                              stdin=subprocess.DEVNULL,stdout=log,stderr=log,start_new_session=True)
        print(proc.pid,flush=True)


if __name__=='__main__':
    {'freeze':freeze,'run':run,'start':start,'report':report}[sys.argv[1]]()
