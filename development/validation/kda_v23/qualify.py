"""Use full Chrome phases and core shards to qualify each physical launch."""
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
import re

from series import probe, content, multiset_digest
from score import tables, checked

FAMILIES={'ChunkKdaFwdPrepare':'chunk_kda_fwd_prepare','ChunkFwdH':'chunk_fwd_h','ChunkKdaFwdFinalize':'chunk_kda_fwd_finalize'}


def counters(path):
    dma=Counter();mmad=Counter();queues=Counter();op_count=0
    for e in json.loads(path.read_bytes())['traceEvents']:
        if e['ph']!='X':continue
        d=json.loads(e['args']['detail'] or '{}')
        if d.get('is_sync') or d.get('is_wait') or d.get('is_duration_excluded'):continue
        queues[e['tid']]+=1
        if 'processed_bytes' in d and 'src_mem' in d and 'dst_mem' in d:
            dma[d['src_mem']+'->'+d['dst_mem']]+=d['processed_bytes']
        if e['name']=='MMAD':
            key=tuple(d[dim] for dim in ('m','n','k'))
            mm=d['m']*d['n']*d['k'];op_count+=2*mm;mmad['x'.join(map(str,key))]+=1
    return dict(descriptor_bytes=dict(dma),mmad_issued_shapes=dict(mmad),issued_mmad_ops=op_count,queue_instruction_counts=dict(queues))


def stage(path,name,index,official,reg):
    binary=path/'dump/aicore_binary.o';family=FAMILIES.get(name.split('_',1)[0]);text=hashlib.sha256(probe.text_bytes(binary)).hexdigest()
    match=[] if family is None else [p for p,h in reg['published_device_text'][family].items() if h==text]
    agg=path/'simulator/trace.json'
    merged=content(agg,True);shards={p.parent.name:p for p in agg.parent.glob('core*/trace.json')}
    checked(set(merged)==set(shards)==set(official) and bool(shards),'official/merged/shard core roster mismatch')
    details={}
    for c,r in merged.items():
        s=content(shards[c],False)[c]
        for field in ('by_instr','sync','first','last','ends','labels','mmads'):
            checked(r[field]==s[field],f'{name} {c} shard/aggregate {field}')
        checked(len(s['ends'])==1 and len(s['labels'])<=1,f'{name} {c} missing physical END')
        if s['labels']:
            label,=s['labels'];checked(int(label[0],16)==0x11111111 and label[2]==s['last'],'CA label not terminal')
        duration=(s['last']-s['first'])/1650
        checked(abs(duration-float(official[c][0]))<=.00501,f'{name} {c} official duration disagrees')
        details[c]=dict(first_tick=s['first'],last_tick=s['last'],end=s['ends'][0],end_labels=s['labels'],
            official_us=official[c][0],mmads=s['mmads'],phys_digest=multiset_digest(s['by_instr']),sync_digest=multiset_digest(s['sync']),
            shard_sha256=probe.digest(shards[c]))
    lo=min(d['first_tick'] for d in details.values());hi=max(d['last_tick'] for d in details.values())
    return dict(kernel=name,ordinal=index,family=family,label='official FLA V2 stage' if family else 'auxiliary device launch',
        first_tick=lo,last_tick=hi,ca_us=(hi-lo)/1650,active_aic=sum('cubecore'in c for c in details),active_aiv=sum('veccore'in c for c in details),
        mmads=sum(c['mmads'] for c in details.values()),resource=counters(agg),cores=details,captured_binary=probe.rel(binary),
        device_text_sha256=text,published_names=match,aggregate=probe.rel(agg),aggregate_sha256=probe.digest(agg))


def verify(reg,case,folder,cost):
    log_path=folder/'run.log';log=log_path.read_text(errors='replace')
    names=re.findall(r'Start profiling on kernel: (\S+)',log);tab=tables(log)
    groups={k:[int(n),int(b)] for k,n,b in re.findall(r'^correctness name=(\S+) count=(\d+) bad=(\d+)',log,re.M)}
    bindings=dict(re.findall(r'^bound_api=(\S+) library=(\S+)',log,re.M));api=str(probe.VENDOR/'op_api/lib/libcust_opapi.so')
    sku_ok=re.findall(r'^runtime_soc=(\S+)',log,re.M)==[case['sku']] and case['physical_memory'] in log
    fault_lines=[l for l in log.splitlines() if re.search(r'\[ERROR\].*(?:execute_(?:set|wait)_flag|deadlock|invalid instruction)',l)]
    published=bindings=={n:api for n in ['aclnnChunkKdaFwdV2','aclnnChunkKdaFwdV2GetWorkspaceSize']}
    host=groups=={k:[v,0] for k,v in case['oracle'].items()} and 'input_vk_unchanged=1' in log and 'SMOKE_PASS' in log and 'SMOKE_FAIL' not in log
    process=cost['exit_code']==0 and cost['stop'] is None and 'Profiling running finished. All task success.' in log and 'Profiling data parse finished.' in log
    root=list(folder.glob('profile/OPPROF_*'));rows=[];ordinals=Counter()
    for idx,name in enumerate(names):
        i=ordinals[name];ordinals[name]+=1
        try:
            checked(idx<len(tab),'missing official duration table')
            checked(len(root)==1,'no unique profile root')
            paths=[root[0]/name/str(i),root[0]/'device0'/name/str(i)]
            if len(names)==1:paths.append(root[0])
            path=next((p for p in paths if (p/'dump/aicore_binary.o').is_file()),None)
            checked(path is not None,'missing captured ELF')
            rows.append(stage(path,name,i,tab[idx],reg))
        except (ValueError,KeyError,IndexError,OSError,TypeError) as exc:
            rows.append(dict(kernel=name,ordinal=i,ca_us=None,reason=str(exc)))
    timed=bool(rows) and len(rows)==len(tab) and all(r['ca_us'] is not None for r in rows)
    ordered=timed and all(a['last_tick']<b['first_tick'] for a,b in zip(rows,rows[1:]))
    main=[r for r in rows if r.get('family')]
    identity=[r['family'] for r in main]==list(FAMILIES.values()) and all(r['published_names'] for r in main)
    eligible=bool(published and sku_ok and host and process and not fault_lines and timed and ordered and identity)
    return dict(qualified=eligible,actual_sku_verified=sku_ok,memory_model=case['physical_memory'],host_api_verified=published,
        golden_qualified=host,output_groups=groups,correct_initial_state='input_vk_unchanged=1'in log,sync_errors=len(fault_lines),sync_examples=fault_lines[:5],
        exact_stage_elfs=identity,full_windows=timed,chronological=ordered,process_complete=process,observed_kernels=names,
        stages=rows,serialized_device_us=sum(r['ca_us'] for r in rows) if eligible else None,
        main_stages_us=sum(r['ca_us'] for r in main) if eligible else None,device_span_us=(rows[-1]['last_tick']-rows[0]['first_tick'])/1650 if eligible else None,
        log_sha256=probe.digest(log_path))
