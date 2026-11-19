"""Check the one FLA capture against release identity, four goldens and per-core CA retirement."""
from collections import Counter, defaultdict
from hashlib import blake2b, sha256
from pathlib import Path
import json
import re
import sys
import time

from probe import CAPTURE, CHIP, COST, OUT, RAW, REGISTER, ROOT, VENDOR, digest, put, rel
sys.path.insert(0, str(ROOT/'development/validation/rms_ordered_verify'))
from verify import checked, events, tables, text_bytes

TICKS = 1650
MOD = 1 << 128


def shape_span(events_source, physical_core=None):
    result = {}
    for e in events_source:
        core = physical_core if physical_core is not None else e['pid']
        name = e['name']
        detail = e['args']['detail']
        a = round(e['ts']*TICKS)
        b = round((e['ts']+e['dur'])*TICKS)
        checked(b >= a, 'negative instruction duration')
        v = result.setdefault(core, {'lo':a,'hi':b,'end':[],'marker':[],'events':0,'fingerprint':0,'mmad':0})
        v['lo'] = min(v['lo'],a)
        v['hi'] = max(v['hi'],b)
        if name=='END':
            v['end'].append([a,b,e['args']['pc_addr']])
        if name=='END_LABEL':
            v['marker'].append([a,b,e['args']['pc_addr']])
        if name=='MMAD':
            v['mmad']+=1
        if '"is_sync":1' in detail or '"is_wait":1' in detail:
            continue
        stamp = f"{name}|{e['args']['pc_addr']}|{a}|{b}".encode()
        v['fingerprint'] = (v['fingerprint'] + int.from_bytes(blake2b(stamp,digest_size=16).digest(),'little')) % MOD
        v['events']+=1
    return result


def timing(directory, table):
    sim = directory/'simulator'
    trace = sim/'trace.json'
    merged = shape_span(events(trace))
    shards = {p.parent.name:p for p in sim.glob('core*/trace.json')}
    checked(set(merged)==set(table)==set(shards) and bool(merged), 'mismatched official core roster/shards')
    measured = {}
    for core,record in merged.items():
        checked(shape_span(events(shards[core]),core)=={core:record}, 'aggregation mismatch on '+core)
        checked(len(record['end'])==len(record['marker'])==1 and
                record['end'][0][1] <= record['marker'][0][1] == record['hi'] and
                int(record['marker'][0][2],16)==0x11111111, 'physical END / terminal label absent on '+core)
        span = (record['hi']-record['lo'])/TICKS
        checked(abs(span-float(table[core][0])) <= .00501, 'per-core duration mismatch on '+core)
        measured[core] = dict(first_tick=record['lo'],last_tick=record['hi'],end_tick=record['end'][0][1],
                              official_duration_us=table[core][0],mmad=record['mmad'],
                              instruction_count_without_sync=record['events'],shard_sha256=digest(shards[core]))
    first=min(v['first_tick'] for v in measured.values());last=max(v['last_tick'] for v in measured.values())
    return dict(trace=rel(trace),trace_sha256=digest(trace),ca_us=(last-first)/TICKS,first_tick=first,last_tick=last,cores=measured)


def main():
    costs=json.loads(COST.read_bytes()); plan=json.loads(REGISTER.read_bytes())
    data=(CAPTURE/'run.log').read_text(errors='replace')
    names=re.findall(r'Start profiling on kernel: (\S+)',data)
    durations=tables(data);ordinals=defaultdict(int);stages=[]
    root,=CAPTURE.glob('profile/OPPROF_*')
    [api] = (VENDOR/'op_api/lib').glob('libcust_opapi.so')
    evidence=(RAW/'loader-bindings.txt').read_text().splitlines()
    binding_ok=all(any(str(api) in l and 'normal symbol `'+name+"'" in l for l in evidence)
                   for name in ('aclnnChunkKdaFwd','aclnnChunkKdaFwdGetWorkspaceSize'))
    counts={'attention':16384,'aqk':8192,'akk':8192,'state_vk':32768}
    groups={s:(int(n),int(b)) for s,n,b in re.findall(r'^correctness name=(\S+) count=(\d+) bad=(\d+)',data,re.M)}
    golden_ok=(groups=={k:(n,0) for k,n in counts.items()} and 'input_vk_unchanged=1' in data and 'SMOKE_PASS' in data and 'SMOKE_FAIL' not in data)
    errors=re.findall(r'already has same set_flag! pc:(0x[0-9a-f]+)',data)
    for idx,name in enumerate(names):
        i=ordinals[name];ordinals[name]+=1
        candidates=[root/name/str(i),root/'device0'/name/str(i)]
        d=next((p for p in candidates if (p/'dump/aicore_binary.o').exists()),None)
        stage=dict(kernel=name,index=idx,ordinal=i,duration_qualified=False,ca_us=None)
        if d:
            binary=d/'dump/aicore_binary.o'
            checksum=sha256(text_bytes(binary)).hexdigest()
            stage.update(raw_directory=rel(d),binary_sha256=digest(binary),captured_text_sha256=checksum)
            if name.startswith('ChunkKdaFwd_'):
                stage['wheel_objects_with_same_text']=[k for k,v in plan['published_kda_text'].items() if v==checksum]
                stage['same_text_as_cann_builtin']=checksum in plan['cann_builtin_kda_text'].values()
            try:
                checked(idx<len(durations),'missing official per-core duration table')
                stage.update(timing(d,durations[idx]),duration_qualified=True)
            except (ValueError,OSError,IndexError,KeyError,TypeError) as exc:
                stage['incomplete']=str(exc)
        else:
            stage['incomplete']='missing retained per-kernel ELF/directory'
        stages.append(stage)
    main_ks=[s for s in stages if s['kernel'].startswith('ChunkKdaFwd_')]
    identity_ok=bool(main_ks) and all(s.get('wheel_objects_with_same_text') and not s['same_text_as_cann_builtin'] for s in main_ks)
    sku=re.findall(r'^runtime_soc=(\S+)',data,re.M)==[CHIP] and 'hbm3_8000M.ini' in data
    trace_ok=bool(stages) and len(durations)==len(stages) and all(s['duration_qualified'] for s in stages)
    complete=(costs['exit_code']==0 and costs['stop_reason'] is None and 'Profiling data parse finished.' in data and 'Profiling running finished. All task success.' in data)
    ok=bool(sku and complete and binding_ok and identity_ok and golden_ok and trace_ok and not errors)
    result=dict(registration_sha256=digest(REGISTER),host_elf=rel(RAW/'kimi-fla'),host_binding_loader_dry_check=evidence,host_api_qualified=binding_ok,
        host_plt_note="The data collection executable's dladdr result points at its own PLT. Loader resolution above was checked by running the same executable with no application arguments, so ACL was not initialized during the loader check.",
        actual_sku_qualified=sku,actual_device_release_qualified=identity_ok,process=costs,process_complete=complete,
        main_count=len(main_ks),stages=stages,official_duration_tables=len(durations),trace_qualified=trace_ok,
        checks=groups,expected_output_elements=counts,golden_qualified=golden_ok,duplicate_set_total=len(errors),
        duplicate_set_unique_pcs=dict(Counter(errors)),timing_qualified=ok,
        device_complete_serial_us=sum(s['ca_us'] for s in stages) if ok else None,
        device_main_only_us=sum(s['ca_us'] for s in main_ks) if ok else None,
        log_sha256=digest(CAPTURE/'run.log'))
    put(OUT/'receipt.json',result)
    print('timing_qualified',ok,[(s['kernel'],s.get('ca_us'),s.get('incomplete')) for s in stages])

if __name__=='__main__':
    if sys.argv[1:]==['--wait']:
        while not COST.is_file():time.sleep(12)
    main()
