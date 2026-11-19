"""Physically qualify a complete official FLA V2 chain. A partial capture has no timing."""
from collections import Counter, defaultdict
import json
import re
from pathlib import Path
import sys

from probe_v2 import OUT,RAW,REG,COST,LOG,probe
from score import events,shape_span,tables,TICKS,checked

STAGES={'ChunkKdaFwdPrepare':'chunk_kda_fwd_prepare','ChunkFwdH':'chunk_fwd_h','ChunkKdaFwdFinalize':'chunk_kda_fwd_finalize'}


def stage_windows(d, official):
    sim=d/'simulator';aggregate=sim/'trace.json';observed=shape_span(events(aggregate))
    shards={p.parent.name:p for p in sim.glob('core*/trace.json')}
    checked(set(observed)==set(official)==set(shards) and bool(shards),'shard/official roster')
    details={}
    for core,window in observed.items():
        checked(shape_span(events(shards[core]),core)=={core:window},'shard/aggregate content on '+core)
        checked(len(window['end'])==1,'missing physical END on '+core)
        if window['marker']:
            checked(len(window['marker'])==1 and window['marker'][0][1]==window['hi'] and
                    int(window['marker'][0][2],16)==0x11111111,'malformed CA endpoint on '+core)
        duration=(window['hi']-window['lo'])/TICKS
        checked(abs(duration-float(official[core][0]))<=.00501,'official per-core duration on '+core)
        details[core]=dict(first_tick=window['lo'],last_tick=window['hi'],physical_end_tick=window['end'][0][1],
                           terminal_label=bool(window['marker']),mmads=window['mmad'],official_us=official[core][0],
                           trace_sha256=probe.digest(shards[core]))
    lo=min(c['first_tick'] for c in details.values());hi=max(c['last_tick'] for c in details.values())
    return dict(ca_us=(hi-lo)/TICKS,first_tick=lo,last_tick=hi,cores=details,trace=probe.rel(aggregate),trace_sha256=probe.digest(aggregate))


def main():
    plan=json.loads(REG.read_bytes());cost=json.loads(COST.read_bytes());log=LOG.read_text(errors='replace')
    names=re.findall(r'Start profiling on kernel: (\S+)',log);durations=tables(log);ordinals=Counter();records=[]
    dirs=list(RAW.glob('profile/OPPROF_*'))
    for index,name in enumerate(names):
        ordinal=ordinals[name];ordinals[name]+=1
        family=STAGES.get(name.split('_',1)[0])
        row=dict(kernel=name,index=index,ordinal=ordinal,release_family=family,ca_us=None,duration_qualified=False)
        if len(dirs)==1:
            candidates=[dirs[0]/name/str(ordinal),dirs[0]/'device0'/name/str(ordinal)]
            if len(names)==1:candidates.append(dirs[0])
            path=next((q for q in candidates if (q/'dump/aicore_binary.o').is_file()),None)
            if path:
                binary=path/'dump/aicore_binary.o';h=probe.hashlib.sha256(probe.text_bytes(binary)).hexdigest()
                row.update(captured_binary=probe.rel(binary),binary_sha256=probe.digest(binary),device_text_sha256=h)
                if family:
                    row['published_names']=[k for k,v in plan['released_device_text'][family].items() if v==h]
                try:
                    checked(index<len(durations),'missing official duration table')
                    row.update(stage_windows(path,durations[index]),duration_qualified=True)
                except (ValueError,KeyError,TypeError,OSError,IndexError) as e:row['incomplete']=str(e)
            else:row['incomplete']='no captured device binary'
        records.append(row)
    implementations=[r for r in records if r['release_family']]
    family_ok=[r['release_family'] for r in implementations]==plan['expected_stages'] and all(r.get('published_names') for r in implementations)
    api=str(probe.VENDOR/'op_api/lib/libcust_opapi.so')
    bindings=dict(re.findall(r'^bound_api=(\S+) library=(\S+)',log,re.M))
    bound=bindings=={'aclnnChunkKdaFwdV2':api,'aclnnChunkKdaFwdV2GetWorkspaceSize':api}
    gold={k:(int(n),int(b)) for k,n,b in re.findall(r'^correctness name=(\S+) count=(\d+) bad=(\d+)',log,re.M)}
    golden=(gold=={k:(n,0) for k,n in plan['oracle_elements'].items()} and 'input_vk_unchanged=1' in log and 'SMOKE_PASS' in log)
    runtime=(re.findall(r'^runtime_soc=(\S+)',log,re.M)==[probe.CHIP] and 'hbm3_8000M.ini' in log)
    completed=(cost['exit_code']==0 and cost['stop'] is None and 'Profiling running finished. All task success.' in log and 'Profiling data parse finished.' in log)
    sync_issues=[line for line in log.splitlines() if re.search(r'\[ERROR\].*(?:execute_set_flag|execute_wait_flag|deadlock|invalid instruction)',line)]
    timing=bool(records) and all(r['duration_qualified'] for r in records) and len(records)==len(durations)
    qualified=bool(golden and runtime and bound and family_ok and completed and timing and not sync_issues)
    data=dict(registration_sha256=probe.digest(REG),role=plan['role'],process=cost,host_actual_bindings=bindings,
              correct_host=bound,runtime_verified=runtime,golden_qualified=golden,actual_numeric_groups=gold,
              initial_unchanged='input_vk_unchanged=1' in log,host_smoke_fail=[l for l in log.splitlines() if 'SMOKE_FAIL' in l],
              actual_launches=records,stages_verified=family_ok,complete_device_windows=timing,process_complete=completed,
              sync_issue_count=len(sync_issues),sync_examples=sync_issues[:12],timing_qualified=qualified,
              serialized_device_us=sum(r['ca_us'] for r in records) if qualified else None,log_sha256=probe.digest(LOG))
    probe.put(OUT/'receipt.json',data)
    print('qualified',qualified,'kernels',names,'golden',gold,'warnings',len(sync_issues))

if __name__=='__main__':main()
