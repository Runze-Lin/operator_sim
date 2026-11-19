"""One same-stream prefix per SKU, no retries. Reuse the original CANN runner and cutoff."""
import fcntl
from hashlib import sha256
import json
import os
from pathlib import Path
import re
import sys

ROOT=Path(__file__).resolve().parents[3]
sys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT/'development/validation/sku17'))
from development.validation.rms_dispatch_audit.ordered_ca import capture
from development.validation.sku19.campaign import now
import verify

OUT=verify.OUT/'prefix-recovery'; RAW=OUT/'raw'; PLAN=OUT/'register.json'
TARGETS={'Ascend950DT_9582':6,'Ascend950PR_957b':7}


def read(p):return json.loads(p.read_bytes())


def put(p, obj):
    tmp=p.with_suffix('.tmp');tmp.write_text(json.dumps(obj,indent=2)+'\n');tmp.replace(p)


def register():
    if PLAN.exists():raise ValueError('prefix plan already registered')
    frozen=verify.OUT/'freeze.json';settings=read(frozen);specs={};files=[frozen,verify.VENDOR, Path(__file__).resolve(),Path(__file__).with_name('verify.py'),ROOT/'development/validation/rms_dispatch_audit/ordered_ca.py',ROOT/'scripts/cann-run.sh']
    for name,sha in settings['files'].items():
        if verify.digest(ROOT/name)!=sha:raise ValueError('original freeze changed: '+name)
        files.append(ROOT/name)
    for chip,target in TARGETS.items():
        verdict=verify.OUT/f'verified-ordered-{chip}.json';source=verify.RAW/chip/'inputs.txt';files.extend([verdict,source]);rows=[j for j in settings['rows'] if j['protocol']=='ordered new sequence' and j['chip']==chip][:target+1]
        hash_inputs=sha256((''.join(source.read_text().splitlines(keepends=True)[:target+1])).encode()).hexdigest()
        old=read(verdict);assert not old['rows'][target]['timing_qualified'] and all(r['timing_qualified'] for r in old['rows'][:target]);
        specs[chip]=dict(target_i=target,prefix_inputs_hash=hash_inputs,scored_shape=rows[-1]['shape'],rows=rows,original_missing=old['rows'][target]['reason'])
    packet=dict(utc=now(),frozen_model=verify.rel(frozen),source_hashes={verify.rel(p):verify.digest(p) for p in files},
                physical_elf_text_sha256=sha256(verify.text_bytes(verify.VENDOR)).hexdigest(),queue=specs,order=list(TARGETS),maximum_attempts_per_chip=1,
                wall_per_batch_s=settings['wall_per_batch_s'],raw_mib_per_batch=settings['raw_mib_per_batch'],concurrency=1,
                role='Only final launch per prefix can replace one N/A on that SKU. All preceding launches are unscored protocol controls.',
                success='Same ordered inputs, ELF, physical SKU, active core roster and 2 numeric goldens per call; scored launch must have all-core actual END/CA terminal, matching aggregate and shard fingerprints, shape-bound write bytes and official per-core durations. Preserve original disposition.',
                trace_semantics='Original msopprof simulator --dump=false, same binary, memory model, one ACL stream, --launch-count equal to input prefix length.')
    with PLAN.open('x') as f:json.dump(packet,f,indent=2);f.write('\n')


def host(folder,chip,jobs):
    log=(folder/'run.log').read_text();checks=verify.checked
    checks(re.findall(r'^runtime_soc=(\S+)',log,re.M)==[chip] and ('lpddr5_6400M_bailu.ini' if 'PR' in chip else 'hbm3_8000M.ini') in log,'physical SKU')
    checks('SMOKE_PASS' in log and 'SMOKE_FAIL' not in log and 'Profiling running finished. All task success.' in log,'host or device collection did not finish')
    checks(not re.search(r'\[ERROR\].*(?:execute_set_flag|execute_wait_flag|deadlock|invalid instruction)',log),'CA instruction error')
    spans=re.findall(r'^BATCH_BEGIN (\d+)\s*$\n(.*?)^BATCH_END \1\s*$',log,re.M|re.S);tables=verify.tables(log)
    checks(len(spans)==len(tables)==len(jobs) and log.count('Start profiling on kernel: ')==len(jobs),'incomplete prefix lengths');out=[]
    for job,(i,body) in zip(jobs,spans):
        n,w=job['shape'];i=int(i);checks(i==job['i'] and f'case=rmsnorm api=aclnnRmsNorm R={n} C={w} dtype=BF16' in body and re.findall(r'Start profiling on kernel: (\S+)',body)==[verify.KERNEL],'input or kernel order')
        gold=re.findall(r'^correctness [^\n]+',body,re.M);checks([int(re.match(r'correctness count=(\d+) bad=0\b',s)[1]) for s in gold]==[n*w,n],'goldens')
        begins=re.findall(r'\[(\d+)\] \[TASK_BEGIN\].*stream_id:(\d+), task_id:(\d+) full_task_id:(\d+) BEGIN',body);dones=re.findall(r'\[(\d+)\] \[TASK_DONE\].*stream_id:(\d+), task_id:(\d+) TASK_DONE',body)
        b,=set(begins);d,=set(dones);checks(len(begins)==len(dones) and b[1:3]==d[1:3] and int(b[3])==i and int(d[0])>=int(b[0]),'device task completion or ordinal')
        out.append(dict(golden=gold,begin_tick=int(b[0]),done_tick=int(d[0]),full_task_id=i,raw_stream_id=int(b[1])))
    return out,tables


def assess(chip,spec,plan,folder,cost):
    verify.checked(verify.digest(folder/'inputs.txt')==spec['prefix_inputs_hash'],'copied input differs from original prefix');jobs=spec['rows'];actual,tables=host(folder,chip,jobs)
    old=read(verify.OUT/f'verified-ordered-{chip}.json');prior_tables=verify.tables((verify.RAW/chip/'run.log').read_text());elf=verify.text_bytes(verify.VENDOR)
    roots=list(folder.glob('profile/OPPROF_*/device0/'+verify.KERNEL));verify.checked(len(roots)==1,'device profile path');controls=[];identity=True;target=None
    for job,task,table in zip(jobs,actual,tables):
        i=job['i'];d=roots[0]/str(i);oldrow=old['rows'][i];match=set(prior_tables[i])==set(table);elfmatch=verify.text_bytes(d/'dump/aicore_binary.o')==elf;identity=identity and match and elfmatch
        try:r=verify.launch(d,chip,job,task,table,elf)
        except (ValueError,KeyError,TypeError,OSError,IndexError) as error:
            reason=str(error);r=dict(**job,timing_qualified=False,stage=None,reason=reason,correctness=task['golden'],incomplete_evidence=verify.incomplete_evidence(d,table,reason))
        delta=(r['stage']['ca_us']-oldrow['stage']['ca_us']) if oldrow['stage'] and r['stage'] else None
        per_core={}
        if r['timing_qualified'] and oldrow['timing_qualified'] and match:
            for c,core in r['stage']['cores'].items():
                oc=oldrow['stage']['cores'][c];per_core[c]=dict(duration_delta_ticks=(core['last_tick']-core['first_tick'])-(oc['last_tick']-oc['first_tick']),
                    start_relative_delta_ticks=(core['first_tick']-r['stage']['first_tick'])-(oc['first_tick']-oldrow['stage']['first_tick']))
        r.update(original_timing_qualified=oldrow['timing_qualified'],original_ca_us=(oldrow['stage'] or {}).get('ca_us'),whole_window_delta_us=delta,
                 same_actual_core_roster=match,same_published_elf_text=elfmatch,per_core_control_comparison=per_core)
        if i==spec['target_i']:target=r
        else:controls.append(r)
    return dict(chip=chip,target_index=spec['target_i'],target_shape=spec['scored_shape'],qualified=bool(identity and target['timing_qualified']),matched_prefix_identity=identity,
                scored_rows=[target] if identity and target['timing_qualified'] else [],target=target,controls=controls,scored_count=int(identity and target['timing_qualified']),
                process=cost,complete_host_golden_groups=2*len(jobs),registration_sha256=verify.digest(PLAN),input_sha256=verify.digest(folder/'inputs.txt'),log_sha256=verify.digest(folder/'run.log'))


def run():
    plan=read(PLAN)
    for name,val in plan['source_hashes'].items():
        verify.checked(verify.digest(ROOT/name)==val,'frozen file changed '+name)
    RAW.mkdir(exist_ok=True);os.environ.update(OMP_WAIT_POLICY='PASSIVE',GOMP_SPINCOUNT='0')
    with (RAW/'worker.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        verify.checked(read(verify.OUT/'ordered-status.json')['state']=='finished','prior serial worker busy')
        for chip,spec in plan['queue'].items():
            f=OUT/f'{chip}.json';folder=RAW/chip
            if f.exists():continue
            put(OUT/'status.json',dict(state='running',chip=chip,target_i=spec['target_i'],utc=now(),registration_sha256=verify.digest(PLAN)))
            cost=capture(chip,folder,spec['rows'],plan)
            try:out=assess(chip,spec,plan,folder,cost)
            except (OSError,ValueError,KeyError,IndexError,TypeError) as e:
                out=dict(chip=chip,qualified=False,target_index=spec['target_i'],target_shape=spec['scored_shape'],scored_count=0,scored_rows=[],process=cost,reason=repr(e),registration_sha256=verify.digest(PLAN))
            put(f,out)
        put(OUT/'status.json',dict(state='finished',utc=now(),scored_count=sum(read(OUT/f'{c}.json')['scored_count'] for c in plan['queue'])))

if __name__=='__main__':
    {'register':register,'run':run}[sys.argv[1]]()
