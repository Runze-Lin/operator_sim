"""Read-only, aggregate-only extraction of npusim helper contract v7.

No JSON timeline or analysis DB is generated. Raw helper facts remain untouched.
"""
import math
from pathlib import Path
import re
import sqlite3
from .ca import sha256


def extract_compact(directory,log_path,*,native_returncode=None):
    root=Path(directory);log=Path(log_path).read_text(errors='replace')
    socs=set(re.findall(r'runtime_soc=(\S+)',log))
    if socs!={'Ascend950DT_9582'}:raise ValueError('Wrong or missing runtime SoC: '+repr(socs))
    native=native_returncode is not None
    if 'SMOKE_PASS' not in log or 'SMOKE_FAIL' in log or (native_returncode!=0 if native else 'Simulation SUCCESS' not in log):
        raise ValueError('Incomplete application/correctness/record run')
    if 'timeout has reached' in log:raise ValueError('Profiler timeout')
    paths=list(root.glob('helper/*.db' if native else 'npusim_*/record/*.db'))
    if len(paths)!=1:raise ValueError('Expected exactly one helper kernel database')
    path=paths[0]
    with sqlite3.connect(path.resolve().as_uri()+'?mode=ro',uri=True) as db:
        if db.execute('PRAGMA quick_check').fetchone()[0]!='ok':raise ValueError('Corrupt helper database')
        info=dict(db.execute('SELECT KernelInfoAttr,KernelInfoVal FROM KernelInfo'))
        if str(info.get('helper_version'))!='0.8.0' or int(info.get('contract_version',-1))!=7:
            raise ValueError('Unsupported helper version/contract')
        freq=float(info['frequency_ghz'])
        if not math.isclose(freq,1.65,abs_tol=1e-12):raise ValueError('Unexpected frequency')
        count,begin,end,bad=db.execute('SELECT COUNT(*),MIN(ExecInstrTickStart),MAX(ExecInstrTickEnd),SUM(ExecInstrTickEnd<ExecInstrTickStart) FROM ExecutedInstructions').fetchone()
        if not count or bad or int(info['instrs_exec_num'])!=count:raise ValueError('Invalid instruction population')
        if int(info['exec_time_ticks'])!=end-begin:raise ValueError('Helper makespan disagrees with instructions')
        if not math.isclose(float(info['exec_time_ns']), (end-begin)/freq,rel_tol=1e-10):raise ValueError('Tick/ns conversion mismatch')
        cores=[]
        for cid,typ,n,a,z,mmad in db.execute('''SELECT ExecInstrCoreId,CoreTypeId,COUNT(*),MIN(ExecInstrTickStart),MAX(ExecInstrTickEnd),SUM(ExecInstrName='MMAD')
             FROM ExecutedInstructions GROUP BY ExecInstrCoreId,CoreTypeId'''):
            if typ not in (1,2,3):raise ValueError('Unknown core type')
            cores.append({'core_id':cid,'core_type_id':typ,'name':f'core{cid}.'+('cubecore0' if typ==1 else f'veccore{typ-2}'),
                'instruction_records':n,'start_ticks':a-begin,'span_ticks':z-a,'span_us':(z-a)/(freq*1000),'mmad_count':mmad})
        if len(cores)!=int(info['ai_core_active']):raise ValueError('Active core mismatch')
        # Streaming unions: retain only one interval per core/queue, not a JSON timeline.
        # Exclusion flags are helper contract facts, not an inferred peak utilization.
        columns={r[1] for r in db.execute('PRAGMA table_info(ExecutedInstructions)')}
        if {'ExtendParams','ExecInstrIsDurationExcluded'}<=columns:
            intervals={};totals={}
            for cid,typ,queue,a,z in db.execute('''SELECT ExecInstrCoreId,CoreTypeId,
                    json_extract(ExtendParams,'$.queue_type'),ExecInstrTickStart,ExecInstrTickEnd
                    FROM ExecutedInstructions WHERE ExecInstrIsDurationExcluded=0
                    ORDER BY ExecInstrTickStart'''):
                key=(cid,typ,queue)
                if not queue:raise ValueError('Missing instruction queue classification')
                if key not in intervals:intervals[key]=(a,z);totals[key]=0
                else:
                    left,right=intervals[key]
                    if a>right:totals[key]+=right-left;intervals[key]=(a,z)
                    else:intervals[key]=(left,max(right,z))
            for key,(a,z) in intervals.items():totals[key]+=z-a
            for core in cores:
                core['nonexcluded_instruction_union_us_by_queue']={q:t/(freq*1000) for (cid,typ,q),t in totals.items()
                    if (cid,typ)==(core['core_id'],core['core_type_id'])}
    return {'reference_us':(end-begin)/(freq*1000),'reference_scope':'cross-core instruction makespan',
        'kernel':info['name'],'frequency_ghz':freq,'block_dim':info.get('block_dim'),'helper_info':info,
        'cores':cores,'active_aic':sum(c['core_type_id']==1 for c in cores),'active_aiv':sum(c['core_type_id']!=1 for c in cores),
        'max_core_mmad':max(c['mmad_count'] for c in cores),'db_sha256':sha256(path),'db_bytes':path.stat().st_size,'db_path':str(path),
        'correctness':re.findall(r'correctness elements=[^\r\n]+',log),
        'kernel_binary_sha256':None,'kernel_identity_limit':'helper DB identifies entry name and blockDim; runtime kernel object was not exported',
        'capture_scope':'full instruction DB, aggregate-only SQL; no new JSON trace',
        'pipeline_scope':'Union of helper non-duration-excluded instruction intervals per queue; not independent ISA efficiency'}
