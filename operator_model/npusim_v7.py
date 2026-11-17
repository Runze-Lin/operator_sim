"""Read complete helper DBs, keeping kernel scopes separate from API wall time."""
import math
from pathlib import Path
import re
import sqlite3
from .ca import sha256


def extract_db(path):
    path=Path(path)
    with sqlite3.connect(path.resolve().as_uri()+'?mode=ro',uri=True) as db:
        if db.execute('PRAGMA quick_check').fetchone()[0]!='ok':raise ValueError('Corrupt DB')
        info=dict(db.execute('SELECT KernelInfoAttr,KernelInfoVal FROM KernelInfo'))
        if str(info.get('helper_version'))!='0.8.0' or int(info.get('contract_version',-1))!=7:
            raise ValueError('Unknown helper contract')
        freq=float(info['frequency_ghz']);n,start,end,bad=db.execute('SELECT COUNT(*),MIN(ExecInstrTickStart),MAX(ExecInstrTickEnd),SUM(ExecInstrTickEnd<ExecInstrTickStart) FROM ExecutedInstructions').fetchone()
        if not math.isclose(freq,1.65) or not n or bad or int(info['instrs_exec_num'])!=n:raise ValueError('Invalid instruction population')
        if int(info['exec_time_ticks'])!=end-start:raise ValueError('Makespan mismatch')
        if not math.isclose(float(info['exec_time_ns']),(end-start)/freq,rel_tol=1e-10):raise ValueError('Clock mismatch')
        cores=[]
        for cid,typ,num,a,z,mmad,rv in db.execute("SELECT ExecInstrCoreId,CoreTypeId,COUNT(*),MIN(ExecInstrTickStart),MAX(ExecInstrTickEnd),SUM(ExecInstrName='MMAD'),SUM(substr(ExecInstrName,1,3)='RV_') FROM ExecutedInstructions GROUP BY ExecInstrCoreId,CoreTypeId"):
            if typ not in (1,2,3):raise ValueError('Unknown core type')
            cores.append({'core_id':cid,'core_type_id':typ,'instruction_records':num,'start_ticks':a-start,'end_ticks':z-start,'span_us':(z-a)/1650,'mmad_count':mmad,'rv_count':rv})
        if len(cores)!=int(info['ai_core_active']):raise ValueError('Active core mismatch')
        spans={};sums={}
        for cid,typ,queue,a,z in db.execute("SELECT ExecInstrCoreId,CoreTypeId,json_extract(ExtendParams,'$.queue_type'),ExecInstrTickStart,ExecInstrTickEnd FROM ExecutedInstructions WHERE ExecInstrIsDurationExcluded=0 ORDER BY ExecInstrTickStart"):
            if not queue:raise ValueError('Missing queue')
            key=(cid,typ,queue)
            if key not in spans:spans[key]=(a,z);sums[key]=0
            else:
                l,r=spans[key]
                if a>r:sums[key]+=r-l;spans[key]=(a,z)
                else:spans[key]=(l,max(r,z))
        for key,(a,z) in spans.items():sums[key]+=z-a
        for c in cores:c['pipeline_union_us']={q:v/1650 for (cid,typ,q),v in sums.items() if (cid,typ)==(c['core_id'],c['core_type_id'])}
    return {'reference_us':(end-start)/1650,'reference_scope':'cross-core instruction makespan','kernel':info['name'],
        'frequency_ghz':freq,'block_dim':info.get('block_dim'),'helper_info':info,'cores':cores,
        'active_aic':sum(c['core_type_id']==1 for c in cores),'active_aiv':sum(c['core_type_id']!=1 for c in cores),
        'max_core_mmad':max(c['mmad_count'] for c in cores),'max_core_rv_count':max(c['rv_count'] for c in cores),
        'rv_count_scope':'helper ExecutedInstructions records; not historical msopprof paired log events',
        'db_path':str(path),'db_sha256':sha256(path),'db_bytes':path.stat().st_size,'kernel_binary_sha256':None}


def extract_run(profile,log_path,expected_groups):
    log=Path(log_path).read_text(errors='strict')
    if set(re.findall(r'runtime_soc=(\S+)',log))!={'Ascend950DT_9582'}:raise ValueError('Wrong SoC')
    if 'Simulation SUCCESS' not in log or 'SMOKE_PASS' not in log or 'SMOKE_FAIL' in log:raise ValueError('Incomplete simulation')
    if re.search(r'\[ERROR\].*(?:execute_set_flag|execute_wait_flag|deadlock|invalid instruction)',log):raise ValueError('CA synchronization/instruction error: timing unqualified')
    checks=re.findall(r'correctness [^\r\n]+',log)
    if len(checks)!=expected_groups or any(not re.search(r'\bbad=0(?:\s|$)',c) for c in checks):raise ValueError('Missing or failed correctness groups')
    if any(not re.search(r'\bcount=[1-9]\d*(?:\s|$)',c) for c in checks):raise ValueError('Empty numeric check')
    paths=sorted(Path(profile).glob('npusim_*/record/*.db'))
    if not paths:raise ValueError('No DB')
    kernels=[extract_db(p) for p in paths]
    stages=[]
    raw_paths=list(Path(profile).glob('npusim_*/npusim.log'))
    if len(raw_paths)>1:raise ValueError('Ambiguous runtime log')
    stage_log=raw_paths[0].read_text() if raw_paths else log
    if re.search(r'\[ERROR\].*(?:execute_set_flag|execute_wait_flag|deadlock|invalid instruction)',stage_log):raise ValueError('Raw CA synchronization/instruction error: timing unqualified')
    stage_matches=list(re.finditer(r'(?<![A-Za-z_])stage=(\w+) api=(\w+)',stage_log))
    if stage_matches:
        for ordinal,match in enumerate(stage_matches):
            name=match.group(1);end=stage_log.find('stage_complete='+name,match.end())
            if end<0:raise ValueError('Incomplete stage '+name)
            segment=stage_log[match.end():end]
            runtime_tasks=set(re.findall(r'\[TASK_DONE\].*?stream_id:(\d+), task_id:(\d+) TASK_DONE',segment))
            if raw_paths:
                if len(runtime_tasks)!=1:raise ValueError('Stage requires exactly one attributable runtime task: '+name)
                launch=ordinal
            else:
                ids=re.findall(r'phase → task (\d+) done',segment)
                if len(ids)!=1:raise ValueError('Stage requires exactly one attributable kernel: '+name)
                launch=int(ids[0])
            matches=[k for k in kernels if int(k['helper_info']['launch_id'])==launch]
            if len(matches)!=1:raise ValueError('Ambiguous launch id '+str(launch))
            stages.append({'op':name,'api':match.group(2),'launch_id':launch,'kernel':matches[0]['kernel'],'db_sha256':matches[0]['db_sha256'],
                           'runtime_tasks':sorted(runtime_tasks),'attribution':'synchronized API stage; unique runtime task; helper chronological launch id' if raw_paths else 'unique CLI phase launch id'})
        if len(stages)!=len(kernels) or len({s['launch_id'] for s in stages})!=len(stages):raise ValueError('Unattributed or duplicated stage kernels')
    return {'reference_us':sum(k['reference_us'] for k in kernels),
        'reference_scope':'cross-core instruction makespan' if len(kernels)==1 else 'serial kernel instruction sum (inter-launch gaps excluded)',
        'kernels':kernels,'kernel_count':len(kernels),'correctness':checks,'stages':stages,
        'scope_limit':'Sequential single-stream caller; sum excludes host launch, inter-launch idle and API setup; no cross-DB global-clock claim.'}
