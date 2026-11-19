"""Keep every original outcome and permit a same-input retry for official parser transient output caps."""
from datetime import datetime
import sys
import time
from register import DT, PR, OUT, RAW, ROOT, digest, read, utc, write
from collect import capture, qualify, error_metrics, make_pair, summarize


def original_jobs(freeze):
    return [(j,c,RAW/(j['id']+'_'+c)) for j in freeze['jobs'] for c in (DT,PR)]


def add_retries(registration,freeze):
    regfile=OUT/'recovery-registration.json'
    policy=read(regfile) if regfile.exists() else dict(utc=utc(),original_freeze_sha256=digest(OUT/'freeze.json'),
        authorization='Same input and previously frozen forecasts only. Keep original failure and wall bound. Permit one extra capture per transient raw-cap interruption; no numeric failures or alternative shapes. Limit all live occupancy to campaign 24 GiB, retain <=4 concurrent jobs.',jobs=[])
    registered={r['base_attempt'] for r in policy['jobs']}
    for j,chip,d in original_jobs(freeze):
        if d.name in registered or not(d/'outcome.json').exists():continue
        failed=read(d/'outcome.json');log=(d/'run.log').read_text(errors='replace')
        if failed['eligible'] or failed['stop_reason']!='raw cap' or 'SMOKE_FAIL' in log:continue
        plan=dict(base_attempt=d.name,op=j['op'],shape=j['shape'],chip=chip,base_cost_sha256=digest(d/'cost.json'),
            original_raw_mib=j['raw_mib'],retry_raw_mib=int(j['raw_mib']*1.5),wall_s=j['wall_s'],utc=utc(),
            reason='The original process exited on a raw-output cap. The partially produced files are not a complete eligible execution.',
            original_finished_bytes=failed['raw_bytes'],original_parser_finish_logged='Profiling data parse finished.' in log)
        policy['jobs'].append(plan)
    write(regfile,policy)
    return policy


def measure(freeze,policy,registration):
    alljobs=original_jobs(freeze)
    for plan in policy['jobs']:
        j,chip,d=next(x for x in alljobs if x[2].name==plan['base_attempt'])
        dest=RAW/'recovery'/d.name
        if (dest/'outcome.json').exists():continue
        pending=[x for x in alljobs if not(x[2]/'outcome.json').exists()]
        # The primary scheduler may fill all remaining slots. Start only when the primary
        # queue has <=3 remaining inputs and every remaining upper bound fits beside this retry.
        charged=sum(read(path)['raw_bytes']for path in RAW.glob('*/cost.json'))
        charged+=sum(read(path)['raw_bytes']for path in RAW.glob('recovery/*/cost.json'))
        reserved=sum(row['raw_mib']*2**20 for row,_,base in pending if not(base/'cost.json').exists())
        if len(pending)>3 or charged+reserved+plan['retry_raw_mib']*2**20>registration['total_raw_gib']*2**30:continue
        elapsed=(datetime.fromisoformat(utc())-min(datetime.fromisoformat(read(path)['started_utc'])for path in RAW.glob('*/cost.json'))).total_seconds()
        if elapsed+j['wall_s']>registration['total_wall_s']:raise RuntimeError('combined campaign wall cap')
        dest.parent.mkdir(exist_ok=True)
        recovery_job=dict(j,raw_mib=plan['retry_raw_mib']);cost=capture(recovery_job,chip,dest,freeze)
        try:
            row,=qualify(j,chip,dest)
            contract=[l for l in (dest/'run.log').read_text(errors='replace').splitlines()if l.startswith('case=')]
            assert len(contract)==1
            row.update(role='both DT and PR fresh frozen holdout; same-input raw-cap recovery',backend='msopprof official full trace',dtype='bf16',input_contract_log=contract,runner_sha256=j['runner_sha256'],host_args=j['args'],run_log=str((dest/'run.log').relative_to(ROOT)))
            write(dest/'qualified.json',row);reason=None
        except (ValueError,AssertionError,OSError,KeyError,IndexError)as exc:reason=repr(exc)
        write(dest/'outcome.json',dict(id=j['id'],op=j['op'],shape=j['shape'],chip=chip,**cost,eligible=reason is None,reason=reason))
        print('recovery',d.name,reason or 'qualified',flush=True)


def final(freeze,policy):
    original=OUT/'primary-summary.json'
    if not original.exists():write(original,read(OUT/'campaign-completion.json'))
    rs=[];ps=[];costs=[]
    for j in freeze['jobs']:
        side={}
        for chip in (DT,PR):
            base=RAW/(j['id']+'_'+chip)
            for d in (base,RAW/'recovery'/base.name):
                if not(d/'outcome.json').exists():continue
                o=read(d/'outcome.json');o['attempt']=str(d.relative_to(RAW));costs.append(o)
                if o['eligible']:side[chip]=read(d/'qualified.json')
            if chip in side:rs.append(side[chip])
        if len(side)==2:ps.append(make_pair(j,side[DT],side[PR]))
    assert len(costs)==2*len(freeze['jobs'])+len(policy['jobs'])
    rejects=[r for r in costs if not r['eligible']]
    for n,obj in [('records',rs),('pairs',ps),('all-costs',costs),('rejections',rejects)]:write(OUT/(n+'.json'),obj)
    by_op={}
    for op in ('rmsnorm','quant'):
        subset=[p for p in ps if p['op']==op];by_op[op]={'all':error_metrics(subset)}
        for grp in sorted({p['regime']for p in subset}):by_op[op][grp]=error_metrics([p for p in subset if p['regime']==grp])
    write(OUT/'paired-errors.json',by_op)
    from trace_evidence import main as reduce_traces
    reduce_traces()
    assert len(read(OUT/'trace-evidence.json')['rows'])==len(rs)
    summary=dict(utc=utc(),**summarize(costs),pairs=len(ps),new_both_ends=True,original_physical=2*len(freeze['jobs']),registered_extra_attempts=len(policy['jobs']),failed_attempts=len(rejects),
        counts={op:len([p for p in ps if p['op']==op])for op in by_op}, files={n:digest(OUT/n)for n in ['registration.json','freeze.json','recovery-registration.json','pairs.json','records.json','all-costs.json','rejections.json','paired-errors.json','query-wall.json','trace-evidence.json','primary-summary.json']})
    write(OUT/'campaign-completion.json',summary);write(OUT/'final-completion.json',summary)


def main():
    b=read(OUT/'freeze.json');reg=read(OUT/'registration.json')
    while True:
        policy=add_retries(reg,b);measure(b,policy,reg)
        if (OUT/'campaign-completion.json').exists()and all((RAW/'recovery'/p['base_attempt']/'outcome.json').exists()for p in policy['jobs']):
            # The last original attempt may have failed while another retry was running.
            updated=add_retries(reg,b)
            if len(updated['jobs'])==len(policy['jobs']):final(b,updated);return
        time.sleep(5)

if __name__=='__main__':main()
