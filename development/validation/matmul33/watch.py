"""Observe a previously frozen 64Ki slot without touching its budget or provider files."""
from datetime import datetime, timezone
import hashlib,json,os,re,sys,time
from pathlib import Path

ROOT=Path(__file__).resolve().parents[3];OUT=ROOT/'results/v0.33/matmul'
REG=ROOT/'results/v0.30/matmul-dispatch/device/freeze.json';MODEL=ROOT/'results/v0.32/matmul/freeze.json';INPUT=ROOT/'runs/matmul_dispatch30/device/524288-dt';DT='Ascend950DT_9582';SHAPE=[524288,128,512]
def read(p):return json.loads(p.read_bytes())
def digest(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def save(filename,obj):
    p=OUT/filename;tmp=p.with_suffix(f'.{os.getpid()}.tmp');tmp.write_text(json.dumps(obj,indent=2,ensure_ascii=False)+'\n');tmp.replace(p)
def now():return datetime.now(timezone.utc).isoformat()
def register():
    job,=[r for r in read(REG)['rows'] if r['shape']==SHAPE];md,=[r for r in read(MODEL)['rows'] if r['shape']==SHAPE and r['chip']==DT]
    with (OUT/'monitor-freeze.json').open('x') as f:json.dump(dict(utc=now(),shape=SHAPE,chip=DT,gold_count=SHAPE[0]*SHAPE[1],acquisition_budget=dict(seconds=job['wall_s'],mib=job['raw_mib']),price_us=md['us'],expected_mmad=md['plan']['mmads_total'],aic=md['aic'],aiv=md['aiv'],acquisition_sha256=digest(REG),model_sha256=digest(MODEL),official_elf=read(ROOT/'results/v0.31/matmul/kimi-prefill/freeze.json')['installed_matmul_elf'],phase='kernel already running and no official receipts were present',qualification='Only a previously frozen provider receipt plus verified original shape, all exact BF16 outputs, exact SKU/installed ELF, physical per-core END and retirement, official stage table and clean bounded exit. STARS, progress counter, inherited time and partial DB do not qualify. The monitor does not launch, suspend or prolong any CA.',process_chain=[235651,309994,309995,309999,310050],diagnostic_comparators='Existing completed devices only: STARS task span and official longest per-core duration as distinct proxies; compare per-core/aggregate timestamp window on matched trace; npusim record requires identical hardware/config/ELF and gold before substitution.'),f,indent=2,ensure_ascii=False);f.write('\n')
def recover_original_receipt():
    # One frozen exporter omitted its own case label when rechecking asynchronous drains.
    # Reapply exactly that archived checker and keep its supplement local to this version.
    local=OUT/'64ki-exporter-recovery.json'
    if local.exists():return local,read(local)
    initial=REG.parent/'524288-dt-initial.json'
    if not initial.exists():return None,None
    original=read(initial);assert original['chip']==DT and original['shape']==SHAPE
    if original.get('reason')!="ValueError('end marker disagrees with all-core window')":
        raise ValueError(f'Unexpected original validation error: {original.get("reason")}')
    sys.path.insert(0,str(ROOT))
    from development.validation.matmul_dispatch30.device import recover
    job,=[row for row in read(REG)['rows'] if row['shape']==SHAPE]
    d=recover(job,dict(original,case=initial.stem),INPUT,acquisition=initial)
    assert d and d['qualified']
    d.update(registration_sha256=digest(REG),shape=SHAPE,chip=DT,original_result=str(initial.relative_to(ROOT)),
             original_sha256=digest(initial),cost=read(INPUT/'cost.json'))
    save(local.name,d)
    return local,d

def status():
    reg=read(OUT/'monitor-freeze.json');assert reg['model_sha256']==digest(MODEL) and reg['acquisition_sha256']==digest(REG)
    log=(INPUT/'run.log').read_text(errors='replace');gold=f"correctness count={reg['gold_count']} bad=0" in log
    cycles=re.findall(r'Cycle Count: (\d+)',log)
    ps=[]
    for pid in reg['process_chain']:
        stat=Path(f'/proc/{pid}/stat');comm=Path(f'/proc/{pid}/comm')
        if stat.exists() and comm.exists():
            state=stat.read_text().split(') ')[-1].split();ps.append(dict(pid=pid,comm=comm.read_text().strip(),state=state[0],user_jiffies=int(state[11]),system_jiffies=int(state[12])))
    costpath=INPUT/'cost.json';cost=read(costpath) if costpath.exists() else None
    receipt=None;reason=None;phase='simulator running';source=None;timing=None;checks=None
    for s in ('524288-dt-retired','524288-dt'):
        q=REG.parent/(s+'.json')
        if not q.exists():continue
        d=read(q)
        if d['qualified']:receipt=d;source=q;break
        reason=d.get('reason')
    if receipt is None and reason=="KeyError('case')" and cost and not cost['exit_code'] and not cost['stop_reason']:
        source,receipt=recover_original_receipt()
    if receipt:
        tr,=receipt['stages'];terminal=receipt.get('terminal');assert cost and not cost['exit_code'] and not cost['stop_reason'] and gold
        assert receipt['registration_sha256']==digest(REG) and receipt['shape']==SHAPE and receipt['chip']==DT
        assert tr['binary_sha256']==reg['official_elf'] and receipt['correctness']==[f"correctness count={reg['gold_count']} bad=0"]
        assert tr['active_aic']+tr['active_aiv']==len(tr['windows']) and abs(tr['ca_us']-receipt['ca_us'])<1e-9
        if terminal is not None:assert set(terminal)==set(tr['windows']) and all(v['physical_end']<=v['retire_tick'] for v in terminal.values())
        assert abs(max(max(v) for v in tr['windows'].values())-min(min(v) for v in tr['windows'].values())-tr['ca_us'])<1e-5
        timing=receipt['ca_us'];checks=dict(expected_mmad=reg['expected_mmad'],observed_mmad=tr['mmad'],aic=tr['active_aic'],aiv=tr['active_aiv'],roster_matches=(reg['aic'],reg['aiv'])==(tr['active_aic'],tr['active_aiv']),observed_kernel=tr['kernel'],all_output_gold=True,trace_sha256=tr['trace_sha256']);phase='qualified'
    elif cost and (cost['exit_code'] or cost['stop_reason']):phase='finished without a qualified full device window';reason=repr((cost['exit_code'],cost['stop_reason']))
    elif reason and reason!="KeyError('case')":phase='finished without a qualified full device window'
    elif cost:phase='waiting for official per-core retirement qualification'
    elif 'TASK_DONE' in log:phase='task completion marker present; export or numerical readback in progress'
    snapshot=dict(utc=now(),phase=phase,status_process_chain=ps,model_us=reg['price_us'],ca_us=timing,signed_pct=100*(reg['price_us']/timing-1) if timing else None,last_emitted_simulator_counter=int(cycles[-1]) if cycles else None,log_mtime_utc=datetime.fromtimestamp((INPUT/'run.log').stat().st_mtime,timezone.utc).isoformat(),kernel_started='[TASK_BEGIN]' in log,gold_seen=gold,official_report=cost,failure=reason,checks=checks,receipt=str(source.relative_to(ROOT)) if source else None,receipt_sha256=digest(source) if source else None)
    save('64ki-state.json',snapshot)
    if phase=='qualified':
        from reduce import run
        from report import main as report
        name='524288-128-512-dt-certificate.json'
        if not (OUT/name).exists():save(name,run(source,INPUT))
        cert=read(OUT/name)
        assert cert['source_log_sha256']==digest(INPUT/'run.log')
        assert cert['compact_us']==timing or abs(cert['compact_us']-timing)<2/1650
        snapshot['reduced_certificate']=name
        snapshot['reduced_certificate_sha256']=digest(OUT/name)
        report()
    if phase=='qualified' or phase.startswith('finished without'):
        save('64ki-state.json',snapshot)
        save('64ki-final.json',snapshot)
        return True
    return False
if __name__=='__main__':
    if sys.argv[1]=='register':register()
    elif sys.argv[1]=='once':status()
    else:
        while True:
            try:
                if status():break
            except (ValueError,KeyError,IndexError,AssertionError,OSError) as exc:
                import traceback
                snapshot=dict(utc=now(),phase='observer encountered validation error',error=repr(exc),
                              detail=traceback.format_exc(),ca_us=None)
                save('64ki-observer-error.json',snapshot)
                if (INPUT/'cost.json').exists():
                    save('64ki-state.json',snapshot);save('64ki-final.json',snapshot);break
            time.sleep(45)
