"""Publish terminal diagnostic receipts with exact simulated chip and stage eligibility."""
import json
import re
import sqlite3
from pathlib import Path
from probe import ART, ROOT, RUN, save, sha

rows=[]
for registration in sorted(ART.glob('*-registration.json')):
    tag=registration.name.removesuffix('-registration.json')
    contract=json.loads(registration.read_text())
    chips=contract['chips'] if 'chips' in contract else [contract['chip']]
    for chip in chips:
        receipt=ART/f'{tag}-{chip}.json'
        if not receipt.exists():
            continue
        data=json.loads(receipt.read_text())
        path=RUN/tag/chip/'run.log'
        if data['log_sha256']!=sha(path):
            raise ValueError(f'changed log {path}')
        log=path.read_text(errors='replace')
        runtime=re.search(r'runtime_soc=([\w]+)',log)
        launcher=f'use simulator {chip.split("_")[0]} {chip.split("_")[1]} by --soc-version'
        compact = contract.get('backend',data.get('backend','')).startswith('npusim')
        verified_backend = (f'SIM_SKU_ADAPTER target={chip} physical_model=Ascend950DT' in log
                            if compact else launcher in log)
        sku_ok=bool(runtime and runtime[1]==chip and verified_backend)
        completed='SMOKE_PASS' in log and data['exit']==0
        failed_pcs=re.findall(r'already has same set_flag! pc:(0x[0-9a-f]+)',log)
        timeout='The timeout has reached' in log or data.get('stop')=='wall budget'
        kernels=re.findall(r'Start profiling on kernel: (\w+)',log)
        if compact:
            for dbpath in sorted(path.parent.glob('profile/*/record/*.db')):
                with sqlite3.connect(dbpath.resolve().as_uri()+'?mode=ro',uri=True) as db:
                    info=dict(db.execute('select KernelInfoAttr,KernelInfoVal from KernelInfo'))
                    kernels.append(info.get('name',dbpath.stem+' (unfinished)'))
        rows.append(dict(tag=tag,shape=contract.get('cube_shape',contract.get('shape')),
            intended_chip=chip,runtime=runtime[1] if runtime else None,sku_verified=sku_ok,
            timeout=timeout,completed=completed,duplicate_set_count=len(failed_pcs),
            unique_set_error_pcs=sorted(set(failed_pcs)),kernels=kernels,wall_s=data.get('seconds',data.get('elapsed_s')),
            actual_checks=[v for v in log.splitlines() if 'correctness name=' in v or 'input_vk_unchanged=1' in v],
            ca_us=None, timing_qualified=False, evidence_sha256=sha(receipt),
            reason=('runtime SKU mismatch and unavailable profiler IPC' if not sku_ok else
                    'simulator duplicate SET_FLAG' if failed_pcs else 'registered wall budget' if data.get('stop')=='wall budget' else 'profiler timed out' if timeout else
                    'requires complete trace audit' if completed else 'host checks did not finish')))
save(ART/'probe-summary.json',{'roles':'mechanism diagnostics; no frozen timing predictions', 'rows':rows})
