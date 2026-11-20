"""Preserve independently checkable first-launch observations from timed-out pairs."""
import hashlib
import json
from pathlib import Path
import re
import sys

ROOT=Path(__file__).resolve().parents[3]
sys.path.insert(0,str(ROOT))
from development.validation.post24.collect import inputs,inspect_window,tables

OUT=ROOT/'results/v0.28/post'
EXPECTED='MhcPost_59130ccee6717ff2c0f15b485e731162_5'


def main():
    registered=json.loads((OUT/'registration.json').read_text())['jobs']
    previous=OUT/'partial-evidence.json'
    seen=json.loads(previous.read_text())['jobs'] if previous.exists() else {}
    for job in registered:
        path=OUT/'receipts'/(job['id']+'.json')
        if job['id'] in seen or not path.exists():
            continue
        receipt=json.loads(path.read_text())
        if receipt.get('rows'):
            continue
        folder=ROOT/receipt['raw']; raw=(folder/'run.log').read_bytes(); log=raw.decode(errors='replace')
        launched=re.findall(r'Start profiling on kernel: (\S+)',log)
        durations=tables(log)
        scopes=dict(re.findall(r'^BATCH_BEGIN (\d+)\s*$\n(.*?)^BATCH_END \1\s*$',log,re.M|re.S))
        sku=job['sku']
        sku_ok=(re.findall(r'^runtime_soc=(\S+)',log,re.M)==[sku] and
                ('lpddr5_6400M_bailu.ini' if 'PR' in sku else 'hbm3_8000M.ini') in log)
        pair=dict(receipt=str(path.relative_to(ROOT)), host_log_sha256=hashlib.sha256(raw).hexdigest(),
                  final_stop_reason=receipt['cost']['stop_reason'], launched=launched, tables=len(durations),
                  closed_host_scopes=sorted(scopes), observed=[])
        seen[job['id']]=pair
        if not sku_ok or (folder/'batch.txt').read_text()!=inputs(job['rows']):
            pair['reason']='target or input sequence mismatch'
            continue
        for i,row in enumerate(job['rows']):
            t,n,d=row['shape']
            if str(i) not in scopes or i>=len(launched) or i>=len(durations) or launched[i]!=EXPECTED:
                continue
            scope=scopes[str(i)]
            expected=f'case=mhc_post api=aclnnMhcPost T={t} N={n} D={d} dtype=BF16'
            golden=re.findall(r'^correctness name=output count=(\d+) bad=(\d+)\b',scope,re.M)
            if expected not in scope or golden!=[(str(t*n*d),'0')]: continue
            paths=list(folder.glob(f'profile/**/{EXPECTED}/{i}/simulator/trace.json'))
            if len(paths)!=1: continue
            try:stage=inspect_window(paths[0].parents[1],durations[i],EXPECTED,row['shape'])
            except (ValueError,KeyError,OSError) as err:
                pair['inspection_error']=str(err);continue
            if stage['active_aiv']!=row['active']:continue
            pair['observed'].append(dict(ordinal=i, role=row['role'], shape=row['shape'], full_output_elements=t*n*d,
                                         frozen_prediction_us=row['candidate_us'],stage=stage))
    result=dict(scope='Diagnostic only. Locally closed host scope, correct whole output, installed ELF, per-lane ends, durations and write volume. Process and pair failed the frozen global completion gate. All entries excluded from formal validation.',
                jobs=seen)
    tmp=previous.with_suffix('.tmp')
    tmp.write_text(json.dumps(result,indent=2)+'\n');tmp.replace(previous)


if __name__=='__main__':
    main()
