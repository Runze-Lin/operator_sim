"""Freeze / replay the first two read issues separately from their service times."""
from datetime import datetime,timezone
from hashlib import sha256
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[3]
sys.path.insert(0,str(ROOT))
from operator_model import estimate,load_chip

OUT=ROOT/'results/v0.26/memory'
BASE=ROOT/'results/v0.23/prefill'
PRO=ROOT/'results/v0.23/pro-prefill'


def predict(shape,chip):
    e=estimate('cast_bf16_fp32',shape,chip=chip)
    d=e.projection['timing_diagnostics']
    return dict(shape=shape,sku=chip,issues_us=d['initial_read_issue_us'],cycles=d['initial_read_pair_cycles'],device_us=e.latency_us)


def main():
    params=BASE/'flash-32ki-dt.json'
    r=json.loads(params.read_text())
    observed=(r['ca_tiles'][1]['Read'][0]-r['ca_tiles'][0]['Read'][0])*1000*load_chip(r['chip']).frequency_ghz
    assert round(observed)==91
    if sys.argv[1]=='freeze':
        dst=OUT/'initial-issues-freeze.json'
        inputs=dict(pro256=([2097152,64],PRO,'pro-b8-256ki'),cast156=([851968,32],ROOT/'results/v0.25/cache','cast_bf16_fp32'))
        cases={}
        for name,(shape,folder,stem) in inputs.items():
            for sku,suffix in [('Ascend950DT_9582','dt'),('Ascend950PR_957b','pr'),('Ascend970DT','future')]:
                if suffix!='future' and any((folder/f'{stem}-{suffix}{end}.json').exists() for end in ('','-result')):
                    raise ValueError('outcome already available')
                cases[name+'-'+suffix]=predict(shape,sku)
        data=dict(utc=datetime.now(timezone.utc).isoformat(),calibration_source=str(params.relative_to(ROOT)),
                  registered='Both issue times are unconditional predictions relative to the whole-chip time origin. Compare them and the consecutive interval; predictions cover the initial pair only.',
                  parameter_source='Two independent UB input slots. Earliest Flash 32Ki DT control yields 91 target frontend cycles (round to nearest cycle). Same binary assumed for 970DT.',
                  sources={str(f.relative_to(ROOT)):sha256(f.read_bytes()).hexdigest() for f in (params,Path(__file__),ROOT/'operator_model/backend.py')},cases=cases)
        with dst.open('x') as f:
            json.dump(data,f,indent=2);f.write('\n')
    else:
        records=[]
        for src in sorted(BASE.glob('*.json'))+sorted(PRO.glob('pro-b8-128ki-*-blocks.json')):
            r=json.loads(src.read_text())
            if r['op']!='cast_bf16_fp32':
                continue
            e=predict(r['shape'],r['chip'])
            observed=[x['Read'][0] for x in r['ca_tiles'][:2]]
            service=[x['Read'][0] for x in r['model_tiles'][:2]]
            e.update(source=str(src.relative_to(ROOT)),role='calibration' if src==params else 'historical replay',
                     measured=observed,old_service=service,issue_errors_us=[x-y for x,y in zip(e['issues_us'],observed)])
            records.append(e)
        (OUT/'initial-issues-replay.json').write_text(json.dumps(records,indent=2)+'\n')


if __name__=='__main__':
    main()
