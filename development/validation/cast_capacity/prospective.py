"""Freeze a supplementary candidate against future starts in an existing CA queue."""
from datetime import datetime, timezone
from hashlib import sha256
import json
from pathlib import Path
import sys

from replay import ROOT, OUT, DT, PR, calibration, read
from two_slot import query
FUTURE='Ascend970DT'
VECTOR=ROOT/'results/v0.21/vector'
RAW=ROOT/'runs/vector21'
FREEZE=OUT/'two-slot-freeze.json'


def digest(p):
    return sha256(p.read_bytes()).hexdigest()


def write_exclusive(path,data):
    with path.open('x') as stream:
        stream.write(json.dumps(data,indent=2)+'\n')


def freeze():
    if FREEZE.exists():raise FileExistsError(FREEZE)
    old=calibration();pars={k:old[k] for k in ('read_latency_factor','write_latency_factor')}
    reg=read(VECTOR/'freeze.json');runner=Path(__file__).resolve().parent
    hashes={str(p.relative_to(ROOT)):digest(p) for p in [
        runner/'two_slot.py',runner/'replay.py',Path(__file__).resolve(),VECTOR/'freeze.json',
        ROOT/'results/v0.20/large/tile-intervals.json',ROOT/'operator_model/cast_resource.py',
        ROOT/'operator_model/vector_memory.py',ROOT/'operator_model/rms_resource.py',ROOT/'operator_model/chips.py',
        ROOT/'operator_model/backend.py',*(ROOT/f'configs/chips/{c}.json' for c in (DT,PR,FUTURE))]}
    forecasts={}
    for j in reg['jobs']:
        if j['op']!='cast_bf16_fp32':continue
        fore={}
        for c in (DT,PR,FUTURE):
            pred=query(j['shape'],c,pars['read_latency_factor'],pars['write_latency_factor'])
            assert abs(pred['baseline_us']-j['targets'][c]['us'])<1e-8
            fore[c]={**pred,'v020_us':j['targets'][c]['old_us']}
        forecasts[j['id']]=dict(id=j['id'],shape=j['shape'],targets=fore)
    # Read only observation identity for eligibility. Do this after computing
    # forecasts to ensure both raw directories are still absent at sealing.
    receipts={(r['id'],r['chip']) for r in read(VECTOR/'holdout.json')}
    now=datetime.now(timezone.utc).isoformat()
    included=[];excluded=[]
    for id,row in forecasts.items():
        presence={c:dict(raw_dir=(RAW/(id+'_'+c)).exists(),receipt=(id,c) in receipts) for c in (DT,PR)}
        pair_started=any(any(end.values()) for end in presence.values())
        target=excluded if pair_started else included
        target.append({**row,'preexisting':presence,'role':'historical replay' if pair_started else 'prospective'})
    write_exclusive(FREEZE,dict(frozen_utc=now,model='two separate UB input/output slot response floors',
        parameters=pars,training='v0.20 DT [262144,32] full-tile descriptor latency only',
        source_sha256=hashes,registered_shapes=len(forecasts),new_ca_launches=0,
        decision='All shapes without a raw directory and a receipt at either endpoint are eligible. Before scoring, also require both actual start times later than the freeze. PR cache capacity hypothesis excluded.',included=included,excluded=excluded))
    print('frozen',now,'eligible',[(r['shape'],r['targets'][DT]['candidate_us']-r['targets'][DT]['baseline_us']) for r in included])


def metrics(rows,key):
    return None if not rows else dict(n=len(rows),wape=sum(abs(r[key]-r['ca_us']) for r in rows)/sum(r['ca_us'] for r in rows),
            max_ape=max(abs(r[key]/r['ca_us']-1) for r in rows))


def score():
    frozen=read(FREEZE)
    captures={(r['id'],r['chip']):r for r in read(VECTOR/'holdout.json')}
    rows=[];failed=[];pending=[]
    for role in ('included','excluded'):
        for j in frozen[role]:
            obs={c:captures.get((j['id'],c)) for c in (DT,PR)}
            # Refuse to call either side prospective if its paired start is uncertain.
            starts={}
            for c in (DT,PR):
                cost=(obs[c] or {}).get('cost')
                f=RAW/(j['id']+'_'+c)/'cost.json'
                if not cost and f.is_file():cost=read(f)
                if cost and cost.get('started_utc'):starts[c]=cost['started_utc']
            comparable=all(datetime.fromisoformat(t)>datetime.fromisoformat(frozen['frozen_utc']) for t in starts.values())
            for c in (DT,PR):
                r=obs[c];row=dict(id=j['id'],shape=j['shape'],chip=c,targets=j['targets'][c])
                if r is None:pending.append(row);continue
                if not r.get('qualified'):failed.append({**row,'reason':r.get('reason')});continue
                if role=='excluded' or not comparable:
                    label='historical replay'
                elif len(starts)!=2:
                    label='awaiting paired start confirmation'
                else:label='prospective'
                pred=j['targets'][c]
                row.update(role=label,ca_us=r['ca_us'],trace_sha256=r['stage']['trace_sha256'],api=r['api'],
                    correctness=r['correctness'],started_utc=starts.get(c),source='results/v0.21/vector/holdout.json',
                    baseline_us=pred['baseline_us'],candidate_us=pred['candidate_us'],v020_us=pred['v020_us'],
                    sensitive=abs(pred['baseline_us']-pred['candidate_us'])>1e-8,
                    projected_970_us=j['targets'][FUTURE]['candidate_us'])
                rows.append(row)
    groups={}
    for c in (DT,PR):
        groups[c]={r:{k:metrics([x for x in rows if x['role']==r and x['chip']==c],k) for k in ('baseline_us','candidate_us')}
                  for r in ('historical replay','prospective','awaiting paired start confirmation')}
    outfile=OUT/'two-slot-scores.json';tmp=outfile.with_suffix('.tmp')
    tmp.write_text(json.dumps(dict(freeze_sha256=digest(FREEZE),summary=groups,rows=rows,unqualified=failed,pending=pending),indent=2)+'\n');tmp.replace(outfile)
    print('scored',len(rows),len(failed),len(pending))


if __name__=='__main__':
    {'freeze':freeze,'score':score}[sys.argv[1]]()
