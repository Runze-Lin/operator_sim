"""Score the frozen schedule against authenticated archived and prospective windows."""
from hashlib import sha256
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[3]
OUT = ROOT / 'results/v0.34/indexer'
if len(sys.argv) > 1:
    sys.path.insert(0, str(Path(sys.argv[1]).resolve()))
else:
    sys.path.insert(0, str(ROOT))
from operator_model import estimate
from operator_model.indexer_pipeline import _prices

CHIPS = ('Ascend950DT_9582', 'Ascend950PR_957b', 'Ascend970DT')


def read(path):
    return json.loads(path.read_bytes())


def sha(path):
    return sha256(path.read_bytes()).hexdigest()


def totals(rows, field):
    if not rows:
        return None
    residuals = [row[field]-row['ca_us'] for row in rows]
    total_ca = sum(r['ca_us'] for r in rows)
    return dict(windows=len(rows), shapes=len({tuple(r['shape']) for r in rows}),
                wape=100*sum(map(abs,residuals))/total_ca,
                max_error=100*max(abs(d)/r['ca_us'] for d,r in zip(residuals,rows)),
                bias=100*sum(residuals)/total_ca)


def archive():
    previous = {r['identity'][0]:r for r in read(ROOT/'results/v0.32/replay.json')['rows']
                if r['op']=='lightning_indexer'}
    rows = []
    for obs in read(OUT/'physical.json'):
        baseline = previous.pop(obs['sha'])
        chip, shape = obs['chip'], obs['shape']
        result = estimate('lightning_indexer',shape,baseline['attrs'],chip=chip)
        plan = result.projection['target_plan']
        assert baseline['chip']==chip and baseline['shape']==shape
        assert plan['mmads_by_shape']==obs['instruction_shapes']
        assert (result.active_aic,result.active_aiv)==(obs['useful_aic'],obs['histogram_aiv'])
        assert plan['modeled_histogram_instructions']==obs['total_histogram_instructions']
        assert abs(baseline['ca_us']-obs['duration_ticks']/1650) < 2/1650
        rows.append(dict(shape=shape,chip=chip,kernel=obs['kernel'],trace_sha256=obs['sha'],
                         role='DT stage calibration' if obs['sha'] in _prices()['source_sha256'] else 'archive replay',
                         prediction_us=result.latency_us,ca_us=baseline['ca_us'],v032_us=baseline['us'],
                         future_970dt_us=estimate('lightning_indexer',shape,chip=CHIPS[2]).latency_us,
                         signed_error_pct=100*(result.latency_us/baseline['ca_us']-1),
                         useful_aic=result.active_aic,histogram_aiv=result.active_aiv,launched=obs['launched'],
                         mmads=plan['mmad_count'],histograms=plan['modeled_histogram_instructions']))
    assert not previous
    return rows


def prospective():
    reg_path = OUT/'freeze.json'
    reg = read(reg_path)
    other = ROOT/'results/v0.29/large/fusion'
    for forecast in reg['forecasts']:
        shape = forecast['shape']
        for chip in CHIPS[:2]:
            local = shape==reg['job']['shape']
            path = OUT/(chip+'.json') if local else other/('index-32ki-'+chip+'.json')
            price = forecast['prices'][chip]['us']
            estimate_now = estimate('lightning_indexer',shape,chip=chip)
            assert abs(price-estimate_now.latency_us)<1e-8
            row = dict(shape=shape,chip=chip,prediction_us=price,
                       future_970dt_us=forecast['prices'][CHIPS[2]]['us'],ca_us=None,
                       role='pending',receipt_path=str(path.relative_to(ROOT)))
            if path.exists():
                receipt=read(path)
                assert receipt['chip']==chip
                if local:
                    assert receipt['freeze_sha256']==sha(reg_path) and receipt['shape']==shape
                else:
                    registration=other/'freeze.json'
                    assert receipt['registration_sha256']==sha(registration)
                    job,=[j for j in read(registration)['jobs'] if j['id']==receipt['id']]
                    assert job['shape']==shape and job['op']=='lightning_indexer'
                verdict=receipt['verdict'] if local else receipt
                row.update(capture_seconds=receipt['cost']['seconds'],receipt_sha256=sha(path),role='unqualified')
                if verdict['qualified']:
                    assert not receipt['cost']['stop_reason'] and receipt['cost']['exit_code']==0
                    stage,=verdict['stages']; assert stage['physical_end_all_lanes'] and stage['provider_matches']
                    assert verdict['golden']=={'topk_set':shape[1]*512}
                    ca=verdict['ca_us']['lightning_indexer']
                    row.update(ca_us=ca,role='independent' if receipt['cost']['started_utc']>reg['utc'] else 'archive replay',
                               signed_error_pct=100*(price/ca-1),kernel=stage['kernel'])
            yield row


def main():
    rows=archive()
    new=list(prospective())
    stats={chip:{label:totals([r for r in rows if r['chip']==chip],field) for label,field in
                 (('v032','v032_us'),('current','prediction_us'))} for chip in CHIPS[:2]}
    for chip,st in stats.items():
        st['independent']=totals([r for r in new if r['chip']==chip and r['role']=='independent'],'prediction_us')
    obj=dict(freeze_sha256=sha(OUT/'freeze.json'),summary=stats,rows=rows,prospective=new)
    (OUT/'score.json').write_text(json.dumps(obj,indent=2)+'\n')
    for chip in CHIPS[:2]:
        print(chip,stats[chip])
    print('Prospective endpoints:',[r['role'] for r in new])


if __name__=='__main__':
    main()
