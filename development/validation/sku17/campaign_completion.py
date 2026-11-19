"""Validate all registered outcomes and issue the final marker after collection and retries."""
from collections import Counter
from datetime import datetime, timedelta
from campaign import DT, PR, OUT, RAW, ROOT, forecast, read, sha, utc
from campaign_publish import publish, atomic_record
from campaign_control_report import controls


def summarize(costs):
    start = [datetime.fromisoformat(x['started_utc']) for x in costs]
    end = [t + timedelta(seconds=x['seconds']) for t, x in zip(start, costs)]
    return {
        'physical_attempts': len(costs),
        'qualified_attempts': sum(x['eligible'] for x in costs),
        'process_wall_s_sum': sum(x['seconds'] for x in costs),
        'calendar_span_s_including_inter_attempt_gaps': (max(end)-min(start)).total_seconds(),
        'cpu_s_accounted_lower_bound': sum(x['cpu_seconds'] or 0 for x in costs),
        'cpu_accounting_missing_attempts': sum(x['cpu_seconds'] is None for x in costs),
        'raw_bytes_at_process_exit': sum(x['raw_bytes'] for x in costs),
        'failed_attempt_wall_s_sum': sum(x['seconds'] for x in costs if not x['eligible']),
        'failed_attempt_raw_bytes': sum(x['raw_bytes'] for x in costs if not x['eligible']),
    }


def complete():
    expected = set()
    expected_pairs = Counter()
    audit = []
    for group in ('core', 'extension'):
        b = read(OUT / (group+'-freeze.json'))
        read(OUT / (group+'-cost.json'))
        for a in read(OUT/(group+'-recovery-list.json'))['ids']:
            expected.add((group, a+'-recovery'))
        for j in b['jobs']:
            assert sha(ROOT / j['runner']) == j['runner_sha256']
            for o in j['outputs']:
                current=forecast(o['op'],o['shape'],o['attrs'])
                assert all(v==o[k] for k,v in current.items()), (o['op'],o['shape'],'frozen forecast drift')
                if o['op'] != 'kda_chunk':expected_pairs[o['op']]+=1
            expected.add((group,j['id']+'_'+PR))
            if group=='core' and not j['outputs'][0]['dt']:
                expected.add((group,j['id']+'_'+DT))
        audit.append({'group':group,'frozen_jobs':len(b['jobs']),'freeze_sha256':sha(OUT/(group+'-freeze.json'))})
    for name in read(OUT/'paired-control-budget.json')['ids']:
        expected.add(('extension',name+'_'+DT))
    pending=[(g,d)for g,d in expected if not(RAW/g/d/'cost.json').exists()]
    if pending:raise SystemExit(f'{len(pending)} registered physical attempts still pending: {sorted(pending)}')
    publish();ctrl=controls()
    costs=read(OUT/'all-costs.json');pairs=read(OUT/'pairs.json')
    actual={(x['group'],x['attempt']) for x in costs}
    assert actual==expected and len(actual)==len(costs), (actual^expected)
    for p in pairs:
        assert p['dt']['chip']==DT and p['pr']['chip']==PR
        assert all(p['dt'][k]==p['pr'][k] for k in ('op','shape','attrs','api'))
        assert p['dt_ca_us']>0 and p['pr_ca_us']>0 and p['speedup_ca']==p['dt_ca_us']/p['pr_ca_us']
    counts=Counter(p['op'] for p in pairs)
    gaps={op:n-counts[op] for op,n in expected_pairs.items() if counts[op]!=n}
    result={'utc':utc(),'expected_pairs_by_op':expected_pairs,'qualified_pairs_by_op':counts,'gaps_excluding_structural_chunk':gaps,
      'qualified_pairs':len(pairs),'matching_backend_pairs':sum(p['backend_match'] for p in pairs),
      'fresh_DT_controls_logical_rows':len(ctrl),'failed_attempts':len(read(OUT/'rejections.json')),
      'frozen_prediction_drift_count':0,'registration':audit,'overall':summarize(costs),
      'by_group':{g:summarize([x for x in costs if x['group']==g]) for g in ('extension','core')},
      'file_sha256':{p:sha(OUT/p) for p in ('all-costs.json','pairs.json','rejections.json','records.json','dt-control-comparison.json','campaign-mmad.json')},
      'chunk_status':'diagnostic attempted on PR; no qualified timing on failure','capture_scope':'Physical attempt totals include new DT controls and retries, excluding reused historical CA.'}
    assert all((OUT/k).is_file() for k in result['file_sha256'])
    atomic_record(OUT/'campaign-completion.json',result)
    print(result)

if __name__ == '__main__':complete()
