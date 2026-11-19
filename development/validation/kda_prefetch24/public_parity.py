"""Compare the installed public scheduling contract to the independent frozen source."""
from pathlib import Path
import json
import sys
from time import perf_counter_ns

from evidence import ROOT, OUT, read, sha
from frozen import engine, predict

sys.path.insert(0, str(ROOT))
from operator_model import estimate

SKUS = ('Ascend950DT_9582', 'Ascend950PR_957b', 'Ascend970DT')
UNTOUCHED = ('dma_bytes', 'dma_descriptors', 'mmads', 'issued_cube_ops_by_dtype',
             'estimated_rvec_subqueue_records', 'active_aic', 'launched_aiv',
             'first_payload_cycles', 'critical_core_head_chunks', 'max_core_head_chunks',
             'hbm_spill_scenario_us', 'representative_tile_phase_cycles')


def verify():
    comparison = read(OUT/'comparison.json')
    shapes = {tuple(r['shape']) for r in comparison['rows']+comparison['timings']}
    archival = engine()[0]
    rows = []
    for shape in sorted(shapes):
        for sku in SKUS:
            old_plan = archival['schedule'](shape, archival['load_chip'](sku))
            registered = predict(shape, sku)
            started = perf_counter_ns()
            out = estimate('kda_chunk', shape, {'implementation': 'fla_26_9_1_v2'}, chip=sku)
            wall_ms = (perf_counter_ns()-started)/1e6
            stages = out.projection['target_plan']['stages']
            assert out.latency_us == registered['us']
            assert [s['device_us'] for s in stages] == registered['stages_us']
            assert stages[1]['prefetch_timeline_cycles'] == registered['fwd_h_trace_model_ticks']
            assert out.status == 'extrapolated' and out.projection['qualification'] == 'experimental'
            assert all(s[k] == old[k] for s,old in zip(stages,old_plan['stages']) for k in UNTOUCHED)
            for i in (0,2):
                assert stages[i] == old_plan['stages'][i]
            rows.append(dict(shape=shape,sku=sku,device_us=out.latency_us,stage_us=registered['stages_us'],
                             fwdh_proof_domain=stages[1]['prefetch_domain'],wall_ms=wall_ms))
    return dict(register_sha256=sha(OUT/'register.json'), unchanged_fields=UNTOUCHED,
                identical_public_and_frozen_cases=len(rows), max_query_ms=max(r['wall_ms'] for r in rows), rows=rows)


if __name__=='__main__':
    result=verify()
    (OUT/'public-parity.json').write_text(json.dumps(result,indent=2)+'\n')
    print(result['identical_public_and_frozen_cases'],result['max_query_ms'])
