"""Register matched same-input repeats before any v0.23 CA launch."""
from hashlib import sha256
import json
from pathlib import Path
import sys
from time import perf_counter_ns
from datetime import datetime,timezone
ROOT=Path(__file__).resolve().parents[3];sys.path.insert(0,str(ROOT))
from operator_model import load_chip
from development.validation.rms_unknown23.candidate import forecast
from development.validation.sku19.campaign import write
OUT=ROOT/'results/rms-unknown23'
CHIPS=('Ascend950DT_9582','Ascend950PR_957b','Ascend970DT')
PROCESSES={
 'sector':[(1,672),(4,608),(17,2400),(34,1664),(55,2272)],
 'wide':[(2,2944),(3,3264),(4,6080)],
 'two-rows':[(128,1184)],
 'multi-UB':[(563,4048)],
}

def freeze():
    out=OUT/'freeze.json'
    if out.exists():raise FileExistsError(out)
    obs=json.loads((ROOT/'results/v0.22/rms-unknown/audit.json').read_text())['rows'];targets={s for g in PROCESSES.values() for s in g}
    assert len(targets)==10 and not targets.intersection(tuple(r['shape']) for r in obs)
    # Include all previously frozen/replayed RMS shapes in the published v0.22 matrix.
    matrix=json.loads((ROOT/'results/v0.22/evaluation.json').read_text())['rows'];known={tuple(r['shape']) for r in matrix if r['op']=='rmsnorm'}
    assert not targets&known
    rows=[]
    for sku in CHIPS:
        chip=load_chip(sku)
        for process,targets in PROCESSES.items():
            sequence=[]
            if process=='wide':sequence.append(('prefix',(1,672)))
            for shape in targets:sequence.extend([('entry',shape),('immediate_repeat',shape)])
            H=0
            for i,(role,shape) in enumerate(sequence):
                start=perf_counter_ns();pred=forecast(shape,chip);ms=(perf_counter_ns()-start)/1e6
                active=pred['active_aiv']; known=pred['conditional_us_by_H'][min(H,active)]
                rows.append(dict(process=process,i=i,role=role,chip=sku,shape=shape,prior_aiv=H,known_us=known,
                    runtime_ms=ms,measurement='extrapolation only' if sku==CHIPS[-1] else 'planned CA',**pred))
                H=max(H,active)
    for sku in CHIPS[:2]:
        for shape in {tuple(r['shape']) for r in rows}:
            pair=[r for r in rows if tuple(r['shape'])==shape and r['chip']==sku and r['role']!='prefix']
            assert len(pair)==2 and pair[0]['prior_aiv']<pair[0]['active_aiv']<=pair[1]['prior_aiv']
    files=['development/validation/rms_unknown23/candidate.py','development/validation/rms_unknown23/freeze.py','build/vector-v8/NORM','operator_model/rms_dispatch.py','operator_model/rms_lane_schedule.py','operator_model/rms_resource.py','operator_model/rms_target_cost.py','operator_model/rms_v12.py','operator_model/wide_v10.py','results/v0.22/rms-lane-parameters.json','results/v0.12/rms/profile.json','results/v0.22/rms-unknown/audit.json',*[f'configs/chips/{s}.json' for s in CHIPS]]
    files+=['results/v0.10/wide/phase-model.json'] if (ROOT/'results/v0.10/wide/phase-model.json').is_file() else []
    result=dict(utc=datetime.now(timezone.utc).isoformat(),op='rmsnorm',candidate='shape-only resource-state tolerance envelope',tolerance=0.20,measured_skus=CHIPS[:2],
        files={f:sha256((ROOT/f).read_bytes()).hexdigest() for f in files},rows=rows,targets_per_chip=10,processes_per_chip=4,
        unique_inputs='Within a registered process, per-shape BF16 x and gamma follow NORM deterministic period-17 generator. Independent all-element y and rstd goldens and separate all-AIV complete device windows required.',
        scoring='Count entry and immediate_repeat independently and as per-shape pairs. Prefix is unscored. Retain fails, actual ELF, trace/window identity, runtime SKU and host costs; only distinct qualified target windows count.',
        collection='One serial collector slot; do not replace any shape or predecessor. Split the measured 21 launches per chip across four fresh processes, --dump=false. Keep existing data if storage caps trigger. No 970 CA.')
    write(out,result)
    print(out, len(rows),'max ms',max(x['runtime_ms'] for x in rows))
if __name__=='__main__':freeze()
