"""Compare frozen 32k first-MMAD and per-stream steady launch predictions with PR."""
from collections import defaultdict
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[3]
sys.path.insert(0,str(ROOT))
from development.validation.sku15.fa_independent import events, slopes, phases
from operator_model import load_chip
from operator_model.ca import sha256


def main():
    out=ROOT/'results/v0.16'
    row,=[r for r in json.loads((out/'sku/holdout.json').read_text()) if r['shape']==[9,32767,64,1,1]]
    trace=ROOT/row['pr']['trace']
    assert sha256(trace)==row['pr']['trace_sha256']
    streams=defaultdict(list)
    for e in events(trace):
        if e['ph']=='X' and e['name']=='MOV_OUT_TO_L1_MULTI_ND2NZ':
            meta=json.loads(e['args']['detail'])
            if meta['n'] in (255,256) and meta['d']==64:
                streams[e['args']['pc_addr']].append((e['ts'],meta['src_addr']))
    cadence=[]
    for steps in sorted(streams.values(),key=min):
        steps.sort()
        if len(steps)<120:
            continue
        assert all(y[1]-x[1]==32768 for x,y in zip(steps,steps[1:]))
        cadence.append(slopes([t for t,_ in steps])['slope_8_120'])
    assert len(cadence)==2
    clock=load_chip(row['pr']['soc']).frequency_ghz*1000
    diag=row['projection']['timing_diagnostics']; wave,=diag['wave_types']
    obs=phases(row)
    data=dict(role='post-freeze stage comparison; microseconds',shape=row['shape'],
              kernel=row['kernel'],trace_sha256=row['pr']['trace_sha256'],
              predicted_first_mmad=row['projection']['stages_us']['launch']+wave['qk_start_cycles']/clock,
              actual_first_mmad=obs['prefix'],predicted_full_tile_ii=wave['first']['ii_cycles']/clock,
              k_steady_issue_us=cadence[0],v_steady_issue_us=cadence[1],slope_window='tile indices 8..119',
              actual_mmads=obs['mmads'],predicted_mmads=diag['mmads_total'],
              total_ca=row['pr_ca_us'],total_predicted=row['pr_us'])
    (out/'fa-pr-32k.json').write_text(json.dumps(data,indent=2)+'\n')


if __name__=='__main__':
    main()
