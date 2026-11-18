"""Use frozen predictions and the critical PR AIV's DMA/VF events for attribution."""
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[3]
sys.path.insert(0,str(ROOT))
from development.validation.sku15.fa_independent import events
from operator_model.ca import sha256
from operator_model import load_chip


def audit(row):
    physical=row['projection']['timing_diagnostics']
    target_tile=physical['first_tile_cycles']
    pr=row['pr']
    latest=max(pr['core_windows'],key=lambda w:w['end_us'])['core']
    trace=ROOT/pr['trace']
    assert sha256(trace)==pr['trace_sha256']
    first=float('inf')
    dma_in=[]; dma_out=[]; vf=[]
    for e in events(trace):
        if e['ph']!='X':
            continue
        first=min(first,e['ts'])
        if e['pid']!=latest:
            continue
        if e['name']=='MOV_SRC_TO_DST_ALIGNv2':
            item=(e['ts'],e['ts']+e['dur'],json.loads(e['args']['detail'])['processed_bytes'])
            (dma_in if e['tid']=='MTE2' else dma_out).append(item)
        if e['name']=='VF' and e['tid']=='PUSHQ':
            vf.append(e['ts'])
    dma_in.sort()
    dma_out.sort()
    vf.sort()
    assert dma_in[0][2]==160
    payload=dma_in[1:]
    assert len(vf)==3*len(payload) and len(dma_out)==2*len(payload)
    read=payload[0]
    write=dma_out[:2]
    measured=dict(dispatch=read[0]-first,read_latency=read[1]-read[0],
                  vector=write[0][0]-read[1],write=max(w[1] for w in write)-write[0][0],
                  first_input_bytes=read[2],first_output_bytes=sum(w[2] for w in write),
                  ub_loops=len(payload),read_issue=payload[1][0]-read[0] if len(payload)>1 else None)
    clock=load_chip('Ascend950PR_957b').frequency_ghz*1000
    predicted={k:target_tile[k]/clock for k in ('read_latency','vector','write')}
    predicted.update(dispatch=row['projection']['stages_us']['dispatch'],
        read_issue=target_tile['read_ii']/clock, first_input_bytes=target_tile['read_bytes'],
        first_output_bytes=target_tile['write_bytes'],ub_loops=physical['target_plan']['iterations'])
    return dict(id=row['id'],shape=row['shape'],ca_us=pr['ca_us'],prediction_us=row['pr_us'],
                critical_core=latest,predicted=predicted,observed=measured,trace_sha256=pr['trace_sha256'])


if __name__=='__main__':
    out=ROOT/'results/v0.16'
    group=[r for r in json.loads((out/'sku/holdout.json').read_text()) if r['op']=='quant']
    assert len(group)==10
    rows=[audit(r) for r in group]
    (out/'quant-pr-phases.json').write_text(json.dumps(dict(role='frozen posthoc',rows=rows),indent=2)+'\n')
