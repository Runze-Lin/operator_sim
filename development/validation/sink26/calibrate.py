"""Estimate local observable issue milestones in cycles, only from historical 950DT."""
from pathlib import Path
from statistics import median
from evidence import DT,OUT,read,save,sha

def annotated(rows):
    h={}
    records=[]
    for r in sorted(rows,key=lambda v:v['source']):
        p=Path(r['source'])
        previous=0 if r['isolated'] else h.get(str(p.parent),0)
        h[str(p.parent)]=max(previous,r['active'])
        records.append((r,previous))
    return records

def phases(r):
    vf=[e for e in r['events'] if e['name']=='VF']
    tile,payload,write=[e for e in r['events'] if e['name'].startswith('MOV_SRC_TO_DST')]
    return dict(tiling_issue=tile['begin'],tiling_wait=tile['end']-tile['begin'],
                setup_from_tile=vf[0]['begin']-tile['end'],input_issue=payload['begin']-vf[0]['begin'],
                input_wait=payload['end']-payload['begin'],soft_after_input=vf[1]['begin']-payload['end'],
                first_col_from_setup=vf[2]['begin']-vf[0]['begin'],write_gap=write['begin']-vf[-1]['begin'],
                write_wait=write['end']-write['begin'],end=r['full_ticks']-write['end'])

def param(rows,stage,fit_slope):
    xy=[(r['active'],phases(r)[stage]) for r in rows]
    slope=0
    if fit_slope:
        slopes=[(y2-y1)/(x2-x1) for x1,y1 in xy for x2,y2 in xy if x2>x1]
        if slopes:slope=max(0.,median(slopes))
    return [round(max(0,median(y-slope*x for x,y in xy)),3),round(slope,4)]

def fit():
    src=OUT/'trace-evidence.json'; annotated_rows=annotated([r for r in read(src) if r['chip']==DT])
    groups={'first':[],'expand':[],'resident':[]}
    for r,h in annotated_rows:groups['first' if h==0 else 'expand' if h<r['active'] else 'resident'].append(r)
    results={}
    for state,rs in groups.items():
        results[state]={k:param(rs,k,k in {'tiling_issue','tiling_wait','first_col_from_setup','write_wait'}) for k in phases(rs[0])}
    return dict(unit='950DT cycles, cycle rates are calibrated to observed issue intervals, not end-to-end targets.',
                reference_hz=1650_000_000,reference_vec_fp32_ops_per_cycle=27e12/64/1.65e9,
                training_samples={k:len(v) for k,v in groups.items()},source_sha256=sha(src),phase_prices=results,
                softmax_col_floor=231,first_col_to_row=378,first_pair=248,steady_pair=254,
                row_col=56,steady_row_col=62,aligned_rows_per_vreg=8,
                added_pass_scope='Extra passes extrapolate the serial dependency body at one steady pair per group; issue/setup are paid once per helper. No old measured local tile exceeded 4 rows.')
if __name__=='__main__':save(OUT/'prices.json',fit())
