"""Obtain costs from marked phase intervals on 950DT. PR has no fitting role."""
from pathlib import Path
import json
from statistics import median
from math import ceil

ROOT=Path(__file__).resolve().parents[3]
OUT=ROOT/'results/v0.26/presink'
src=json.loads((OUT/'archive.json').read_bytes())
anchors=[]
for r in src:
    if 'DT' not in r['sku'] or r['shape'][0] not in (1,20,33,56,80):continue
    t,_,d=r['shape'];mm=r['cube']['milestones']['MMAD']
    v=r['vec']['milestones']['VF'];post=next(e for e in v if e['pc'].endswith('14cb0'))
    pre=[e for e in v if e['us'][0]<post['us'][0]]
    span=1650*(mm[-1]['us'][1]-mm[0]['us'][0])
    vv=1650*(pre[-1]['us'][1]-pre[0]['us'][0])
    transition=1650*(post['us'][0]-pre[-1]['us'][1])-span
    anchors.append(dict(shape=r['shape'],trace=r['trace_sha256'],front=1650*pre[0]['us'][0],cast_elements=ceil(t/2)*512,
        cast_span=vv,mmad_count=len(mm),aligned_kmac=ceil(t/16)*16*32*512/1024,mmad_span=span,
        residual=transition,post=1650*(r['ca_us']-post['us'][0]),stage2_rows=ceil(t/64)))
# Estimate per-element cost with median pairwise slope, shared setup with median stage residual.
slopes=[(b['cast_span']-a['cast_span'])/(b['cast_elements']-a['cast_elements']) for i,a in enumerate(anchors) for b in anchors[i+1:]]
cast_b=median(slopes);cast_a=median(r['cast_span']-cast_b*r['cast_elements'] for r in anchors)
# Least-squares counts use only MMAD first issue..last retire; no fit to total kernel time.
ss=sum(r['mmad_count']**2 for r in anchors);xx=sum(r['aligned_kmac']**2 for r in anchors);sx=sum(r['mmad_count']*r['aligned_kmac']for r in anchors)
sz=sum(r['mmad_count']*r['mmad_span']for r in anchors);xz=sum(r['aligned_kmac']*r['mmad_span']for r in anchors)
cm=(sz*xx-xz*sx)/(ss*xx-sx*sx);cw=(ss*xz-sx*sz)/(ss*xx-sx*sx)
coeff=dict(first_vf_cycles=median(r['front'] for r in anchors),cast_start_cycles=cast_a,cast_cycles_per_element=cast_b,
    cube_issue_cycles=cm,cube_cycles_per_aligned_kmac=cw,rendezvous_cycles=median(r['residual']for r in anchors),
    post_cycles_per_row=median(r['post']/r['stage2_rows'] for r in anchors))
(OUT/'prices.json').write_text(json.dumps(dict(anchors=anchors,prices=coeff,units='950DT cycles at 1.65 GHz'),indent=2)+'\n')
print(coeff)
