"""Register ten prospective shapes, ordered launches, prices and acceptance before collection."""
import json, sys, time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[3];sys.path.insert(0,str(ROOT))
from development.validation.post24.freeze import known_shapes, sha, CHIPS
from development.validation.post24.collect import LIB, BIN
from development.validation.post25.calibrate import OUT, predict
from development.validation.sku19.campaign import read, write, now
from operator_model import estimate, load_chip

GROUPS={'boundary':[(57,192),(67,384),(113,768)],'steady':[(181,1024),(196,1536)],
        'width':[(65,320),(121,1216),(65,5120)],'many':[(193,896),(449,512)]}
def main():
    reg=OUT/'freeze.json'
    if reg.exists():raise FileExistsError(reg)
    old=known_shapes()|{tuple(r['shape']) for r in read(ROOT/'results/v0.24/post/freeze.json')['rows']}
    shapes={(t,4,d) for pairs in GROUPS.values() for t,d in pairs}
    assert len(shapes)==10 and not old&shapes
    params=read(OUT/'prices.json');rows=[]
    for s in CHIPS:
        chip=load_chip(s)
        for group,pairs in GROUPS.items():
            prior=0;previous=None
            for i,(role,t,d) in enumerate((role,t,d) for t,d in pairs for role in ('entry','repeat')):
                shape=[t,4,d];start=time.perf_counter_ns()
                pred=predict(shape,chip,prior,params,previous=previous)
                price_ms=(time.perf_counter_ns()-start)/1e6
                unknown=predict(shape,chip,None,params)
                base=estimate('mhc_post',shape,{'resident_aiv':prior},chip=chip,dtype='bf16')
                assert abs(pred['baseline']-base.latency_us)<1e-7 and pred['active']==base.active_aiv
                rows.append(dict(chip=s,group=group,i=i,role=role,shape=shape,prior_aiv=prior,previous_shape=previous,
                                 old_known_us=base.latency_us,old_unknown_us=unknown['baseline'],candidate_unknown_us=unknown['us'],
                                 state=base.diagnostics['context'] if hasattr(base,'diagnostics') and 'context' in base.diagnostics else ('first' if not prior else ('expand' if prior<pred['active'] else 'reuse')),predict_ms=price_ms,**pred))
                prior=max(prior,pred['active']);previous=shape
    files=[Path(__file__).resolve(),Path(__file__).with_name('calibrate.py'),Path(__file__).with_name('collect.py'),OUT/'prices.json',OUT/'archive.json',ROOT/'operator_model/post_token_pipeline.py',ROOT/'operator_model/post_resource.py',ROOT/'operator_model/rms_resource.py',ROOT/'operator_model/post_v12.py',ROOT/'operator_model/wide_v10.py',ROOT/'results/v0.12/post/profile.json',ROOT/'development/validation/post24/collect.py',ROOT/'development/validation/post24/physical.py',ROOT/'development/validation/post_candidate/extract.py',ROOT/'development/validation/rms_ordered_verify/verify.py',ROOT/'smoke/vector_v8.cpp',ROOT/'smoke/extension_indexer_mhc.cpp',ROOT/'scripts/cann-run.sh',LIB,BIN]+[ROOT/'configs/chips'/f'{s}.json' for s in CHIPS]
    write(reg,dict(utc=now(),inputs={str(f.relative_to(ROOT)):sha(f) for f in files},archive_shapes=len(old),
          measured=CHIPS[:2],extrapolated=CHIPS[2:],groups=list(GROUPS),primary='All active AIVs earliest instruction through last END_LABEL incl. one real END and all numerically checked output values; same definition as published v0.24. If the synthetic marker is incomplete, the full ELF-entry to physical END envelope is an optional separate measure.',wall_s=5400,mib=12288,schedule='One serial simulator host across the four groups and two SKUs; each group has its own new stream, with an immediate exact repeat of each input. Feasibility error targets WAPE<=20%, max<=40%. Both signed and absolute error reported. 970DT predicts only.',rows=rows))
if __name__=='__main__':main()
