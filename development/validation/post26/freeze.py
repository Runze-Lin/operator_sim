"""Register deployment decisions and prices for additional unseen shapes before either SKU runs."""
import sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[3];sys.path.insert(0,str(ROOT))
from development.validation.sku19.campaign import read,write,now
from development.validation.post24.collect import LIB,BIN
from development.validation.post24.freeze import CHIPS,sha,known_shapes
from development.validation.post25.repeat_policy import Contract,estimate_repeat
from operator_model import load_chip
from operator_model.post_resource import resource_post
OUT=ROOT/'results/v0.26/post';OLD=ROOT/'results/v0.25/post'
GROUPS={'wide':[(39,4096),(81,2560)],'multi':[(127,448),(157,1088),(337,640)],'reserve':[(95,1280),(151,768)]}
SOURCES=['development/validation/post25/repeat_policy.py','development/validation/post25/repeat_policy_freeze.py','operator_model/post_token_pipeline.py','operator_model/post_resource.py','operator_model/post_v12.py','operator_model/wide_v10.py','results/v0.12/post/profile.json','results/v0.25/post/prices.json','results/v0.25/post/repeat-policy-freeze.json','development/validation/post24/collect.py','development/validation/rms_ordered_verify/verify.py','development/validation/post_candidate/extract.py','scripts/cann-run.sh','smoke/vector_v8.cpp','smoke/extension_indexer_mhc.cpp']
def main():
    reg=OUT/'registration.json'
    if reg.exists():raise FileExistsError(reg)
    old=known_shapes()|{tuple(r['shape']) for p in ('v0.24','v0.25') for r in read(ROOT/'results'/p/'post/freeze.json')['rows']}
    target={(t,4,d) for g in GROUPS.values() for t,d in g};assert len(target)==7 and not target&old
    data=read(OLD/'prices.json');rows=[]
    for sku in CHIPS:
        chip=load_chip(sku)
        for group,pairs in GROUPS.items():
            h=0;last=None
            for i,(role,t,d) in enumerate((role,t,d) for t,d in pairs for role in ('entry','repeat')):
                shape=[t,4,d];c=Contract((0,group),chip,'mhc_post','aclnnMhcPost',tuple(shape),'bf16','contiguous_ND',True)
                old_us,_,active,diag=resource_post(shape,chip,{'resident_aiv':h});unknown=resource_post(shape,chip,{})[0]
                st=time.perf_counter_ns();us,mode=estimate_repeat(c,last,h,data);wall=(time.perf_counter_ns()-st)/1e6
                rows.append(dict(chip=sku,group=group,i=i,shape=shape,role=role,state=diag['context'],previous_shape=list(last.shape) if last else None,prior_aiv=h,mode=mode,us=us,old_known_us=old_us,old_unknown_us=unknown,candidate_unknown_us=unknown,
                                 active=active,plan=diag['target_plan'],predict_ms=wall));h=max(h,active);last=c
    files=[ROOT/f for f in SOURCES]+[LIB,BIN,Path(__file__).resolve(),Path(__file__).with_name('collect.py')]+[ROOT/'configs/chips'/f'{s}.json' for s in CHIPS]
    write(reg,dict(utc=now(),inputs={str(f.relative_to(ROOT)):sha(f) for f in files},measured=CHIPS[:2],extrapolated=CHIPS[2:],order=list(GROUPS),primary=['wide','multi'],reserve=['reserve'],wall_s=5400,mib=12288,
        primary_evidence='Old policy-freeze groups width/many and new wide/multi; reserve is registered before data, collected only when at least one SKU has fewer than 10 complete different exact repeats after primary groups. Never stop based on error.',
        acceptance='Deployment evidence for intended exact-repeat subset: >=10 distinct all-lane complete shapes per SKU; repeat-only WAPE<=20%, max<=40% at both, improve WAPE vs original. Also report all known calls, first/ordinary repeats/shape-only separately and DT/PR speed ratios. Unmeasured 970DT extrapolated. If unavailable or mispriced, keep explicit experimental policy.',
        scope='One slot shared with prior v0.25 Post. No retuning, frozen physical prices already tested on new boundary shapes. Numerically check each element and demand complete ELF/official core timing, END and writes. Aggregate and real-END contracts never mixed.',rows=rows))
if __name__=='__main__':main()
