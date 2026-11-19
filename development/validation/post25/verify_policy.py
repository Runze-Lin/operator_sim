"""Check exact-contract selection boundary, including identical shape with a different predecessor."""
from dataclasses import replace
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[3];sys.path.insert(0,str(ROOT))
from development.validation.post25.repeat_policy import Contract,estimate_repeat
from development.validation.post25.calibrate import OUT
from development.validation.sku19.campaign import read
from operator_model import load_chip
def main():
    source=read(OUT/'repeat-policy-freeze.json');params=read(OUT/'prices.json');n=0
    for r in source['rows']:
        c=Contract((0,r['group']),load_chip(r['chip']),'mhc_post','aclnnMhcPost',tuple(r['shape']),'bf16','contiguous_ND',True)
        if r['role']!='repeat':continue
        us,label=estimate_repeat(c,c,r['prior_aiv'],params)
        assert abs(us-r['policy_us'])<1e-10 and label=='immediate identical builtin Post'
        cold,label=estimate_repeat(c,c,0,params);assert label=='published v0.24'
        for prev in [None,replace(c,dtype='fp16'),replace(c,op='matmul'),replace(c,shape=(r['shape'][0]+1,*c.shape[1:])),replace(c,rank_stream=(1,r['group'])),replace(c,layout='view'),replace(c,has_h_res=False),replace(c,implementation='aclnnHcPost'),replace(c,chip=replace(c.chip,frequency_ghz=c.chip.frequency_ghz+1))]:
            us,label=estimate_repeat(c,prev,r['prior_aiv'],params);assert abs(us-r['public_us'])<1e-10 and label=='published v0.24'
        us,label=estimate_repeat(c,c,None,params);assert abs(us-r['no_history_us'])<1e-10 and label=='published v0.24'
        n+=1
    print('exact/impostor/cold/unknown contracts verified across',n,'registered repeats')
if __name__=='__main__':main()
