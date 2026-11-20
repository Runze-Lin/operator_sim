"""Independent CPU-only verifier regression against archived multi-kernel, BF16 and state dispatches."""
from pathlib import Path
import json
import sys
from campaign import ROOT,OUT,V2_OLD,read,save
from verify import qualify

cases=[]
for stem,host,args,expected,prefix,contract in [
    ('moe_ffn_1_64_64_1','moe',[1,64,64,1],{'gmm1_output':64,'gmm1_mx_scale':2,'gmm2_output':64},['GroupedMatmulSwigluQuantV2','GroupedMatmul'],'case=moe_ffn M=1 K=64 F=64 E=1'),
    ('addmm_128_256_384','addmm',[],{'output':128*256},['MatMulV3'],'case=addmm api=aclnnAddmm M=128 N=256 K=384'),
    ('kda_recurrent_1_1_1_1_128_128','recurrent',[],{'attention':128,'state':128*128},['RecurrentKda'],'case=kda_recurrent api=aclnnRecurrentKda')]:
    folder=ROOT/'runs/sku17/extension'/(stem+'_Ascend950PR_957b')
    if not folder.exists():continue
    if host=='recurrent':
        import re
        log=(folder/'run.log').read_text();expected={k:int(v) for k,v in re.findall(r'correctness name=(\S+) count=(\d+)',log)}
    job=dict(id=host,prefix=prefix,contract=contract,golden=expected)
    verdict=qualify(job,'Ascend950PR_957b',folder,read(folder/'cost.json'),read(V2_OLD),ROOT)
    assert verdict['qualified'],verdict
    cases.append(dict(old_capture=str(folder.relative_to(ROOT)),op=host,groups=verdict['actual'],kernel=[r['kernel'] for r in verdict['stages']],ca_us=verdict['ca_us']))
save(OUT/'verifier-controls.json',cases)
