"""Offline coverage and numerical negative controls. No device launch."""
import json
import math
import struct
from pathlib import Path
from acquire import OUT, ROOT, DT, PR, save
from certificate import certify, no_bias_rejections, sha


def brute(shape):
    m,n,k=shape
    def val(i): return (i%17-8)/16
    fail=0
    for i in range(m):
        for j in range(n):
            dot=sum(val(i*k+q)*val(q*n+j+2) for q in range(k));target=dot+val(j+5)
            bits=struct.unpack('I',struct.pack('f',dot))[0]; out=struct.unpack('f',struct.pack('I',((bits+32767+(bits>>16&1))>>16)<<16))[0]
            fail+=abs(out-target)>(1+abs(target))/64
    return fail


def main():
    controls=[]
    for sku,shape,target in ((DT,[16,64,256],'control-dt16'),(PR,[128,192,256],'control-128'),(PR,[512,512,256],'control-512')):
        m,n,k=shape;raw=ROOT/f'runs/sku17/extension/addmm_{m}_{n}_{k}_{sku}';cost=json.loads((raw/'cost.json').read_bytes());r=certify(raw,sku,shape,cost)
        save(OUT/(target+'.json'),r); controls.append(dict(shape=shape,sku=sku,ca_us=r['ca_us'],active_aic=r['active_aic'],bt_moves=r['bias_bt_moves'],
            output_checked=r['output_checked'],rejected_without_bias=r['outputs_rejecting_bias_free_bf16_dot'],mmads=r['mmads'],receipt=target+'.json'))
    small=ROOT/'results/v0.29/large/cube/addmm-verifier-control.json';ref=json.loads(small.read_bytes());assert abs(controls[1]['ca_us']-ref['ca_us'])<1/1650
    brute_results=[]
    for case in ((1,1,1),(2,32,7),(19,21,32),(17,64,256)):
        exact=brute(case);assert exact==no_bias_rejections(case);brute_results.append({'shape':case,'bias_free_bad':exact})
    plain=ROOT/'runs/matmul28/32768-128-512-dt';cost=json.loads((plain/'cost.json').read_bytes());caught=None
    try:certify(plain,DT,[32768,128,512],cost)
    except ValueError as e:caught=str(e)
    assert caught=='wrong input or SKU'
    save(OUT/'controls.json',dict(archived=controls,strict_comparison={'original':str(small.relative_to(ROOT)),'sha256':sha(small),'absolute_delta_us':abs(controls[1]['ca_us']-ref['ca_us'])},
        dense_oracle_checks=brute_results,rejected_similar_matmul={'source':str(plain.relative_to(ROOT)),'reason':caught}))

if __name__=='__main__':main()
