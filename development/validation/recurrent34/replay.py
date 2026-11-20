"""Offline positive-control of the endpoint evidence using immutable v0.17 receipts."""
from pathlib import Path
import single

def run():
    c=single.PR
    d=single.ROOT/'runs/sku17/extension'/f'kda_recurrent_1_2_1_1_128_256_{c}'
    path=d/'strict.json'
    old,=single.read(path)
    assert old['shape']==[1,2,1,1,128,256] and old['chip']==c
    target=single.read(single.REG)['job']
    j=dict(target,shape=old['shape'],golden=dict(attn_out=512,final_state=32768,initial_state_unchanged=32768))
    v=single.inspect(j,c,d,old['cost'])
    assert v['qualified'] and abs(v['ca_us']-old['ca_us'])<.001
    stage,=v['stages']; old_stage,=old['stages']
    assert stage['trace_sha256']==old_stage['trace_sha256']
    assert v['actual']=={k:[n,0] for k,n in j['golden'].items()}
    record=dict(role='Archived PR S2 / V256 verifier control, not a new S8 measurement.',shape=j['shape'],chip=c,
                old_receipt=single.rel(path),old_receipt_sha256=single.digest(path),merged_trace_sha256=stage['trace_sha256'],
                expected_and_verified_golden=v['actual'],released_device_text_sha256=v['released_text_sha256'],kernel=stage['kernel'],
                strict_s8_verifier_on_old_shape_us=v['ca_us'],old_us=old['ca_us'],active_aiv=stage['active_aiv'],cost=old['cost'],
                actual_core_windows=stage['cores'])
    single.write(single.OUT/'verifier-control.json',record)
if __name__=='__main__':run()
