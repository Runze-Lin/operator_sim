"""Assert the public integration still returns every frozen compact forecast."""
import hashlib
from check import FREEZE, MODEL, RES, SKUS, digest, read, save
from operator_model import estimate
from operator_model.kda_recurrent_chains import estimate_recurrent_chains

reg=read(FREEZE)
assert reg['files']['operator_model/kda_recurrent_chains.py']==digest(RES/'frozen-model.py')
contract=dict(layout='BSND',dtype='BF16',gate_dtype='FP32',state_dtype='FP32',state_v_first=False,use_gate_in_kernel=False,chunk_size=None,export_gk=False,output_final_state=True)
rows=[]
for c in reg['cases'].values():
    shape=c['job']['shape']
    for chip in SKUS:
        old=c['predictions'][chip];ref=estimate_recurrent_chains(shape,{},chip=chip); explicit=estimate_recurrent_chains(shape,contract,chip=chip)
        compare={k:old[k] for k in ref};assert ref==compare and ref==explicit
        for mode in ('auto','resource','calibrated'):
            out=estimate('kda_recurrent',shape,contract,chip=chip,timing_mode=mode)
            assert out.status==('unavailable' if mode=='calibrated' else 'extrapolated')
            assert out.latency_us==(None if mode=='calibrated' else old['latency_us'])
        rows.append(dict(shape=shape,chip=chip,all_frozen_prediction_fields_identical=True,latency_us=ref['latency_us']))
save(RES/'contract-check.json',dict(freeze_sha256=digest(FREEZE),frozen_model_sha256=digest(RES/'frozen-model.py'),current_model_sha256=digest(MODEL),scope='Accept documented neutral attributes and gracefully propagate unavailable single-chain price; unchanged central forecast and sensitivity for all six frozen rows.',rows=rows))
