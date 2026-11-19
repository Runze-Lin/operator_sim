"""Replay packed Kimi prefill contracts through the public API; no allocated activation tensors."""
from pathlib import Path
import json
import sys
import time

ROOT=Path(__file__).resolve().parents[3]
sys.path.insert(0,str(ROOT))
from operator_model import estimate
from operator_model.extensions import plan

def run():
    conf=json.loads((ROOT/'configs/models/Kimi-K3.json').read_text())
    tc=conf['text_config']
    kda=tc['linear_attn_config']
    heads=kda['num_heads']//8
    dim=kda['head_dim']
    output=[]
    for batch,length in [(1,64),(8,32768),(8,131072),(8,262144)]:
        cu=[i*length for i in range(batch+1)]
        shape=[1,cu[-1],heads,heads,dim,dim]
        attrs=dict(cu_seqlens=cu,dt_bias=True,initial_state=True,gate_dtype='bf16',beta_dtype='fp32',
                   output_final_state=True,export_gk=False,state_v_first=True,use_gate_in_kernel=True,
                   safe_gate=True,lower_bound=kda['gate_lower_bound'],chunk_size=64)
        schedule=plan('kda_chunk',shape,cu_seqlens=cu,dt_bias=True,initial_state=True,chunk_size=64)['tiling']
        results=[]
        for chip in ['Ascend950DT_9582','Ascend950PR_957b','Ascend970DT']:
            start=time.perf_counter_ns()
            value=estimate('kda_chunk',shape,attrs,chip=chip)
            elapsed=time.perf_counter_ns()-start
            if value.status!='unavailable' or not value.work: raise ValueError(value)
            results.append({'chip':chip,'predicted_us':value.latency_us,'status':value.status,'wall_ms':elapsed/1e6})
        output.append(dict(shape=shape,batch=batch,context_per_request=length,attrs=attrs,partition=schedule,
                           work=value.work,estimates=results))
    return {'method':'public estimate, single call per shape per chip in one fresh interpreter, process startup excluded',
            'evidence':'workload arithmetic, no CA latency', 'tp':8,'cases':output}

if __name__=='__main__':
    (ROOT/'results/v0.21/chunk/kimi-workloads.json').write_text(json.dumps(run(),indent=2)+'\n')
