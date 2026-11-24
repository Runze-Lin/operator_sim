"""Attach the system model to each retained native fixture and summarize both 950 SKUs."""
import argparse
import json
from math import prod
from pathlib import Path
from types import SimpleNamespace

from operator_model.system_routes.vector import price


def tensor(shape, dtype):
    return SimpleNamespace(shape=shape, stride=tuple(prod(shape[i+1:]) for i in range(len(shape))),
                           dtype=dtype, layout="torch.strided", device="meta")


def features(op, shape):
    r, d = shape[0], shape[-1]
    sem = {}
    if op == "add":
        name = "aten.add.Tensor"
        x = tensor((r,d),"bfloat16")
        ins, outs = (x,x),(x,)
    elif op == "mul":
        name = "aten.mul.Tensor"
        x = tensor((r,d),"float32")
        ins, outs = (x,tensor((r,1),"float32")),(x,)
    elif op == "swiglu":
        name = "npu::npu_swiglu"
        ins, outs = (tensor((r,2*d),"bfloat16"),),(tensor((r,d),"bfloat16"),)
    elif op == "rope":
        name = "_C_ascend::inplace_partial_rotary_mul"
        sem = {"partial_slice": (d-64,d), "rotary_mode": "interleave"}
        table = tensor((r,1,1,64),"bfloat16")
        ins, outs = (tensor((r,1,8,d),"bfloat16"), table, table), ()
    elif op == "mhc_pre_sinkhorn":
        r,n,d = shape
        c=n*n+2*n
        name="_C_ascend::npu_hc_pre_v2"
        sem={"sinkhorn_iterations":20}
        ins=(tensor((r,n,d),"bfloat16"),tensor((c,n*d),"float32"),tensor((3,),"float32"),tensor((c,),"float32"))
        outs=(tensor((r,d),"bfloat16"),tensor((r,n),"float32"),tensor((r,n,n),"float32"))
    elif op == "moe_gating_top_k_hash":
        r,e,k=shape
        name="_C_ascend::moe_gating_top_k_hash"
        sem={"out_flag":False, "norm_type":2}
        ins=(tensor((r,e),"float32"),tensor((r,),"int32"),tensor((129280,k),"int32"))
        outs=(tensor((r,k),"float32"),tensor((r,k),"int32"),tensor((r,e),"float32"))
    elif op == "cache_load_fp8":
        name = "inference::cache_load"
        sem = {"storage_precision":"fp8","logical_shape":tuple(shape)}
        ins = (tensor(tuple(shape),"float8_e4m3fn"),tensor(tuple(shape[:-1]),"float32"))
        outs = (tensor(tuple(shape),"bfloat16"),)
    else:
        raise ValueError(op)
    return SimpleNamespace(raw_name=name, inputs=ins, outputs=outs, attributes=(("semantic",tuple(sem.items())),))


def compare(row):
    result = price(features(row["op"], row["shape"]), row["chip"])
    earlier = {}
    if row["op"] == "moe_gating_top_k_hash" and row["shape"][0]==2:
        # Frozen pre-correction estimates, preserved from the first probe run.
        prior = {'Ascend950DT_9582': 2.6705888571070506, 'Ascend950PR_957b': 2.6705958136287897}[row["chip"]]
        earlier = {"prior_k_only_us": prior, "prior_signed_error_pct":100*(prior/row["measured_us"]-1)}
    return dict(row, **earlier, predicted_us=result.latency_us, signed_error_pct=100*(result.latency_us/row["measured_us"]-1),
                predicted_aiv=result.work["schedule"]["active_aiv"],
                sample_role="hash_control_calibration" if earlier else "development_replay" if row["chip"]=="Ascend950DT_9582" and
                            (row["op"] in ("rope","swiglu") or row["shape"] in ([9,7168],[9,4],[1,128,512]))
                            else "postfreeze_native_check" if row["op"] in ("mhc_pre_sinkhorn","moe_gating_top_k_hash")
                            else "diagnostic_replay")


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input",nargs="+",type=Path)
    parser.add_argument("--output",required=True,type=Path)
    args=parser.parse_args()
    rows = [compare(row) for path in args.input for row in json.loads(path.read_text())]
    args.output.write_text(json.dumps(rows, indent=2)+"\n")
