"""Recover fixed front-end residuals using only nine-row primitives, and score each window."""
import argparse
import csv
import json
from pathlib import Path
from types import SimpleNamespace

from calibration.system_vector_ca.report import tensor
from operator_model.chips import load_chip
from operator_model.system_routes.vector import _PRIMITIVE_API, _primitive, _schedule

NAMES = {api:name for name,api in _PRIMITIVE_API.items()}


def fixture(api, rows):
    raw = NAMES[api]
    ids = tensor((rows,), "int64")
    flag = tensor((rows,), "bool")
    f32 = tensor((rows,64), "float32")
    bf16 = tensor((rows,64), "bfloat16")
    args = {
        "aclnnLtScalar":((ids,),(flag,)), "aclnnGeScalar":((ids,),(flag,)),
        "aclnnBitwiseOrTensor":((flag,flag),(flag,)), "aclnnSubs":((ids,),(ids,)),
        "aclnnInplaceMaskedFillScalar":((ids,flag),(ids,)),
        "aclnnNeg":((bf16,),(bf16,)), "aclnnPowTensorScalar":((f32,),(f32,)),
        "aclnnMean":((f32,),(tensor((rows,),"float32"),)),
        "aclnnReduceSum":((f32,),(tensor((rows,),"float32"),)),
        "aclnnRsqrt":((f32,),(f32,)),
        "aclnnEmbedding":((tensor((128,64),"bfloat16"),ids),(bf16,)),
        "aclnnGatherV2":((tensor((128,64),"bfloat16"),ids),(bf16,)),
        "aclnnCast":((bf16,),(f32,)),
        "aclnnCat":((bf16,bf16),(tensor((rows,128),"bfloat16"),)),
        "aclnnConstantPadNd":((bf16,),(tensor((rows+7,64),"bfloat16"),)),
    }
    ins, outs=args[api]
    if api=="aclnnNeg":
        ins=outs=(tensor((rows,1,1,64),"bfloat16"),)
    return SimpleNamespace(raw_name=raw,inputs=ins,outputs=outs,attributes=())


def evaluate(records, *, fit):
    fits = {}
    rows = []
    for r in records:
        if fit and r["rows"] != 9:
            continue
        chip=load_chip(r["chip"])
        for stage in r["stages"]:
            api=stage["api"]
            f=fixture(api,r["rows"])
            result=_schedule(_primitive(f),chip)
            s=result.work["schedule"]
            payload=max(s["core_pipeline_cycles"],s["shared_payload_floor_cycles"])
            fits.setdefault(api,{})[chip.name]=stage["cycles"]-payload
            observed=stage["cycles"]/(1000*chip.frequency_ghz)
            rows.append(dict(api=api,raw_name=f.raw_name,chip=chip.name,rows=r["rows"],
                             input_shapes=str([t.shape for t in f.inputs]),output_dtype=f.outputs[0].dtype,
                             role="control_calibration" if r["rows"]==9 else ("postfreeze_size_check" if r["group"]=="memory" else "size_held_out_from_fit_replay"),
                             predicted_us=result.latency_us,ca_us=observed,
                             signed_error_pct=(result.latency_us/observed-1)*100,
                             predicted_aiv=s["active_aiv"],ca_aiv=stage["aiv"],kernel=stage["kernel"]))
    if fit:
        for api,devices in fits.items():
            print(f'    "{api}": ({devices["Ascend950DT_9582"]:.6f}, {devices["Ascend950PR_957b"]:.6f}),')
    return rows


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("receipts",type=Path)
    p.add_argument("--fit",action="store_true")
    p.add_argument("--out",type=Path)
    a=p.parse_args()
    rows=evaluate(json.loads(a.receipts.read_text()),fit=a.fit)
    if a.out:
        with a.out.open("w") as h:
            w = csv.DictWriter(h, fieldnames=rows[0], lineterminator="\n")
            w.writeheader()
            w.writerows(rows)
