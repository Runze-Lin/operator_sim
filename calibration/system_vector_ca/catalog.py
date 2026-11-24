"""Export a rank-local shape ledger for the vector-family examples using archived native graphs."""
import argparse
from collections import Counter
import csv
import gzip
import json
from pathlib import Path
from types import SimpleNamespace

from calibration.system_compare import cases
from operator_model.chips import load_chip
from operator_model.system_adapter import estimate_features
from operator_model.system_routes.vector import NAMES, price


def freeze(value):
    if isinstance(value,dict):
        return tuple((k,freeze(v)) for k,v in sorted(value.items()))
    if isinstance(value,list):
        return tuple(freeze(v) for v in value)
    return value


def collect(root):
    counts,examples={},{}
    for case,_,_,_ in cases(root):
        for path in sorted((case/"native").glob("*rank0_*.json.gz")):
            with gzip.open(path) as file:
                graph=json.load(file)
            tensors={v["id"]:v for v in graph["tensors"]}
            for node in graph["nodes"]:
                raw=node["name"]
                if raw not in NAMES:
                    continue
                def descriptors(side):
                    return tuple((tensors[i]["dtype"], tuple(tensors[i].get("storage_shape",tensors[i]["shape"])),
                                  tuple(tensors[i]["stride"])) for i in node[side])
                inputs, outputs=descriptors("inputs"),descriptors("outputs")
                sem=dict(node["attributes"].get("semantic",{}))
                for k in ("component","reference","reference_operator"):
                    sem.pop(k,None)
                # Preserve call-specific scalars and semantic parameters. IR
                # tensor identities are not model parameters or shape keys.
                params=dict(node["attributes"].get("parameter_input",{}))
                key=(raw,inputs,outputs,freeze(sem),freeze(params))
                counts[key]=counts.get(key,0)+1
                if key not in examples:
                    examples[key]=f'{case.name}/{node["id"]}/{path.name}'
    return counts,examples


def evaluate(counts,examples):
    targets=[load_chip(k) for k in ("Ascend950DT_9582","Ascend950PR_957b","Ascend970DT")]
    for key,count in counts.items():
        raw,ins,outs,sem,params=key
        def ts(desc):
            return tuple(SimpleNamespace(dtype=d,shape=s,stride=st,device="meta",layout="torch.strided") for d,s,st in desc)
        tensors=ts(ins),ts(outs)
        attrs=(("semantic",sem),("parameter_input",params))
        if raw=="aten._to_copy.default" and outs:
            attrs+=(("kwargs",("mapping",(("dtype",outs[0][0]),))),)
        f=SimpleNamespace(raw_name=raw,inputs=tensors[0],outputs=tensors[1],category="vec",attributes=attrs)
        estimates=[]
        for t in targets:
            result=estimate_features(f,chip=t)
            if result.latency_us is None:
                result=price(f,t)
            if result.latency_us is None:
                raise ValueError((key,t.name,result.reason))
            estimates.append(result)
        yield dict(raw_name=raw,example=examples[key],calls=count,inputs=json.dumps(ins),outputs=json.dumps(outs),
                   semantic=json.dumps(sem),parameters=json.dumps(params),contract=estimates[0].contract,
                   dt_us=estimates[0].latency_us,pr_us=estimates[1].latency_us,extrapolated_970dt_us=estimates[2].latency_us)


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("matrix",type=Path);p.add_argument("output",type=Path)
    a=p.parse_args()
    records=list(evaluate(*collect(a.matrix)))
    with gzip.open(a.output,"wt",newline="") as f:
        w=csv.DictWriter(f,fieldnames=records[0])
        w.writeheader();w.writerows(records)
