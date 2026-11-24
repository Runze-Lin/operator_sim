"""Summarize DES anchors, validate interpolation, and aggregate complete requests."""
import argparse
import csv
import json
from bisect import bisect_right
from collections import Counter
from math import gcd
from pathlib import Path

from calibration.context_sweep import ACCEPT, EXACT_DECODE, OUTPUT, case_id, inputs, profiles

METRICS = ("legacy", "peak", "effective", "full")


def linear(samples, x):
    """Piecewise linear time at an integer position, using the last segment outside the preset."""
    keys = sorted(samples)
    i = min(len(keys)-2, max(0, bisect_right(keys,x)-1))
    x0,x1 = keys[i:i+2]
    return samples[x0]+(x-x0)*(samples[x1]-samples[x0])/(x1-x0)


def error_percent(value, reference):
    return 100*abs(value-reference)/reference


def write(path, rows):
    with path.open("w",newline="") as out:
        w = csv.DictWriter(out,fieldnames=list(rows[0]),lineterminator="\n")
        w.writeheader()
        w.writerows(rows)


def summarize(scratch, output):
    records, graphs, windows, plans = [], [], [], []
    periods = []
    coverage = Counter()
    for profile in profiles():
        model,length,b,ep,chunk = profile
        profile_id = dict(model=model,prompt=length,output=OUTPUT,batch=b,tp=8,ep=ep)
        row = dict(**profile_id,prefill_chunks=length//chunk,decode_rounds=(OUTPUT-2)//ACCEPT+1)
        ys = {phase:{k:{} for k in METRICS} for phase in ("prefill","decode")}
        peaks = Counter()
        for spec in inputs(profile):
            q = json.loads((scratch/case_id(spec)/"quote.json").read_text())
            if q["failures"] or q["below"]:
                raise ValueError(f"{case_id(spec)}: failed/unpriced/below aligned peak: {q['failures']}, {q['below']}")
            assert list(spec) == q["spec"]
            phase,history,query = spec[4:]
            coverage.update(q["operators"])
            pos = history//chunk if phase=="prefill" else history
            peaks[phase] = max(peaks[phase],q["peak_bytes"])
            for stage, r in enumerate(q["stages"]):
                assert r["full_us"] >= r["peak_us"]*(1-1e-10)
                graphs.append(dict(**profile_id,phase=phase,kv_history=history,query=query,
                    stage=("prefill" if phase=="prefill" else ("target","draft")[stage]),peak_bytes=q["peak_bytes"],
                    **{k:v for k,v in r.items() if k!="legacy_status"},**{"legacy_"+s:r["legacy_status"].get(s,0)
                    for s in ("success","placeholder","miss","fail")}))
            if spec == next(s for s in inputs(profile) if s[4]==phase):
                for plan in q["plans"]:
                    plans.append(dict(**profile_id,phase=phase,kv_history=history,name=plan["name"],
                                contract=plan["contract"],input_shapes=json.dumps(plan["input_shapes"]),
                                plan=json.dumps(plan["plan"],sort_keys=True)))
            for k in METRICS:
                ys[phase][k][pos] = sum(r[k+"_us"] for r in q["stages"])
        for phase,measures in ys.items():
            for k,series in measures.items():
                xs = sorted(series)
                if phase=="prefill":
                    count = row["prefill_chunks"]
                    hot={x:series[x] for x in (1,count//2,count-1)}
                    held=[x for x in xs if x and x not in hot]
                    total=sum(series[x] if x in series else linear(series,x) for x in range(count))
                else:
                    hot={x:series[x] for x in (xs[0],xs[-1])}
                    held=[xs[len(xs)//2]]
                    total=sum(linear(series,length+i*ACCEPT) for i in range(row["decode_rounds"]))
                for x in held:
                    windows.append(dict(**profile_id,phase=phase,backend=k,position=x,reference_us=series[x],
                        interpolated_us=linear(hot,x),ape_pct=error_percent(linear(hot,x),series[x])))
                row[phase+"_"+k+"_s"]=total/1e6
            row[phase+"_peak_GB"]=peaks[phase]/1e9
        if profile[:4] in EXACT_DECODE:
            # In these runs rows advance by B*7 and the quant tiler can split
            # over 64 AIVs. Test the resulting parity/residue hypothesis against
            # every remaining real round; keep the final total fully enumerated.
            nrounds=row["decode_rounds"]
            period=64//gcd(64,b*ACCEPT)
            series=ys["decode"]["full"]
            values={i:series[length+i*ACCEPT] for i in range(nrounds)}
            held_errors=[]
            predicted=0.
            for residue in range(period):
                ids=list(range(residue,nrounds,period))
                anchors={i:values[i] for i in (ids[0],ids[-1])}
                estimates=[linear(anchors,i) for i in ids]
                held_errors.extend(error_percent(z,values[i]) for i,z in zip(ids[1:-1],estimates[1:-1]))
                predicted+=sum(estimates)
            actual=sum(values.values())
            periods.append(dict(**profile_id,rounds=nrounds,period=period,anchors=2*period,
                held_rounds=len(held_errors),held_max_ape_pct=max(held_errors),
                exact_decode_s=actual/1e6,interpolated_decode_s=predicted/1e6,
                total_error_pct=100*(predicted/actual-1)))
        for k in METRICS:
            row["request_"+k+"_s"]=row["prefill_"+k+"_s"]+row["decode_"+k+"_s"]
        row["fits_144GB"]=max(peaks.values())<=144e9
        row["decode_sampling"]="all_rounds" if profile[:4] in EXACT_DECODE else "three_rounds"
        row["decode_positions"]="beyond_preset_extrapolation" if length==1048576 else "in_preset"
        records.append(row)
    output.mkdir(parents=True,exist_ok=True)
    for filename,data in (("requests",records),("graphs",graphs),("interpolation-check",windows),("plans",plans),("decode-tiling-check",periods)):
        write(output/(filename+".csv"),data)
    write(output/"coverage.csv",[dict(raw_name=k,nodes=v) for k,v in sorted(coverage.items())])


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("scratch",type=Path)
    p.add_argument("output",type=Path)
    a=p.parse_args()
    summarize(a.scratch,a.output)
