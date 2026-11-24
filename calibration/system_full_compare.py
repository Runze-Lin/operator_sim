"""Replay both DES backends and verify complete attribution against the 27 native graphs."""
import argparse
import csv
import gzip
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor
from hashlib import sha256
import json
from pathlib import Path
from time import perf_counter

from calibration.system_compare import cases, near

FULL = "operator_sim_full"


def capture(job):
    import tomllib
    from sim_platform.api.inference import run_inference
    from sim_platform.backend.prediction import convert_node_features
    from sim_platform.hardware.loading import load_hardware
    from operator_model.sim_backend import FullOperatorSimBackend

    case, model, layout, batch, mode, root = job
    directory = root/mode/case.name
    backend = "roofline" if mode == "roofline" else FULL
    samples = {}
    if mode == "full":
        config = tomllib.loads((case/"task.toml").read_text())
        hardware = load_hardware(case / config["platform"]["hardware"]).to_model_hardware()
        backend_probe = FullOperatorSimBackend()
    start = perf_counter()
    result = run_inference(case/"task.toml",overrides={"output.directory":str(directory),
                           "exports.enabled": False, "prediction.backends":[backend]})
    wall = perf_counter()-start
    [stat], [memory] = result.statistics, result.static_memory
    archive = case/"results/l1"
    previous = json.loads((archive/"inference.json").read_text())["statistics"][0]
    assert stat.local_batch_size==batch and near(memory.peak_bytes/1e9, previous["memory"]["peak_gb"])
    assert len(result.evaluations)==2 and stat.accepted_length==7
    ops = defaultdict(Counter)
    methods = defaultdict(Counter)
    errors = {}
    graph_hash = sha256()
    for idx, ((stage,_,repeat),ev) in enumerate(zip(stat.stage_durations_us,result.evaluations,strict=True)):
        assert repeat==1 and stage==("target","draft")[idx]
        with (archive/f"{idx:03d}_verify_{stage}.csv").open(newline="") as inp:
            original = {r["id"]:r for r in csv.DictReader(inp) if r["record_type"]=="operator"}
        assert len(original)==len(ev.graph.nodes)
        for entity in (*ev.graph.nodes,*ev.graph.dependencies):
            graph_hash.update(repr(entity).encode()+b"\n")
        priced, costs = dict(ev.predictions), dict(ev.execution_costs)
        for node in ev.graph.nodes:
            r=original[node.ref.node_id]
            assert node.raw_name==r["raw_name"] and ev.graph.graph_id==r["graph_id"]
            pred=priced.get(node.ref)
            state=pred.status if pred else "skipped_"+node.category
            value=costs[node.ref].duration_us
            old=float(r["duration_us"])
            if backend=="roofline" and not near(old,value):
                raise ValueError(f"Changed original node {node.ref}: {old} vs {value}")
            name=node.raw_name
            tally=ops[name]
            tally.update(dict(nodes=1,old_us=old,new_us=value))
            if pred and state=="success" and pred.backend_name==backend:
                tally["hits"]+=1
                tally["zero_nodes"]+=value==0
                detail=pred.estimate.details
                method=detail.model_name if detail else pred.backend_name
                methods[name][method]+=1
                if mode=="full":
                    def geometry(tensors):
                        return json.dumps([[t.dtype.removeprefix("torch."),list(t.shape)] for t in tensors],separators=(",",":"))
                    key=(stage,name,method,geometry(node.inputs),geometry(node.outputs))
                    if key not in samples:
                        features=convert_node_features(node)
                        begin=perf_counter()
                        quote=backend_probe.predict(features,hardware)
                        query_ms=(perf_counter()-begin)*1000
                        assert near(quote.duration_us,pred.estimate.duration_us),node.ref
                        samples[key]=[0,0.,value,value,query_ms]
                    sample=samples[key]
                    sample[0]+=1;sample[1]+=value;sample[2]=min(sample[2],value);sample[3]=max(sample[3],value)
            else:
                tally[state]+=1
                if backend==FULL and name not in errors:
                    errors[name]=[state, pred.error_message if pred else None, pred.error_context if pred else None]
    total=sum(c["new_us"] for c in ops.values())
    assert near(sum(c["old_us"] for c in ops.values()),previous["duration_us"])
    assert stat.duration_us <= total + 1e-4, (case,stat.duration_us,total)
    row=dict(case=case.name,model=model,layout=layout,batch=batch,wall_s=wall,nodes=sum(o["nodes"] for o in ops.values()),
             us=stat.duration_us,serial_us=total,target_us=stat.stage_durations_us[0][1],draft_us=stat.stage_durations_us[1][1],
             tpot_us=stat.tpot_us,peak_gb=memory.peak_bytes/1e9,graph_sha256=graph_hash.hexdigest(),
             task_sha256=sha256((case/"task.toml").read_bytes()).hexdigest(),ops=ops,methods=methods,errors=errors)
    (directory/"comparison.json").write_text(json.dumps(row,indent=1)+"\n")
    if samples:
        with gzip.open(directory/"shapes.csv.gz","wt",newline="") as f:
            writer=csv.writer(f,lineterminator="\n")
            writer.writerow(("stage","raw_name","method","physical_inputs","physical_outputs","nodes","sum_us","min_us","max_us","first_uncached_query_ms"))
            writer.writerows((*key,*v) for key,v in sorted(samples.items()))
    if backend == FULL and errors:
        raise ValueError(f"{case.name}: incomplete operator coverage. See {directory / 'comparison.json'}")
    return f"{case.name}: {stat.duration_us:.1f} µs, {len(errors)} unpriced names: {list(errors)}"


def write(path, rows):
    with path.open("w",newline="") as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]),lineterminator="\n");w.writeheader();w.writerows(rows)


def compare(matrix, scratch, output):
    output.mkdir(parents=True,exist_ok=True)
    reference=Path(__file__).with_name("reference")/"system-candidate-comparison.csv"
    with reference.open() as f:
        old={r["case"]:r for r in csv.DictReader(f)}
    rows=[]; totals=defaultdict(Counter); contracts=defaultdict(Counter)
    signatures=[]
    for case,model,layout,batch in cases(matrix):
        a,b=[json.loads((scratch/mode/case.name/"comparison.json").read_text()) for mode in ("roofline","full")]
        assert not b["errors"], (case,b["errors"])
        assert near(float(old[case.name]["roofline_us"]),a["us"])
        for key in ("nodes","graph_sha256","task_sha256","peak_gb"):
            assert a[key]==b[key],(key,case)
        assert a["task_sha256"]==old[case.name]["task_sha256"]
        with gzip.open(scratch/"full"/case.name/"shapes.csv.gz","rt",newline="") as inp:
            packet=[dict(case=case.name,**v) for v in csv.DictReader(inp)]
        assert sum(int(p["nodes"]) for p in packet)==b["nodes"]
        signatures.extend(packet)
        hits=0; structural=0; net=0
        for name,data in b["ops"].items():
            assert data["hits"]==data["nodes"]
            assert near(a["ops"][name]["new_us"],data["old_us"])
            hits+=data["hits"]; structural+=data["zero_nodes"]
            net+=data["new_us"] if name.startswith("collective::") or "moe_distribute_" in name else 0
            totals[name].update(data);contracts[name].update(b["methods"][name])
        rows.append(dict(case=case.name,model=model,layout=layout,batch=batch,nodes=b["nodes"],full_hits=hits,
                         zero_duration=structural,original_us=a["us"],hybrid_v18_us=old[case.name]["hybrid_new_us"],
                         full_us=b["us"],full_serial_us=b["serial_us"],overlap_us=b["serial_us"]-b["us"],delta_pct=100*(b["us"]/a["us"]-1),target_us=b["target_us"],
                         draft_us=b["draft_us"],tpot_us=b["tpot_us"],comm_operator_us=net,peak_gb=b["peak_gb"],fits_144gb=b["peak_gb"]<=144,
                         graph_sha256=b["graph_sha256"],task_sha256=b["task_sha256"],roofline_wall_s=a["wall_s"],
                         full_wall_s=b["wall_s"]))
    op_rows=[dict(raw_name=name,nodes=c["nodes"],full_hits=c["hits"],zero_duration=c["zero_nodes"],
                  roofline_us=c["old_us"],full_us=c["new_us"],delta_us=c["new_us"]-c["old_us"],
                  methods=json.dumps(dict(sorted(contracts[name].items())),sort_keys=True))
             for name,c in sorted(totals.items(),key=lambda pair:-pair[1]["new_us"])]
    assert len(rows)==27 and len(op_rows)==80
    # Reproducible gzip header for the all-shapes ledger.
    import io
    with gzip.GzipFile(str(output/"system-full-shapes.csv.gz"),"wb",mtime=0) as raw:
        with io.TextIOWrapper(raw,encoding="utf-8",newline="") as f:
            writer=csv.DictWriter(f,fieldnames=list(signatures[0]),lineterminator="\n")
            writer.writeheader();writer.writerows(signatures)
    write(output/"system-full-comparison.csv",rows);write(output/"system-full-operators.csv",op_rows)


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("matrix",type=Path);parser.add_argument("scratch",type=Path)
    parser.add_argument("mode",choices=["roofline","full","compare"]);parser.add_argument("--jobs",type=int,default=3)
    parser.add_argument("--case",action="append");parser.add_argument("--output",type=Path,default=Path(__file__).with_name("reference"))
    a=parser.parse_args()
    matrix,scratch=a.matrix.resolve(),a.scratch.resolve()
    if matrix==scratch or matrix in scratch.parents or scratch in matrix.parents: parser.error("Use separate scratch and source.")
    if a.mode=="compare": compare(matrix,scratch,a.output)
    else:
        jobs=[(*item,a.mode,scratch) for item in cases(matrix) if not a.case or item[0].name in a.case]
        with ProcessPoolExecutor(max_workers=a.jobs) as pool:
            for report in pool.map(capture,jobs): print(report,flush=True)
