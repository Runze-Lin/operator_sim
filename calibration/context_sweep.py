"""Capture and replay rank-local prefill chunks and speculative decode at several contexts."""
import argparse
import gzip
import json
import math
import pickle
import re
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from dataclasses import replace
from pathlib import Path

MODELS = ("v4-pro", "v41-flash", "kimi-k31")
# Length, per-request prefill chunk, second local batch and EP (TP is 8).
GRID = ((32768,4096,16,32), (131072,4096,4,32), (524288,4096,4,64), (1048576,4096,4,64))
OUTPUT = 1024
ACCEPT = 7
# Initial endpoint-to-midpoint check exceeded 5% in these deployment scenarios.
# This refines only the offline experiment; the operator schedules see no case IDs.
EXACT_DECODE = {("kimi-k31",131072,32,64),("kimi-k31",32768,16,32),("kimi-k31",32768,8,32)}


def profiles():
    for model in MODELS:
        for length, chunk, batch, ep in GRID:
            cohorts = [(1,32), (batch,ep)]
            cohorts += {32768:[(8,32)],131072:[(32,64)],524288:[(4,32)]}.get(length,[])
            for b,e in cohorts:
                yield model, length, b, e, chunk


def points(length, chunk):
    count = length//chunk
    # Capture the distinct first chunk explicitly. Interior midpoint checks are
    # reserved for validation of the interpolation between the three hot anchors.
    hot = (1, count//2, count-1)
    sample = range(count) if count <= 8 else sorted({0,*hot,(1+hot[1])//2,(hot[1]+hot[2])//2})
    return list(sample)


def inputs(profile):
    model, length, batch, ep, chunk = profile
    for i in points(length,chunk):
        yield model,length,batch,ep,"prefill",i*chunk,chunk
    # Prefill emits output token one. 1023 more tokens consume 147 rounds at 7.
    rounds = math.ceil((OUTPUT-1)/ACCEPT)
    # All three presets end at 1 Mi positions. Sample nearby *valid* decode
    # shapes, leaving room for all draft positions. Extrapolate only in the report.
    start = length - 2*OUTPUT if length == 1048576 else length
    for i in (range(rounds) if profile[:4] in EXACT_DECODE else (0,rounds//2,rounds-1)):
        yield model,length,batch,ep,"decode",start+i*ACCEPT,1


def case_id(spec):
    m,length,b,ep,phase,history,q = spec
    return f"{m}-{length}-b{b}-tp8ep{ep}-{phase}-kv{history}-q{q}"


def capture(job):
    from sim_platform.api.inference import run_inference
    from sim_platform.api.inference_evaluation import evaluate_inference_workload
    from sim_platform.backend.factory import build_predictor
    from sim_platform.config.inference import load_inference_task
    matrix, scratch, spec = job
    m,length,b,ep,phase,history,q = spec
    dest = scratch/case_id(spec)
    dest.mkdir(parents=True, exist_ok=True)
    source = matrix/f"{m}-tp8-ep32-b1/task.toml"
    raw = source.read_text()
    if phase == "prefill":
        raw = re.sub(r"\[inference.speculation\][\s\S]*?(?=\[platform\])", "", raw)
    for name, value in (("phase", json.dumps(phase)), ("input_seq_len",q), ("kv_seq_len",history),
                        ("kv_capacity",length+OUTPUT+18), ("batch_size",b), ("dp",ep//8)):
        raw, replacements = re.subn(rf"^{name} = .*$", f"{name} = {value}", raw, flags=re.M)
        if replacements != 1:
            raise ValueError(f"{source}: ambiguous {name}")
    task = dest/"task.toml"
    saved = dest/"graph.pkl.gz"
    if saved.exists() and task.read_text()==raw:
        return
    task.write_text(raw)
    config = load_inference_task(task, overrides={"output.directory":str(dest/"run"), "exports.enabled":False,
                        "prediction.backends":["roofline"],"logging.console_level":"ERROR"})
    # The optional timeline attribution in the preview rejects long clocks at a
    # fixed 1e-9 us absolute tolerance. Use its graph capture and DES entry points
    # directly, and check each interval to the precision of its absolute clock.
    run = run_inference(replace(config,options=None))
    options = replace(config.options,hardware=config.hardware_variants[0].hardware.to_model_hardware())
    predictor = build_predictor(config.prediction, (),hardware=options.hardware)
    evaluations = evaluate_inference_workload(run.workload,predictor=predictor,options=options)
    original = []
    for ev in evaluations:
        costs = dict(ev.execution_costs)
        for entry in ev.l1_result.timeline:
            assert math.isclose(entry.end_us-entry.start_us,costs[entry.ref].duration_us,rel_tol=1e-9,
                                abs_tol=max(1e-9,2*math.ulp(entry.end_us)))
        statuses = Counter(p.status for _,p in ev.predictions)
        original.append(dict(raw_us=ev.duration_us,statuses=statuses,nodes=len(ev.graph.nodes)))
    assert len(evaluations) == (1 if phase == "prefill" else 2)
    assert all(s.execution_count == 1 for s in run.summaries)
    tmp = dest/"graph.tmp"
    # Local scratch only. Load only this run's trusted native capture.
    with gzip.open(tmp,"wb",compresslevel=1) as f:
        pickle.dump((run.workload,options,original,run.static_memory[0].peak_bytes),f)
    tmp.replace(saved)


def replay(job):
    from calibration.roofline_audit import nominal_engine_us
    from calibration.system_compare import near
    from operator_model.system_routes import STRUCTURAL, communication
    from sim_platform.api.inference_evaluation import evaluate_inference_workload
    from sim_platform.backend.factory import build_predictor
    from sim_platform.backend.l1 import simulate_graph
    from sim_platform.config.execution import PredictionOptions
    from sim_platform.hardware.loading import load_hardware

    _, scratch, spec = job
    dest = scratch/case_id(spec)
    with gzip.open(dest/"graph.pkl.gz","rb") as f:
        workload, options, old, memory = pickle.load(f)
    hw = options.hardware
    if not hasattr(getattr(hw, "chip", None), "hbm"):
        hw = hw.to_model_hardware() if hasattr(hw, "to_model_hardware") else load_hardware(hw).to_model_hardware()
    name = "operator_sim_full"
    predictor = build_predictor(PredictionOptions((name,),8192), (), hardware=hw)
    caps = dict(options.capacities)
    if not options.cube_vec_overlap:
        caps["compute"] = 1
    evaluations = evaluate_inference_workload(workload, predictor=predictor, options=options)
    peaks = [dict(unit.tflops_by_bits) for unit in (hw.chip.cube,hw.chip.vec)]
    operators = Counter()
    failures = Counter()
    below = Counter()
    plans = {}
    rows = []
    for ev, baseline in zip(evaluations,old,strict=True):
        costs = dict(ev.execution_costs)
        prices = costs.copy()
        effects = costs.copy()
        preds = dict(ev.predictions)
        assert len(ev.graph.nodes) == len(costs) == baseline["nodes"]
        local = adjusted = 0
        for n in ev.graph.nodes:
            op = n.raw_name
            operators[op] += 1
            pred = preds.get(n.ref)
            if not pred or not pred.estimate or pred.backend_name != name:
                failures[(op,str(pred.error if hasattr(pred,"error") else getattr(pred,"status","absent")))] += 1
                continue
            if op in STRUCTURAL or op in communication.NAMES:
                continue
            d = pred.estimate.details
            tc = nominal_engine_us(d.cube_work_by_dtype, peaks[0])
            tv = nominal_engine_us(d.vec_work_by_dtype, peaks[1])
            tm = (d.read_bytes+d.write_bytes)/(hw.chip.hbm.peak_bytes_per_second/1e6)
            if tc is None or tv is None:
                raise ValueError(f"Unpriced computation for {op}")
            price = max(tc,tv,tm)
            eff = max(tc/hw.chip.cube.utilization.ratio,tv/hw.chip.vec.utilization.ratio,tm/hw.chip.hbm.utilization.ratio)
            cost = costs[n.ref]
            prices[n.ref] = replace(cost,duration_us=price)
            effects[n.ref] = replace(cost,duration_us=eff)
            local += 1
            params = dict(d.model_parameters)
            window = json.loads(params["memory_window"])
            if window["exposed_io_us"] > 1e-9:
                adjusted += 1
            if cost.duration_us < price and not near(cost.duration_us,price):
                term = ("cube","vector","memory")[(tc,tv,tm).index(price)]
                below[(op,d.model_name,term)] += 1
            if op in {"npu::npu_quant_matmul","npu::npu_grouped_matmul","npu::npu_dynamic_mx_quant","inference::candidate_filtered_indexer",
                      "vllm_ascend::chunk_kda_fwd_reference","npu::npu_rms_norm","aten.mm.default"}:
                key = (op,d.model_name)
                shape = [list(t.shape) for t in n.inputs[:2]]
                plan = {}
                for k,v in params.items():
                    if k == "target_plan":
                        z=json.loads(v)
                        if isinstance(z,dict):
                            plan.update({a:w for a,w in z.items() if type(w) in (float,int,str,bool) or
                                         (isinstance(w,list) and len(w)<=4 and all(type(x) is int for x in w))})
                    elif k in ("active_aic","active_aiv","waves","tile_elements","query_groups","query_tile",
                               "kv_page","groups","tile_mnk","k_pages_per_split","output_tiles","split_k",
                               "state_phase_aic","chunk_tokens","chunk_head_tasks","max_serial_chunks","state_v_stripe"):
                        plan[k]=json.loads(v) if isinstance(v,str) and v.startswith("[") else v
                if key not in plans and plan:
                    plans[key] = dict(name=op,contract=d.model_name,input_shapes=shape,plan=plan)
        rows.append(dict(legacy_us=baseline["raw_us"],full_us=ev.duration_us,
                         peak_us=simulate_graph(ev.graph,prices,caps).duration_us,
                         effective_us=simulate_graph(ev.graph,effects,caps).duration_us,
                         nodes=len(costs),local=local,extended=adjusted,legacy_status=baseline["statuses"]))
    receipt=dict(spec=spec,stages=rows,peak_bytes=memory,operators=dict(operators),
                 failures=[[a,b,c] for (a,b),c in failures.items()],
                 below=[[a,b,c,v] for (a,b,c),v in below.items()],plans=list(plans.values()))
    (dest/"quote.json").write_text(json.dumps(receipt,allow_nan=False))
    return len(failures),len(below)


def study(job):
    capture(job)
    return replay(job)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode",choices=["capture","replay","study"])
    parser.add_argument("matrix",type=Path)
    parser.add_argument("scratch",type=Path)
    parser.add_argument("--workers",type=int,default=5)
    args=parser.parse_args()
    specs=[s for profile in profiles() for s in inputs(profile)]
    jobs=[(args.matrix.resolve(),args.scratch.resolve(),s) for s in specs]
    with ProcessPoolExecutor(args.workers) as pool:
        worker = {"capture":capture,"replay":replay,"study":study}[args.mode]
        for spec,result in zip(specs,pool.map(worker,jobs)):
            if result and any(result):
                raise ValueError(f"{case_id(spec)} needs investigation: incomplete contracts or below-peak calls {result}")


if __name__=="__main__":
    main()
