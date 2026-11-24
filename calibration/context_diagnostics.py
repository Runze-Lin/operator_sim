"""Trace a same-graph price difference and the Kimi tiling change back to captured calls."""
import argparse
from collections import Counter
from dataclasses import replace
import gzip
import json
from pathlib import Path
import pickle
from calibration.context_sweep import case_id


def run(scratch, output):
    from sim_platform.api.inference_evaluation import evaluate_inference_workload
    from sim_platform.backend.factory import build_predictor
    from sim_platform.backend.l1 import simulate_graph
    from sim_platform.config.execution import PredictionOptions
    from sim_platform.hardware.loading import load_hardware

    receipt = {}
    specs = [("v41-flash",1048576,1,32,"prefill",1044480,4096)]
    specs += [("kimi-k31",131072,32,64,"decode",131072+i*7,1) for i in (0,73,146)]
    for spec in specs:
        with gzip.open(scratch/case_id(spec)/"graph.pkl.gz","rb") as f:
            w,o,_,_ = pickle.load(f)
        hw=o.hardware
        if not hasattr(getattr(hw,"chip",None),"hbm"):
            hw=hw.to_model_hardware() if hasattr(hw,"to_model_hardware") else load_hardware(hw).to_model_hardware()
        def evaluate(name):
            p=build_predictor(PredictionOptions((name,),8192),(),hardware=hw)
            return evaluate_inference_workload(w,predictor=p,options=o)[0]
        full=evaluate("operator_sim_full")
        costs=dict(full.execution_costs)
        preds=dict(full.predictions)
        if spec[0]=="v41-flash":
            earlier=evaluate("roofline")
            legacy=dict(earlier.execution_costs)
            deltas=Counter()
            for n in full.graph.nodes:
                deltas[n.raw_name]+=legacy[n.ref].duration_us-costs[n.ref].duration_us
            capacities=dict(o.capacities)
            if not o.cube_vec_overlap:
                capacities["compute"]=1
            counterfactual = []
            for op,delta in deltas.most_common(3):
                matched=[n for n in full.graph.nodes if n.raw_name==op]
                new=dict(costs)
                for node in matched:
                    new[node.ref]=replace(costs[node.ref],duration_us=legacy[node.ref].duration_us)
                counterfactual.append(dict(name=op,nodes=len(matched),serial_delta_us=delta,
                     full_with_legacy_price_us=simulate_graph(full.graph,new,capacities).duration_us,
                     example_input_shapes=[t.shape for t in matched[0].inputs]))
            receipt[case_id(spec)]=dict(legacy_us=earlier.duration_us,full_us=full.duration_us,restore=counterfactual)
        else:
            node=next(n for n in full.graph.nodes if n.ref.node_id=="202")
            assert node.raw_name=="npu::npu_dynamic_mx_quant"
            detail=preds[node.ref].estimate.details
            receipt[case_id(spec)]=dict(stage="target",node=node.ref.node_id,raw_name=node.raw_name,
                contract=detail.model_name,input_shapes=[x.shape for x in node.inputs],
                time_us=costs[node.ref].duration_us,plan=json.loads(dict(detail.model_parameters)["target_plan"]))
    output.write_text(json.dumps(receipt,indent=2)+"\n")


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("scratch",type=Path)
    p.add_argument("output",type=Path)
    a=p.parse_args()
    run(a.scratch,a.output)
