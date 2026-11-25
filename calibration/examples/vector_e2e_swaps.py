"""Switch one operator's corrected node costs and rerun the same graph scheduler."""

import argparse
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor
from functools import partial
from pathlib import Path

from calibration.candidate_indexer.context_sensitivity import bundle, caps, hardware, page_table
from calibration.context_report import write
from calibration.context_sweep import MODELS
from calibration.examples.context_vector_e2e import MODES, ROOT


def swap(spec, scratch):
    from operator_model.sim_backend import FullOperatorSimBackend
    from sim_platform.api.inference_evaluation import evaluate_inference_workload
    from sim_platform.backend.factory import build_predictor
    from sim_platform.backend.l1 import simulate_graph
    from sim_platform.config.execution import PredictionOptions

    workload, options, _, _ = bundle(scratch, spec)
    hw = hardware(options)
    _, pages = page_table(scratch, spec)
    model, prompt, batch, ep, phase, history, _ = spec
    by_mode = {}
    for mode in MODES:
        backend = FullOperatorSimBackend(
            vector_model=mode,
            memory_policy="cold_io",
            candidate_implementation="a5_qsli_projection",
            candidate_options={"pa_block_size": 128, "table_pages": pages},
        )
        predictor = build_predictor(PredictionOptions((backend.name,), 8192), (backend,), hardware=hw)
        evaluations = evaluate_inference_workload(workload, predictor=predictor, options=options)
        by_mode[mode] = [(r.duration_us, dict(r.execution_costs)) for r in evaluations]
        if any(p.status != "success" for r in evaluations for _, p in r.predictions):
            raise ValueError("Unpriced node in counterfactual")
    results = []
    for base, changed in (("calibrated", "auto"), ("auto", "instruction_classes")):
        refs = defaultdict(list)
        diffs = Counter()
        for i, graph in enumerate(workload.graphs):
            for node in graph.nodes:
                a, b = (by_mode[m][i][1][node.ref] for m in (base, changed))
                if a.resources != b.resources or a.category != b.category:
                    raise ValueError("A method changed the node schedule resources")
                refs[node.raw_name].append((i, node.ref))
                diffs[node.raw_name] += b.duration_us - a.duration_us
        before, after = (sum(d for d, _ in by_mode[mode]) for mode in (base, changed))
        for name in sorted(diffs, key=lambda n: -abs(diffs[n]))[:3]:
            costs = [dict(c) for _, c in by_mode[base]]
            for i, ref in refs[name]:
                costs[i][ref] = by_mode[changed][i][1][ref]
            elapsed = sum(
                simulate_graph(g, c, caps(options)).duration_us
                for g, c in zip(workload.graphs, costs, strict=True)
            )
            results.append(
                dict(
                    model=model,
                    prompt=prompt,
                    batch=batch,
                    ep=ep,
                    phase=phase,
                    history=history,
                    base_mode=base,
                    changed_mode=changed,
                    raw_name=name,
                    base_us=before,
                    changed_all_us=after,
                    nodes=len(refs[name]),
                    serial_delta_us=diffs[name],
                    swap_us=elapsed,
                    swap_delta_us=elapsed - before,
                )
            )
    return results


def main():
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("scratch", type=Path)
    cli.add_argument("--workers", type=int, default=3)
    cli.add_argument("--output", type=Path, default=ROOT / "results/v2.3/system-e2e")
    args = cli.parse_args()
    cases = [
        (m, l, 1, 32, p, h, q)
        for m in MODELS
        for l, p, h, q in ((1048576, "prefill", 1044480, 4096), (131072, "decode", 131072, 1))
    ]
    with ProcessPoolExecutor(args.workers) as pool:
        rows = [r for group in pool.map(partial(swap, scratch=args.scratch), cases) for r in group]
    args.output.mkdir(exist_ok=True, parents=True)
    write(args.output / "counterfactuals.csv", rows)


if __name__ == "__main__":
    main()
