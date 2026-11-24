"""Reprice the four Flash consumers in one captured DES graph with a measured-wave scenario."""
import argparse
import gzip
import json
import pickle
from dataclasses import replace
from pathlib import Path

from calibration.candidate_indexer.context_sensitivity import bundle, caps, case_id

SPEC = ("v41-flash", 1048576, 1, 32, "prefill", 1044480, 4096)
HERE = Path(__file__).resolve().parent


def replay(scratch, cache, wave):
    from sim_platform.backend.l1 import simulate_graph

    workload, options, _, _ = bundle(scratch, SPEC)
    graph, = workload.graphs
    with gzip.open(cache / (case_id(SPEC) + ".gz"), "rb") as stream:
        stage, = pickle.load(stream)
    base = {node.ref: stage[node.ref.node_id][0] for node in graph.nodes}
    consumers = [ref for ref in base if stage[ref.node_id][3] is not None]
    if len(consumers) != 4:
        raise ValueError("Expected exactly four Flash candidate consumers")
    if wave["shape"] != [1, 32, 32, 128, SPEC[1], 2048, 512]:
        raise ValueError("Measured wave must have 32 queries, the same context and candidate cap")

    waves = SPEC[-1] // wave["shape"][1]
    wave_us = wave["device_us"]
    scenarios = []
    # For successive waves, retain the full standalone cost or credit the
    # model's ordinary-QLI startup proxy (8568 cycles at 1650 cycles/us).
    for credit in (0.0, 8568 / 1650):
        consumer_us = waves * wave_us - (waves - 1) * credit
        costs = base.copy()
        for ref in consumers:
            costs[ref] = replace(costs[ref], duration_us=consumer_us)
        duration = simulate_graph(graph, costs, caps(options)).duration_us
        scenarios.append(dict(standalone_wave_ca_us=wave_us, waves=waves,
                              credit_per_additional_wave_us=credit,
                              projected_single_consumer_us=consumer_us, des_last_chunk_us=duration))
    return dict(case=case_id(SPEC), wave_trace_set_sha256=wave["trace_set_sha256"], scenarios=scenarios,
                scope="Repeat the measured native 32-row wave in a system projection. The fixture shares "
                      "candidate locations across rows. Other graph nodes retain their prices. Shared HBM, "
                      "cache and cross-wave stalls can change. These scenarios do not bound the full chunk.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("scratch", type=Path)
    parser.add_argument("cache", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    receipt = json.loads((HERE / "prefill-wave.json").read_text())
    args.output.write_text(json.dumps(replay(args.scratch, args.cache, receipt), indent=2) + "\n")
