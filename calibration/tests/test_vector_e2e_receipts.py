"""Check published system comparisons independently of the SIMPlatform and CA runtime."""

import csv
import json
from collections import defaultdict
from pathlib import Path
import unittest

from calibration.context_report import linear
from calibration.context_sweep import ACCEPT, EXACT_DECODE, OUTPUT, profiles


ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "results/v2.3/system-e2e"
MODES = ("calibrated", "auto", "instruction_classes")


def rows(name):
    with (OUT / name).open() as f:
        return list(csv.DictReader(f))


def key(row):
    return row["model"], int(row["prompt"]), int(row["batch"]), int(row["ep"])


class SystemComparisons(unittest.TestCase):
    def test_reintegrate_all_published_stages(self):
        requests = {key(r): r for r in rows("requests.csv")}
        entries = rows("graphs.csv")
        summary = json.loads((OUT / "sweep.json").read_text())
        self.assertEqual(len(entries), summary["stage_graphs"])
        self.assertEqual(
            sum(int(e["nodes"]) for e in entries), summary["priced_node_instances_per_fine_mode"]
        )
        grouped = defaultdict(list)
        self.assertEqual(set(requests), {p[:4] for p in profiles()})
        for e in entries:
            grouped[key(e), e["phase"]].append(e)
            for mode in MODES:
                self.assertGreaterEqual(float(e[mode + "_us"]) + 1e-6, float(e["peak_us"]))
        for profile in profiles():
            model, length, batch, ep, chunk = profile
            request = requests[profile[:4]]
            for lane in (*MODES, "peak", "effective", "legacy"):
                totals = []
                for phase in ("prefill", "decode"):
                    graph = defaultdict(dict)
                    for e in grouped[profile[:4], phase]:
                        pos = int(e["history"])
                        self.assertNotIn(e["stage"], graph[pos])
                        graph[pos][e["stage"]] = float(e[lane + "_us"])
                    required = {"prefill"} if phase == "prefill" else {"target", "draft"}
                    self.assertTrue(all(set(values) == required for values in graph.values()))
                    times = {h: sum(stages.values()) for h, stages in graph.items()}
                    positions = (
                        range(0, length, chunk)
                        if phase == "prefill"
                        else range(length, length + ACCEPT * int(request["decode_rounds"]), ACCEPT)
                    )
                    if phase == "decode" and profile[:4] in EXACT_DECODE:
                        self.assertEqual(set(times), set(positions))
                    elapsed = sum(times[h] if h in times else linear(times, h) for h in positions) / 1e6
                    self.assertAlmostEqual(elapsed, float(request[f"{phase}_{lane}_s"]), places=7)
                    totals.append(elapsed)
                self.assertAlmostEqual(sum(totals), float(request[f"request_{lane}_s"]), places=7)
            self.assertEqual(int(request["output"]), OUTPUT)

    def test_counterfactuals_use_same_graph_as_primary_replay(self):
        groups = defaultdict(lambda: defaultdict(float))
        for e in rows("graphs.csv"):
            g = groups[key(e), e["phase"], int(e["history"])]
            for m in MODES:
                g[m] += float(e[m + "_us"])
        for r in rows("counterfactuals.csv"):
            stage = groups[key(r), r["phase"], int(r["history"])]
            self.assertAlmostEqual(float(r["base_us"]), stage[r["base_mode"]], places=6)
            self.assertAlmostEqual(float(r["changed_all_us"]), stage[r["changed_mode"]], places=6)
            self.assertAlmostEqual(
                float(r["swap_delta_us"]), float(r["swap_us"]) - float(r["base_us"]), places=6
            )

    def test_original27_rounds(self):
        current = rows("original27.csv")
        with (ROOT / "calibration/reference/context-950dt/original-27.csv").open() as f:
            old = {r["case"]: r for r in csv.DictReader(f)}
        self.assertEqual(len(current), 27)
        self.assertEqual(set(old), {r["case"] for r in current})
        for r in current:
            self.assertAlmostEqual(float(r["historical_us"]), float(old[r["case"]]["original_us"]), places=5)
            for mode in MODES:
                self.assertAlmostEqual(
                    float(r[mode + "_us"]),
                    float(r[mode + "_target_us"]) + float(r[mode + "_draft_us"]),
                    places=7,
                )
