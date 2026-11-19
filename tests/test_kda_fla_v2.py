import json
from dataclasses import replace
from pathlib import Path
from unittest import TestCase

from operator_model import OperatorStream, estimate, load_chip

ROOT = Path(__file__).resolve().parents[1]
ATTR = {"implementation": "fla_26_9_1_v2"}


class FLAChunkTest(TestCase):
    def test_v023_work_and_other_stage_prices_match_original_freeze(self):
        freeze = json.loads((ROOT / "results/kda-resource23/freeze.json").read_text())
        for row in freeze["predictions"]:
            with self.subTest(shape=row["shape"], chip=row["chip"]):
                result = estimate("kda_chunk", row["shape"], ATTR, chip=row["chip"])
                self.assertEqual(result.status, "extrapolated")
                for new, old in zip(
                    result.projection["target_plan"]["stages"], row["stages"]
                ):
                    if old["family"] != "chunk_fwd_h":
                        self.assertEqual(new["device_us"], old["device_us"])
                    self.assertEqual(new["dma_bytes"], old["dma_bytes"])
                    self.assertEqual(
                        new["mmads"],
                        {
                            k.replace("f32_tf32", "fp32"): v
                            for k, v in old["mmads"].items()
                        },
                    )
                    self.assertEqual(
                        new["estimated_rvec_subqueue_records"],
                        old["estimated_rvec_subqueue_records"],
                    )
                    self.assertEqual(
                        new["issued_mmad_ops"],
                        sum(new["issued_cube_ops_by_dtype"].values()),
                    )
                    self.assertGreaterEqual(
                        new["max_core_head_chunks"], new["head_chunk_waves_lower_bound"]
                    )

    def test_prefetched_stage_and_public_trace_match_prospective_freeze(self):
        path = ROOT / "results/v0.24/kda-prefetch/register.json"
        cases = json.loads(path.read_bytes())["cases"]
        for case in cases[::2]:
            for sku, expected in case["predictions"].items():
                with self.subTest(shape=case["shape"], sku=sku):
                    out = estimate("kda_chunk", case["shape"], ATTR, chip=sku)
                    stages = out.projection["target_plan"]["stages"]
                    self.assertEqual(out.latency_us, expected["us"])
                    self.assertEqual([s["device_us"] for s in stages], expected["stages_us"])
                    self.assertEqual([s["active_aic"] for s in stages], expected["active_aic"])
                    self.assertEqual(stages[1]["prefetch_timeline_cycles"], expected["fwd_h_trace_model_ticks"])
                    self.assertNotEqual(out.projection["prediction_freeze_sha256"], out.projection["resource_price_freeze_sha256"])
                    for before, after in zip(expected["fwd_h_trace_model_ticks"], expected["fwd_h_trace_model_ticks"][1:]):
                        self.assertEqual(after["H"][0], before["write_state"][1])
                        self.assertEqual(after["W"][0], before["mm2"][1])
                        self.assertLess(after["U"][0], before["write_state"][1])

    def test_identical_full_chunks_skip_without_losing_tail_or_transient(self):
        from operator_model.kda_fla_v2 import FAMILIES, Tile
        from operator_model.kda_fwdh_schedule import _advance, _prices, _repeat, schedule_chain
        tile = Tile(load_chip(), FAMILIES[1], 1)
        costs = _prices(64, tile)
        state = (0., 1200000., 2.)
        direct = state
        for _ in range(37):
            direct, _ = _advance(direct, costs)
        for actual, expected in zip(_repeat(state, costs, 37), direct):
            self.assertAlmostEqual(actual, expected, places=7)
        t, bars = schedule_chain(262145, 500, tile)
        self.assertEqual(len(bars), 4)
        self.assertEqual(bars[-1]["write_state"][1], t)

    def test_work_contract_and_scaling(self):
        shape = (1, 2, 1, 1, 128, 128)
        out = estimate("kda_chunk", shape, ATTR)
        self.assertEqual(out.projection["qualification"], "experimental")
        self.assertEqual(out.work["logical_read_bytes"], 68108)
        self.assertEqual(out.work["logical_write_bytes"], 66560)
        self.assertIsNone(out.work["physical_hbm_bytes"])
        stage = out.projection["target_plan"]["stages"]
        self.assertEqual(
            [(x["active_aic"], x["launched_aiv"]) for x in stage],
            [(1, 2), (1, 2), (1, 0)],
        )
        self.assertEqual(stage[2]["dma_bytes"]["OUT->L1"], 33800)
        self.assertEqual(stage[1]["issued_mmad_ops"], 1048576)
        self.assertEqual(OperatorStream().preview("kda_chunk", shape, ATTR), out)
        self.assertEqual(
            estimate("kda_chunk", shape, ATTR, timing_mode="resource"), out
        )
        self.assertIsNone(
            estimate("kda_chunk", shape, ATTR, timing_mode="calibrated").latency_us
        )
        self.assertIsNone(estimate("kda_chunk", shape).latency_us)
        chip = load_chip()
        renamed = estimate(
            "kda_chunk", shape, ATTR, chip=replace(chip, name="same-resources")
        )
        self.assertEqual(renamed.latency_us, out.latency_us)
        slow = estimate(
            "kda_chunk",
            shape,
            ATTR,
            chip=replace(chip, vector_tops={**chip.vector_tops, "32": 10}),
        )
        self.assertGreater(slow.latency_us, out.latency_us)

    def test_large_shape_mover_and_equal_packed_offsets(self):
        logical = (8, 32768, 12, 12, 128, 128)
        shape = (1, 262144, 12, 12, 128, 128)
        offsets = [32768 * i for i in range(9)]
        single = estimate("kda_chunk", logical, ATTR, chip="Ascend970DT")
        packed = estimate(
            "kda_chunk", shape, {**ATTR, "cu_seqlens": offsets}, chip="Ascend970DT"
        )
        self.assertEqual(packed.latency_us, single.latency_us)
        stages = single.projection["target_plan"]["stages"]
        self.assertEqual(stages[1]["head_chunks"], 49152)
        self.assertIn("multiple head", stages[1]["prefetch_domain"])
        self.assertEqual(stages[1]["prefetch_timeline_cycles"], [])
        self.assertTrue(stages[2]["aiv_mover"])
        self.assertGreater(stages[2]["dma_bytes"]["UB->L1"], 0)
        self.assertLessEqual(stages[2]["launched_aiv"], 120)
        for opts in (
            {"cu_seqlens": [0, 1, 3]},
            {"cu_seqlens": None, "safe_gate": 0},
            {"dtype": "fp16"},
            {"gate_dtype": "fp32"},
            {"implementation": "unknown"},
            {"chunk_size": 128},
        ):
            self.assertEqual(
                estimate("kda_chunk", (1, 3, 1, 1, 128, 128), {**ATTR, **opts}).status,
                "unsupported",
            )
