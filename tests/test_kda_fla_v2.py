import json
from dataclasses import replace
from pathlib import Path
from unittest import TestCase

from operator_model import OperatorStream, estimate, load_chip

ROOT = Path(__file__).resolve().parents[1]
ATTR = {"implementation": "fla_26_9_1_v2"}


class FLAChunkTest(TestCase):


    def test_identical_full_chunks_skip_without_losing_tail_or_transient(self):
        from operator_model.operators.kda.fla import FAMILIES, Tile
        from operator_model.operators.kda.fwdh import _advance, _prices, _repeat, schedule_chain
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
