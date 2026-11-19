import json
from dataclasses import replace
from pathlib import Path
from unittest import TestCase

from operator_model import OperatorStream, estimate, load_chip

ROOT = Path(__file__).resolve().parents[1]
ATTR = {"implementation": "fla_26_9_1_v2"}


class FLAChunkTest(TestCase):
    def test_all_preobserved_shapes_match_freeze_on_three_skus(self):
        freeze = json.loads((ROOT / "results/kda-resource23/freeze.json").read_text())
        for row in freeze["predictions"]:
            with self.subTest(shape=row["shape"], chip=row["chip"]):
                result = estimate("kda_chunk", row["shape"], ATTR, chip=row["chip"])
                self.assertEqual(result.status, "extrapolated")
                self.assertEqual(result.latency_us, row["serialized_device_us"])
                for new, old in zip(
                    result.projection["target_plan"]["stages"], row["stages"]
                ):
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
