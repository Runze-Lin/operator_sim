from dataclasses import replace
import unittest

from operator_model import estimate, load_chip


class SigmoidResourceTest(unittest.TestCase):
    def test_flattening_work_and_ub_tail(self):
        shape = (262144, 12)
        r = estimate("sigmoid_fp32", shape, chip="Ascend950DT_9582")
        n = shape[0] * shape[1]
        self.assertEqual(
            r.latency_us, estimate("sigmoid_fp32", (131072, 24)).latency_us
        )
        self.assertEqual(
            r.work["vec_ops_by_class"],
            {op: n for op in ("negate", "exp", "add", "divide")},
        )
        self.assertEqual(
            (r.work["logical_read_bytes"], r.work["logical_write_bytes"]),
            (4 * n, 4 * n),
        )
        self.assertIsNone(r.work["physical_hbm_bytes"])
        plan = r.projection["target_plan"]
        self.assertEqual(
            (r.active_aiv, plan["critical_core_tiles"], plan["last_tile_elements"]),
            (64, 4, 1536),
        )
        self.assertAlmostEqual(r.latency_us, sum(r.projection["stages_us"].values()))

    def test_new_chip_fields_drive_large_workload_and_small_saturates_one_aiv(self):
        chip = load_chip()
        slow = replace(chip, hbm_bandwidth_gbs=chip.hbm_bandwidth_gbs / 2)

        def run(c):
            return estimate("sigmoid_fp32", (2097152, 12), chip=c).latency_us

        self.assertGreater(run(slow), run(chip))
        self.assertEqual(run(chip), run(replace(chip, name="unpublished")))
        self.assertEqual(estimate("sigmoid_fp32", (9, 12)).active_aiv, 1)
        self.assertEqual(estimate("sigmoid_fp32", (0, 12)).status, "unsupported")
        self.assertEqual(
            estimate("sigmoid_fp32", (24, 12), dtype="bf16").status, "unsupported"
        )
