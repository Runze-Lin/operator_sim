"""MX first-operand dependencies and steady-state composition."""
from dataclasses import replace
import unittest

from operator_model import estimate, load_chip


class MX8NativePipeline(unittest.TestCase):
    def predict(self, shape, chip):
        return estimate("matmul", shape, {"weight_dtype": "mxfp8"}, dtype="mxfp8", chip=chip)

    def test_sideband_response_is_paid_once(self):
        chip = load_chip("Ascend950DT_9582")
        one = self.predict((16, 256, 512), chip).projection["target_plan"]
        four = self.predict((16, 256, 2048), chip).projection["target_plan"]
        self.assertEqual((one["k_pages"], four["k_pages"]), (1, 4))
        self.assertEqual(one["cycles"]["remaining_overlap"], 0)
        for stage in ("first_dma", "first_mmad", "scalar_setup", "drain", "operand_ready"):
            self.assertEqual(one["cycles"][stage], four["cycles"][stage])
        slow_resource = max(four["cycles"][s] - one["cycles"][s] for s in ("cube", "panel_mte2"))
        self.assertAlmostEqual(slow_resource, four["cycles"]["remaining_overlap"])

    def test_bandwidth_changes_whole_program_critical_path(self):
        chip = replace(load_chip("Ascend950DT_9582"), hbm_bandwidth_gbs=10)
        estimate_ = self.predict((512, 2048, 7168), chip)
        costs = estimate_.projection["target_plan"]["cycles"]
        local = sum(costs[s] for s in ("first_dma", "operand_ready", "first_mmad", "remaining_overlap"))
        self.assertGreater(costs["hbm"], local)
        total = sum(costs[s] for s in ("hbm", "core_start_spread", "scalar_setup", "drain"))
        self.assertAlmostEqual(estimate_.latency_us, total / (1000 * chip.frequency_ghz))

    def test_first_fragment_serializes_before_second(self):
        chip = load_chip("Ascend950DT_9582")
        prediction = self.predict((9, 256, 128), chip)
        plan = prediction.projection["target_plan"]
        c = plan["cycles"]
        local = sum(c[s] for s in ("first_dma", "operand_ready", "first_mmad", "remaining_overlap"))
        self.assertGreater(local, c["hbm"])
        cycles = sum(c[s] for s in ("scalar_setup", "core_start_spread", "drain")) + local
        self.assertAlmostEqual(prediction.latency_us, cycles / (1000 * chip.frequency_ghz))
        self.assertGreater(c["first_dma"], c["panel_mte2"])
        self.assertGreater(c["first_mmad"], c["cube"])
