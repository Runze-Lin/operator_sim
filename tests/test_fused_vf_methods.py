from dataclasses import replace
import unittest

from operator_model import estimate, load_chip


class FusedVFMethods(unittest.TestCase):
    def test_matrix_register_cost_and_unchanged_native(self):
        chip = load_chip()
        sh = (512, 64, 7168)

        def price(chip, dtype, **extra):
            return estimate("matmul", sh, dict(weight_dtype="mxfp4", **extra), chip=chip, dtype=dtype)

        slow = replace(chip, details={**chip.details, "vf_convert_ii_cycles": 80})
        a4 = price(slow, "mxfp4")
        same = price(slow, "mxfp4", vector_model="instruction_classes")
        self.assertEqual(same.latency_us, a4.latency_us)
        self.assertEqual(same.projection["vector_model"], "not_applicable")
        a8 = price(chip, "fp8_e4m3fn", vector_model="instruction_classes")
        delayed = price(slow, "fp8_e4m3fn", vector_model="instruction_classes")
        self.assertEqual(delayed.active_aic, a8.active_aic)
        self.assertGreater(delayed.latency_us, a8.latency_us)
        self.assertEqual(
            delayed.projection["target_plan"]["page_schedule"], a8.projection["target_plan"]["page_schedule"]
        )

    def test_chunk_keeps_three_kernel_edge_and_untimed_v1(self):
        shape = (1, 128, 1, 1, 128, 128)
        kw = {"vector_model": "instruction_classes"}
        e = estimate("kda_chunk", shape, {**kw, "implementation": "fla_26_9_1_v2"})
        kernels = e.projection["target_plan"]["stages"]
        self.assertEqual(3, len(kernels))
        self.assertAlmostEqual(sum(k["device_us"] for k in kernels), e.latency_us)
        self.assertFalse(kernels[-1]["instruction_class_counts"])  # finalize has no arithmetic VF
        self.assertTrue(kernels[0]["instruction_class_counts"])
        self.assertIsNone(estimate("kda_chunk", shape, kw).latency_us)

    def test_larger_register_shrinks_counts_but_not_dependency_latency(self):
        from operator_model.operators.fused.family_programs import Programs

        chip = load_chip()
        key = "W4 unpack 4x1024"
        p = Programs(chip)
        p.unpack(4, 1024)
        wider = Programs(replace(chip, details={**chip.details, "vector_register_bytes": 512}))
        wider.unpack(4, 1024)
        self.assertEqual(wider.records[key]["counts"]["load"] * 2, p.records[key]["counts"]["load"])
        self.assertEqual(
            wider.records[key]["bounds_cycles"]["dependency"], p.records[key]["bounds_cycles"]["dependency"]
        )
        longer = Programs(replace(chip, details={**chip.details, "vf_convert_latency_cycles": 100}))
        longer.unpack(4, 1024)
        self.assertGreater(longer.records[key]["cycles"], p.records[key]["cycles"])
