"""Integration invariants: different SIMD lowering; unchanged contracts, launches and IO."""

import unittest
from dataclasses import replace

from operator_model import OperatorStream, estimate, load_chip
from operator_model.operators.vector.family_programs import Pricer


class NativeVectorPrograms(unittest.TestCase):
    def test_opt_in_preserves_algorithm_and_qualified_only_is_not_invented(self):
        for op, shape, attrs, dtype in [
            ("quant", [5, 320], {}, None),
            ("quant_mxfp4", [9, 128], {}, None),
            ("rmsnorm", [64, 2048], {}, None),
            ("mhc_post", [1, 4, 4096], {}, None),
            ("mhc_post", [1, 1, 4, 512], {"implementation": "vllm_hc_post"}, None),
            ("mhc_pre", [8, 4, 512], {}, None),
            ("mhc", [8, 4, 512], {}, None),
            ("mhc_pre_sinkhorn", [8, 4, 4096], {}, None),
            ("mhc_sinkhorn", [24, 4], {"num_iters": 8}, None),
            ("attn_res_fwd", [8, 256, 8, 1], {"addend": True, "optimize_prefill": True}, None),
            ("moe_gating_top_k", [4, 256, 8], {}, None),
            ("cache_load", [9, 1, 128], {"implementation": "aclnnAntiMxQuant"}, "mxfp4"),
        ]:
            with self.subTest(op=op, attrs=attrs):
                base = estimate(op, shape, attrs, dtype=dtype)
                opts = dict(attrs, vector_model="instruction_classes")
                other = estimate(op, shape, opts, dtype=dtype)
                self.assertEqual(base.work, other.work)
                self.assertGreater(other.latency_us, 0.0)
                self.assertEqual(other.kernel_count, base.kernel_count)
                self.assertTrue(other.projection["instruction_samples"])
                self.assertEqual(
                    estimate(op, shape, dict(attrs, vector_model="calibrated"), dtype=dtype).latency_us,
                    base.latency_us,
                )
                self.assertIsNone(estimate(op, shape, opts, dtype=dtype, timing_mode="calibrated").latency_us)

    def test_pipeline_pressure_and_cache_scope(self):
        ref = load_chip()
        slow = replace(ref, details={**ref.details, "vf_reduce_latency_cycles": 512})
        base = Pricer(ref)
        limited = Pricer(slow)
        self.assertGreater(limited("mx4", 1, 128), base("mx4", 1, 128))
        self.assertEqual(limited("rms_norm", 1, 256), base("rms_norm", 1, 256))
        opts = {"vector_model": "instruction_classes"}
        self.assertGreater(
            estimate("quant_mxfp4", [9, 128], opts, chip=slow).latency_us,
            estimate("quant_mxfp4", [9, 128], opts, chip=ref).latency_us,
        )
        self.assertIsNone(estimate("cache_load", [9, 1, 128], opts, dtype="mxfp4").latency_us)

    def test_stream_uses_chosen_graph_for_partial_prefix_and_post_repeat(self):
        cfg = load_chip()
        opts = {"vector_model": "instruction_classes"}
        s = OperatorStream(cfg)
        s.forget_history()
        s.advance("rmsnorm", [9, 2048], opts)
        p = s.preview("rmsnorm", [34, 2048], opts)
        ctx = p.projection["partial_history"]
        ends = [
            estimate("rmsnorm", [34, 2048], dict(opts, resident_aiv=h), chip=cfg).latency_us for h in (9, 34)
        ]
        self.assertEqual(ctx["sampled_endpoint_range_us"], sorted(ends))
        s.reset()
        a = s.advance("mhc_post", [64, 4, 2048], opts)
        b = s.advance("mhc_post", [64, 4, 2048], opts)
        self.assertEqual(a.projection["timing_diagnostics"]["vector_model"], "instruction_classes")
        self.assertEqual(b.projection["timing_diagnostics"]["model"], "post_token_immediate_repeat")
        self.assertIn("post_mix", b.projection["timing_diagnostics"]["instruction_samples"])
        f = s.advance("mhc_post", [64, 4, 8192], opts)
        self.assertIn("post_mix", f.projection["timing_diagnostics"]["instruction_samples"])
