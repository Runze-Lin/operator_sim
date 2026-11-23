import json
import unittest

from operator_model import OperatorStream, estimate, load_chip
from operator_model.resources import chip_file


class DispatchBudgetTests(unittest.TestCase):
    def setUp(self):
        self.config = json.loads(chip_file("Ascend950DT_9582").read_text())

    def test_dispatches_are_not_tiles_or_cores(self):
        cases = [
            ("quant", (5, 320), {}, 1),
            ("matmul", (1280, 2048, 512), {}, 1),
            ("mhc_pre_sinkhorn", (80, 4, 4096), {"need_backward": False, "num_iters": 20}, 1),
            ("mhc", (2, 4, 512), {}, 3),
            ("groupedmatmulswiglugmm", (8, 256, 128, 2), {"expert_counts": [1, 7]}, 2),
            ("kda_chunk", (1, 145, 2, 2, 128, 128), {"implementation": "fla_26_9_1_v2"}, 3),
        ]
        for op, shape, attrs, count in cases:
            with self.subTest(op=op):
                base = estimate(op, shape, attrs)
                charged = estimate(op, shape, attrs, chip={**self.config, "kernel_overhead_us": 1.5})
                self.assertIsNotNone(base.latency_us)
                self.assertEqual(base.scheduled_us, base.latency_us)
                self.assertEqual((base.status, base.latency_us, base.rates, base.work),
                                 (charged.status, charged.latency_us, charged.rates, charged.work))
                self.assertEqual(charged.kernel_count, count)
                self.assertEqual(charged.scheduled_us, base.latency_us + count * 1.5)

    def test_unknowns_and_invalid_overhead(self):
        for value in (-1, float('nan'), float('inf'), True, '2', None):
            with self.subTest(value=value), self.assertRaises(ValueError):
                load_chip({**self.config, "kernel_overhead_us": value})
        for op, shape, opts in [('matmul', (1, 64, 64), {'dtype': 'int4'}),
                                ('kda_chunk', (1, 64, 1, 1, 128, 128), {})]:
            r = estimate(op, shape, chip={**self.config, "kernel_overhead_us": 2}, **opts)
            self.assertIsNone(r.kernel_count)
            self.assertIsNone(r.scheduled_us)
            self.assertIsNone(r.dispatch_overhead_us)

    def test_stream_preserves_overhead_on_repeat(self):
        default = OperatorStream()
        stream = OperatorStream({**self.config, "kernel_overhead_us": 1.5})
        args = ('mhc_post', (65, 4, 5120))
        for _ in range(2):
            preview = stream.preview(*args)
            issued = stream.advance(*args)
            self.assertEqual(preview, issued)
            self.assertEqual(issued.scheduled_us, default.advance(*args).latency_us + 1.5)

    def test_target_capacities_change_supported_schedules(self):
        cases = [
            ('quant', (8192, 5120), 'ub_bytes', 65536, 'iterations'),
            ('matmul', (1280, 2048, 512), 'l1_bytes', 262144, 'read_page_k'),
            ('cast_bf16_fp32', (262144, 128), 'l2_bytes', 33554432, None),
        ]
        for op, shape, key, smaller, plan_field in cases:
            with self.subTest(key=key):
                config = {**self.config, 'details': {**self.config['details'], key: smaller}}
                base = estimate(op, shape, timing_mode='resource')
                result = estimate(op, shape, chip=config, timing_mode='resource')
                self.assertIsNotNone(result.latency_us)
                self.assertNotEqual(result.latency_us, base.latency_us)
                if plan_field:
                    self.assertNotEqual(base.projection['target_plan'][plan_field],
                                        result.projection['target_plan'][plan_field])
