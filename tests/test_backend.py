import unittest

from operator_model.backend import estimate


class TestSystemBackend(unittest.TestCase):
    def test_converts_units_at_the_system_boundary(self):
        result = estimate('addmm', (9, 128, 256))
        self.assertEqual(result.status, 'estimated')
        self.assertGreater(result.latency_us, 0)

    def test_unsupported_and_untimed_are_distinct(self):
        for result in (estimate('unknown_op', (1,)),
                       estimate('quant', (9, 17)),
                       estimate('rmsnorm', (1, 128), soc='A8')):
            self.assertEqual(result.status, 'unsupported')
            self.assertIsNone(result.latency_us)
        chunk = estimate('kda_chunk', (1, 64, 1, 1, 128, 128))
        self.assertIsNone(chunk.latency_us)
        self.assertEqual(chunk.status, 'unavailable')

    def test_invalid_input_returns_a_fallback_signal(self):
        for op, shape, attrs in [('rmsnorm', (), {}), ('fa', (1,), {}), ('rmsnorm', None, {}),
                                 ('mhc_post', (1, 4, 4096), []), ('rmsnorm', (1, 6144), {'resident_aiv': 65}),
                                 ('kda_chunk', (1, 64, 1, 1, 128, 128), {'beta_dtype': None})]:
            with self.subTest(op=op, shape=shape, attrs=attrs):
                result = estimate(op, shape, attrs)
                self.assertEqual(result.status, 'unsupported')
                self.assertIsNone(result.latency_us)

    def test_marks_long_context_timing_as_extrapolation(self):
        result = estimate('fa', (9, 524288, 64, 1, 1))
        self.assertEqual(result.status, 'extrapolated')
        self.assertGreater(result.latency_us, 0)

    def test_scheduler_can_track_vector_residency(self):
        result = estimate('rmsnorm', (2, 6144), {'resident_aiv': 0})
        self.assertEqual(result.active_aiv, 2)
        self.assertEqual(estimate('matmul', (2, 64, 128)).active_aiv, 0)

    def test_fused_contract_uses_expert_distribution(self):
        result = estimate('groupedmatmulswiglugmm', (32, 256, 128, 2),
                          {'expert_counts': [1, 31]})
        self.assertEqual(result.status, 'estimated')
        self.assertIsNone(estimate('moe_ffn', (32, 256, 128, 2)).latency_us)
