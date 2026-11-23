"""Explicitly materialized inference cache ops carry valid-row and physical padding contracts."""
import unittest

from operator_model import estimate
from operator_model.operators.vector.cache_load import describe


class CacheLoadTests(unittest.TestCase):
    def test_packed_odd_width_and_padded_e8m0_pair(self):
        work, plan = describe((3, 65), 'mxfp4', {'storage_shape': [3, 33], 'scale_shape': [3, 2, 2]})
        self.assertEqual((work['cache_value_bytes'], work['cache_scale_bytes'], work['logical_write_bytes']),
                         (99, 12, 390))
        self.assertEqual((work['logical_scale_values'], work['unused_scale_slots'], work['padded_value_lanes']),
                         (9, 3, 3))
        self.assertEqual(plan['logical_shape'], [3, 65])
        with self.assertRaises(ValueError):
            describe((3, 65), 'mxfp4', {'scale_shape': [3, 3]})
        with self.assertRaises(ValueError):
            describe((3, 65), 'mxfp4', {'storage_shape': [3, 65]})

    def test_public_rank_local_kimi_capture_counts_without_timing(self):
        shape = [64, 131081, 2, 16, 64]
        attrs = dict(storage_shape=[64, 131081, 2, 16, 32], scale_shape=[64, 131081, 2, 16, 1, 2])
        result = estimate('cache_load', shape, attrs, dtype='mxfp4', chip='Ascend950DT_9582')
        self.assertEqual(result.status, 'unavailable')
        self.assertIsNone(result.latency_us)
        self.assertIsNone(result.kernel_count)
        self.assertEqual(result.work['valid_values'], 17181048832)
        self.assertEqual(result.work['logical_read_bytes'], 9127432192)
        self.assertEqual(result.work['logical_write_bytes'], 34362097664)
        self.assertIsNone(result.work['physical_hbm_bytes'])

    def test_rowwise_fp8_and_invalid_contracts(self):
        work, plan = describe((1, 1024, 512), 'fp8_e4m3fn', {'scale_shape': [1, 1024]})
        self.assertEqual((work['cache_value_bytes'], work['cache_scale_bytes'], work['valid_values']),
                         (524288, 4096, 524288))
        self.assertEqual(work['logical_write_bytes'], 1048576)
        self.assertIsNone(plan['scale_group_size'])
        for dtype, attrs in [('bf16', {}), ('mxfp8', {}), ('mxfp4', {'group_size': 64}),
                             ('mxfp4', {'output_dtype': 'fp16'}), ('fp8', {'materialize': False}),
                             ('fp8', {'scale_dtype': 'e8m0'}), ('fp8', {'storage_precision': 'fp4'})]:
            with self.subTest(dtype=dtype, attrs=attrs), self.assertRaises(ValueError):
                describe((1, 1024, 512), dtype, attrs)

    def test_empty_valid_rows_do_not_charge_the_allocation(self):
        work, _ = describe((2, 0, 129), 'mxfp4', {})
        self.assertEqual((work['valid_values'],work['logical_read_bytes'],work['logical_write_bytes']), (0,0,0))
