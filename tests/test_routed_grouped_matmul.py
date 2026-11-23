"""Coverage of rank-local routed A4W4 GEMM and its system-facing work-only contract."""
import unittest

from operator_model import estimate
from operator_model.workload import account_work


class RoutedGroupedMatmulTests(unittest.TestCase):
    def test_gate_up_from_pro_rank_zero(self):
        inputs = {'expert_counts': [1] * 48, 'dispatch_capacity_rows': 54}
        shape = (48, 6144, 7168)
        for chip in ('Ascend950DT_9582', 'Ascend950PR_957b', 'Ascend970DT'):
            result = estimate('routed_grouped_matmul', shape, inputs, dtype='mxfp4', chip=chip)
            self.assertEqual(result.status, 'unavailable', result.reason)
            self.assertIsNone(result.latency_us)
            self.assertIsNone(result.active_aic)
            self.assertIsNone(result.active_aiv)
            self.assertIsNone(result.kernel_count)
            self.assertIsNone(result.work['physical_hbm_bytes'])
            self.assertEqual(result.work, account_work('routed_grouped_matmul', shape, inputs))
            self.assertEqual(result.work['cube_ops_by_dtype'], {'mxfp4': 4227858432})
            self.assertEqual(result.work['logical_read_bytes'], 1123207680)
            self.assertEqual(result.work['logical_write_bytes'], 589824)
            self.assertEqual(result.projection['unused_dispatch_capacity_rows'], 6)
            self.assertEqual(result.projection['weight_scale_shape'], [48, 112, 6144, 2])

    def test_same_rows_different_histograms_and_paired_scale_slot(self):
        # K=32 uses one of two E8M0 slots. Same FLOP count, fewer distinct weights with one expert.
        results = [estimate('routed_grouped_matmul', (3, 5, 32), {'expert_counts': counts})
                   for counts in ([1, 2], [0, 3])]
        for result in results:
            self.assertEqual(result.work['cube_ops'], 960)
            self.assertEqual(result.work['logical_write_bytes'], 30)
            self.assertIsNone(result.projection['unused_dispatch_capacity_rows'])
            self.assertEqual(result.work['all_expert_weight_and_scale_storage_bytes'], 180)
        dense, sparse = (r.work for r in results)
        self.assertEqual((dense['logical_read_bytes'], sparse['logical_read_bytes']), (234, 144))
        self.assertEqual((dense['unused_scale_slots'], sparse['unused_scale_slots']), (13, 8))

    def test_empty_rank_does_no_math_or_tensor_reads(self):
        result = estimate('routed_grouped_matmul', (0, 5, 32),
                          {'expert_counts': [0, 0], 'dispatch_capacity_rows': 8})
        self.assertEqual([result.work[key] for key in
                          ('cube_ops', 'logical_read_bytes', 'logical_write_bytes')], [0, 0, 0])
        self.assertEqual(result.work['all_expert_weight_and_scale_storage_bytes'], 180)
        self.assertIsNone(result.latency_us)

    def test_rejects_missing_or_inconsistent_routing_and_precision(self):
        default = {'expert_counts': [3]}
        for shape, attrs, dtype in [
            ((3, 5, 32), {}, 'mxfp4'), ((3, 5, 32), {'expert_counts': [1, 1]}, 'mxfp4'),
            ((3, 5, 32), {'expert_counts': [True, 2]}, 'mxfp4'),
            ((3, 5, 32), {**default, 'dispatch_capacity_rows': 1}, 'mxfp4'),
            ((3, 5, 32), {**default, 'activation_scale_layout': 'token_fp32'}, 'mxfp4'),
            ((3, 5, 31), default, 'mxfp4'), ((3, 5, 32), default, 'fp8_e4m3fn'),
        ]:
            with self.subTest(shape=shape, attrs=attrs, dtype=dtype):
                result = estimate('routed_grouped_matmul', shape, attrs, dtype=dtype)
                self.assertEqual(result.status, 'unsupported', result.reason)
                self.assertIsNone(result.latency_us)


class RoutedGroupedNativeScheduleTests(unittest.TestCase):
    def test_last_expert_spreads_over_free_cores(self):
        shape=(3,128,256)
        attrs={'expert_counts':[1,2],'implementation':'aclnnGroupedMatmulV5'}
        native=estimate('routed_grouped_matmul',shape,attrs,chip='Ascend950DT_9582')
        logical=estimate('routed_grouped_matmul',shape,{'expert_counts':[1,2]},chip='Ascend950DT_9582')
        self.assertEqual(native.work,logical.work)
        self.assertEqual((native.active_aic,native.active_aiv,native.kernel_count),(17,0,1))
        self.assertGreater(native.latency_us,0)
        self.assertEqual(native.projection['target_plan']['base_output_tiles'],2)
        self.assertIsNone(logical.latency_us)

    def test_unpriced_geometry_and_new_chip_qualification(self):
        options={'expert_counts':[1,0,2],'implementation':'aclnnGroupedMatmulV5'}
        for dims in ((3,96,32),(3,96,2)):
            result=estimate('routed_grouped_matmul',dims,options)
            self.assertEqual(result.status,'unavailable')
            self.assertIsNone(result.latency_us)
        future=estimate('routed_grouped_matmul',(3,128,256),options,chip='Ascend970DT')
        self.assertEqual(future.status,'extrapolated')
        self.assertGreater(future.latency_us,0)
        strict=estimate('routed_grouped_matmul',(3,128,256),options,chip='Ascend970DT',timing_mode='calibrated')
        self.assertIsNone(strict.latency_us)
