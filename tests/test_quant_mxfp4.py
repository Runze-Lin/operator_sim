from dataclasses import replace
import unittest

from operator_model import OperatorStream, estimate, load_chip
from operator_model.workload import account_work


class MXFP4QuantTest(unittest.TestCase):
    def test_captured_strided_input_keeps_only_logical_work(self):
        contiguous = estimate('quant_mxfp4', (9, 128), {'input_stride': [128, 1]})
        view = estimate('quant_mxfp4', (9, 128), {'input_stride': [6288, 1]})
        self.assertEqual(contiguous.latency_us, estimate('quant_mxfp4', (9, 128)).latency_us)
        self.assertEqual(view.work, contiguous.work)
        self.assertEqual((view.status, view.latency_us, view.rates), ('unavailable', None, None))
        self.assertEqual(estimate('quant_mxfp4', (1, 128), {'input_stride': [6288, 1]}).status, 'extrapolated')
        self.assertEqual(estimate('quant_mxfp4', (9, 128), {'input_stride': [1]}).status, 'unsupported')

    def test_rank_and_packed_first_and_tail_groups(self):
        price = estimate('quant_mxfp4', (2, 9, 66), {'group_size':32, 'round_mode':'round'})
        self.assertEqual(price.status, 'extrapolated')
        work = price.work
        self.assertEqual(work['packed_shape'], [2, 9, 33])
        self.assertEqual(work['scale_shape'], [2, 9, 2, 2])
        self.assertEqual(work['groups'], 54)
        self.assertEqual(work['logical_write_bytes'], 18*(33+4))
        self.assertEqual(work['logical_read_bytes'], 18*66*2)
        self.assertEqual(work, account_work('quant_mxfp4', (2, 9, 66)))
        self.assertEqual(price.latency_us, estimate('quant_mxfp4', (18, 66)).latency_us)
        self.assertEqual(estimate('quant_mxfp4', (64,)).work['scale_shape'], [1, 2])

    def test_frame_separates_the_precision_and_calibration_contracts(self):
        for attrs in ({'group_size':64}, {'axis':0}, {'scale_alg':False}, {'round_mode':'rint'}, {'output_dtype':'mxfp8'}):
            self.assertEqual(estimate('quant_mxfp4', (9, 128), attrs).status, 'unsupported')
        for shape in ((9, 31), (64, 65), (True, 64), (), (1,)*7+(128,)):
            self.assertEqual(estimate('quant_mxfp4', shape).status, 'unsupported')
        self.assertEqual(estimate('quant_mxfp4', [9, 128], dtype='fp16').status, 'unsupported')
        unpriced = estimate('quant_mxfp4', [9, 128], timing_mode='calibrated')
        self.assertIsNone(unpriced.latency_us)
        self.assertIsNotNone(unpriced.work)
        self.assertEqual(unpriced.status, 'unavailable')
        self.assertEqual(estimate('quant_mxfp4', [9, 128]).work['packed_value_bytes'], 576)
        self.assertEqual(estimate('quant', [9, 128]).work['logical_write_bytes'], 1152+36)

    def test_source_tile_cycles_and_parallel_shared_link(self):
        e = OperatorStream().advance('quant_mxfp4', (9, 128))
        self.assertEqual(e.active_aiv, 9)
        self.assertEqual(e.active_aic, 0)
        self.assertEqual(e.kernel_count, 1)
        self.assertEqual(e.projection['first_tile']['vector_source_calls'], 31)
        self.assertAlmostEqual(e.latency_us, sum(e.projection['stages_us'].values()))
        huge = estimate('quant_mxfp4', [16, 131081, 576])
        self.assertGreater(huge.projection['target_plan']['iterations'], 500)
        slow = replace(load_chip(), hbm_bandwidth_gbs=10)
        self.assertGreater(estimate('quant_mxfp4', [16, 131081, 576], chip=slow).latency_us, huge.latency_us)
        self.assertGreater(estimate('quant_mxfp4', [9, 128], chip='Ascend970DT').latency_us, 0)
