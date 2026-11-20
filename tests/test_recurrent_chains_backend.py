import unittest

from operator_model import estimate, estimate_recurrent_chains


class DenseRecurrentBackendTests(unittest.TestCase):
    def test_types_and_qualification_through_public_backend(self):
        shape = [128, 1, 12, 12, 128, 128]
        for chip in ('Ascend950DT_9582', 'Ascend950PR_957b', 'Ascend970DT'):
            with self.subTest(chip=chip):
                forecast = estimate_recurrent_chains(shape, chip=chip)
                for mode in ('auto', 'resource'):
                    row = estimate('kda_recurrent', shape, chip=chip, timing_mode=mode)
                    self.assertEqual(row.status, 'extrapolated')
                    self.assertEqual(row.latency_us, forecast['latency_us'])
                    self.assertEqual(row.active_aiv, forecast['active_aiv'])
                    self.assertEqual(row.work['logical_read_bytes'], forecast['work']['logical_read_bytes'])
                calibrated = estimate('kda_recurrent', shape, chip=chip, timing_mode='calibrated')
                self.assertIsNone(calibrated.latency_us)
                self.assertEqual(calibrated.status, 'unavailable')

    def test_gqa_and_token_window(self):
        for shape in ([128, 1, 12, 4, 128, 128], [1, 9, 12, 12, 128, 128]):
            self.assertIsNone(estimate('kda_recurrent', shape).latency_us)
        self.assertEqual(estimate('kda_recurrent', [1,1,1,1,128,128]).status, 'estimated')
