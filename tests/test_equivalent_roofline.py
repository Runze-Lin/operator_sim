import unittest

from operator_model import estimate


class EquivalentRooflineTests(unittest.TestCase):
    def test_mixed_engines_use_the_largest_nominal_time(self):
        fa = estimate('fa', [9, 131072, 64, 1, 1], chip='Ascend950PR_957b')
        roof = fa.rates['roofline']
        self.assertAlmostEqual(roof['cube_us'], 301989888 / 378e6)
        self.assertAlmostEqual(roof['vector_us'], 7077870 / 23e6)
        self.assertAlmostEqual(roof['ideal_us'], (33555584 + 1728) / 1.4e6)
        self.assertEqual(roof['bound'], 'HBM')
        self.assertAlmostEqual(roof['ideal_us'] / roof['utilization'], fa.latency_us)

    def test_unknown_active_peak_and_logical_ratios_above_one(self):
        self.assertIsNone(estimate('mhc_pre', [2, 4, 1024]).rates['roofline']['utilization'])
        rms = estimate('rmsnorm', [8192, 5120], chip='Ascend950PR_957b')
        self.assertGreater(rms.rates['roofline']['utilization'], 1)
        self.assertEqual(rms.rates['roofline']['cube_us'], 0)
