import unittest

from operator_model import estimate, physical_roofline


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

    def test_a_large_logical_factor_has_no_measured_dram_utilization(self):
        cast = estimate('cast_bf16_fp32', [262144, 64], chip='Ascend950PR_957b')
        self.assertAlmostEqual(cast.rates['logical_roofline']['time_factor'], 2.8727, places=3)
        physical = cast.rates['physical_roofline']
        self.assertIn('HBM', physical['missing'])
        self.assertIsNone(physical['utilization'])
        matmul = estimate('matmul', [262144, 128, 512], chip='Ascend950PR_957b')
        floor = matmul.rates['physical_roofline']
        self.assertAlmostEqual(floor['known_lower_bound_us'], 2*262144*128*512 / 378e6)
        self.assertIs(floor['below_known_bound'], False)

    def test_complete_physical_window_flags_impossible_time(self):
        measured = physical_roofline(4, 1000, cube_us=1, vector_us=0, physical_hbm_bytes=2_000_000)
        self.assertEqual(measured['ideal_us'], 2)
        self.assertEqual(measured['utilization'], .5)
        partial = physical_roofline(1, 1000, physical_hbm_bytes=2_000_000)
        self.assertIsNone(partial['utilization'])
        self.assertTrue(partial['below_known_bound'])
        self.assertIsNone(physical_roofline(1, 1000)['below_known_bound'])
