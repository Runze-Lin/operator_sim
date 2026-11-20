from dataclasses import replace
from unittest import TestCase

from operator_model.chips import load_chip
from operator_model.matmul_supply30 import estimate_matmul_supply


class SupplyCandidateTests(TestCase):
    def test_low_pressure_and_dirty_eviction_use_different_resources(self):
        little=estimate_matmul_supply([262144,128,512])
        self.assertEqual({g['limiter'] for g in little['segments']},{'core'})
        large=estimate_matmul_supply([1048576,128,512])
        self.assertEqual(large['pricing']['service_write_decile'],2)
        segments=large['segments']
        self.assertEqual(segments[0]['first_wave'],0)
        self.assertEqual(segments[-1]['exclusive_end_wave'],large['waves'])
        self.assertEqual(sum((s['exclusive_end_wave']-s['first_wave'])*s['output_tiles_per_wave'] for s in segments),4096)
        self.assertEqual(sum((s['exclusive_end_wave']-s['first_wave'])*s['cached_store_tiles_per_wave'] for s in segments),
                         large['pricing']['output_tiles_before_spill'])
        self.assertEqual(segments[-1]['limiter'],'memory_supply')
        self.assertAlmostEqual(segments[-1]['cycles_per_wave'],32*(262144+65536)/large['pricing']['hbm_bytes_per_cycle']/large['pricing']['mixed_efficiency'])

    def test_cache_price_and_target_override_have_physical_monotonicity(self):
        s=[1048576,128,512]
        chip=load_chip()
        reduced=replace(chip,details={**chip.details,'l2_bytes':chip.details['l2_bytes']//4})
        self.assertGreater(estimate_matmul_supply(s,chip=reduced)['latency_us'],estimate_matmul_supply(s)['latency_us'])
        slow=estimate_matmul_supply(s,mixed_efficiency=.5)
        quick=estimate_matmul_supply(s,mixed_efficiency=1)
        self.assertGreater(slow['latency_us'],quick['latency_us'])
        unknown=replace(chip,name='future-design')
        self.assertEqual(estimate_matmul_supply(s,chip=unknown)['pricing']['logical_ingress_ratio'],1.)
        with self.assertRaises(ValueError):estimate_matmul_supply(s,mixed_efficiency=1.01)

    def test_partial_last_wave_and_credit_only_once(self):
        s=[1048576,128,512];p=estimate_matmul_supply(s,chip='Ascend950PR_957b')
        self.assertEqual(p['segments'][-1]['active_aic'],4096%28)
        self.assertLessEqual(p['pricing']['maximum_initial_read_credit_bytes'],p['pricing']['l2_bytes'])
        cold=estimate_matmul_supply(s,chip='Ascend950PR_957b',input_mode='cold')
        self.assertEqual(cold['pricing']['maximum_initial_read_credit_bytes'],0)
        self.assertLess(p['latency_us'],cold['latency_us'])
        self.assertIsNone(estimate_matmul_supply([32768,128,512],chip='Ascend970DT')['latency_us'])
