import json
from pathlib import Path
from time import perf_counter
import unittest

from operator_model.chips import load_chip
from operator_model.hc_post_timing import resource_hc_post


ROOT = Path(__file__).resolve().parents[1]
ATTRS = {'implementation': 'vllm_hc_post'}
DT = 'Ascend950DT_9582'


def custom(**fields):
    config = json.loads((ROOT / 'configs/chips' / (DT+'.json')).read_text())
    details = fields.pop('details', {})
    config.update(name='alternative', **fields)
    config['details'].update(details)
    return load_chip(config)


class ExperimentalHcPostTest(unittest.TestCase):
    def test_all_preregistered_targets(self):
        register = json.loads((ROOT/'results/v0.23/hc-timing/holdout-register.json').read_text())
        for case in register['cases'][::2]:
            for sku, frozen in case['predictions'].items():
                with self.subTest(shape=case['shape'], sku=sku):
                    latency, work, diag = resource_hc_post(case['shape'], load_chip(sku), ATTRS)
                    self.assertEqual(work['vec_ops'], frozen['fp32_useful_ops'])
                    self.assertEqual(diag['active_aiv'], frozen['active_aiv'])
                    if case['shape'][-1] == 4112:
                        self.assertIsNone(latency)
                        self.assertIn('64 elements', ' '.join(diag['warnings']))
                        continue
                    self.assertAlmostEqual(latency, frozen['device_us'], places=10)
                    self.assertEqual(diag['pipeline']['vf_commands'], frozen['vf_commands'])

    def test_capacity_and_unpriced_fp32(self):
        fp = dict(load_chip(DT).vector_tops)
        fp['32'] = None
        for chip, message in ((custom(details={'ub_bytes': 192*1024}), 'UB'),
                              (custom(vector_tops=fp), 'FP32')):
            latency, work, diag = resource_hc_post([1,8,4,1024], chip, ATTRS)
            self.assertIsNone(latency)
            self.assertGreater(work['vec_ops'], 0)
            self.assertIn(message, ' '.join(diag['warnings']))
        self.assertIsNone(resource_hc_post([1,1,4,4095], load_chip(DT), ATTRS)[0])
        for width in (16,32,48,4112,4128,4144):
            self.assertIsNone(resource_hc_post([1,1,4,width], load_chip(DT), ATTRS)[0])
        self.assertIsNotNone(resource_hc_post([1,1,4,4160], load_chip(DT), ATTRS)[0])

    def test_mte_override_changes_payload_and_preserves_fanout(self):
        normal = load_chip(DT)
        # Supply a slower effective per-core transfer rate for both directions.
        slow = custom(details={'mte2_bandwidth_gbs': 5, 'mte3_bandwidth_gbs': 5})
        fast = custom(details={'mte2_bandwidth_gbs': 500, 'mte3_bandwidth_gbs': 500})
        shape = [1,3,4,7168]
        results = [resource_hc_post(shape, chip, ATTRS) for chip in (slow, normal, fast)]
        self.assertGreater(results[0][0], results[1][0])
        self.assertGreater(results[1][0], results[2][0])
        self.assertEqual(len({r[2]['pipeline']['first_coeff_issue_tick'] for r in results}), 1)

    def test_configured_clock_bandwidth_and_vector_rates(self):
        shape = [1,3,4,7168]
        chip = load_chip(DT)
        fp = {bits: rate*2 if rate else None for bits, rate in chip.vector_tops.items()}
        base = resource_hc_post(shape, chip, ATTRS)[0]
        faster = custom(frequency_ghz=2*chip.frequency_ghz, vector_tops=fp)
        self.assertAlmostEqual(resource_hc_post(shape, faster, ATTRS)[0], base/2, places=10)
        self.assertLess(resource_hc_post(shape, custom(vector_tops=fp), ATTRS)[0], base)
        slow_memory = custom(hbm_bandwidth_gbs=1.)
        latency, work, diag = resource_hc_post(shape, slow_memory, ATTRS)
        self.assertAlmostEqual(latency, work['logical_read_bytes']/1000.)
        self.assertGreater(diag['stages_us']['exposed_shared_limit'], 0)
        self.assertFalse(diag['target_has_archived_ca'])

    def test_exact_two_slot_skip_for_long_narrow_prefill(self):
        # Frozen scheduler reference for B=8, S=256Ki. Includes an imbalanced core group.
        start = perf_counter()
        latency, work, diag = resource_hc_post((8,262144,4,1280),load_chip('Ascend970DT'),ATTRS)
        elapsed = perf_counter()-start
        self.assertAlmostEqual(latency,10509.106817118663,places=6)
        self.assertLess(elapsed,.1)
        self.assertEqual(work['logical_write_bytes'],8*262144*8*1280)
        self.assertGreater(diag['pipeline']['skipped_steady_rows'],17000)
        self.assertTrue(any('extrapolation' in s for s in diag['warnings']))
