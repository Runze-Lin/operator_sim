import math
import unittest
from dataclasses import replace

from operator_model.chips import load_chip
from operator_model.operators.mhc.history import estimate_post_history


class PostHistoryTest(unittest.TestCase):
    def test_channel_conditional_and_published_reference(self):
        shape = [3, 4, 7168]
        histories = estimate_post_history(shape, dtype='bfloat16')
        self.assertEqual(histories.route, 'post_channel')
        self.assertFalse(histories.target_extrapolated)
        cold = histories.for_predecessor(0)
        hot = histories.for_predecessor(48, shape)
        other = histories.for_predecessor(48, [1, 4, 192])
        self.assertEqual(cold, histories.scenarios['first'])
        self.assertEqual(hot, histories.scenarios['repeat'])
        self.assertEqual(other, histories.scenarios['reuse'])
        self.assertAlmostEqual(hot.latency_us, 2.191537319617995)
        self.assertAlmostEqual(sum(hot.stages_us.values()), hot.latency_us)
        self.assertLess(hot.latency_us, cold.latency_us)
        self.assertEqual(histories.expected_us({'first': 1}), cold.latency_us)
        self.assertAlmostEqual(histories.expected_us({'first': .125, 'repeat': .875}), cold.latency_us/8 + 7*hot.latency_us/8)
        with self.assertRaises(ValueError):
            histories.for_predecessor(1, shape)
        with self.assertRaises(ValueError):
            histories.for_predecessor(48)
        self.assertNotIn('expand', estimate_post_history([1,4,192]).scenarios)

    def test_no_silent_prior(self):
        h = estimate_post_history([2,4,448])
        for p in ({}, {'first': .5}, {'first': 2}, {'unknown': 1.0}, {'expand': math.nan}, {'reuse': math.inf},
                  {'first': True}, {'first': '1.0'}, {'first': 1.1, 'expand': -.1}):
            with self.subTest(p=p), self.assertRaises(ValueError):
                h.expected_us(p)
        with self.assertRaises(TypeError):
            h.scenarios['first'] = h.scenarios['repeat']

    def test_predecessor_entry_and_exact_repeat_differ(self):
        new = [157,4,1088]
        h = estimate_post_history(new)
        self.assertEqual(h.for_predecessor(64,[127,4,448]),h.scenarios['reuse_after_token'])
        self.assertEqual(h.for_predecessor(64,[3,4,7168]),h.scenarios['reuse_after_channel'])
        self.assertEqual(h.for_predecessor(64,new),h.scenarios['repeat'])
        self.assertNotEqual(h.scenarios['reuse_after_channel'].latency_us,h.scenarios['reuse_after_token'].latency_us)
        self.assertGreater(h.scenarios['reuse_after_token'].latency_us,h.scenarios['repeat'].latency_us)
        self.assertNotIn('expand_after_token',estimate_post_history([32,4,512]).scenarios)

    def test_target_configuration_and_contract(self):
        shape=[29,4,320]
        self.assertEqual(estimate_post_history(shape).route,'post_channel')
        self.assertEqual(estimate_post_history(shape,chip='Ascend950PR_957b').route,'post_token')
        self.assertTrue(estimate_post_history(shape,chip='Ascend970DT').target_extrapolated)
        chip = load_chip()
        modified = replace(chip,frequency_ghz=1.7)
        self.assertTrue(estimate_post_history(shape,chip=modified).target_extrapolated)
        no_price=replace(chip,vector_tops={**chip.vector_tops,'32':None})
        cold=estimate_post_history(shape,chip=no_price)
        self.assertEqual(cold.scenarios['repeat'].model,'occupancy_resource')
        for bad in ([1,3,192],[1,4,129],[1,4,16384]):
            with self.subTest(bad=bad), self.assertRaises(ValueError):
                estimate_post_history(bad)
        with self.assertRaises(ValueError):
            estimate_post_history(shape,dtype='fp16')



class PostUnknownDispatchTest(unittest.TestCase):
    def test_public_point_uses_state_endpoints_without_learning_a_predecessor(self):
        from operator_model import OperatorStream, estimate
        shape = [9, 4, 3840]
        for chip in ('Ascend950DT_9582','Ascend950PR_957b','Ascend970DT'):
            result = estimate('mhc_post', shape, chip=chip)
            diag = result.projection['timing_diagnostics']
            lo, hi = diag['state_range_us']
            self.assertEqual(diag['state_scenarios_us']['repeat'],lo)
            self.assertAlmostEqual(result.latency_us/lo-1,1-result.latency_us/hi)
            self.assertAlmostEqual(result.latency_us,sum(result.projection['stages_us'].values()))
            stream=OperatorStream(chip)
            stream.advance('mhc_post',shape)
            stream.forget_history()
            self.assertEqual(stream.preview('mhc_post',shape),result)
            self.assertEqual(estimate('mhc_post',shape,{'resident_aiv':None},chip=chip),result)

    def test_two_fragment_scenario_matches_default_stream_when_history_proves_it(self):
        from operator_model import OperatorStream
        shape=[65,4,5120]
        for chip in ('Ascend950DT_9582','Ascend950PR_957b','Ascend970DT'):
            stream=OperatorStream(chip)
            previous=[load_chip(chip).vector_cores,4,1216]
            prefix=stream.advance('mhc_post',previous)
            conditional=estimate_post_history(shape,chip=chip).for_predecessor(prefix.active_aiv,previous)
            self.assertAlmostEqual(conditional.latency_us,stream.preview('mhc_post',shape).latency_us)
            self.assertEqual(conditional.model,'post_two_fragment_slot_reuse')


if __name__ == '__main__':
    unittest.main()
