import json
from dataclasses import replace
from pathlib import Path
import unittest

from operator_model import OperatorStream, estimate, load_chip

ROOT = Path(__file__).resolve().parents[1]
DT = 'Ascend950DT_9582'
PR = 'Ascend950PR_957b'
FUT = 'Ascend970DT'


class SinkHistoryTests(unittest.TestCase):

    def test_history_is_explicit_only(self):
        opts = {'num_iters': 10}
        stream = OperatorStream()
        base = estimate('mhc_sinkhorn', [67,4], opts)
        for attrs in (opts, {**opts,'resident_aiv':None}):
            self.assertEqual(stream.advance('mhc_sinkhorn', [67,4], attrs), base)
        first = estimate('mhc_sinkhorn', [67,4], {**opts,'resident_aiv':0})
        warm = estimate('mhc_sinkhorn', [67,4], {**opts,'resident_aiv':46})
        self.assertGreater(first.latency_us, warm.latency_us)
        self.assertEqual(stream.advance('mhc_sinkhorn', [67,4], opts), base)


    def test_stream_unknown_prefix_recovers_only_provable_coverage(self):
        stream = OperatorStream(track_sinkhorn_history=True)
        stream.forget_history()
        for t in (13,46):
            self.assertEqual(stream.advance('mhc_sinkhorn',(t,4)), estimate('mhc_sinkhorn',(t,4)))
        # H is still unknown in [46,64], but every such state covers 34 AIVs.
        projected = stream.preview('mhc_sinkhorn',(67,4)).projection
        plan = projected['target_plan']
        self.assertEqual((plan['resident_aiv'],plan['history_state']), (46,'resident'))
        self.assertEqual(projected['partial_history'], {'minimum_resident_aiv':46, 'maximum_resident_aiv':64})
        self.assertEqual(stream.preview('mhc_sinkhorn',(57,4)), estimate('mhc_sinkhorn',(57,4)))
        # The preceding preview of a fresh shape must not expand residency.
        stream.advance('mhc_sinkhorn',(64,4))
        wide = stream.preview('mhc_sinkhorn',(4097,4))
        self.assertEqual(wide.projection['target_plan']['resident_aiv'],64)
        self.assertEqual(wide.status,'extrapolated')
        stream.reset()
        self.assertEqual(stream.preview('mhc_sinkhorn',(257,4)).projection['target_plan']['resident_aiv'],0)
        stream.forget_history()
        self.assertEqual(stream.preview('mhc_sinkhorn',(257,4)),estimate('mhc_sinkhorn',(257,4)))

    def test_stream_cannot_inherit_fusion_and_invalid_calls_do_not_advance(self):
        stream = OperatorStream(track_sinkhorn_history=True)
        stream.advance('mhc_sinkhorn',(20,4))
        for op,shape,attrs in [('mhc_sinkhorn',(300,8),{}),('mhc_sinkhorn',(256,4),{'return_intermediates':True}),
                               ('mhc_sinkhorn',(64,4),{'implementation':'other'}),('mhc_sinkhorn',(0,4),{})]:
            self.assertIsNone(stream.advance(op,shape,attrs).latency_us)
        self.assertIsNone(stream.advance('mhc_sinkhorn',(32,4),dtype='bf16').latency_us)
        with self.assertRaises(ValueError):
            stream.advance('mhc_sinkhorn',(64,4),aic_only=True)
        self.assertEqual(stream.preview('mhc_sinkhorn',(21,4)).projection['target_plan']['resident_aiv'],20)
        self.assertIsNotNone(stream.advance('mhc_pre_sinkhorn',(72,4,4096)).latency_us)
        self.assertEqual(stream.preview('mhc_sinkhorn',(21,4)),estimate('mhc_sinkhorn',(21,4)))
        stream.advance('mhc_sinkhorn',(30,4))
        stream.advance('matmul',(64,64,64),aic_only=True)
        self.assertEqual(stream.preview('mhc_sinkhorn',(21,4)),estimate('mhc_sinkhorn',(21,4)))

    def test_stream_explicit_undefined_and_unsupported_hardware(self):
        stream = OperatorStream(track_sinkhorn_history=True)
        stream.advance('mhc_sinkhorn',(46,4),{'resident_aiv':52})
        self.assertEqual(stream.preview('mhc_sinkhorn',(32,4)).projection['target_plan']['resident_aiv'],52)
        self.assertEqual(stream.preview('mhc_sinkhorn',(32,4),{'resident_aiv':0}).projection['target_plan']['history_state'],'first')
        stream.advance('mhc_sinkhorn',(16,4),{'resident_aiv':None})
        self.assertEqual(stream.preview('mhc_sinkhorn',(30,4)),estimate('mhc_sinkhorn',(30,4)))
        self.assertEqual(stream.preview('mhc_sinkhorn',(16,4)).projection['target_plan']['history_state'],'resident')
        chip = load_chip()
        # A launch that does not fit the issue model cannot advance the stream.
        # The next supported launch therefore still observes a clean prefix.
        small = replace(chip,details={**chip.details,'ub_bytes':1800})
        stream = OperatorStream(small,track_sinkhorn_history=True)
        rejected = stream.advance('mhc_sinkhorn',(80,4))
        self.assertEqual(rejected,estimate('mhc_sinkhorn',(80,4),chip=small))
        self.assertIsNone(rejected.latency_us)
        self.assertEqual(stream.preview('mhc_sinkhorn',(16,4)),
                         estimate('mhc_sinkhorn',(16,4),{'resident_aiv':0},chip=small))
        no_vec = replace(chip,vector_tops={**chip.vector_tops,'32':None})
        stream = OperatorStream(no_vec,track_sinkhorn_history=True)
        self.assertEqual(stream.advance('mhc_sinkhorn',(67,4)),estimate('mhc_sinkhorn',(67,4),chip=no_vec))
        self.assertEqual(stream.preview('mhc_sinkhorn',(67,4),{'resident_aiv':0}).status,'unavailable')
        with self.assertRaises(TypeError):
            OperatorStream(track_sinkhorn_history=1)

    def test_exact_qualification_filters_and_custom_resources(self):
        chip = load_chip(PR)
        attrs = {'num_iters':10, 'resident_aiv':0}
        def result(t, target=chip, mode='auto', options=attrs):
            return estimate('mhc_sinkhorn', [t,4], options, chip=target, timing_mode=mode)
        self.assertEqual(result(67, mode='calibrated').status, 'estimated')
        outside = result(257)
        self.assertEqual(outside.status, 'extrapolated')
        qualified = result(257, mode='calibrated')
        self.assertIsNone(qualified.latency_us)
        self.assertIsNone(qualified.projection['latency_us'])
        self.assertEqual(qualified.status, 'unavailable')
        self.assertEqual(result(10, options={**attrs,'num_iters':100}).status, 'extrapolated')
        custom = replace(chip, name='future-variant')
        self.assertEqual(result(130,custom).latency_us, result(130).latency_us)
        self.assertEqual(result(130,custom).status,'extrapolated')
        self.assertGreater(result(130,replace(custom, hbm_bandwidth_gbs=.1)).latency_us, result(130).latency_us)
        slow_vec = replace(custom, vector_tops={**custom.vector_tops,'32':.1})
        self.assertGreater(result(130,slow_vec).latency_us, result(130).latency_us)
        slow_write = replace(custom, details={**custom.details,'mte3_bandwidth_gbs':.01})
        self.assertGreater(result(130,slow_write).latency_us, result(130).latency_us)
        low = replace(chip, details={**chip.details,'ub_bytes':1024})
        self.assertEqual(result(130,low).status, 'unavailable')
        absent = replace(chip, vector_tops={**chip.vector_tops,'32':None})
        self.assertIsNone(result(1,absent).latency_us)

    def test_only_valid_same_family_contract(self):
        for attrs in ({'resident_aiv':False},{'resident_aiv':57},{'resident_aiv':-1},
                      {'resident_aiv':1,'return_intermediates':True},{'resident_aiv':1,'num_iters':False},
                      {'resident_aiv':2,'same_shape':True}):
            self.assertEqual(estimate('mhc_sinkhorn',[64,4],attrs,chip=PR).status,'unsupported')
        self.assertEqual(estimate('mhc_pre_sinkhorn',[64,4,4096],{'resident_aiv':46}).status,'unsupported')
        self.assertEqual(estimate('mhc_sinkhorn',[1,4],{'resident_aiv':0},dtype='bf16').status,'unsupported')


if __name__ == '__main__':
    unittest.main()
