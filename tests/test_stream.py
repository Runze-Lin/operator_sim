from unittest import TestCase
from operator_model import OperatorStream, estimate


class StreamTest(TestCase):
    def test_preview_commit_alias_and_reset(self):
        s = OperatorStream('Ascend950PR_957b')
        first = s.preview('mhc_post',[3,4,512])
        self.assertEqual(first,s.advance('mhc_post',[3,4,512]))
        self.assertEqual(first.projection['timing_diagnostics']['context'],'first')
        self.assertEqual(s.preview('mhc_post',[1,4,128]).projection['timing_diagnostics']['context'],'reuse')
        before = s.preview('mhc_post',[1,4,128])
        self.assertIsNone(s.advance('missing',[1]).latency_us)
        self.assertEqual(before,s.preview('mhc_post',[1,4,128]))
        s.advance('matmul',[16,64,256])
        self.assertEqual(s.preview('mhc_post',[1,4,128]),estimate('mhc_post',[1,4,128],chip=s.chip))
        s.reset()
        self.assertEqual(s.advance('rms_norm',[8,1536]).projection['timing_diagnostics']['residency'],'first')
        self.assertEqual(s.preview('rmsnorm',[4,1536]).projection['timing_diagnostics']['residency'],'reuse')

    def test_foreign_launch_leaves_only_a_lower_bound(self):
        s = OperatorStream('Ascend950PR_957b')
        s.advance('matmul', [16,64,256])
        s.advance('mhc_post', [1,4,128])
        self.assertEqual(s.preview('mhc_post', [3,4,128]), estimate('mhc_post', [3,4,128], chip=s.chip))
        s.advance('mhc_post', [3,4,128])
        self.assertEqual(s.preview('mhc_post', [2,4,128]).projection['timing_diagnostics']['context'], 'reuse')
        s.advance('mhc_post', [3,4,128], {'resident_aiv': 3})
        self.assertEqual(s.preview('mhc_post', [4,4,128]).projection['timing_diagnostics']['context'], 'expand')

    def test_partial_rms_clips_point_without_committing_an_inferred_history(self):
        stream = OperatorStream('Ascend950PR_957b')
        stream.forget_history()
        self.assertEqual(stream.advance('rmsnorm', [17,2400]), estimate('rmsnorm', [17,2400], chip=stream.chip))
        partial = stream.preview('rms_norm', [34,1664])
        history = partial.projection['partial_history']
        self.assertEqual(history['minimum_resident_aiv'], 17)
        self.assertGreater(history['shape_only_us'], history['possible_us'][1])
        self.assertAlmostEqual(partial.latency_us, history['possible_us'][1])
        equivalent = estimate('rmsnorm', [34,1664], {'resident_aiv':history['endpoint_scenario_aiv']}, chip=stream.chip)
        self.assertEqual(partial.rates, equivalent.rates)
        self.assertEqual(stream.advance('rmsnorm', [34,1664]), partial)
        self.assertEqual(stream.preview('rmsnorm',[47,2592]).projection['partial_history']['minimum_resident_aiv'], 34)
        self.assertIsNone(stream.advance('rmsnorm', [47,2592], dtype='int4').latency_us)
        stream.forget_history()
        self.assertEqual(stream.preview('rmsnorm',[34,1664]),estimate('rmsnorm',[34,1664],chip=stream.chip))
        stream.reset()
        self.assertEqual(stream.preview('rmsnorm',[34,1664]),estimate('rmsnorm',[34,1664],{'resident_aiv':0},chip=stream.chip))
