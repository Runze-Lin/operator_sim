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
