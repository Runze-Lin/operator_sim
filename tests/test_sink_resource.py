import unittest

from operator_model import estimate, load_chip
from operator_model.operators.mhc.sink_resources import partition


class SinkResourceTests(unittest.TestCase):
    def test_equal_vector_groups_across_pr_boundary(self):
        def predict(t):
            return estimate('mhc_sinkhorn',(t,4),chip='Ascend950PR_957b')
        first, second = predict(56), predict(57)
        self.assertEqual((first.active_aiv,second.active_aiv),(56,29))
        for r,rows in [(first,1),(second,2)]:
            self.assertEqual(r.projection['target_plan']['rows_per_core'],rows)
            self.assertEqual(r.projection['target_plan']['register_passes'],1)
        self.assertAlmostEqual(first.projection['stages_us']['normalization_loop'],
                               second.projection['stages_us']['normalization_loop'])

    def test_pre_mixed_launch_vs_stage_two_payload(self):
        shape=(1,4,4096)
        for chip,k,blocks in [('Ascend950DT_9582',512,32),('Ascend950PR_957b',768,22)]:
            p=partition(shape,load_chip(chip))
            self.assertEqual((p['tile'],p['k_blocks']),([16,k],blocks))
            self.assertEqual((p['active_aic'],p['active_aiv'],p['stage2_payload_aiv']),(blocks,2*blocks,1))

    def test_resident_depth_only_repeats_weighted_part_of_tail(self):
        def tail(n):
            r=estimate('attn_res_fwd',(1,7168,8,n),dict(addend=True,write_idx=n,materialize=True))
            return r.projection['stages_us']['weighted_norm_write']
        # A 2x deeper bank repeats the weighted multiply-adds. The output
        # normalization and gamma row transfer are still single-pass.
        self.assertLess(tail(3),1.25*tail(1))
        self.assertGreater(tail(3),tail(1))
