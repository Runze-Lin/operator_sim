"""Fused MIX has a different entry point and scheduling contract from standalone Sinkhorn."""
import unittest

from operator_model import load_chip
from operator_model.operators.mhc.pre_sinkhorn import cube_fragments, fused
from operator_model.operators.mhc.sink_resources import fused as legacy

ATTRS={'num_iters':20,'need_backward':False}


class PreSinkResources(unittest.TestCase):
    def test_partial_l1_and_k_split_grid(self):
        self.assertEqual(cube_fragments(80,512,24),(6,384,96))
        self.assertEqual(cube_fragments(80,768,24),(8,384,96))
        self.assertEqual(cube_fragments(80,256,24)[0],3)
        t=load_chip('Ascend950PR_957b')
        us,aic,aiv,data=fused((80,4,4096),t,ATTRS)
        self.assertEqual((aic,aiv),(22,44))
        self.assertAlmostEqual(sum(data['stages_us'].values()),us)
        self.assertEqual(data['target_plan']['stage2_rows'],2)

    def test_anchor_and_distinct_unverified_template(self):
        for sku in ('Ascend950PR_957b','Ascend950DT_9582'):
            target=load_chip(sku)
            self.assertAlmostEqual(fused((1,4,4096),target,ATTRS)[0],legacy((1,4,4096),target,ATTRS)[0])
            us,_,_,d=fused((128,4,4096),target,ATTRS)
            self.assertEqual(us,legacy((128,4,4096),target,ATTRS)[0])
            self.assertEqual(d['schedule'],'prior_overlap_template')
        self.assertEqual(fused((80,4,4096),load_chip('Ascend970DT'),ATTRS)[-1]['schedule'],'prior_overlap_template')
