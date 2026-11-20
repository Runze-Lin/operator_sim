import unittest
from dataclasses import replace
from operator_model.chips import load_chip
from operator_model.operators.attention.projection import project_fa

class FlashTransferTests(unittest.TestCase):
    def price(self, shape, factors=None, chip=None):
        default = dict.fromkeys(('CUBE','SIMD','MTE1','MTE2','MTE3','FIXP','SCALAR'),1.)
        return project_fa(shape,default|dict(factors or {}),load_chip(),chip or load_chip(),
                          {'predicted_us':20.})

    def test_common_resource_model_and_calibration_scope(self):
        shape=(17,1025,64,1,1)
        self.assertAlmostEqual(self.price(shape)[0],20.)
        self.assertGreater(self.price(shape,{'SIMD':2.})[0],20.)
        pr=load_chip('Ascend950PR_957b')
        self.assertLess(self.price(shape,chip=pr)[0],20.)
        self.assertAlmostEqual(self.price(shape,chip=replace(pr,frequency_ghz=2.))[0],20.)
        self.assertAlmostEqual(self.price((129,128,128,32,32),chip=pr)[0],20.)

    def test_effective_issue_and_instruction_width_are_distinct(self):
        pr=load_chip('Ascend950PR_957b')
        _,d=self.price((9,4097,64,1,1),chip=pr)
        self.assertEqual((d['kv_tiles'],d['mmads_per_core']),(17,66))
        self.assertEqual(d['parameter_source'],'paired PR CA')
        self.assertGreater(self.price((9,8192,64,1,1),{'SIMD':4.})[0],20.)
