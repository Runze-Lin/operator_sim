from dataclasses import replace
import unittest
from operator_model import load_chip
from operator_model.matmul_multisplit import estimate_matmul_retile


class RetileGeometryTest(unittest.TestCase):
    def test_geometry_uses_two_input_double_buffers(self):
        expected=[
            ([144,64,7168],(1,32,224,64,128),(1,28,256,64,112)),
            ([9,1280,5120],(5,6,854,64,420),(5,5,1024,64,400)),
        ]
        for shape,dt,pr in expected:
            for chip,physical in [('Ascend950DT_9582',dt),('Ascend950PR_957b',pr)]:
                x=estimate_matmul_retile(shape,chip=chip)
                self.assertEqual(tuple(x[k] for k in ('output_tiles','k_splits','max_segment_k','base_k','mmads_total')),physical)
                self.assertEqual(x['productive_aic'],x['output_tiles']*x['k_splits'])
                self.assertEqual(shape[2],(x['k_splits']-1)*x['max_segment_k']+x['tail_segment_k'])
    def test_no_renaming_or_interfering_with_released_branches(self):
        dt=load_chip()
        x=estimate_matmul_retile([72,64,7168],chip=dt)
        self.assertEqual(x,estimate_matmul_retile([72,64,7168],chip=replace(dt,name='scenario-copy')))
        self.assertEqual(estimate_matmul_retile([16,64,7168])['route'],'one_output_split_k')
        self.assertEqual(estimate_matmul_retile([72,64,7168],chip='Ascend970DT')['route'],'ordinary')
        self.assertEqual(estimate_matmul_retile([262144,32,5120])['route'],'ordinary')
