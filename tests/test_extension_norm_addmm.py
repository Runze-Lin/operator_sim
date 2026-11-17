import unittest
from operator_model.extensions.norm_addmm import plan


class NormAddmmPlans(unittest.TestCase):
    def test_rms_work_and_core_tail(self):
        p=plan('rmsnorm',(65,5120))
        self.assertEqual(p['predicted_us'],None)
        self.assertEqual(p['tiling']['active_aiv'],33)
        self.assertEqual(p['tiling']['last_core_rows'],1)
        self.assertEqual(p['work']['rsqrt'],65)
        self.assertEqual(p['work']['flops_excluding_rsqrt'],65*(4*5120+1))
        self.assertEqual(p['output_shapes']['rstd'],[65,1])

    def test_rms_large_reduction_does_not_claim_full_row_route(self):
        p=plan('rms_norm',(1,65536))
        self.assertEqual(p['tiling']['status'],'unresolved_dispatch')
        self.assertIsNone(p['tiling']['active_aiv'])
        self.assertIsNone(p['predicted_us'])

    def test_addmm_broadcast_is_not_an_extra_full_tensor_read(self):
        bias=plan('addmm',(9,128,64),self_shape=(128,))
        matrix=plan('addmm',(9,128,64),self_shape=(9,128))
        self.assertEqual(matrix['work']['logical_tensor_bytes']-bias['work']['logical_tensor_bytes'],2*8*128)
        self.assertEqual(bias['work']['matmul_flops'],2*9*128*64)
        self.assertIsNone(bias['predicted_us'])
        self.assertIsNone(bias['tiling']['active_aic'])
        self.assertFalse(bias['tiling']['fused_add_source_predicates_match'])
        self.assertTrue(matrix['tiling']['fused_add_source_predicates_match'])
        self.assertFalse(plan('addmm',(9,128,64),self_shape=(9,128),dtype='fp32')['tiling']['fused_add_source_predicates_match'])
        with self.assertRaises(ValueError):plan('addmm',(9,128,64),self_shape=(2,128))

    def test_invalid_and_explicit_scope(self):
        for op,shape,attrs in [('rmsnorm',(0,64),{}),('rmsnorm',(1,64),{'epsilon':-1}),
                                ('addmm',(8,64,128),{'beta':float('nan')}),('rmsnorm',(1,64),{'causal':True})]:
            with self.assertRaises(ValueError):plan(op,shape,**attrs)


if __name__=='__main__':unittest.main()
