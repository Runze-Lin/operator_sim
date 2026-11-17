import unittest
from operator_model.model_v2 import Case,plan,nnls


class ExpandedTests(unittest.TestCase):
    def test_ub_capacity_transition(self):
        a=plan(Case('quant',(64,1024)))
        b=plan(Case('quant',(1024,4096)))
        c=plan(Case('quant',(768,4096)))
        self.assertEqual(a['iterations'],1)
        self.assertEqual((b['iterations'],c['iterations']),(2,2))
        self.assertEqual((b['active_aiv'],c['active_aiv']),(64,64))
        self.assertLess(c['units']['vector'],b['units']['vector'])

    def test_matmul_core_saturation_retiles(self):
        p=plan(Case('matmul',(128,512,512)))
        self.assertEqual(p['tile'],[32,64,256])
        self.assertEqual(p['active_aic'],32)
        self.assertEqual(plan(Case('matmul',(65,128,129)))['branch'],'k_tail')

    def test_fa_heads_and_q_boundary(self):
        self.assertEqual(Case('fa',(16,128,64)).shape,(16,128,64,1,1))
        p=plan(Case('fa',(16,128,64,1,8)))
        self.assertEqual((p['active_aic'],p['active_aiv']),(8,16))
        p=plan(Case('fa',(129,128,64)))
        self.assertEqual((p['active_aic'],p['active_aiv']),(3,6))
        self.assertNotEqual(plan(Case('fa',(128,128,64)))['branch'],p['branch'])
        self.assertEqual(plan(Case('fa',(16,128,128)))['branch'],'d128_kv128')

    def test_nonnegative_multifeature_fit(self):
        xs=[[1,0,0],[1,1,0],[1,0,1],[1,1,1]];ys=[2,5,6,9]
        weights=nnls(xs,ys)
        for a,b in zip(weights,[2,3,4]):self.assertAlmostEqual(a,b,places=6)
        self.assertGreaterEqual(min(nnls([[1,1],[1,2],[1,3]],[4,3,2])),0)


if __name__=='__main__':unittest.main()
