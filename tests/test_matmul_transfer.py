from dataclasses import replace
import unittest
from operator_model.chips import load_chip
from operator_model.operators.matmul.projection import project_matmul
from operator_model.scheduling.tiling import matmul_plan
from operator_model.scheduling.reference import predict


class CubeLandmarkTests(unittest.TestCase):
    def calculate(self,shape,chip):
        dt=load_chip();f=dict.fromkeys(('SCALAR','MTE1','FIXP','CUBE'),1.)
        return project_matmul(shape,f,dt,chip,matmul_plan(shape,chip.cube_cores))

    def test_paired_calibration_requires_sku_and_physics_and_preserves_identity(self):
        dt,pr=load_chip(),load_chip('Ascend950PR_957b');sh=(129,256,256)
        self.assertEqual(self.calculate(sh,dt)[0],predict('matmul',sh)['predicted_us'])
        self.assertEqual(self.calculate(sh,replace(pr,name='custom'))[1]['mode'],'reference_anchored_landmarks')
        self.assertEqual(self.calculate(sh,replace(pr,hbm_bandwidth_gbs=100))[1]['mode'],'reference_anchored_landmarks')

    def test_second_read_page_and_critical_tail_count_once(self):
        pr=load_chip('Ascend950PR_957b');a=self.calculate((64,256,512),pr)[1];b=self.calculate((64,256,513),pr)[1]
        self.assertEqual((a['read_pages'],b['read_pages']),(1,2));self.assertEqual(a['page_intervals_us'],0.)
        self.assertGreater(b['page_intervals_us'],0.);self.assertLess(a['drain_us'],.5)
        us,diag=self.calculate((128,256,512),pr);self.assertAlmostEqual(us,sum(diag[k] for k in ('prefix_us','first_pair_us','l0_us','page_intervals_us','last_compute_us','drain_us','extra_waves_us')))
