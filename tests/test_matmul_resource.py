import unittest
from dataclasses import replace
from operator_model.chips import load_chip
from operator_model.operators.matmul.resources import resource_matmul


class TestMatmulResource(unittest.TestCase):
    def test_counts_reuse_and_no_product_label(self):
        dt=load_chip();shape=(256,512,512)
        us,a,v,d=resource_matmul(shape,dt)
        self.assertEqual(d['logical_cube_ops'],2*256*512*512)
        self.assertGreater(d['l2_to_l1_read_bytes'],d['hbm_read_bytes_scenario'])
        self.assertAlmostEqual(us,sum(d['stages_us'].values()))
        self.assertEqual(resource_matmul(shape,replace(dt,name='anonymous')), (us,a,v,d))

    def test_hbm_and_compute_can_independently_limit(self):
        dt=load_chip();shape=(257,512,1024)
        base=resource_matmul(shape,dt)[0]
        read=resource_matmul(shape,replace(dt,hbm_bandwidth_gbs=10))[0]
        cube=resource_matmul(shape,replace(dt,cube_tops={**dt.cube_tops,'16':5}))[0]
        self.assertGreater(read,3*base)
        self.assertGreater(cube,3*base)

    def test_long_k_does_not_replay_events(self):
        us,_,_,d=resource_matmul((64,128,2**24),load_chip())
        self.assertGreater(d['pages_per_output_tile'],1000)
        self.assertGreater(us,1000)
