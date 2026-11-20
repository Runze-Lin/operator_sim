from collections import Counter
from dataclasses import replace
import unittest
from operator_model.operators.matmul.waves import dispatch
from operator_model.operators.matmul.resources import resource_matmul
from operator_model.chips import load_chip
from operator_model.operators.matmul.tiling import matmul_plan


class TailWaveTests(unittest.TestCase):
    pr=load_chip('Ascend950PR_957b')

    def test_final_wave_splits_output_only(self):
        sh=(1024,2048,32);plan=matmul_plan(sh,self.pr)
        full,tail,split=dispatch(sh,plan)
        self.assertEqual((split,full,tail),((3,2),Counter({(256,256):28}),Counter({(86,128):16,(84,128):8})))
        self.assertEqual(sum(x*y*n for h in (full,tail) for (x,y),n in h.items()),sh[0]*sh[1])
        us,_,_,d=resource_matmul(sh,self.pr)
        self.assertEqual(d['mmads_total'],52);self.assertAlmostEqual(us,sum(d['stages_us'].values()))

    def test_second_wave_is_not_serial_drain(self):
        sh=(1536,2048,16)
        t,_,_,d=resource_matmul(sh,self.pr);f=d['wave_cost_cycles']['full']
        self.assertEqual(d['mmads_total'],56)
        self.assertLess(d['stages_us']['additional_output_waves'],sum(f[k] for k in ('R','L','P','C','W'))/1650)
        for sh in ((1599,3462,33),(9517,2045,1031),(1,349,17)):
            full,tail,_=dispatch(sh,matmul_plan(sh,self.pr))
            self.assertEqual(sum(x*y*n for h in (full,tail) for (x,y),n in h.items()),sh[0]*sh[1])

    def test_write_spill_and_share(self):
        sh=(1536,2048,16);dt=load_chip()
        _,_,_,small=resource_matmul((1,32,16),dt)
        _,_,_,large=resource_matmul(sh,dt)
        self.assertLess(small['shared_traffic']['l2_payload_fraction_of_nominal'],large['shared_traffic']['l2_payload_fraction_of_nominal'])
        self.assertLessEqual(large['shared_traffic']['l2_payload_fraction_of_nominal'],1.)
        _,_,_,spilled=resource_matmul(sh,replace(dt,details={**dt.details,'l2_bytes':32768}))
        self.assertGreater(spilled['shared_traffic']['hbm_write_bytes_exposed'],large['shared_traffic']['hbm_write_bytes_exposed'])
