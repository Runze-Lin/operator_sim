"""Release-limited A8W4 page schedules, including large-K fast evaluation."""

import unittest

from operator_model import estimate
from operator_model.operators.matmul.mx_handoff import schedule


class MxHandoffTest(unittest.TestCase):
    def test_ring_power_matches_explicit_event_schedule(self):
        for slots in (2,4):
            for lanes in (1,2):
                for speed in (.001,1,1000):
                    rates = dict(aiv=lanes,mte2=47*speed,mte1=181/speed,mte3=158,cube=16384,
                                 shared_per_byte=.005,read_lat=1200,load_lat=36,cube_lat=83,
                                 handoff_lat=158,setup=50,vector_scale=speed,mmad_ii=98)
                    plan = dict(l1_k=1024,ub_n=48,l1_slots=slots,base_tile=[64,96,256],tail_split=[1,2])
                    for k in (257*1024,257*1024+8):
                        fast = schedule(k,64,96,3,plan,rates)
                        direct = schedule(k,64,96,3,plan,rates,fast=False)
                        for key in direct:
                            self.assertGreaterEqual(fast[key],0)
                            self.assertAlmostEqual(fast[key],direct[key],delta=max(1e-7,1e-12*abs(direct[key])))

    def test_old_two_sku_handoff_direction_and_short_kernel(self):
        opts = dict(dtype='mxfp8',attrs={'weight_dtype':'mxfp4'})
        dt,pr = [estimate('matmul',[512,64,7168],chip=s,**opts) for s in ('Ascend950DT_9582','Ascend950PR_957b')]
        self.assertGreater(dt.latency_us,pr.latency_us)
        self.assertEqual([dt.projection['target_plan']['pages_per_tile'],pr.projection['target_plan']['pages_per_tile']],[4,7])
        one = estimate('matmul',[9,128,256],chip='Ascend950DT_9582',**opts)
        self.assertNotIn('page_schedule',one.projection['target_plan'])
        self.assertAlmostEqual(one.latency_us,3.9457,places=3)
