"""Offline regression tests for the experimental schedule; no CA or GPU required."""
from dataclasses import replace
import unittest

from evidence import DT, PR, read, OUT
from model import forecast
from operator_model import load_chip


class SinkhornScheduleTests(unittest.TestCase):
    def test_kernel_register_pass_boundary(self):
        # N=4 packs eight matrices on a 256-byte VReg over 32-byte UB blocks.
        for sku in (DT, PR):
            c=load_chip(sku)
            for rows, expected in [(4,1),(5,1),(8,1),(9,2)]:
                t=(rows-1)*c.vector_cores+1
                pred=forecast((t,4),10,c,0)
                self.assertEqual((pred['rows_per_core'],pred['register_passes']),(rows,expected))
                self.assertEqual(pred['bytes_written'],64*t)

    def test_same_family_order_holds_same_work(self):
        c=load_chip(DT)
        cold=forecast((67,4),10,c,0)
        hot=forecast((67,4),10,c,c.vector_cores)
        self.assertEqual(cold['vf_calls_per_core'],hot['vf_calls_per_core'])
        self.assertEqual(cold['vf_setup_epochs']['steady_pair_period'],hot['vf_setup_epochs']['steady_pair_period'])
        self.assertGreater(cold['vf_setup_epochs']['setup_vf'],hot['vf_setup_epochs']['setup_vf'])

    def test_first_iteration_and_more_cycles(self):
        c=load_chip(DT)
        first=forecast((67,4),1,c,0)
        second=forecast((67,4),2,c,0)
        third=forecast((67,4),3,c,0)
        self.assertEqual(first['vf_calls_per_core'],4)
        self.assertAlmostEqual((second['latency_us']-first['latency_us'])*1650,248)
        self.assertAlmostEqual((third['latency_us']-second['latency_us'])*1650,254)

    def test_unseen_chip_responds_to_resource(self):
        c=load_chip(PR)
        p=forecast((130,4),10,c,0)
        renamed=replace(c,name='custom')
        self.assertEqual(p,forecast((130,4),10,renamed,0))
        slow=replace(renamed,hbm_bandwidth_gbs=1.)
        self.assertGreater(forecast((130,4),10,slow,0)['latency_us'],p['latency_us'])

    def test_unsupported_ub_and_layout(self):
        with self.assertRaises(ValueError):
            forecast((4097,4),10,replace(load_chip(DT),details={'ub_bytes':8192}),0)
        with self.assertRaises(ValueError):
            forecast((128,8),10,load_chip(DT),0)

    def test_all_locked_model_outputs(self):
        for j in read(OUT/'register.json')['jobs']:
            p=forecast(j['shape'],j['iters'],load_chip(j['chip']),j['prior_h'])
            self.assertAlmostEqual(p['latency_us'],j['known']['latency_us'],places=11)


if __name__=='__main__':unittest.main()
