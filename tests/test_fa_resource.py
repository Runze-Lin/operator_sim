from dataclasses import replace
import unittest

from operator_model.chips import load_chip
from operator_model.fa_resource import resource_fa


class FAResourceTest(unittest.TestCase):
    def setUp(self):
        self.dt=load_chip()

    def test_sku_names_do_not_affect_stages(self):
        shape=(9,32768,64,1,1)
        us, aic, aiv, d = resource_fa(shape,self.dt)
        self.assertEqual(resource_fa(shape,replace(self.dt,name='anonymous')),(us,aic,aiv,d))
        self.assertEqual((aic,aiv,d['target_plan']['iterations'],d['mmads_total']),(1,2,128,512))
        self.assertAlmostEqual(sum(d['stages_us'].values()),us)

    def test_tail_is_priced_at_its_true_extent(self):
        _,_,_,out=resource_fa((9,16385,64),self.dt)
        w=out['wave_types'][0]
        self.assertEqual((w['first']['k_bytes'],w['last']['k_bytes']),(32768,128))
        self.assertEqual(out['mmads_total'],258)
        self.assertEqual(out['k_read_bytes_scenario'],2*16385*64)
        self.assertGreater(w['first']['ii_cycles'],w['last']['ii_cycles'])

    def test_shared_ports_and_vector_price_critical_path(self):
        shape=(64,2048,128,1,64)
        base=resource_fa(shape,self.dt)[0]
        self.assertGreater(resource_fa(shape,replace(self.dt,hbm_bandwidth_gbs=50))[0],base)
        rates=dict(self.dt.vector_tops, **{'32':self.dt.vector_tops['32']/4})
        self.assertGreater(resource_fa(shape,replace(self.dt,vector_tops=rates))[0],base)
        rates=dict(self.dt.cube_tops, **{'16':self.dt.cube_tops['16']/8})
        self.assertGreater(resource_fa(shape,replace(self.dt,cube_tops=rates))[0],base)

    def test_waves_and_vector_ratio(self):
        shape=(129,512,64,1,67)
        _, aic, aiv, d=resource_fa(shape,replace(self.dt,vector_cores=32))
        self.assertEqual((aic,aiv),(32,32))
        self.assertEqual(d['target_plan']['jobs'],134)
        self.assertEqual(sum(w['occupied']*w['waves'] for w in d['wave_types']),134)
        self.assertEqual(d['target_plan']['waves'],5)

    def test_single_pair_with_partial_q_wave(self):
        _,_,_,d=resource_fa((129,512,64,1,3),replace(self.dt,cube_cores=1))
        self.assertEqual([(w['q_rows'],w['waves']) for w in d['wave_types']],[(128,3),(1,3)])

    def test_launch_cycles_scale_with_clock(self):
        shape=(9,8192,64)
        s=resource_fa(shape,self.dt)[3]['stages_us']['launch']
        s2=resource_fa(shape,replace(self.dt,frequency_ghz=2*self.dt.frequency_ghz))[3]['stages_us']['launch']
        self.assertAlmostEqual(s/2,s2)
