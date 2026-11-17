import json
from pathlib import Path
import unittest

from operator_model.model_v5 import Case,Predictor,schedule,supported
from operator_model.model_v4 import Predictor as V4

ROOT=Path(__file__).resolve().parents[1]


class LongFATests(unittest.TestCase):
    def test_latency_is_fill_not_repeated_interval(self):
        costs={'prefix_us':2,'first_k_latency_us':10,'first_v_start_lag_us':1,'first_v_latency_us':8,
            'compute_fill_us':3,'drain_us':4,'k_initiation_interval_us':.5,'v_initiation_interval_us':.6,
            'cube_service_per_tile_us':.1,'vector_service_per_tile_us':.2,'mte1_service_per_tile_us':.3,'fix_service_per_tile_us':.1}
        one=schedule(256,costs);two=schedule(512,costs)
        self.assertAlmostEqual(two['predicted_us']-one['predicted_us'],.6)
        longer={**costs,'first_k_latency_us':20}
        self.assertAlmostEqual(schedule(131072,longer)['predicted_us']-schedule(131072,costs)['predicted_us'],10)
        self.assertAlmostEqual(schedule(512,{**costs,'vector_service_per_tile_us':2})['steady_interval_us'],2)

    def test_dispatch_guard_and_tail(self):
        self.assertTrue(supported(Case('fa',(9,16385,64,1,1))))
        for s in [(9,1024,64,1,1),(17,32768,64,1,1),(9,32768,128,1,1),(9,32768,64,4,1)]:
            self.assertFalse(supported(Case('fa',s)))

    @unittest.skipUnless((ROOT/'results/v0.5/model.json').exists(),'model not frozen yet')
    def test_frozen_model_preserves_other_routes_and_cache(self):
        new=Predictor(ROOT/'results/v0.5/model.json',cache_size=1);old=V4(ROOT/'results/v0.4/model.json')
        for op,s in [('quant',(9,5120)),('matmul',(32,128,512)),('fa',(17,32768,64,1,1)),('fa',(9,32768,64,4,1))]:
            c=Case(op,s);self.assertEqual(new.predict(c),old.predict(c))
        c=Case('fa',(9,131072,64,1,1));value=new.predict(c)
        self.assertEqual(value,new.predict(c,use_cache=False));self.assertEqual(new.predict_us(c,False),value['predicted_us'])
        self.assertIs(new.predict(c),value)
        with self.assertRaises(TypeError):value['phase_schedule']['iterations']=0
        new.predict(Case('fa',(9,16385,64,1,1)))
        self.assertEqual(len(new.cache),1)

if __name__=='__main__':unittest.main()
