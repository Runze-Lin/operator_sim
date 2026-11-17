import json
from pathlib import Path
import random
import unittest
from operator_model.model_v6 import Case,Predictor,applies
from operator_model.model_v5 import Predictor as V5
ROOT=Path(__file__).resolve().parents[1]

class QuantEnvelopeTests(unittest.TestCase):
    def setUp(self):
        self.new=Predictor(ROOT/'results/v0.6/model.json',cache_size=2)
        self.old=V5(ROOT/'results/v0.5/model.json')

    def test_scalar_full_parity_across_tiles_and_tails(self):
        rng=random.Random(6)
        cases=[(9,5120),(144,7168),(1024,7168),(33,768),(1,64)]
        cases += [(rng.randint(1,4096),64*rng.randint(1,256)) for _ in range(100)]
        for shape in cases:
            c=Case('quant',shape)
            self.assertAlmostEqual(self.new.predict_us(c,False),self.new.predict(c,False)['predicted_us'],places=11)
        self.assertEqual(len(self.new.cache),0)

    def test_only_guarded_timing_changes_and_cache_is_immutable(self):
        for op,shape in [('fa',(9,131072,64,1,1)),('matmul',(17,96,128)),('quant',(1024,7168)),('quant',(36,512))]:
            c=Case(op,shape)
            self.assertEqual(self.new.predict_us(c,False),self.old.predict_us(c,False))
        c=Case('quant',(9,5120));p=self.new.predict(c);self.assertTrue(applies(p['plan']))
        self.assertAlmostEqual(p['phase_schedule']['write_finish_us'][-1]+self.new.quant_model['quant']['costs']['suffix_us'],p['predicted_us'])
        self.assertIs(p,self.new.predict(c))
        with self.assertRaises(TypeError):p['phase_schedule']['predicted_us']=0
        for s in [(144,7168),(36,5120)]:self.new.predict(Case('quant',s))
        self.assertEqual(len(self.new.cache),2)

    def test_other_operators_historical_frozen_values_preserved(self):
        d=json.loads((ROOT/'results/v0.5/evaluation.json').read_text())
        for r in d['historical']+d['rows']:
            if r['op']=='quant':continue
            self.assertAlmostEqual(self.new.predict_us(Case(r['op'],r['shape']),False),r['v05_us'],places=12)
