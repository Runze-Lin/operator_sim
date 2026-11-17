import random
import unittest
from pathlib import Path
from operator_model.fast_query import Predictor
from operator_model.model_v6 import Case,Predictor as Frozen
ROOT=Path(__file__).resolve().parents[1]

class StreamingQueryTests(unittest.TestCase):
    def test_streaming_equals_frozen_equations(self):
        fast=Predictor(ROOT/'results/v0.6/model.json');frozen=Frozen(ROOT/'results/v0.6/model.json')
        rng=random.Random(26)
        shapes=[(36,5120),(288,7168),(1024,7168),(1048576,64),(144,7168),(33,768)]
        shapes += [(rng.randint(1,4096),64*rng.randint(1,1024)) for _ in range(150)]
        for shape in shapes:
            c=Case('quant',shape)
            self.assertAlmostEqual(fast.predict_us(c,False),frozen.predict_us(c,False),places=10)
        self.assertEqual(len(fast.cache),0)

    def test_other_paths_and_cached_results_preserved(self):
        fast=Predictor(ROOT/'results/v0.6/model.json');frozen=Frozen(ROOT/'results/v0.6/model.json')
        for op,s in [('fa',(9,131072,64,1,1)),('matmul',(129,256,256)),('quant',(36,5120))]:
            c=Case(op,s);self.assertEqual(fast.predict(c),frozen.predict(c))
            self.assertEqual(fast.predict_us(c),frozen.predict_us(c))
