import json
import unittest
from operator_model.extensions import plan

class RegistryTests(unittest.TestCase):
    def test_six_families_expose_structure_without_fake_latency(self):
        for op,shape,attrs in [
            ('rmsnorm',(9,7168),{}),('addmm',(9,128,256),{}),
            ('lightning_indexer',(1,9,131072,8,128),{}),('mhc',(9,4,7168),{}),
            ('kda_chunk',(1,2048,4,4,128,128),{}),
            ('groupedmatmulswiglugmm',(8,128,128,4),{'expert_counts':[1,0,7,0]})]:
            with self.subTest(op=op):
                p=plan(op,shape,**attrs)
                self.assertEqual(p['status'],'uncalibrated');self.assertIsNone(p['predicted_us']);json.dumps(p)
    def test_user_spelling_and_invalid_speculation_route(self):
        self.assertEqual(plan('lightening_indexer',(1,9,32768,8,128)),plan('lightning_indexer',(1,9,32768,8,128)))
        self.assertEqual(plan('kda_recurrent',(1,9,4,4,128,128))['status'],'unsupported')
