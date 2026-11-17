import json
from pathlib import Path
import unittest
from operator_model.model_v3 import Predictor as V3
from operator_model.model_v4 import Case,Predictor,fit
ROOT=Path(__file__).resolve().parents[1]

class V4Tests(unittest.TestCase):
 @classmethod
 def setUpClass(cls):
  cls.records=json.loads((ROOT/'results/v0.4/imported-calibration.json').read_text())['records']
  cls.model=fit(cls.records,json.loads((ROOT/'results/v0.4/trace-analysis.json').read_text()),json.loads((ROOT/'results/v0.3/model.json').read_text()))
 def test_single_pass_quant_does_not_invoke_legacy_model(self):
  p=Predictor(self.model)
  def fail(*a,**k):raise AssertionError('duplicate old prediction')
  p.base.predict=fail
  for shape in [(1,128),(9,5120),(1152,5120),(32768,7168)]:
   c=Case('quant',shape);out=p.predict(c,False)
   self.assertEqual(out['predicted_us'],p.predict_us(c,False))
   self.assertGreater(out['predicted_us'],0)
 def test_retained_fa_and_matmul_on_all_old_shapes(self):
  p=Predictor(self.model);old=V3(self.model['base_v3_model'])
  for r in self.records:
   if r['case']['op']=='quant':continue
   c=Case(r['case']['op'],r['case']['shape'])
   self.assertEqual(p.predict(c)['predicted_us'],old.predict(c)['predicted_us'])
   self.assertEqual(p.predict(c)['estimated_max_core_selected_opcode_count'],old.predict(c)['estimated_max_core_selected_opcode_count'])
 def test_large_shapes_are_explicit_extrapolations(self):
  p=Predictor(self.model)
  for c in [Case('quant',(2304,7168)),Case('fa',(9,525312,64,1,1)),Case('fa',(9,4096,64,16,1))]:
   self.assertEqual(p.predict(c)['coverage'],'extrapolation')
  for shape in [(9,0,64,1,1),(9,1048577,64,1,1),(9,512,512,1,1),(9,512,64,257,1)]:
   with self.assertRaises(ValueError):Case('fa',shape)
 def test_immutable_bounded_cache_and_uncached_bypass(self):
  p=Predictor(self.model,2);c=Case('quant',(9,5120));a=p.predict(c)
  with self.assertRaises(TypeError):a['plan']['ub_tiles'][0]['rows']=0
  self.assertIs(a,p.predict(c));self.assertIsNot(a,p.predict(c,False))
  for rows in [1,4]:p.predict(Case('quant',(rows,5120)))
  self.assertNotIn(c,p.cache);self.assertEqual(len(p.cache),2)
