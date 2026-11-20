"""Public-contract checks against the selected -I wheel, from outside the checkout."""
from pathlib import Path
import json,sys,unittest
HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[2];prefix=Path(sys.argv.pop(1)).resolve();sys.path.insert(0,str(prefix))
import operator_model as api
class InstalledAdmission(unittest.TestCase):
 def test_sink_frozen_unknown_and_observed(self):
  src=json.loads((ROOT/'results/v0.32/large/freeze.json').read_bytes())
  for chip,rs in src['predictions'].items():
   for r in rs:
    attrs={'num_iters':r['iterations']};x=api.estimate('mhc_sinkhorn',r['shape'],attrs,chip=chip)
    self.assertAlmostEqual(x.latency_us,r['unknown_history_mean_us'],7)
    y=api.estimate('mhc_sinkhorn',r['shape'],{**attrs,'resident_aiv':r['resident_aiv']},chip=chip);self.assertAlmostEqual(y.latency_us,r['explicit_history']['us'],7)
 def test_public_multichain_mode_and_frozen_prices(self):
  f=json.loads((ROOT/'results/v0.32/large/recurrent-freeze.json').read_bytes())
  for chip,v in f['prices'].items():
   p=api.estimate('kda_recurrent',f['job']['shape'],chip=chip)
   self.assertEqual(p.status,'extrapolated');self.assertEqual(p.work,v['work']);self.assertAlmostEqual(p.latency_us,v['latency_us'],7)
   self.assertIsNone(api.estimate('kda_recurrent',f['job']['shape'],chip=chip,timing_mode='calibrated').latency_us)
 def test_matmul_and_post_independent_freezes(self):
  m=json.loads((ROOT/'results/v0.32/matmul/kimi-head/freeze.json').read_bytes())
  for chip,p in m['predictions'].items():self.assertAlmostEqual(api.estimate('matmul',m['shape'],chip=chip).latency_us,p['us'],7)
  reg=json.loads((ROOT/'results/v0.32/post/freeze.json').read_bytes());groups={}
  for p in reg['predictions']:
   seq=groups.setdefault((p['chip'],p['group']),api.OperatorStream(p['chip']))
   self.assertAlmostEqual(seq.advance('mhc_post',p['shape']).latency_us,p['ordered_us'],7)
 def test_distribution_location(self):
  self.assertEqual(api.__version__,'0.32.0');self.assertTrue(Path(api.__file__).is_relative_to(prefix))
if __name__=='__main__':unittest.main()
