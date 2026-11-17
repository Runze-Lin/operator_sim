import runpy
from pathlib import Path
import unittest
ROOT=Path(__file__).resolve().parents[1]

class WorkloadTests(unittest.TestCase):
 @classmethod
 def setUpClass(cls):cls.data=runpy.run_path(str(ROOT/'scripts/workload-v4.py'))['build']()
 def test_user_batch_and_context_sets_are_preserved(self):
  rows=self.data['scenarios']
  self.assertEqual({r['prompt_tokens'] for r in rows},{32768,131072,262144,524288})
  self.assertEqual({r['batch_per_card'] for r in rows if r['phase']=='prefill'},{1,4,16})
  self.assertEqual({r['batch_per_card'] for r in rows if r['phase']=='decode'},{1,4,16,32,128,256})
 def test_tokenwise_quant_deduplicates_context_without_dividing_tokens_by_tp(self):
  rows=[r for r in self.data['scenarios'] if r['model']=='DeepSeek-V4.1-Flash' and r['phase']=='decode' and r['batch_per_card']==256 and r['q_per_request']==9]
  self.assertEqual(len(rows),8)
  self.assertEqual({r['quant_key'] for r in rows},{'quant:2304x5120'})
  self.assertTrue(all(r['output_tokens']==1024 for r in rows))
 def test_dense_fa_is_explicitly_unvalidated_proxy(self):
  rows=[r for r in self.data['scenarios'] if r['model']=='DeepSeek-V4-Pro' and r['phase']=='decode' and r['batch_per_card']==32 and r['q_per_request']==9 and r['prompt_tokens']==524288 and r['generated_progress']==1024]
  self.assertEqual(len(rows),1)
  pred=self.data['unique_predictions'][rows[0]['dense_fa_proxy_key']]
  self.assertEqual(pred['shape'],[9,525312,128,32,16])
  self.assertEqual(pred['status'],'analytical_only_not_ca_validated')
