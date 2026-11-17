import copy
import json
from pathlib import Path
import unittest
from operator_model.model_v3 import Case,Predictor,fit,quant_plan,quant_schedule

ROOT=Path(__file__).resolve().parents[1]

class IterationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.records=json.loads((ROOT/'results/v0.3/imported-calibration.json').read_text())['records']
        cls.analysis=json.loads((ROOT/'results/v0.3/trace-analysis.json').read_text())
        cls.base=json.loads((ROOT/'results/v0.2/model.json').read_text())
        cls.model=fit(cls.records,cls.analysis,cls.base)

    def test_double_buffer_blocks_third_prefetch_until_slot_release(self):
        const=lambda x:{'intercept_us':x,'slope_us_per_unit':0}
        costs={'prefix':const(0),'first_read':const(10),'read_interval':const(2),
               'vector':const(1),'control_floor':const(0),'write_drain_us':1,'suffix_us':0}
        tile={'read_kib':1,'main_loops':1,'scale_loops':1}
        result=quant_schedule({'active_aiv':1,'ub_tiles':[tile]*3},costs)
        self.assertEqual(result['read_ready_us'],[10,12,21])
        self.assertEqual(result['vector_finish_us'],[11,13,22])
        self.assertEqual(result['predicted_us'],23)

    def test_tail_work_is_not_rounded_to_full_ub_iteration(self):
        p=quant_plan(Case('quant',(832,4096)))
        self.assertEqual([t['rows'] for t in p['ub_tiles']],[7,6])
        self.assertEqual(sum(t['rows']*t['cols'] for t in p['ub_tiles']),p['rows_per_core']*p['cols_per_core'])
        self.assertLess(p['ub_tiles'][-1]['main_loops'],p['ub_tiles'][0]['main_loops'])

    def test_fa_counts_match_every_existing_ca_case_and_timing_is_retained(self):
        p=Predictor(self.model)
        for r in self.records:
            if r['case']['op']!='fa':continue
            c=Case('fa',r['case']['shape']);pred=p.predict(c)
            observed=max(x['dynamic_opcodes'].get('MMAD',0) for x in r['measurement']['cores'])
            self.assertEqual(pred['estimated_max_core_selected_opcode_count'],observed,r['case']['id'])
            self.assertEqual(pred['predicted_us'],p.base.predict(c)['predicted_us'])
        self.assertIsNone(p.predict(Case('fa',(128,128,64,1,32)))['estimated_max_core_selected_opcode_count'])

    def test_cache_is_immutable_and_bounded(self):
        p=Predictor(self.model,cache_size=2)
        c=Case('quant',(64,1024));result=p.predict(c)
        self.assertEqual(result,p.predict(c,use_cache=False))
        with self.assertRaises(TypeError):result['phase_schedule']['read_ready_us'][0]=0
        for rows in (65,66):p.predict(Case('quant',(rows,256)))
        self.assertEqual(len(p.cache),2)
        self.assertNotIn(c,p.cache)

    def test_reject_holdout_or_stale_phase_evidence(self):
        bad=copy.deepcopy(self.records);bad[0]['case']['split']='holdout'
        with self.assertRaisesRegex(ValueError,'leakage'):fit(bad,self.analysis,self.base)
        analysis=copy.deepcopy(self.analysis);analysis['quant'][0]['trace_sha256']='bad'
        with self.assertRaisesRegex(ValueError,'Stale'):fit(self.records,analysis,self.base)
