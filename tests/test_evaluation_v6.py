import copy
import unittest
from operator_model.evaluation_v6 import evaluate_record


class AdmissionTests(unittest.TestCase):
    def setUp(self):
        p={'predicted_us':10,'constant_baseline_us':5,'plan':{'active_aic':1,'active_aiv':2},'estimated_max_core_selected_opcode_count':2048,'phase_schedule':{}}
        self.case={'id':'new','op':'fa','shape':[9,131072,64,1,1],'split':'holdout','timeout_seconds':960,'prediction':p,'v05_prediction':p}
        self.frozen={'model_sha256':'model','frozen_utc':'2026-11-17T00:00:00+00:00','artifact_sha256':{'build/smoke/three_ops':'host','build/capture/libsim_soc_adapter.so':'adapter'}}
        self.record={'case':{k:v for k,v in self.case.items() if k not in ('prediction','v05_prediction')},
            'identity':{'model_sha256':'model','frozen_predictions_sha256':'freeze','host_binary_sha256':'host','sku_adapter_sha256':'adapter','soc_version':'Ascend950DT_9582','cann_version':'9.2.1'},
            'started_utc':'2026-11-17T01:00:00+00:00','status':'ok','returncode':0,'timed_out':False,
            'wall_seconds':20,'cpu_seconds':100,'raw_bytes':1000,'extraction_seconds':1,
            'measurement':{'reference_us':8,'reference_scope':'cross-core instruction makespan','correctness':['correctness elements=576 bad=0'],
                'active_aic':1,'active_aiv':2,'max_core_rv_count':2048,'kernel':'entry','db_sha256':'db'}}
    def test_partial_success_never_becomes_measurement(self):
        self.assertIsNotNone(evaluate_record(self.case,self.record,self.frozen,'freeze'))
        for changes in ({'timed_out':True},{'status':'failed'},{'returncode':124}):
            self.assertIsNone(evaluate_record(self.case,{**self.record,**changes},self.frozen,'freeze'))
    def test_reject_shape_environment_and_freeze_substitution(self):
        for field,value in [('model_sha256','other'),('soc_version','Ascend950PR_9589'),('frozen_predictions_sha256','other'),('host_binary_sha256','other')]:
            r=copy.deepcopy(self.record);r['identity'][field]=value
            with self.assertRaises(ValueError):evaluate_record(self.case,r,self.frozen,'freeze')
        r=copy.deepcopy(self.record);r['case']['shape'][1]=32768
        with self.assertRaises(ValueError):evaluate_record(self.case,r,self.frozen,'freeze')
        r=copy.deepcopy(self.record);r['started_utc']='2026-11-16T00:00:00+00:00'
        with self.assertRaises(ValueError):evaluate_record(self.case,r,self.frozen,'freeze')
