import copy
from pathlib import Path
import runpy
import unittest

utilities = runpy.run_path(str(Path(__file__).resolve().parents[1] / 'scripts/evaluate-v2.py'))
validate, replay_equal = utilities['validate_dataset'], utilities['replay_equal']


class EvaluationIntegrityTests(unittest.TestCase):
    def test_replay_tolerance_is_only_roundoff(self):
        self.assertTrue(replay_equal({'x':[9.080718520695882]}, {'x':[9.08071852069588]}))
        self.assertFalse(replay_equal({'x':[9.08]}, {'x':[9.081]}))
        self.assertFalse(replay_equal({'cores':3}, {'cores':4}))
        self.assertFalse(replay_equal({'x':float('nan')}, {'x':float('nan')}))

    def setUp(self):
        identity = dict.fromkeys(('soc_version','cann_version','package_manifest_sha256',
                                  'platform_sha256','sim_config_sha256','host_binary_sha256'), 'fixed')
        self.data = {'manifest_sha256':'manifest', 'records':[
            {'case':{'id':'train','op':'quant','shape':[4,256],'split':'calibration'},
             'cache_key':'training-key','status':'ok','identity':identity},
            {'case':{'id':'test','op':'quant','shape':[8,256],'split':'holdout'},
             'cache_key':'test-key','status':'ok','identity':copy.deepcopy(identity)}]}
        self.frozen = {'manifest_sha256':'manifest',
                       'predictions':[copy.deepcopy(r['case']) for r in self.data['records']]}
        self.model = {'training_keys':['training-key']}

    def test_valid_and_missing_holdout(self):
        validate(self.data,self.frozen,self.model)
        self.data['records'].pop()  # Missing test is reported separately, never used for fitting.
        validate(self.data,self.frozen,self.model)

    def test_reject_environment_mix(self):
        self.data['records'][1]['identity']['sim_config_sha256']='changed'
        with self.assertRaisesRegex(ValueError,'environments'):
            validate(self.data,self.frozen,self.model)

    def test_reject_replaced_training(self):
        self.data['records'][0]['cache_key']='replacement'
        with self.assertRaisesRegex(ValueError,'Training'):
            validate(self.data,self.frozen,self.model)

    def test_reject_same_id_different_shape(self):
        self.data['records'][1]['case']['shape']=[16,256]
        with self.assertRaisesRegex(ValueError,'frozen input'):
            validate(self.data,self.frozen,self.model)
