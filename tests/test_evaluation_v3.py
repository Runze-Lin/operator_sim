import copy
from pathlib import Path
import runpy
import unittest

validate=runpy.run_path(str(Path(__file__).resolve().parents[1]/'scripts/evaluate-v3.py'))['validate']

class FrozenValidationTests(unittest.TestCase):
    def setUp(self):
        identity=dict.fromkeys(('soc_version','cann_version','package_manifest_sha256','platform_sha256','sim_config_sha256','host_binary_sha256'),'fixed')
        train={'case':{'id':'train','op':'quant','shape':[1,64],'split':'calibration'},'cache_key':'train','identity':identity,'imported_from':{'version':'0.2'}}
        test={'case':{'id':'test','op':'quant','shape':[512,256],'split':'holdout'},'cache_key':'test','identity':copy.deepcopy(identity),
              'finished_utc':'2026-11-17T01:02:00+00:00','wall_seconds':60}
        self.data={'manifest_sha256':'manifest','records':[train,test]}
        self.frozen={'manifest_sha256':'manifest','predictions':[copy.deepcopy(r['case']) for r in (train,test)],
                     'frozen_utc':'2026-11-17T01:00:00+00:00','artifact_sha256':{'build/smoke/three_ops':'fixed'}}
        self.model={'training_keys':['train']}

    def test_accept_original_and_report_missing_separately(self):
        validate(self.data,self.frozen,self.model)
        self.data['records'].pop()
        validate(self.data,self.frozen,self.model)

    def test_reject_started_before_freeze_even_if_finished_later(self):
        self.data['records'][1]['wall_seconds']=180
        with self.assertRaisesRegex(ValueError,'predates'):validate(self.data,self.frozen,self.model)

    def test_reject_new_caller_binary_or_input(self):
        self.data['records'][1]['identity']['host_binary_sha256']='changed'
        with self.assertRaisesRegex(ValueError,'binary'):validate(self.data,self.frozen,self.model)
        self.data['records'][1]['identity']['host_binary_sha256']='fixed'
        self.data['records'][1]['case']['shape']=[1024,256]
        with self.assertRaisesRegex(ValueError,'Input'):validate(self.data,self.frozen,self.model)
