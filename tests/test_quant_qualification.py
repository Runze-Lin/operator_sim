import importlib.util
import json
from pathlib import Path
import sqlite3
import tempfile
import unittest
from operator_model.ca import sha256
ROOT=Path(__file__).resolve().parents[1]
spec=importlib.util.spec_from_file_location('qualify',ROOT/'scripts/qualify-quant-v6.py');module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)

class QuantQualificationTests(unittest.TestCase):
    def fixture(self,folder,correctness):
        dbpath=folder/'test.db'
        with sqlite3.connect(dbpath) as db:
            db.execute('CREATE TABLE ExecutedInstructions(ExecInstrCoreId INT,CoreTypeId INT,ExecInstrName TEXT)')
            db.executemany('INSERT INTO ExecutedInstructions VALUES(0,2,?)',[('RV_VADD',),('RV_SMOV',),('WAIT_FLAG',)])
        log=folder/'run.log';log.write_text('SMOKE_PASS\nSimulation SUCCESS\n'+correctness+'\n')
        path=folder/'result.json';path.write_text(json.dumps({'case':{'op':'quant','shape':[1,64]},'status':'failed',
            'error':'Missing complete numerical validation','returncode':0,'timed_out':False,'log_sha256':sha256(log),
            'measurement':{'db_path':str(dbpath),'db_sha256':sha256(dbpath),'correctness':[]}}))
        return path

    def test_offline_recovery_preserves_original_and_checks_all_elements(self):
        with tempfile.TemporaryDirectory() as d:
            path=self.fixture(Path(d),'correctness quant_elements=64 scale_elements=2 bad=0 comparison=exact_bytes')
            before=path.read_bytes();r=module.qualify(path)
            self.assertEqual(r['status'],'ok');self.assertEqual(r['measurement']['max_core_rv_count'],2)
            self.assertEqual(path.read_bytes(),before);self.assertEqual(r['recovery']['source_result_sha256'],sha256(path))

    def test_partial_scale_or_nonzero_error_is_rejected(self):
        for text in ['correctness quant_elements=64 scale_elements=1 bad=0 comparison=exact_bytes',
                     'correctness quant_elements=64 scale_elements=2 bad=1 comparison=exact_bytes']:
            with tempfile.TemporaryDirectory() as d:
                path=self.fixture(Path(d),text)
                with self.assertRaises(ValueError):module.qualify(path)
