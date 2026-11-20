"""Run the original v0.29 skip path and report in temporary directories, with CA disabled."""
from hashlib import sha256
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch
import importlib
import json
import sys
import unittest

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[2]
sys.path.insert(0,str(ROOT/'development/validation/large29/vector'))
sys.path.append(str(HERE))
from deduplicate import DT,PR,OUT,FREEZE,RMS,read,disposition
import acquire as original_runner
import report as original_report

class ReusedRmsReceipts(unittest.TestCase):
    def setUp(self):
        self.jobs=[j for j in read(FREEZE)['jobs'] if j['op']=='rmsnorm' and j['shape']==[8192,5120]]
        self.assertEqual(len(self.jobs),2)
    def test_only_exact_pair_and_two_input_skus(self):
        for j in self.jobs:
            dest=OUT/(j['case']+'.json');r=read(dest)
            ext=ROOT/r['external_qualified_receipt']
            self.assertEqual(r['external_receipt_sha256'],sha256(ext.read_bytes()).hexdigest())
            self.assertEqual(r['external_trace_sha256'],read(ext)['trace_sha256'])
            self.assertFalse(r['qualified']);self.assertFalse(r['physical_launch'])
            self.assertIsNone(r['instruction_us']);self.assertIsNone(r['wrapper_us'])
            self.assertEqual(r['registered_prediction_us'],j['prices'][j['chip']]['public_us'])
            wrong=RMS/('rms-8192-5120-'+('pr' if j['chip']==DT else 'dt')+'.json')
            with self.assertRaises(ValueError):
                disposition(j,wrong)
    def test_original_worker_skips_both_before_capture_and_report_keeps_na(self):
        with TemporaryDirectory() as temp:
            out=Path(temp)/'out';out.mkdir();raw=Path(temp)/'raw'
            for j in self.jobs:(out/(j['case']+'.json')).write_bytes((OUT/(j['case']+'.json')).read_bytes())
            test_reg=out/'test-register.json'
            test_reg.write_text(json.dumps(dict(inputs_sha256={},jobs=self.jobs)))
            with patch.multiple(original_runner,OUT=out,RAW=raw,FREEZE=test_reg),patch.object(original_runner,'capture',side_effect=AssertionError('no CA allowed')) as captured,patch.object(original_runner.os,'nice'),patch.object(original_report,'render'):
                original_runner.run()
                captured.assert_not_called()
                self.assertEqual(read(out/'state.json')['phase'],'finished')
                self.assertFalse(any((raw/j['case']).exists() for j in self.jobs))
            with patch.multiple(original_report,OUT=out,FREEZE=FREEZE),patch.object(original_report,'archive',return_value=read(OUT/'archive.json')):
                original_report.render()
            rows=[line for line in (out/'report.md').read_text().splitlines() if '<code>[8192, 5120]</code>' in line and 'instruction-rmsnorm-' in line]
            self.assertEqual(len(rows),2)
            self.assertTrue(all('/ N/A' in line and '[未合格]' in line for line in rows))
    def test_current_scorer_cannot_promote_disposition(self):
        sys.path.append(str(ROOT/'development/validation/release32'))
        qualified = list(importlib.import_module('receipts').other_full_windows())
        blocked={str((OUT/(j['case']+'.json')).relative_to(ROOT)) for j in self.jobs}
        self.assertFalse(any(x[6] in blocked for x in qualified))
if __name__=='__main__':unittest.main()
