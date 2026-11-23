import unittest

from operator_model import OperatorStream, estimate
from operator_model.workload import account_work


class TokenQuantTest(unittest.TestCase):
    def test_per_token_contract_and_schedule(self):
        for chip in ('Ascend950DT_9582','Ascend950PR_957b','Ascend970DT'):
            result=OperatorStream(chip).advance('quant_token_fp8',(9,1024))
            self.assertEqual(result.status,'extrapolated' if chip=='Ascend970DT' else 'estimated')
            self.assertEqual((result.active_aiv,result.active_aic,result.kernel_count),(9,0,1))
            self.assertEqual(result.work['logical_write_bytes'],9*1024+4*9)
            self.assertEqual(result.work,account_work('quant_token_fp8',(9,1024)))
            self.assertAlmostEqual(result.latency_us,sum(result.projection['stages_us'].values()))
        mixed=estimate('quant_token_fp8',(9,1024),dtype='fp16')
        self.assertEqual(mixed.status,'unsupported')
        self.assertEqual(estimate('quant_token_fp8',(9,1024),{'input_stride':[4096,1]}).status,'unavailable')
        self.assertEqual(estimate('quant_token_fp8',(1,131072)).status,'unavailable')

    def test_measured_only_cache_has_no_model_error_or_predicted_price(self):
        attrs={'implementation':'aclnnAscendAntiQuant'}
        r=estimate('cache_load',(1,128,512),attrs,dtype='fp8_e4m3fn')
        self.assertEqual(r.status,'unavailable')
        self.assertIsNone(r.latency_us)
        self.assertEqual(r.work,account_work('cache_load',(1,128,512),{**attrs,'dtype':'fp8_e4m3fn'}))
        self.assertIsNone(estimate('cache_load',(1,128,512),attrs,dtype='mxfp4').latency_us)
