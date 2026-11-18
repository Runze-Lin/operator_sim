import unittest
from operator_model.predictor import predict, _legacy, Case

class TestQueryV8(unittest.TestCase):
    def test_long_quant_keeps_double_buffer_schedule(self):
        for shape in [(1048576,65536),(510001,65472),(700003,51072),(1048573,7104)]:
            expected=_legacy().predict_us(Case('quant',shape),use_cache=False)
            self.assertAlmostEqual(predict('quant',shape)['predicted_us']/expected,1.,places=9)

    def test_chunk_qualification_and_work_are_distinct(self):
        result = predict('kda_chunk', (1, 64, 1, 2, 128, 128))
        self.assertEqual(result['status'], 'unqualified_ca_timing')
        self.assertIsNone(result['predicted_us'])
        self.assertTrue(result['structural_plan']['work'])

    def test_system_entry_rejects_incompatible_contract(self):
        self.assertIsNone(predict('addmm', (9, 128, 256), {'alpha': 2})['predicted_us'])
        self.assertIsNone(predict('moe_ffn', (8, 128, 256, 2), {'expert_counts': [2, 3]})['predicted_us'])

    def test_chain_counts_only_its_two_stages(self):
        attrs = {'expert_counts': [1, 7]}
        shape = (8, 128, 256, 2)
        full = predict('moe_ffn', shape, attrs)
        parts = [predict(op, shape, attrs) for op in ('grouped_matmul_swiglu_quant_v2', 'gmm2')]
        self.assertAlmostEqual(full['predicted_us'], sum(r['predicted_us'] for r in parts))
