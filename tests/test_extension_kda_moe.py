import json
import unittest

from operator_model.extensions.kda_moe import plan


class TestKdaMoe(unittest.TestCase):
    def test_recurrent_speculation_boundary(self):
        for s in (1, 8):
            p = plan('kda_recurrent', (4, s, 2, 4, 128, 256))
            self.assertEqual(p['status'], 'uncalibrated')
            self.assertEqual(p['tiling']['logical_tasks'], 16)
            self.assertEqual(p['tiling']['serial_tokens_per_task'], s)
            self.assertIsNone(p['predicted_us'])
        p = plan('kda_recurrent', (4, 9, 2, 4, 128, 128))
        self.assertEqual(p['status'], 'unsupported')
        self.assertIn('Q=9', p['reason'])

    def test_explicit_route_and_contract(self):
        self.assertEqual(plan('kda', (1, 8, 1, 1, 128, 128))['status'], 'unsupported')
        self.assertEqual(plan('kda_chunk', (1, 64, 3, 4, 128, 128))['status'], 'unsupported')
        self.assertEqual(plan('kda_recurrent', (1, 1, 1, 1, 128, 128), dtype='fp16')['status'], 'unsupported')

    def test_chunk_tail_and_state_dependency(self):
        p = plan('kda_chunk', dict(B=1,S=65,Hq=1,Hv=2,K=128,V=128))
        self.assertEqual(p['tiling']['chunks_per_sequence'], 2)
        self.assertEqual(p['tiling']['tail_tokens'], 1)
        self.assertEqual(p['tiling']['state_chains'], 2)
        self.assertEqual(p['work']['aqk_useful_dot_flops'], 2*256*(64*65//2+1))
        self.assertIsNone(p['work']['total_executed_flops'])
        self.assertIsNone(p['predicted_us'])
        json.dumps(p)

    def test_histogram_counts_cumulative_and_empty_experts(self):
        a = plan('moe_ffn', (10,128,128,4), expert_counts=[1,0,7,0])
        b = plan('moe_ffn', (10,128,128,4), group_list=[1,1,8,8], group_list_type=0)
        self.assertEqual(a['work'], b['work'])
        self.assertEqual(a['work']['active_experts'], 2)
        self.assertEqual(a['work']['unused_capacity_rows'], 2)
        self.assertEqual(a['work']['max_over_mean_expert_load'], 3.5)
        self.assertEqual(a['work']['gmm1_useful_flops'], 4*8*128*128)
        self.assertEqual(a['work']['gmm2_useful_flops'], 2*8*128*128)
        self.assertIsNone(a['predicted_us'])
        json.dumps(a)

    def test_no_uniform_expert_assumption(self):
        self.assertEqual(plan('moe_ffn', (8,128,128,4))['status'], 'unsupported')
        self.assertEqual(plan('moe_ffn', (8,128,128,4), expert_counts=[1,0,9,0])['status'], 'unsupported')
        self.assertEqual(plan('moe_ffn', (8,128,128,4), group_list=[1,0,7,8], group_list_type=0)['status'], 'unsupported')
        self.assertEqual(plan('moe_ffn', (8,128,128,4), expert_counts=[2]*4, layout='NZ')['status'], 'unsupported')

    def test_all_zero_experts_do_not_imply_zero_latency(self):
        p = plan('grouped_matmul_swiglu_quant_v2', (8,128,129,4), expert_counts=[0]*4)
        self.assertEqual(p['work']['gmm1_useful_flops'], 0)
        self.assertIsNone(p['predicted_us'])
        self.assertEqual(p['tensor_shapes']['output_scale'], [8,3,2])


if __name__ == '__main__':
    unittest.main()
