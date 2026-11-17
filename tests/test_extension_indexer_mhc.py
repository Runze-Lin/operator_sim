"""Semantic and contract tests; not CA accuracy tests."""
import json
import unittest
from operator_model.extensions.indexer_mhc import plan


class IndexerMhcTests(unittest.TestCase):
    def test_causal_pairs_match_enumeration_including_q_longer_than_k(self):
        for q, length in ((1,64), (9,32), (32,32), (40,32)):
            expected = sum(k <= length-q+i for i in range(q) for k in range(length))
            result = plan('lightning_indexer', (4,q,length,8,128), sparse_count=16)
            self.assertEqual(result['work']['logical_valid_query_key_pairs'], 4*expected)
            self.assertEqual(result['work']['logical_dot_flops'], 2*4*expected*8*128)

    def test_topk_does_not_reduce_score_work(self):
        shape = (1,9,32768,32,128)
        low = plan('lightning_indexer',shape,sparse_count=512)
        high = plan('lightning_indexer',shape,sparse_count=1024)
        self.assertEqual(low['work']['logical_dot_flops'],high['work']['logical_dot_flops'])
        self.assertEqual(high['work']['output_slots'],2*low['work']['output_slots'])
        self.assertIsNone(low['work']['topk_comparisons'])

    def test_contracts_do_not_conflate_platform_or_api(self):
        cases = [
            ('lightning_indexer',(1,1,64,8,128),{'weights_dtype':'fp32'}),
            ('lightning_indexer',(1,1,64,4,128),{}),
            ('lightning_indexer',(1,1,64,8,128),{'layout':'PA_BSND'}),
            ('lightning_indexer',(1,1,64,8,128),{'sparse_count':2049}),
            ('mhc_pre_sinkhorn',(1,4,128),{}),
            ('mhc_pre_sinkhorn',(1,6,4096),{}),
            ('mhc_pre_sinkhorn',(1,4,4096),{'num_iters':19}),
            ('mhc_pre_sinkhorn',(1,4,4096),{'need_backward':True}),
            ('mhc_pre',(1,4,16385),{}),
            ('mhc_sinkhorn',(1,5),{}),
        ]
        for op,shape,attrs in cases:
            with self.subTest(op=op,shape=shape,attrs=attrs):
                self.assertEqual(plan(op,shape,**attrs)['status'],'unsupported')

    def test_sinkhorn_pass_count_and_padded_optional_outputs(self):
        for n in (4,6,8):
            result=plan('mhc_sinkhorn',(3,n),num_iters=1,return_intermediates=True)
            self.assertEqual(result['work']['normalization_passes_after_softmax'],1)
            self.assertEqual(result['work']['logical_divisions'],2*3*n*n)
            self.assertEqual(result['output_shapes']['norm_out'],[2*3*n*8])
            self.assertEqual(result['output_shapes']['sum_out'],[2*3*8])
            plain=plan('mhc_sinkhorn',(3,n),num_iters=1)
            self.assertEqual(result['work']['logical_tensor_bytes']-plain['work']['logical_tensor_bytes'],4*(2*3*n*8+2*3*8))

    def test_split_and_fused_never_double_count_mhc(self):
        fused=plan('mhc',(9,4,4096))
        split=plan('mhc',(9,4,4096),implementation='split')
        self.assertEqual([x['op'] for x in fused['components']],['mhc_pre_sinkhorn','mhc_post'])
        self.assertEqual([x['op'] for x in split['components']],['mhc_pre','mhc_sinkhorn','mhc_post'])
        self.assertEqual((fused['aclnn_calls'],split['aclnn_calls']),(2,3))
        self.assertFalse(fused['external_block_included'])
        self.assertEqual(plan('mhc',(9,4,128))['status'],'unsupported')
        self.assertEqual(plan('mhc',(9,4,128),implementation='split')['status'],'uncalibrated')

    def test_pre_shape_and_alpha_two_optional_path(self):
        full=plan('mhc_pre',(9,4,4096),return_intermediates=True)
        self.assertEqual(full['input_shapes']['phi'],[24,16384])
        self.assertEqual(full['output_shapes']['h_mix'],[9,24])
        self.assertEqual(full['work']['projection_flops'],2*9*16384*24)
        short=plan('mhc_pre',(9,4,4096),alpha_count=2)
        self.assertEqual(short['input_shapes']['phi'],[8,16384])
        self.assertIsNone(short['output_shapes']['h_res'])

    def test_post_identity_residual_and_non_a2_constraints(self):
        full=plan('mhc_post',(3,6,123))
        identity=plan('mhc_post',(3,6,123),has_h_res=False)
        self.assertEqual(full['status'],'uncalibrated')
        self.assertEqual(full['packaged_core_type'],'VectorCore')
        self.assertEqual(full['work']['residual_multiplications'],3*6*6*123)
        self.assertEqual(identity['work']['residual_multiplications'],0)
        self.assertEqual(full['work']['logical_tensor_bytes']-identity['work']['logical_tensor_bytes'],4*3*6*6)

    def test_all_plans_are_json_safe_and_never_claim_calibration(self):
        cases=[('lightening_indexer',(1,1,64,8,128)),('mhc_pre',(1,4,128)),
               ('mhc_sinkhorn',(1,4)),('mhc_pre_sinkhorn',(1,4,4096)),
               ('mhc_post',(1,4,4096)),('mhc',(1,4,4096))]
        for op,shape in cases:
            result=plan(op,shape)
            self.assertEqual(result['status'],'uncalibrated')
            self.assertIsNone(result['predicted_us'])
            self.assertEqual(result['ca_measurements'],0)
            self.assertFalse(result['kernel_selection_verified'])
            json.dumps(result,allow_nan=False)
        self.assertEqual(plan('mhc_pre',{'T':1,'N':4,'D':128})['shape'],[1,4,128])

    def test_invalid_scalar_types_and_unrecognized_attributes_fail_loudly(self):
        bad=[('mhc_pre',(True,4,128),{}),('mhc_pre',(1,4,128),{'has_gamma':'false'}),
             ('mhc_sinkhorn',(1,4),{'eps':float('nan')}),
             ('mhc',(1,4,4096),{'num_iters':True}),
             ('lightning_indexer',(1,1,64,8,128),{'tp':8})]
        for op,shape,attrs in bad:
            with self.assertRaises(ValueError):
                plan(op,shape,**attrs)


if __name__=='__main__':
    unittest.main()
