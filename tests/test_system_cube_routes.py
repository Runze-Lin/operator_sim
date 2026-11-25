"""Tensor roles and packed scale axes determine the execution precision."""
from dataclasses import replace
import unittest

from operator_model import load_chip
from operator_model.system_routes import estimate_full_features
from test_system_adapter import BF16, F32, FP4, FP8, E8, Features, attributes, tensor


class SystemCubeTests(unittest.TestCase):
    chip = load_chip('Ascend950DT_9582')

    def test_fp32_split_k_and_transposed_storage(self):
        f = Features('aten.mm.default',(tensor(F32,(9,5120)),tensor(F32,(5120,384),(1,5120))),
                     (tensor(F32,(9,384)),))
        p = estimate_full_features(f,chip=self.chip)
        self.assertEqual(p.work['schedule']['split_k'],16)
        self.assertEqual(p.work['schedule']['active_aic'],32)
        self.assertIsNone(p.work['schedule']['active_aiv'])
        self.assertEqual(p.work['vec_ops_by_class']['split_k_combine'],15*9*384)
        self.assertEqual(p.work['cube_ops_by_dtype'],{'fp32': 2*9*384*5120})
        self.assertIsNone(estimate_full_features(replace(f,inputs=(f.inputs[0],tensor(F32,(5120,384)))),
                                                chip=self.chip).latency_us)

    def test_prefill_last_token_matmul_packs_only_selected_rows(self):
        m,n,k=16,16160,7168
        def price(pitch):
            f=Features('aten.mm.default',(tensor(BF16,(m,k),(pitch,1)),tensor(BF16,(k,n),(1,k))),
                       (tensor(BF16,(m,n)),))
            return estimate_full_features(f,chip=self.chip)
        packed, dense = price(4096*k), price(k)
        self.assertEqual(packed.contract,dense.contract+'/pack_rows')
        self.assertEqual(packed.work['logical_read_bytes']-dense.work['logical_read_bytes'],2*m*k)
        self.assertAlmostEqual(packed.latency_us-dense.latency_us,packed.work['schedule']['pack_us'])
        self.assertIsNone(price(k-1).latency_us)

    def test_router_fp32_retains_native_split_k(self):
        m,n,k = 9,896,7168
        f = Features('inference::matmul_rhs_transposed',(tensor(BF16,(m,k)),tensor(BF16,(n,k))),
                     (tensor(F32,(m,n)),),attributes(dict(output_dtype=F32,accumulator_dtype=F32)))
        dt=estimate_full_features(f,chip=self.chip)
        self.assertEqual(dt.work['schedule']['tile_mnk'],(16,224,64))
        self.assertEqual(dt.work['schedule']['split_k'],8)
        self.assertIsNone(dt.work['schedule']['active_aiv'])
        self.assertEqual(dt.work['schedule']['partial_workspace_bytes'],4*8*m*n)
        self.assertLess(abs(dt.latency_us-10.280),0.001)
        self.assertEqual(dt.work['cube_ops_by_dtype'],{'bf16':2*m*n*k})

    def test_kimis_column_group_scale_needs_unpack_before_mmad(self):
        g,m,k,n = 12,9,128,512
        x = tensor(FP4,(g,m,k//2))
        w = tensor(FP4,(g,k,n//2))
        scales = tensor(E8,(g,m,k//64,2)),tensor(E8,(g,k,n//64,2))
        f = Features('inference::quantized_batched_matmul',(x,w,*scales),(tensor(BF16,(g,m,n)),),
                     attributes(dict(compute_precision='fp4',logical_input_shapes=((g,m,k),(g,k,n)),
                                     group_size=32,input_layout='values_then_corresponding_scales',
                                     accumulator_dtype='float32')))
        p = estimate_full_features(f,chip=self.chip)
        self.assertEqual(p.work['cube_ops_by_dtype'],{'bf16': 2*g*m*k*n})
        self.assertEqual(p.status,'extrapolated')
        self.assertIsNone(estimate_full_features(replace(f,inputs=(x,w,scales[0],tensor(E8,(g,n,k//64,2)))),
                                                chip=self.chip).latency_us)

    def test_w4_grouped_head_token_scale_and_group_count(self):
        g,m,k,n = 2,18,4096,1024
        pairs = k//64
        x = tensor(FP8,(m,k))
        w = tensor(FP4,(g,k//2,n),(k//2*n,1,k//2))
        sw = tensor(E8,(g,pairs,n,2),(2*pairs*n,2,2*pairs,1))
        sem = dict(logical_input_shape=(m,k),logical_weight_shape=(g,n,k),weight_precision='fp4',
                   activation_precision='fp8',group_counts=(9,9),gemm_grouping='output_head_groups',
                   group_size=32)
        slots = tuple((('index',i),('role',r)) for i,r in enumerate(('activation','weight','weight_scale','activation_scale')))
        f = Features('npu::npu_grouped_matmul',(x,w,sw,tensor(F32,(m,))),(tensor(BF16,(m,n)),),
                     attributes(sem,tensor_semantics=(('inputs',slots),),input_logical_shapes=(None,(g,k,n),None,None)))
        p = estimate_full_features(f,chip=self.chip)
        self.assertEqual(p.work['vec_ops_by_class']['unpack_weight'],g*k*n)
        self.assertEqual(p.work['cube_ops_by_dtype'],{'fp8_e4m3fn': 2*m*n*k})
        self.assertEqual(p.work['schedule']['groups'],g)
        peaks = dict(self.chip.vector_tops)
        peaks.update({'8': 24, '16': 1, '32': 12})
        target = replace(self.chip, vector_tops=peaks)
        proposal = estimate_full_features(f, chip=target)
        explicit = estimate_full_features(f, chip=target, vector_model='instruction_classes')
        self.assertEqual(proposal.latency_us, explicit.latency_us)
        self.assertTrue(all(stage['method']=='instruction_classes' for stage in proposal.work['schedule']['vector_stages']))
        self.assertIn('permute',proposal.work['schedule']['vector_stages'][0]['counts'])
        schedule = estimate_full_features(f, chip=target, vector_model='calibrated').work['schedule']
        active = schedule['active_aiv']
        per_cycle = 1000 * active / (target.frequency_ghz * target.vector_cores)
        expected = {'fp8_e4m3fn': g*k*n / (24*per_cycle),
                    'fp32': 2*m*n / (12*per_cycle)}
        for dtype, cycles in expected.items():
            self.assertAlmostEqual(schedule['vector_cycles_by_dtype'][dtype], cycles)
        self.assertAlmostEqual(schedule['resource_cycles']['vector'], sum(expected.values()))
        self.assertIsNone(estimate_full_features(replace(f,attributes=attributes(dict(sem,group_counts=(9,8)))),
                                                chip=self.chip).latency_us)
