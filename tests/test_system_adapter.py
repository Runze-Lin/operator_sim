"""System integration uses byte-slot storage geometry and rank-local capture semantics."""
from dataclasses import dataclass, replace
from math import prod
import unittest

from operator_model import estimate, load_chip
from operator_model.system_adapter import BF16, E8, FP4, FP8, F32, estimate_features


@dataclass(frozen=True)
class Tensor:
    dtype: str
    shape: tuple
    stride: tuple
    device: str = 'meta'
    layout: str = 'torch.strided'


def tensor(dtype, shape, stride=None):
    if stride is None:
        stride = tuple(prod(shape[axis + 1:]) for axis in range(len(shape)))
    return Tensor('torch.' + dtype, shape, stride)


@dataclass(frozen=True)
class Features:
    raw_name: str
    inputs: tuple
    outputs: tuple
    attributes: tuple = ()
    category: str = 'vec'


def attributes(sem, **more):
    return (('semantic', tuple(sem.items())), *more.items())


def projection(grouped):
    m, k, n = 9, 128, 64
    batch = (4,) if grouped else ()
    g = 2
    x = tensor(FP4, (m, k//2))
    w = tensor(FP4, (*batch, k//2, n), (*((4096,) if grouped else ()), 1, 64))
    sw = tensor(E8, (*batch, g, n, 2), (*((256,) if grouped else ()), 2, 4, 1))
    sx = tensor(E8, (m, g, 2))
    sem = dict(activation_scale_layout="mx_e8m0", weight_precision="fp4", activation_precision="fp4",
               group_size=32, padding=(0, 0), accumulator_dtype="float32", logical_input_shape=(m,k),
               logical_weight_shape=(*batch, n, k))
    if grouped:
        sem['group_counts'] = (4, 0, 3, 2)
    roles = ("activation", "weight", "weight_scale", "activation_scale")
    slots = tuple((('index', i), ('role', role)) for i, role in enumerate(roles))
    return Features("npu::npu_grouped_matmul" if grouped else "npu::npu_quant_matmul",
                    (x,w,sw,sx), (tensor(BF16,(m,n)),),
                    attributes(sem, tensor_semantics=(('inputs', slots),),
                               input_logical_shapes=((m,k), (*batch,k,n), None, None)))


class SystemAdapterTests(unittest.TestCase):
    chip = load_chip("Ascend950DT_9582")

    def lookup(self, f):
        return estimate_features(f, chip=self.chip)

    def declines(self, f):
        r = self.lookup(f)
        self.assertIsNone(r.latency_us)
        self.assertTrue(r.reason)

    def matches(self, f, op, shape, dtype="bf16", attrs=None):
        ref = estimate(op, shape, attrs, chip=self.chip, dtype=dtype)
        result = self.lookup(f)
        self.assertIsNotNone(result.latency_us, result.reason)
        self.assertEqual((result.latency_us, result.status, result.work), (ref.latency_us, ref.status, ref.work))

    def test_routed_and_dense_transposed_pack_and_rank_local_histogram(self):
        for grouped in (False, True):
            f = projection(grouped)
            if grouped:
                self.matches(f, 'routed_grouped_matmul', (9,64,128), 'mxfp4',
                             dict(implementation='aclnnGroupedMatmulV5', expert_counts=(4,0,3,2)))
            else:
                self.matches(f, 'matmul', (9,64,128), 'mxfp4')
            self.declines(replace(f, inputs=(f.inputs[0], replace(f.inputs[1], stride=tuple(
                reversed(f.inputs[1].stride))), *f.inputs[2:])))
            self.declines(replace(f, attributes=tuple((k,v) for k,v in f.attributes if k != 'tensor_semantics')))
        f = projection(True)
        sem = dict(dict(f.attributes)['semantic'], group_counts=(1,1,1,1))
        self.declines(replace(f, attributes=tuple((k,tuple(sem.items()) if k == 'semantic' else v)
                                                  for k,v in f.attributes)))
        self.declines(replace(f, inputs=(*f.inputs[:3], tensor(F32,(9,)))))

    def test_quantized_storage_and_strided_slices(self):
        shape = (9,128)
        x = tensor(BF16, shape)
        y = tensor(FP4, (9,64))
        scale = tensor(E8, (9,2,2))
        sem = dict(logical_shape=shape, padding=(0,0), logical_dtype='fp4_e2m1',
                   scale_dtype='float8_e8m0fnu', group_size=32, round_mode='round')
        q = Features("npu::npu_dynamic_mx_quant", (x,), (y,scale),
                     attributes(sem, output_logical_shapes=(shape,None)))
        self.matches(q, "quant_mxfp4", shape)
        self.declines(replace(q, inputs=(replace(x,stride=(6288,1)),)))
        self.declines(replace(q, outputs=(replace(y,shape=shape),scale)))
        self.declines(replace(q, attributes=attributes(sem, output_logical_shapes=((9,256),None))))
        c = Features("inference::cache_load",(y,scale),(x,),
                     attributes(dict(storage_precision="fp4", group_size=32, logical_shape=shape,
                                     conversion="unpack and dequantize valid rows to BF16")))
        self.matches(c, "cache_load", shape, "mxfp4", {"implementation":"aclnnAntiMxQuant"})
        self.declines(replace(c, inputs=(replace(y,stride=(512,1)),scale)))
        self.declines(replace(c, inputs=(tensor(FP8,shape),tensor(F32,(9,)))))

    def test_token_scale_and_local_rms_window(self):
        shape = (2,9,512)
        x = tensor(BF16, shape)
        fp8 = Features("inference::dynamic_mx_quant_fp8_reference", (x,),
                       (tensor(FP8,shape),tensor(F32,(2,9))), attributes(dict(
                           logical_shape=shape, padding=(0,0), granularity='per_token', logical_dtype='fp8')))
        self.matches(fp8, "quant_token_fp8", shape)
        self.declines(replace(fp8, outputs=(fp8.outputs[0],tensor(E8,(2,9,8,2)))))
        rms = Features('npu::npu_rms_norm',(x,tensor(BF16,(512,))), (x,tensor(F32,(2,9,1))),
                       attributes(dict(epsilon=1e-6, normalized_shape=(512,), reduction_dtype='float32')))
        self.matches(rms, 'rmsnorm', (18,512))
        self.declines(replace(rms, attributes=attributes(dict(epsilon=1e-6, reduction_dtype='float16'))))
        self.declines(replace(rms, outputs=(x,)))
        self.declines(replace(rms, category='comm'))

    def test_hc_post_and_dense_bf16_matmul_contracts(self):
        inputs = (tensor(BF16, (1,9,5120)), tensor(BF16, (1,9,4,5120)),
                  tensor(F32, (1,9,4)), tensor(F32, (1,9,4,4)))
        hc = Features("_C_ascend::npu_hc_post", inputs, (inputs[1],), attributes(dict(hc_mult=4)))
        self.matches(hc, 'mhc_post', (1,9,4,5120), attrs={"implementation": "vllm_hc_post"})
        self.declines(replace(hc, outputs=(inputs[0],)))
        mm = Features("aten.mm.default", (tensor(BF16, (9,64)), tensor(BF16, (64,32))),
                      (tensor(BF16, (9,32)),))
        self.matches(mm, 'matmul', (9,32,64))
        self.declines(replace(mm, inputs=(mm.inputs[0], tensor(BF16, (64,32), (1,64)))))
        self.declines(replace(mm, outputs=(tensor(F32, (9,32)),)))

    def test_cast_scope_and_device(self):
        x,y = tensor(BF16,(9,512)),tensor(F32,(9,512))
        f = Features("aten._to_copy.default", (x,), (y,))
        self.matches(f, 'cast_bf16_fp32', (9,512))
        for bad in (replace(f, inputs=(replace(x, device='cpu'),)),
                    replace(f, attributes=(('kwargs', ('mapping', (('device', 'cpu'),))),)),
                    replace(f, raw_name='inference::candidate_filtered_indexer'),
                    replace(f, attributes=(('semantic', 'missing'),))):
            self.declines(bad)
        self.matches(Features('aten.sigmoid.default', (y,), (y,)), 'sigmoid_fp32', (9,512), 'fp32')
        self.declines(Features('aten.sigmoid.default', (replace(y,device='cpu'),), (replace(y,device='cpu'),)))


class FlashCandidateTests(unittest.TestCase):
    def test_full_scan_only_for_known_flash_consumer(self):
        q, length = 9, 131081
        sem = dict(index_topk=512,kv_source=20,index_source=24,causal=True,
                   component='model.layers.24.self_attn',
                   reference='vllm_ascend/models/deepseek_v41/model.py:DeepseekV41Attention')
        inp = (tensor(BF16,(q,32,128)), tensor(BF16,(1,length,128)),tensor(BF16,(q,32)),
               tensor('int64',(q,)),tensor('int32',(q,2048)))
        f = Features('inference::candidate_filtered_indexer',inp,(tensor('int32',(q,512)),),attributes(sem))
        def get(node): return estimate_features(node,chip=load_chip())
        e = get(f)
        self.assertEqual(e.status,'extrapolated')
        self.assertEqual(e.work['cube_ops'],2*32*128*q*length)
        self.assertIn('/full_scan/',e.contract)
        def packed(node):
            return estimate_features(node,chip=load_chip(),include_schedule=True,
                                     candidate_implementation='a5_qsli_projection',candidate_options={'table_pages':9000})
        e4 = packed(f)
        self.assertEqual(e4.status,'extrapolated')
        self.assertIn('mxfp4',e4.work['cube_ops_by_dtype'])
        self.assertEqual(e4.work['schedule']['address_path'],'scalar')
        self.assertLess(e4.work['cube_ops'],e.work['cube_ops'])
        self.assertIsNone(packed(replace(f,attributes=attributes({**sem,'kv_source':18}))).latency_us)
        for changes in (dict(causal=False),dict(kv_source=18),dict(candidate_block_size=16),dict(reference='unknown')):
            self.assertIsNone(get(replace(f,attributes=attributes({**sem,**changes}))).latency_us)
        self.assertIsNone(get(replace(f,inputs=(*inp[:4],tensor('int64',(q,2048))))).latency_us)
