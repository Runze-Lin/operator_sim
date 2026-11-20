from dataclasses import dataclass, field, replace
import unittest

from operator_model.scheduling.reference import predict
from operator_model.scheduling.projection import project
from operator_model.workload import account_work


@dataclass
class Hardware:
    frequency_ghz: float = 1.65
    cube_cores: int = 32
    vector_cores: int = 64
    cube_tops: dict = field(default_factory=lambda: {'16':432, '8':865})
    vector_tops: dict = field(default_factory=lambda: {'32':27, '16':54})
    hbm_bandwidth_gbs: float = 4000
    details: dict = field(default_factory=dict)


class ProjectionTests(unittest.TestCase):
    ref = Hardware()

    def price(self, op, shape, target=None, attrs=None):
        result = predict(op,shape,attrs)
        return project(op,shape,attrs,result,account_work(op,shape,attrs),
                       self.ref,target or self.ref)

    def test_identity_and_small_grid_does_not_gain_unused_cores(self):
        shape = (16,64,128)
        bigger = replace(self.ref,cube_cores=64,cube_tops={'16':864})
        before = self.price('matmul',shape)
        after = self.price('matmul',shape,bigger)
        self.assertAlmostEqual(before['reference_latency_us'],after['latency_us'])
        self.assertEqual(after['active_aic'],1)
        self.assertLess(self.price('matmul',(2048,2048,512),bigger)['latency_us'],
                        self.price('matmul',(2048,2048,512))['latency_us'])

    def test_frequency_and_vector_peak_act_on_separate_stages(self):
        shape = (512,4096)
        base = self.price('rmsnorm',shape)['stages_us']
        fast_control = replace(self.ref,frequency_ghz=3.3)
        control = self.price('rmsnorm',shape,fast_control)['stages_us']
        self.assertAlmostEqual(control['SIMD'],base['SIMD'])
        self.assertAlmostEqual(control['SCALAR'],base['SCALAR']/2)
        fast_vector = replace(self.ref,vector_tops={'32':54})
        vector = self.price('rmsnorm',shape,fast_vector)['stages_us']
        self.assertAlmostEqual(vector['SIMD'],base['SIMD']/2)
        self.assertAlmostEqual(vector['SCALAR'],base['SCALAR'])

    def test_bandwidth_and_unknown_precision(self):
        base = self.price('rmsnorm',(512,4096))['stages_us']
        slow = replace(self.ref,hbm_bandwidth_gbs=1000)
        result = self.price('rmsnorm',(512,4096),slow)['stages_us']
        self.assertGreater(result['MTE2'],base['MTE2'])
        self.assertAlmostEqual(result['SIMD'],base['SIMD'])
        unknown = self.price('rmsnorm',(3,4096),replace(self.ref,vector_tops={}))
        self.assertTrue(any('absent comparable peaks' in s for s in unknown['notes']))

    def test_cv_bottleneck_can_change_and_serial_windows_add(self):
        slow = replace(self.ref,vector_tops={'32':1})
        self.assertGreater(self.price('fa',(9,32768,64,1,1),slow)['latency_us'],
                           self.price('fa',(9,32768,64,1,1))['latency_us'])
        shape,attrs = (32,256,128,2),{'expert_counts':[1,31]}
        chain = self.price('moe_ffn',shape,slow,attrs)
        parts = [self.price(op,shape,slow,attrs)['latency_us']
                 for op in ('grouped_matmul_swiglu_quant_v2','gmm2')]
        self.assertAlmostEqual(sum(parts),chain['latency_us'])
        self.assertAlmostEqual(sum(chain['stages_us'].values()),chain['latency_us'])

    def test_target_repartition_changes_local_work(self):
        target = replace(self.ref,cube_cores=28,vector_cores=56)
        q = self.price('quant',(64,1024),target)['target_plan']
        self.assertEqual((q['active_aiv'],q['rows_per_core'],q['cols_per_core']),(44,3,512))
        self.assertEqual(q['units']['read'],3.)
        m = self.price('matmul',(128,256,512),target)['target_plan']
        self.assertEqual((m['active_aic'],m['tile'][:2],m['waves']),(16,[32,64],1))
        self.assertEqual(m['units']['cube']*131072,2*32*64*512)

    def test_row_chunk_leaves_idle_cores(self):
        self.assertEqual(self.price('rmsnorm',(65,128))['active_aiv'],33)
