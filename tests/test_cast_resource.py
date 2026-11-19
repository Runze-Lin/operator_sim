from dataclasses import replace
import unittest

from operator_model import estimate, load_chip


class CastResourceTest(unittest.TestCase):
    def test_exact_conversion_work_and_flattened_partition(self):
        chip=load_chip()
        left=estimate('cast_bf16_fp32',(64,4096),chip=chip)
        right=estimate('cast_bf16_fp32',(128,2048),chip=chip)
        self.assertEqual(left.latency_us,right.latency_us)
        self.assertEqual(left.active_aiv,chip.vector_cores)
        self.assertEqual(left.work['logical_write_bytes'],4*64*4096)
        self.assertEqual(left.work['logical_read_bytes'],2*64*4096)
        self.assertEqual(left.work['vec_ops_by_class']['bf16_to_fp32_conversion'],64*4096)
        self.assertEqual(left.work['cube_ops'],0)
        self.assertAlmostEqual(sum(left.projection['stages_us'].values()), left.latency_us)

    def test_resource_limits_and_contract(self):
        chip=load_chip()
        query=lambda c:estimate('cast_bf16_fp32',(262144,7168),chip=c,timing_mode='resource')
        base=query(chip)
        slower=query(replace(chip,hbm_bandwidth_gbs=chip.hbm_bandwidth_gbs/2))
        self.assertGreater(slower.latency_us,base.latency_us)
        self.assertEqual(query(replace(chip,name='renamed')).latency_us,base.latency_us)
        self.assertEqual(estimate('cast_bf16_fp32',(5,320),dtype='fp16').status,'unsupported')
        self.assertEqual(estimate('cast_bf16_fp32',(1,0)).status,'unsupported')
