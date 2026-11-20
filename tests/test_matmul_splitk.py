import json
from dataclasses import replace
from pathlib import Path
import unittest

from operator_model import load_chip
from operator_model.matmul_splitk import estimate_matmul_splitk


class TestExplicitMatmulStreamK(unittest.TestCase):
    def test_all_original_frozen_pairs(self):
        root=Path(__file__).resolve().parents[1]
        freeze=json.loads((root/'results/v0.28/matmul/streamk-freeze.json').read_text())
        for row in freeze['rows']:
            for name,expected in row['predictions'].items():
                with self.subTest(shape=row['shape'],chip=name):
                    result=estimate_matmul_splitk(row['shape'],chip=name)
                    self.assertEqual(result['route'], expected['route'])
                    self.assertAlmostEqual(result['latency_us'], expected['us'], places=10)
                    self.assertEqual((result['active_aic'],result['active_aiv']),(expected['aic'],expected['aiv']))
                    for key in ('k_splits','max_aligned_depth','partial_fixup_bytes','launch_us','partial_pipeline_us','cold_floor_us','merge_us'):
                        if key in expected:self.assertAlmostEqual(result[key],expected[key],places=10)

    def test_shape_alias_chip_clone_and_resource_changes(self):
        dt=load_chip();base=estimate_matmul_splitk([9,64,7168],chip=dt)
        alias=estimate_matmul_splitk((9*1,64,7168),chip=replace(dt,name='anonymous'))
        self.assertEqual(base,alias)
        slow=estimate_matmul_splitk((1*9,64,7168),chip=replace(dt,hbm_bandwidth_gbs=100.))
        self.assertGreater(slow['latency_us'],base['latency_us'])
        slow_launch=replace(dt,details={**dt.details,'matmul_launch_cycles':5000})
        self.assertGreater(estimate_matmul_splitk([9,64,7168],chip=slow_launch)['launch_us'],base['launch_us'])
        low_vec=replace(dt,vector_tops={**dt.vector_tops,'32':dt.vector_tops['32']/2})
        self.assertGreater(estimate_matmul_splitk((9,64,7168),chip=low_vec)['merge_us'],base['merge_us'])
        threshold=estimate_matmul_splitk((9,64,7168),chip='Ascend970DT')
        self.assertEqual(threshold['route'],'ordinary')
        self.assertEqual(threshold['threshold_aligned_k'],15360)
        self.assertEqual(estimate_matmul_splitk((9,64,7168),deterministic_level=2)['route'],'ordinary')

    def test_contract(self):
        for shape in ((0,64,1),(True,64,1),(64,7168),(2,3,-4)):
            with self.assertRaises(ValueError):estimate_matmul_splitk(shape)
        for opts in ({'dtype':'fp32'},{'layout':'NZ'},{'deterministic_level':True}):
            with self.assertRaises(ValueError):estimate_matmul_splitk((9,64,7168),**opts)
        no_vec=replace(load_chip(),vector_tops={**load_chip().vector_tops,'32':None})
        self.assertIn('Missing FP32',estimate_matmul_splitk((9,64,7168),chip=no_vec)['reason'])

    def test_normal_entry_preserves_useful_work_and_accounts_for_reduction(self):
        from operator_model import estimate
        for sku,splits,aiv in [('Ascend950DT_9582',32,64),('Ascend950PR_957b',28,56)]:
            direct=estimate_matmul_splitk([16,64,7168],chip=sku)
            for mode in ('auto','resource'):
                result=estimate('matmul',[16,64,7168],chip=sku,dtype='bfloat16',timing_mode=mode)
                self.assertAlmostEqual(result.latency_us,direct['latency_us'],places=10)
                self.assertEqual((result.active_aic,result.active_aiv),(splits,aiv))
                self.assertEqual((result.work['cube_ops'],result.work['vec_ops']),(2*16*64*7168,0))
                self.assertEqual(result.work['logical_write_bytes'],2*16*64)
                extra=result.work['implementation_work']
                self.assertEqual(extra['fp32_reduction_adds'],16*64*(splits-1))
                self.assertEqual(extra['fp32_partial_write_bytes'],4*16*64*splits)
                self.assertEqual(extra['fp32_partial_read_bytes'],4*16*64*splits)
                self.assertEqual(result.projection['target_plan']['mmads_total'],splits)
                self.assertAlmostEqual(sum(result.projection['stages_us'].values()),result.latency_us,places=10)
        unchanged=estimate('matmul',[16,64,7168],chip='Ascend970DT')
        self.assertEqual(unchanged.projection['timing_diagnostics']['streamk_selection']['route'],'ordinary')
        self.assertNotIn('implementation_work',unchanged.work)
        bias=estimate('addmm',[16,64,7168])
        self.assertNotIn('streamk_selection',bias.projection['timing_diagnostics'])
        self.assertEqual(estimate('matmul',[16,64,7168],dtype='fp32').status,'unsupported')
