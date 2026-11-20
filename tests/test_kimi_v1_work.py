import random
from dataclasses import replace
from unittest import TestCase

from operator_model.chips import load_chip
from operator_model.kimi_kda import plan_kimi_kda_prefill


class KimiV1WorkTest(TestCase):
    def test_official_packed_finalize_example(self):
        result = plan_kimi_kda_prefill([1,0,1], attention_tp=96)
        work = result['v1_work']
        final = work['finalize']
        self.assertFalse(work['timing_qualified'])
        self.assertFalse(work['target_extrapolated'])
        self.assertIsNone(result['full_dispatch_us'])
        self.assertIsNone(result['main_kernels_us'])
        self.assertEqual(result['main_stages_us'], {})
        self.assertEqual(final['logical_cube_ops'], 66048)
        self.assertEqual(final['logical_ops_per_aic'], (33024,33024)+(0,)*30)
        self.assertEqual(final['valid_rows_per_task_histogram'], {1:2})
        self.assertAlmostEqual(final['cube_useful_work_floor_us'],33024/(432e6/32))

    def test_pr_970_and_missing_peak(self):
        for name,cores in [('Ascend950PR_957b',28),('Ascend970DT',120)]:
            p=plan_kimi_kda_prefill([2048]*8,attention_tp=8,chip=name)['v1_work']
            f=p['finalize']
            self.assertEqual(f['logical_cube_ops'],9663676416)
            self.assertEqual(f['aic_with_output_tasks'],cores)
            self.assertEqual(p['target_extrapolated'],name=='Ascend970DT')
            self.assertAlmostEqual(f['cube_useful_work_floor_us'],max(f['logical_ops_per_aic'])*cores/(f['chip_bf16_cube_tops']*1e6))
        dt=load_chip()
        missing=replace(dt,cube_tops={**dt.cube_tops,'16':None})
        result=plan_kimi_kda_prefill([65],attention_tp=8,chip=missing)['v1_work']
        self.assertTrue(result['target_extrapolated'])
        self.assertIsNone(result['finalize']['cube_useful_work_floor_us'])
        self.assertGreater(result['finalize']['logical_cube_ops'],0)

    def test_other_routes_and_global_task_order(self):
        self.assertIsNone(plan_kimi_kda_prefill([21], attention_tp=8)['v1_work']['finalize'])
        self.assertIsNone(plan_kimi_kda_prefill([0], attention_tp=8)['v1_work'])
        self.assertIsNone(plan_kimi_kda_prefill([2689]*8,attention_tp=8)['v1_work'])
        rng=random.Random(27003)
        for _ in range(24):
            lengths=[rng.randrange(1,170) for _ in range(4)]
            data=plan_kimi_kda_prefill(lengths,attention_tp=8)['v1_work']['finalize']
            logical_tasks=[2*t*128*(128+t) for length in lengths for start in range(0,length,64)
                           for t in [min(length-start,64)] for _ in range(12)]
            expected=tuple(sum(logical_tasks[i::32]) for i in range(32))
            self.assertEqual(data['logical_ops_per_aic'],expected)
