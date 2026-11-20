from dataclasses import replace
from math import isclose
from unittest import TestCase

from operator_model import estimate, load_chip
from operator_model.operators.matmul.aspect_dispatch import resource_matmul_dispatch as old
from operator_model.operators.matmul.dispatch import resident_schedule


class ResidentRouteTests(TestCase):
    def test_real_tail_histogram_and_shared_supply_on_both_devices(self):
        # Raw physical PC/MMAD-address audits, independent of offline timing.
        for sku, workers, count, expected in [
            ('Ascend950DT_9582',32,65536,[[256,128,32]]),
            ('Ascend950PR_957b',28,65792,[[84,128,8],[86,128,16]])]:
            s=[1048576,128,512]; result=estimate('matmul',s,chip=sku)
            p=result.projection['target_plan'];work=result.work['implementation_work']
            self.assertEqual(p['branch'],'b_resident_l1')
            self.assertEqual(p['mmads_total'],count)
            self.assertEqual(p['tail_wave_histogram'],expected)
            self.assertEqual(work['resident_b_gm_to_l1_bytes'],workers*512*128*2)
            self.assertEqual(work['a_gm_to_l1_bytes']+work['logical_output_bytes'],2*1048576*(512+128))
            self.assertAlmostEqual(sum(result.projection['stages_us'].values()),result.latency_us)

    def test_portable_target_scaling_and_credit_scope(self):
        s=[262145,128,512];baseline=load_chip('Ascend950PR_957b')
        rename=replace(baseline,name='experiment')
        self.assertEqual(resident_schedule(s,baseline),resident_schedule(s,rename))
        mixed=resident_schedule(s,baseline);self.assertIsNotNone(mixed)
        diag=mixed[-1];plan=diag['target_plan']
        self.assertEqual(sum(m*n for m,_,n in plan['tail_wave_histogram'])+
                         (plan['waves']-1)*baseline.cube_cores*plan['tile'][0],s[0])
        self.assertLessEqual(diag['internal_work']['scenario_initial_a_credit_bytes'],diag['pricing']['l2_bytes'])
        cold=replace(baseline,details={k:v for k,v in baseline.details.items() if k!='matmul_initial_read_gbs'})
        ct=resident_schedule(s,cold)
        self.assertGreaterEqual(ct[0],mixed[0]);self.assertEqual(ct[-1]['internal_work']['scenario_initial_a_credit_bytes'],0)
        limited=replace(baseline,details={**baseline.details,'matmul_hbm_efficiency':.5})
        self.assertGreater(resident_schedule(s,limited)[0],mixed[0])

    def test_few_cores_and_repeated_block_cost_is_bounded(self):
        dt=load_chip('Ascend950DT_9582')
        one=replace(dt,name='one',cube_cores=1,hbm_bandwidth_gbs=dt.hbm_bandwidth_gbs/32,cube_tops={**dt.cube_tops,'16':dt.cube_tops['16']/32})
        for m in [65536,2097152,2**32]:
            result=estimate('matmul',[m,128,512],chip=one);d=result.projection['timing_diagnostics'];p=d['target_plan']
            self.assertEqual((p['active_aic'],p['tail_wave_pieces']),(1,1))
            self.assertLessEqual(len(d['service_segments']),5)
            self.assertEqual(m < 2**32,d['pricing']['inplace_excluded_tensors_fit_hbm'])
        res=estimate('matmul',[2**64,128,512],chip=dt);self.assertIsNone(res.latency_us)

    def test_decode_and_wide_kimi_keep_previous_kernel(self):
        for s in ([1,2304,1536],[9,1280,5120],[144,2304,1536],[32768,3072,512]):
            for sku in ('Ascend950DT_9582','Ascend950PR_957b','Ascend970DT'):
                chip=load_chip(sku);r=estimate('matmul',s,chip=chip);v=old(s,chip)
                self.assertTrue(isclose(r.latency_us,v[0],rel_tol=1e-10))
                self.assertNotEqual(r.projection['target_plan']['branch'],'b_resident_l1')
