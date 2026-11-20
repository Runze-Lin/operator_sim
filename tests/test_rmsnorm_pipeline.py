import unittest
from dataclasses import replace

from operator_model import estimate, load_chip
from operator_model.operators.rmsnorm.pipeline import RmsGeometry, rmsnorm_pipeline


class RmsNormPipelineTests(unittest.TestCase):
    def test_inspect_same_rows_and_dependencies(self):
        q = rmsnorm_pipeline([539,4000], chip='Ascend950PR_957b', resident_aiv=1, diagnostics=True)
        self.assertEqual(sum(t['rows'] for t in q['stages']), 539)
        for t in q['stages']:
            self.assertGreaterEqual(t['reduce_issue'], max(t['x'][1], t['gamma'][1]))
            self.assertGreater(t['rstd_vf'][0], t['reduce_issue'])
            self.assertGreater(t['normalize'][0], t['rstd_vf'][0])
            self.assertGreater(t['rstd_issue'], t['rstd_vf'][1])
            self.assertGreaterEqual(t['rstd_retire'], t['rstd_issue'])
            self.assertGreater(t['y'][0], t['normalize'][1])
        self.assertAlmostEqual(q['us'] * 1650, max(q['bounds_cycles'].values()))

    def test_public_unknown_and_known_history_contract(self):
        from operator_model.operators.rmsnorm.dispatch import price_rms
        for chip in ('Ascend950DT_9582', 'Ascend950PR_957b', 'Ascend970DT'):
            for shape in ((4,784),(539,4000)):
                public = estimate('rmsnorm',shape,chip=chip)
                self.assertAlmostEqual(public.latency_us,rmsnorm_pipeline(shape,chip=chip)['us'])
                self.assertEqual(public.projection['stages_us'],{'unknown_history_total':public.latency_us})
                for h in (0,1):
                    known = estimate('rmsnorm',shape,{'resident_aiv':h},chip=chip)
                    self.assertEqual(known.latency_us,price_rms(shape,load_chip(chip),{'resident_aiv':h})[0])
            shape = (262144,4096)
            self.assertEqual(estimate('rmsnorm',shape,chip=chip).latency_us,price_rms(shape,load_chip(chip),{})[0])

    def test_repeat_and_prefix_are_distinct(self):
        shape = [4,784]
        q = rmsnorm_pipeline(shape)
        for key,h,repeat in [('first',0,False),('one_lane_prefix',1,False),('repeat',4,True)]:
            explicit = rmsnorm_pipeline(shape,resident_aiv=h,immediate_repeat=repeat)
            self.assertEqual(q['scenarios'][key]['us'],explicit['us'])
        self.assertLess(q['lower_us'],q['us'])
        self.assertLess(q['us'],q['upper_us'])
        with self.assertRaises(ValueError):
            rmsnorm_pipeline(shape,resident_aiv=1,immediate_repeat=True)
        with self.assertRaises(ValueError):
            rmsnorm_pipeline(shape,dtype='fp32')
        self.assertEqual(q['us'],rmsnorm_pipeline(shape,dtype='bfloat16')['us'])

    def test_target_hbm_limit_and_geometry(self):
        target = load_chip('Ascend970DT')
        narrow = replace(target,hbm_bandwidth_gbs=1.)
        ref = rmsnorm_pipeline([539,4000],chip=target,resident_aiv=0)
        slow = rmsnorm_pipeline([539,4000],chip=narrow,resident_aiv=0)
        self.assertGreater(slow['us'],ref['us']*10)
        self.assertTrue(slow['extrapolated'])
        with self.assertRaises(ValueError):
            rmsnorm_pipeline([8,320],prices={'x_issue':[float('nan')]})
        delayed = rmsnorm_pipeline([8,320],resident_aiv=8,immediate_repeat=True,
                                   geometry=RmsGeometry(sector_issue_interval_cycles=2048))
        nominal = rmsnorm_pipeline([8,320],resident_aiv=8,immediate_repeat=True)
        self.assertGreater(delayed['us'],nominal['us'])
        self.assertEqual(RmsGeometry().vector_work(320,2),(2,6,6))
        self.assertEqual(RmsGeometry(vector_register_bytes=128).vector_work(320,2),(2,10,8))
        with self.assertRaises(ValueError):
            RmsGeometry(statistic_sector_bytes=2)

    def test_very_large_rows_use_bounded_steady_path(self):
        q = rmsnorm_pipeline([8*32*1024,4096],resident_aiv=64,immediate_repeat=True,diagnostics=True)
        self.assertEqual(q['route'],'steady multi-UB')
        self.assertIsNone(q['stages'])
        self.assertAlmostEqual(q['us'],sum(q['stages_us'].values()))
