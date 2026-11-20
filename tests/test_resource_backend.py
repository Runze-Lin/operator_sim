from dataclasses import replace
from unittest import TestCase
from unittest.mock import patch

from operator_model import estimate, load_chip
from operator_model.operators.quant.resources import resource_quant


class ResourceBackendTest(TestCase):
    def test_fused_bias_repartitions_on_target_and_loads_bt_once_per_tile(self):
        pr = load_chip('Ascend950PR_957b')
        for m,cores in ((256,24),(512,28)):
            x = estimate('addmm',(m,512,256),chip=replace(pr,name='target-with-same-resources'))
            diag = x.projection['timing_diagnostics']
            self.assertEqual(x.active_aic,cores)
            self.assertEqual(diag['target_plan']['waves'],1)
            self.assertEqual(diag['bias_cold_bytes'],1024)
            self.assertGreaterEqual(diag['l1_to_bt_bytes'],1024)
            self.assertEqual(x.work['logical_read_bytes'],2*((m+512)*256+512))
        self.assertIsNone(estimate('addmm',(256,512,256),{'self_shape':[256,512]}).latency_us)

    def test_resource_never_calls_reference_predictor(self):
        dt, pr = load_chip(), load_chip('Ascend950PR_957b')
        with patch('operator_model.backend.predict', side_effect=AssertionError('reference pricing')):
            for op, shape in [('quant',(768,4096)),('fa',(9,32768,64)),('matmul',(256,512,256)),('rmsnorm',(64,2048))]:
                forced = estimate(op, shape, chip=pr, timing_mode='resource')
                renamed = estimate(op, shape, chip=replace(pr,name='unseen'))
                self.assertEqual(renamed.latency_us,forced.latency_us)
                self.assertEqual(estimate(op,shape,chip=pr),forced)
                self.assertGreater(estimate(op,shape,chip=dt,timing_mode='resource').latency_us,0)
                self.assertIsNone(forced.projection['reference_latency_us'])
                self.assertAlmostEqual(sum(forced.projection['stages_us'].values()), forced.latency_us)

    def test_contract_and_explicit_mode(self):
        for op,shape,kw in [('quant',[1,65],{}),('fa',[9,128,8],{}),('matmul',[1,32,16],{'dtype':'fp16'}),
                            ('mhc_post',[2,5,256],{}),('rmsnorm',[2,255],{}),('quant',[1,128],{'attrs':{'layout':'wrong'}})]:
            self.assertEqual(estimate(op,shape,timing_mode='resource',**kw).status,'unsupported')
        unknown = replace(load_chip('Ascend950PR_957b'),name='other')
        self.assertEqual(estimate('quant',[1,128],chip=unknown,timing_mode='calibrated').status,'unsupported')

    def test_explained_ub_tiles_match_compressed_quant_latency(self):
        for name in ('Ascend950DT_9582','Ascend950PR_957b'):
            chip=load_chip(name)
            for shape in ((2048,5120),(1024,7168),(448,6592)):
                us,_,_,diag=resource_quant(shape,chip,explain_tiles=True)
                tiles=diag['tile_intervals_us']
                self.assertEqual(len(tiles),diag['target_plan']['iterations'])
                body=max(tiles[-1]['MTE3'][1],diag['stages_us']['dispatch']
                         +diag['chip_floor_after_launch_cycles']/(1000*chip.frequency_ghz))
                self.assertAlmostEqual(us,body+diag['stages_us']['drain'])
                for t in tiles:
                    self.assertLessEqual(t['MTE2'][1], t['SIMD'][0])
                    self.assertLessEqual(t['SIMD'][1], t['MTE3'][0])

    def test_monolithic_prefill_uses_bounded_resource_schedule(self):
        for op,shape in [('quant',(8*262144,7168)),('matmul',(8*32768,32,5120))]:
            auto=estimate(op,shape)
            self.assertEqual(auto.status,'extrapolated')
            self.assertEqual(auto,estimate(op,shape,timing_mode='resource'))
            self.assertEqual(estimate(op,shape,timing_mode='calibrated').status,'unsupported')
        self.assertGreater(estimate('quant',(8*262144,7168)).projection['target_plan']['iterations'],100)

    def test_reference_quant_marks_streaming_work_as_extrapolated(self):
        small = estimate('quant', (1024, 7168))
        self.assertEqual(small.status, 'estimated')
        for shape in ((16384, 5120), (262144, 7168)):
            result = estimate('quant', shape)
            self.assertEqual(result.status, 'extrapolated')
            self.assertIn('instruction-window', result.reason)
            self.assertGreater(result.latency_us, 0)
            self.assertEqual(result.latency_us, estimate('quant', shape, timing_mode='calibrated').latency_us)

    def test_quant_slow_resources_and_long_repeats(self):
        chip=load_chip()
        shape=(1024,4096)
        base=resource_quant(shape,chip)[0]
        for target in [replace(chip,details={**chip.details,'mte2_bandwidth_gbs':3}),
                       replace(chip,details={**chip.details,'mte3_bandwidth_gbs':2}),
                       replace(chip,hbm_bandwidth_gbs=120),
                       replace(chip,vector_tops={**chip.vector_tops,'16':1})]:
            self.assertGreater(resource_quant(shape,target)[0],base)
        us,_,_,info=resource_quant((2**20,65536),chip)
        self.assertGreater(us,base)
        self.assertGreater(info['target_plan']['iterations'],1000)
