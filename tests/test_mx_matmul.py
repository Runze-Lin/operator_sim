"""Regression checks for the public dense MX routes and their work contract."""
import json
import subprocess
import sys
import unittest

from operator_model import OperatorStream, estimate
from operator_model.resources import chip_file
from operator_model.workload import account_work


class MxMatmulTests(unittest.TestCase):
    def setUp(self):
        self.shape = (9, 17, 72)
        self.attrs = {'weight_dtype': 'mxfp4'}
        self.config = json.loads(chip_file('Ascend950DT_9582').read_text())

    def test_packed_values_and_paired_e8m0_tail(self):
        m, n, k = self.shape
        for dtype, attrs, a_bytes in [('mxfp4', {}, m*k//2), ('fp8_e4m3fn', self.attrs, m*k)]:
            with self.subTest(dtype=dtype):
                r = estimate('matmul', self.shape, attrs, dtype=dtype)
                self.assertEqual(r.status, 'extrapolated')
                self.assertEqual(r.work['operand_bytes'], {
                    'activation': a_bytes, 'weight': n*k//2, 'activation_scale': m*4, 'weight_scale': n*4})
                work = account_work('matmul', self.shape, {**attrs, 'dtype': dtype})
                self.assertEqual(r.work['logical_read_bytes'], work['logical_read_bytes'])
                self.assertEqual(work['logical_read_bytes'], a_bytes+n*k//2+(m+n)*4)
                self.assertEqual(r.work['logical_write_bytes'], 2*m*n)
                self.assertEqual(r.work['cube_ops_by_dtype'], {dtype: 2*m*n*k})
                self.assertEqual(r.work['implementation_work']['dynamic_instruction_counts'], None)
                self.assertAlmostEqual(sum(r.projection['stages_us'].values()), r.latency_us)
                self.assertAlmostEqual(r.rates['cube_nominal_peak_fraction'],
                                       2*m*n*k/(1e6*self.config['cube_tops']['4' if dtype=='mxfp4' else '8']*r.latency_us))
                self.assertEqual(r.projection['target_plan']['scale_shapes'], [[9, 2, 2], [17, 2, 2]])
                self.assertIsNone(r.work['physical_hbm_bytes'])

    def test_alias_and_contract_validation(self):
        valid = estimate('matmul', self.shape, self.attrs, dtype='fp8_e4m3fn')
        explicit = {'dtype': 'mxfp8', 'weight_dtype': 'float4_e2m1', 'transpose_a': False,
                    'transpose_b': True, 'group_size': 32, 'bias': False, 'layout': 'ND',
                    'scale_dtype': 'e8m0', 'accumulator_dtype': 'float32', 'output_dtype': 'bfloat16'}
        self.assertEqual(valid, estimate('matmul', self.shape, explicit, dtype='float8_e4m3fn'))
        for shape, dtype, attrs in [
                (self.shape, 'mxfp8', {}), (self.shape, 'mxfp4', {'dtype': 'mxfp8'}),
                (self.shape, 'mxfp4', {'weight_dtype': 'int4'}), (self.shape, 'mxfp4', {'layout': 'NZ'}),
                (self.shape, 'mxfp4', {'transpose_b': False}), (self.shape, 'mxfp4', {'transpose_a': 0}),
                (self.shape, 'mxfp4', {'bias': True}), (self.shape, 'mxfp4', {'group_size': 16}),
                (self.shape, 'mxfp4', {'output_dtype': 'fp8'}), ((1, 16, 2), 'mxfp4', {}),
                ((1, 16, 3), 'mxfp4', {}), ((1, 16, 20), 'mxfp8', self.attrs),
                ((1, 0, 64), 'mxfp4', {}), ((True, 16, 64), 'mxfp4', {})]:
            with self.subTest(shape=shape, dtype=dtype, attrs=attrs):
                self.assertEqual(estimate('matmul', shape, attrs, dtype=dtype).status, 'unsupported')

    def test_mixed_weight_stays_on_chip_and_uses_target_precision(self):
        shape = (65536, 256, 512)
        r = estimate('matmul', shape, self.attrs, dtype='mxfp8', chip='Ascend970DT')
        native = estimate('matmul', shape, dtype='mxfp4', chip='Ascend970DT')
        self.assertEqual((r.status, native.status), ('extrapolated', 'extrapolated'))
        self.assertEqual((native.active_aiv, native.kernel_count, r.kernel_count), (0, 1, 1))
        self.assertEqual(r.active_aic, r.active_aiv)
        self.assertEqual(r.projection['target_plan']['aiv_per_aic'], 1)
        grid = r.projection['target_plan']['tile_grid']
        self.assertGreater(grid[0]*grid[1], r.active_aic)
        iw = r.work['implementation_work']
        swat = r.projection['target_plan']['swat']
        self.assertEqual(iw['ub_to_l1_expanded_bytes'], sum(w*count*shape[2] for _,w,count in swat['rectangles']))
        self.assertEqual(sum(h*w*count for h,w,count in swat['rectangles']), shape[0]*shape[1])
        self.assertEqual(iw['weight_values_expanded'], iw['ub_to_l1_expanded_bytes'])
        self.assertEqual(r.work['logical_write_bytes'], native.work['logical_write_bytes'])
        self.assertAlmostEqual(native.projection['prices']['cube_ops_per_cycle_per_aic'],
                               4*r.projection['prices']['cube_ops_per_cycle_per_aic'])
        self.assertGreater(r.projection['stages_us']['inline_unpack'], 0)
        self.assertEqual(native.projection['stages_us']['inline_unpack'], 0)

    def test_swat_grid_matches_prefill_and_explains_non_dyadic_core_count(self):
        r = estimate('matmul', (2048,128,256), self.attrs, dtype='mxfp8', chip='Ascend950PR_957b')
        p = r.projection['target_plan']
        self.assertEqual((r.active_aic,r.active_aiv,p['weight_producer_aiv']), (26,52,26))
        self.assertEqual(p['swat']['rectangles'], [(48,128,1),(80,128,25)])
        self.assertEqual(p['swat']['l1_slots'],2)
        self.assertEqual(r.work['implementation_work']['ub_to_l1_expanded_bytes'],851968)

    def test_multiframe_local_capacity_and_first_page_dependence(self):
        # Two successive L1 pages go to separate producers. Each individual
        # page still traverses one AIV before its dependent FP8 MMAD.
        r = estimate('matmul', (9,4096,7168), self.attrs, dtype='mxfp8', chip='Ascend950PR_957b')
        p = r.projection['target_plan']
        self.assertEqual(p['weight_producer_aiv'],2*r.active_aic)
        self.assertGreater(p['l1_page_k'],p['mmad_k'])
        self.assertGreater(p['pages_per_tile'],1)
        self.assertEqual(p['swat']['l1_slots'],2)
        service = r.projection['page_service_cycles']['first']
        _,w,_ = max(p['swat']['rectangles'],key=lambda row:row[0]*row[1])
        self.assertAlmostEqual(service['unpack'], w*p['l1_page_k']/r.projection['prices']['weights_per_cycle_per_aiv'])
        huge = estimate('matmul', ((1<<31)+1,(1<<31)+3,7168), self.attrs, dtype='mxfp8')
        rects = huge.projection['target_plan']['swat']['rectangles']
        self.assertEqual(sum(h*w*count for h,w,count in rects), ((1<<31)+1)*((1<<31)+3))
        self.assertLess(len(rects),128)
        self.assertEqual(estimate('matmul',(1<<32,128,256),self.attrs,dtype='mxfp8').status,'unsupported')

    def test_extrapolated_only_and_resource_limits(self):
        for chip in (self.config, 'Ascend950PR_957b', 'Ascend970DT'):
            for dtype, attrs in [('mxfp4', {}), ('fp8_e4m3fn', self.attrs)]:
                r = estimate('matmul', self.shape, attrs, chip=chip, dtype=dtype, timing_mode='calibrated')
                self.assertEqual(r.status, 'unavailable')
                self.assertIsNone(r.latency_us)
                self.assertIsNone(r.kernel_count)
                self.assertIsNone(r.rates)
                self.assertEqual(r.work['cube_ops'], 2*9*17*72)
        for resource, field, dtype, attrs in [
                ('cube_tops', '4', 'mxfp4', {}), ('cube_tops', '8', 'mxfp8', self.attrs),
                ('vector_tops', '8', 'mxfp8', self.attrs)]:
            config = {**self.config, resource: {**self.config[resource], field: None}}
            r = estimate('matmul', self.shape, attrs, chip=config, dtype=dtype)
            self.assertEqual(r.status, 'unavailable')
            self.assertIsNotNone(r.work)
        tiny = {**self.config, 'details': {**self.config['details'], 'ub_bytes': 64}}
        self.assertEqual(estimate('matmul', self.shape, self.attrs, dtype='mxfp8', chip=tiny).status, 'unavailable')
        self.assertIsNotNone(estimate('matmul', self.shape, dtype='mxfp4', chip=tiny).latency_us)

    def test_development_partitioning_and_timing_scope(self):
        for chip, native_aic, mixed_aic in [('Ascend950DT_9582', 32, 32), ('Ascend950PR_957b', 24, 16)]:
            for dtype, attrs, aic in [('mxfp4', {}, native_aic), ('mxfp8', self.attrs, mixed_aic)]:
                r = estimate('matmul', (256, 128, 256), attrs, chip=chip, dtype=dtype)
                self.assertEqual(r.status, 'estimated')
                self.assertEqual(r.active_aic, aic)
                self.assertEqual(r.projection['target_plan']['pages_per_tile'], 1)
                self.assertEqual(r.projection['target_plan']['weight_producer_aiv'], 0 if dtype=='mxfp4' else aic)
                self.assertEqual(r.active_aiv, 0 if dtype=='mxfp4' else 2*aic)
                self.assertTrue(r.projection['calibration_scope']['covered'])
                strict = estimate('matmul', (256, 128, 256), attrs, chip=chip, dtype=dtype, timing_mode='calibrated')
                self.assertEqual(r, strict)
        modified = {**self.config, 'hbm_bandwidth_gbs': 2500}
        for chip, shape in [(modified, (9,128,256)), ('Ascend970DT', (9,128,256)),
                            (self.config, (2049,128,256)), (self.config, (1,127,256)),
                            (self.config, (16,128,512))]:
            self.assertFalse(estimate('matmul', shape, dtype='mxfp4', chip=chip).projection['calibration_scope']['covered'])
            self.assertIsNone(estimate('matmul', shape, dtype='mxfp4', chip=chip, timing_mode='calibrated').latency_us)
        absent = {**self.config, 'details': {**self.config['details'], 'l1_bytes': None}}
        no_capacity = estimate('matmul', (1,128,256), dtype='mxfp4', chip=absent)
        self.assertEqual(no_capacity.status, 'unavailable')
        self.assertIsNotNone(no_capacity.work)

    def test_stream_and_cli_price_the_same_launch(self):
        options = {**self.config, 'kernel_overhead_us': 2.5}
        stream = OperatorStream(options)
        kw = {'dtype': 'mxfp8', 'attrs': self.attrs}
        api = estimate('matmul', self.shape, chip=options, **kw)
        self.assertEqual(stream.preview('matmul', self.shape, **kw), api)
        self.assertEqual(stream.advance('matmul', self.shape, **kw), api)
        self.assertAlmostEqual(api.scheduled_us, api.latency_us + 2.5)
        cli = subprocess.run([sys.executable, '-m', 'operator_model', '--op', 'matmul', '--shape',
                              '9', '17', '72', '--attrs-json', json.dumps(self.attrs), '--dtype', 'mxfp8'],
                             capture_output=True, text=True, check=True)
        out = json.loads(cli.stdout)
        self.assertEqual((out['predicted_us'], out['kernel_count'], out['estimate_status']),
                         (api.latency_us, 1, 'extrapolated'))
