from dataclasses import replace
from unittest import TestCase

from operator_model import estimate, load_chip


class FAVectorModelsTest(TestCase):
    shape = (9, 32768, 64, 1, 1)

    def predict(self, attrs=None, **kwargs):
        return estimate('fa', self.shape, attrs, **kwargs)

    def diagnostics(self, result):
        return result.projection['timing_diagnostics']

    def test_default_matches_existing_bf16_route(self):
        default = self.predict()
        selected = self.predict({'vector_model': 'calibrated'})
        self.assertEqual(default.latency_us, selected.latency_us)
        self.assertEqual(default.status, 'estimated')
        other = self.predict({'vector_model': 'instruction_classes'})
        self.assertEqual(default.work, other.work)
        self.assertEqual(default.projection['target_plan'], other.projection['target_plan'])
        self.assertEqual(other.status, 'extrapolated')

    def test_new_design_dispatches_by_program_identity(self):
        design = self.predict({'implementation': 'dense_nd_row'})
        classes = self.predict({'vector_model': 'instruction_classes'})
        self.assertEqual(design.latency_us, classes.latency_us)
        self.assertEqual(self.diagnostics(design)['vector_model'], 'instruction_classes')
        self.assertIsNone(self.predict({'implementation': 'unknown'}).latency_us)
        self.assertIsNone(self.predict({'implementation': 'dense_nd_row', 'vector_model': 'calibrated'}).latency_us)

    def test_first_probability_and_final_conversion(self):
        wave = self.diagnostics(self.predict(dtype='fp16'))['wave_types'][0]
        self.assertLess(wave['p_ready_cycles'], wave['stats_ready_cycles'])
        for key in ('first', 'repeat', 'last'):
            output = wave[key]['vf']['phases']['output']['counts']
            self.assertEqual(output.get('divide', 0), 5 if key == 'last' else 0)
        self.assertEqual(wave['first']['vf']['dtype'], 'fp16')
        once = estimate('fa', (9, 64, 64), dtype='fp16')
        self.assertEqual(once.projection['stages_us']['steady'], 0)
        self.assertGreater(self.diagnostics(once)['wave_types'][0]['first']['vec2'], 0)

    def test_configured_convert_and_970_scenario(self):
        dt = load_chip()
        slow = replace(dt, details={**dt.details, 'fa_vf_fp16_convert_ii_cycles': 160})
        self.assertGreater(self.predict(dtype='fp16', chip=slow).latency_us,
                           self.predict(dtype='fp16', chip=dt).latency_us)
        self.assertEqual(self.predict(dtype='bf16', chip=slow).latency_us,
                         self.predict(dtype='bf16', chip=dt).latency_us)
        portable = self.predict(chip='Ascend970DT', dtype='fp16')
        self.assertEqual(portable.status, 'extrapolated')
        self.assertEqual(portable.active_aiv, 1)

    def test_probe_is_independent_and_scoped(self):
        for model in ('calibrated', 'instruction_classes'):
            result = self.predict({'vector_model': model, 'fa_port_model': 'probe'})
            info = self.diagnostics(result)
            self.assertEqual(info['wave_types'][0]['first']['read_timing'], 'probe')
            self.assertEqual(info['vector_model'], model)
        self.assertIsNone(self.predict({'fa_port_model': 'probe'}, chip='Ascend970DT').latency_us)
        self.assertIsNone(self.predict({'fa_port_model': 'probe'}, dtype='fp16').latency_us)
        self.assertIsNone(self.predict({'causal': True}, dtype='fp16').latency_us)
        self.assertIsNone(self.predict({'vector_model': 'calibrated'}, dtype='fp16').latency_us)
        self.assertIsNone(self.predict(dtype='fp16', timing_mode='calibrated').latency_us)
