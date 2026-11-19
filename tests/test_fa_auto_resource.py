import json
from dataclasses import replace
from pathlib import Path
from unittest import TestCase
from unittest.mock import patch

from operator_model import estimate, load_chip


class FAPublicResourceRoute(TestCase):
    def test_default_reprices_each_page_on_every_frozen_sku(self):
        path = Path(__file__).resolve().parents[1] / 'results/v0.25/coverage/fa-freeze.json'
        jobs = json.loads(path.read_bytes())['jobs']
        for job in jobs[::2]:
            for chip, expected in job['forecasts'].items():
                with self.subTest(shape=job['shape'], chip=chip):
                    with patch('operator_model.backend.predict', side_effect=AssertionError('DT shape surrogate')):
                        result = estimate('fa', job['shape'], chip=chip)
                    self.assertAlmostEqual(result.latency_us, expected['candidate'], delta=1e-12)
                    self.assertEqual((result.active_aic,result.active_aiv), (expected['active_aic'], expected['active_aiv']))
                    self.assertEqual(result.projection['target_plan'], expected['plan'])
                    self.assertEqual(result, estimate('fa', job['shape'], chip=chip, timing_mode='resource'))
                    if chip == 'Ascend950DT_9582':
                        self.assertAlmostEqual(estimate('fa',job['shape'],chip=chip,timing_mode='calibrated').latency_us,
                                               expected['old'], delta=1e-12)

    def test_reference_resources_are_portable_and_long_kv_is_marked(self):
        chip = load_chip()
        shape = (9, 1008, 64)
        reference = estimate('fa',shape,chip=chip)
        anonymous = estimate('fa',shape,chip=replace(chip,name='scenario'))
        self.assertEqual(anonymous.latency_us, reference.latency_us)
        self.assertEqual(anonymous.projection['target_plan'], reference.projection['target_plan'])
        self.assertEqual(reference.status,'estimated')
        self.assertEqual(estimate('fa',(9,262144,64)).status,'extrapolated')
        self.assertEqual(estimate('fa',shape,{'causal':True}).status,'unsupported')
