import json
from pathlib import Path
import unittest

from operator_model import estimate
from operator_model.predictor import predict, predict_v10, predict_v11


class TestFrozenWideModels(unittest.TestCase):
    def test_calibrated_route_and_explicit_history_replay_the_v12_freeze(self):
        root = Path(__file__).resolve().parents[1] / 'results/v0.12'
        for folder in ('rms', 'post'):
            cases = json.loads((root / folder / 'freeze.json').read_text())['cases']
            for case in cases:
                op, shape, attrs = case['op'], case['shape'], case['attrs']
                with self.subTest(op=op, shape=shape):
                    args = {**attrs, 'resident_aiv': case['prior_aiv']}
                    with_history = estimate(op, shape, args, timing_mode='calibrated')
                    self.assertAlmostEqual(with_history.latency_us,
                                           case['sequence_prediction']['predicted_us'])
                    current = predict(op, shape, attrs)
                    self.assertAlmostEqual(sum(current['stages_us'].values()),
                                           case['prediction']['predicted_us'])
                    self.assertAlmostEqual(estimate(op, shape, attrs, timing_mode='calibrated').latency_us,
                                           case['prediction']['predicted_us'])
                    self.assertAlmostEqual(predict_v11(op, shape, attrs)['predicted_us'],
                                           case['previous_us'])

    def test_historical_v11_predictions_remain_replayable(self):
        root = Path(__file__).resolve().parents[1] / 'results/v0.11'
        for folder in ('rms', 'post'):
            for case in json.loads((root / folder / 'freeze.json').read_text())['cases']:
                op, shape, attrs = case['op'], case['shape'], case['attrs']
                with self.subTest(op=op, shape=shape):
                    self.assertAlmostEqual(predict_v11(op, shape, attrs)['predicted_us'],
                                           case['prediction']['predicted_us'])
                    self.assertAlmostEqual(predict_v10(op, shape, attrs)['predicted_us'],
                                           case['previous_us'])
