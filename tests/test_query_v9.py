import unittest

from operator_model.predictor import predict, predict_v8


class TestCurrentModels(unittest.TestCase):
    def test_mhc_chain_prices_pre_once(self):
        shape = (4, 4, 1024)
        chain = predict('mhc', shape, {'implementation': 'split'})
        self.assertAlmostEqual(chain['stages_us']['pre'], predict('mhc_pre', shape)['predicted_us'])
        self.assertAlmostEqual(chain['predicted_us'], sum(chain['stages_us'].values()))

    def test_unmeasured_long_token_path_is_identified(self):
        shape = (512, 4, 256)
        result = predict('mhc', shape)
        self.assertEqual(result['status'], 'extrapolation')
        self.assertEqual(result['predicted_us'], predict_v8('mhc', shape)['predicted_us'])
        self.assertIsNone(predict('mhc', shape, {'num_iters': 7})['predicted_us'])

    def test_addmm_respects_stage_overlap(self):
        result = predict('addmm', (9, 128, 256))
        stages = result['stage_services_us'].values()
        self.assertGreater(result['predicted_us'], max(stages))
        self.assertLess(result['predicted_us'], predict_v8('addmm', (9, 128, 256))['predicted_us'])
        self.assertIsNone(predict('addmm', (9, 128, 256), {'alpha': 0})['predicted_us'])
