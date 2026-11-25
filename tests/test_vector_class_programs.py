from dataclasses import replace
import unittest
from operator_model import estimate, load_chip
from operator_model.operators.vector.instruction_classes import tile_service


class VectorClassPrograms(unittest.TestCase):
    def test_known_routes_and_contracts(self):
        for op in ('cast_bf16_fp32', 'sigmoid_fp32', 'quant_token_fp8'):
            original = estimate(op, (17,1024))
            explicit = estimate(op, (17,1024), {'vector_model':'calibrated'})
            self.assertEqual(original.latency_us, explicit.latency_us)
            options = {'vector_model':'instruction_classes'}
            trial = estimate(op, (17,1024), options)
            self.assertEqual(trial.status, 'extrapolated')
            self.assertEqual(trial.work, original.work)
            self.assertEqual(trial.active_aiv, original.active_aiv)
            self.assertIsNone(estimate(op, (17,1024), options, timing_mode='calibrated').latency_us)
            self.assertIsNone(estimate(op, (17,1024), {'vector_model':'unknown'}).latency_us)

    def test_resource_changes(self):
        nominal = load_chip()
        slow = replace(nominal,details={**nominal.details,'vf_exp_ii_cycles':12})
        self.assertGreater(tile_service('sigmoid',512,slow)[0],tile_service('sigmoid',512,nominal)[0])
        self.assertEqual(tile_service('cast',512,slow),tile_service('cast',512,nominal))
        service, diag = tile_service('token_quant',258,nominal,rows=2)
        self.assertEqual(diag['groups'],6)
        self.assertEqual(diag['counts']['convert'],18)
        self.assertEqual(service,max(diag['bounds_cycles'].values()))
