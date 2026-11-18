from dataclasses import replace
import unittest

from operator_model.chips import load_chip
from operator_model.quant_transfer_v14 import project_quant
from operator_model.target_tiling import quant_plan


class QuantTransfer(unittest.TestCase):
    def price(self, shape, target=None):
        pr = target or load_chip('Ascend950PR_957b')
        return project_quant(shape, {}, load_chip(), pr, quant_plan(shape, pr.vector_cores, pr.details['ub_bytes']))

    def test_guarded_sku_route(self):
        self.assertIsNone(self.price((1, 128)))
        self.assertIsNone(self.price((8192, 8192)))
        pr = load_chip('Ascend950PR_957b')
        self.assertIsNone(self.price((512, 256), replace(pr, frequency_ghz=1.8)))
        self.assertEqual(self.price((512, 256), replace(pr, sources={})), self.price((512, 256)))

    def test_identifiable_intervals_add(self):
        t, diagnostics = self.price((512, 256))
        self.assertAlmostEqual(sum(diagnostics['phases_us'].values()), t)
        self.assertLess(t, 5.42978)
        self.assertGreater(diagnostics['phases_us']['write_window'], .4)
