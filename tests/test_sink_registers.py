"""Installed N4 kernel register work and shape-only candidate eligibility."""
import json
from pathlib import Path
import unittest
from dataclasses import replace

from operator_model import load_chip
from operator_model.operators.mhc.sink_resources import sink as legacy
from operator_model.operators.mhc.sink_registers import price, registers_per_helper

ROOT = Path(__file__).resolve().parents[1]


class SinkRegisterCandidateTest(unittest.TestCase):
    def test_rows_per_register_in_implementation(self):
        self.assertEqual([registers_per_helper(n) for n in (1, 4, 5, 8, 9, 16, 17)],
                         [1, 1, 1, 1, 2, 2, 3])
        self.assertEqual(registers_per_helper(5, 128, 32), 2)
        self.assertEqual(registers_per_helper(9, 512, 32), 1)
        with self.assertRaises(ValueError):
            registers_per_helper(3, 224, 64)

    def test_replacement_only_affects_work_in_simd(self):
        # 233 matrices can cross different register boundaries solely by changing
        # the available cores in an otherwise identical config.
        for cores, rows, groups in ((24, 10, 2), (56, 5, 1), (64, 4, 1), (120, 2, 1)):
            chip = replace(load_chip(), vector_cores=cores)
            shape, attrs = [233, 4], {'num_iters': 16}
            previous, _, old_cores, old = legacy(shape, chip, attrs)
            new, _, new_cores, diag = price(shape, chip, attrs)
            self.assertEqual(new_cores, old_cores)
            self.assertEqual((diag['target_plan']['rows_per_core'], diag['target_plan']['register_passes']), (rows, groups))
            for phase, us in old['stages_us'].items():
                if phase != 'SIMD':
                    self.assertEqual(diag['stages_us'][phase], us)
            self.assertEqual(new, sum(diag['stages_us'].values()))
            if groups == old['target_plan']['packed_chunks']:
                self.assertEqual(new, previous)
            else:
                self.assertLess(diag['stages_us']['SIMD'], old['stages_us']['SIMD'])


    def test_preserve_shape_and_attribute_contract(self):
        chip = load_chip('Ascend950PR_957b')
        for shape, attrs in (([233, 6], {}), ([257, 4], {}), ([233, 4], {'return_intermediates': True})):
            with self.assertRaises(ValueError):
                price(shape, chip, attrs)


if __name__ == '__main__':
    unittest.main()
