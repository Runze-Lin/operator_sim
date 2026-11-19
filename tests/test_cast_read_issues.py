import unittest
from dataclasses import replace

from operator_model import estimate, load_chip


class CastReadIssues(unittest.TestCase):
    def run_case(self, shape, chip):
        result = estimate('cast_bf16_fp32', shape, chip=chip, dtype='bf16')
        self.assertIsNotNone(result.latency_us)
        return result.latency_us, result.projection['timing_diagnostics']

    def test_initial_pair_is_independent_of_the_steady_bandwidth(self):
        reference = load_chip('Ascend950PR_957b')
        slow = replace(reference, hbm_bandwidth_gbs=reference.hbm_bandwidth_gbs/2)
        base_us, base = self.run_case([1048576, 64], reference)
        slow_us, limited = self.run_case([1048576, 64], slow)
        self.assertGreater(slow_us, base_us)
        self.assertEqual(base['initial_read_issue_us'], limited['initial_read_issue_us'])
        a, b = base['initial_read_issue_us']
        self.assertAlmostEqual((b-a)*1000*reference.frequency_ghz, 91)

    def test_single_slot_and_alternate_frontend(self):
        base = load_chip('Ascend970DT')
        self.assertEqual(len(self.run_case([1, 64], base)[1]['initial_read_issue_us']), 1)
        for cycles in (200, None):
            chip = replace(base, details={**base.details, 'cast_initial_read_pair_cycles': cycles})
            _, data = self.run_case([2097152, 64], chip)
            issues = data['initial_read_issue_us']
            if cycles is None:
                self.assertEqual(len(issues), 1)
            else:
                self.assertAlmostEqual((issues[1]-issues[0])*1000*chip.frequency_ghz, cycles)
