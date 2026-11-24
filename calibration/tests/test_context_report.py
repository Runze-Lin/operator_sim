"""Context sweep geometry and integration of device time."""
import unittest

from calibration.context_report import linear
from calibration.context_sweep import EXACT_DECODE, inputs, points, profiles


class ContextTests(unittest.TestCase):
    def test_request_coverage_and_position_ceiling(self):
        cases = list(profiles())
        self.assertEqual(len(cases),len(set(cases)))
        self.assertEqual({p[1] for p in cases},{32768,131072,524288,1048576})
        for p in cases:
            chunks, rounds = [], []
            for *_,phase,history,q in inputs(p):
                if phase=="prefill":
                    chunks.append((history,q))
                else:
                    rounds.append(history)
            self.assertEqual(chunks[-1][0] + chunks[-1][1],p[1])
            self.assertEqual(len(rounds),147 if p[:4] in EXACT_DECODE else 3)
            self.assertLessEqual(max(rounds)+17,1048576)
            self.assertEqual(len(rounds),len(set(rounds)))

    def test_linear_integration_including_last_output_round(self):
        self.assertEqual(points(32768,4096),list(range(8)))
        samples={0:3.,73:76.,146:149.}
        self.assertEqual(sum(linear(samples,x) for x in range(147)),sum(range(3,150)))
        self.assertEqual(linear(samples,147),150)
