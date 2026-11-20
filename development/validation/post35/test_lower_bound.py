from pathlib import Path
import sys,unittest
HERE=Path(__file__).resolve().parent;sys.path[:0]=[str(HERE),str(HERE.parents[2])]
from lower_bound import oracle
def observations(times):
    return [dict(shape=[1,4,256],ca_us=y,validation_only_prior='irrelevant',sha256='-',trace='test') for y in times]
class OracleTest(unittest.TestCase):
    def test_repeated_low_requires_clipping(self):
        o=oracle(observations([1,1,2.9]));g,=o['groups']
        self.assertEqual(g['unconstrained_median'],1)
        self.assertEqual(g['best_achievable_point_us_at_max50'],1.45)
        self.assertAlmostEqual(o['min_wape_at_max50_pct'],100*(.45+.45+1.45)/4.9)
    def test_impossible_relative_cap(self):
        o=oracle(observations([1,3.1]))
        self.assertFalse(o['cap_feasible'])
        self.assertIsNone(o['min_wape_at_max50_pct'])
    def test_singleton_oracle_has_no_predictive_validation(self):
        o=oracle(observations([4.1]))
        self.assertEqual(o['singleton_shapes'],1)
        self.assertEqual(o['min_wape_at_max50_pct'],0)
if __name__=='__main__':unittest.main()
