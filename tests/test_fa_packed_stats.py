from unittest import TestCase

from operator_model import load_chip
from operator_model.operators.attention.vector_classes import vector_class_tile


class FAPackedStatisticsTest(TestCase):
    def tile(self, rows, *, first=False, n=256):
        return vector_class_tile(rows, n, 64, first=first, last=False,
                                 dtype='bf16', chip=load_chip())

    def test_row_scores_and_cross_row_stats_have_different_register_counts(self):
        for rows, registers in ((1,1), (5,1), (64,1), (65,2)):
            phases = self.tile(rows)['phases']
            self.assertEqual(phases['statistics']['counts']['exp'], registers)
            self.assertEqual(phases['probability']['counts']['exp'], rows * 4)

    def test_first_page_initializes_and_tail_keeps_probability(self):
        tile = self.tile(5, first=True, n=1)
        self.assertEqual(tile['phases']['statistics']['counts'], {})
        self.assertEqual(tile['phases']['probability']['counts']['exp'], 5)
        self.assertLess(tile['p_ready'], tile['vec1'])

    def test_canonical_128_page_count_and_busy_cycles(self):
        pages = [self.tile(5, first=True), self.tile(5),
                 vector_class_tile(5,256,64,first=False,last=True,
                                   dtype='bf16',chip=load_chip())]
        exps = sum(n * sum(v['counts'].get('exp',0) for v in p['phases'].values())
                   for n,p in zip((1,126,1),pages))
        self.assertEqual(exps, 2687)
        self.assertAlmostEqual(sum(n*(p['vec1']+p['vec2']) for n,p in zip((1,126,1),pages)),32570.5)
