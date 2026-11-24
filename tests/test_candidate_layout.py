import unittest

from operator_model import summarize_candidate_layout


class CandidateLayoutTest(unittest.TestCase):
    def test_rank_order_is_distinct_from_shared_page_set(self):
        a = [0, 16, 33]
        same = summarize_candidate_layout([a, a])
        rotated = summarize_candidate_layout([a, [16, 33, 0]])
        self.assertEqual(same["candidate_order_overlap"], 1)
        self.assertEqual(rotated["candidate_order_overlap"], 0)
        self.assertEqual(rotated["distinct_pages"], same["distinct_pages"])

    def test_ragged_rows_and_physical_aliases(self):
        result = summarize_candidate_layout([[0, 1], [16, 17], [32]],
                                            page_table=[7, 7, 8])
        self.assertEqual((result["matching_pairs"], result["comparable_pairs"]), (2, 4))
        self.assertEqual(result["candidate_order_overlap"], 0.5)
        self.assertEqual(result["distinct_pages"], 2)

    def test_multiple_length_boundaries_with_empty_and_repeated_rows(self):
        rows = [[], [2, 7, 8], [2], [], [2, 6, 8], [2, 7]]
        result = summarize_candidate_layout(map(iter, rows))
        pairs = [(a, b) for i, row in enumerate(rows) for other in rows[:i]
                 for a, b in zip(row, other)]
        self.assertEqual(result["comparable_pairs"], len(pairs))
        self.assertEqual(result["matching_pairs"], sum(a == b for a, b in pairs))
        self.assertEqual(result["valid_candidates"], sum(map(len, rows)))
        self.assertEqual(result["queries"], len(rows))

    def test_no_comparable_pair_and_invalid_inputs(self):
        self.assertIsNone(summarize_candidate_layout([[], [3]])["candidate_order_overlap"])
        for ids, attrs in (([[-1]], {}), ([[16]], {"page_table": [0]}),
                           ([[0]], {"page_table": [-1]}), ([], {"pa_block_size": 7})):
            with self.assertRaises(ValueError):
                summarize_candidate_layout(ids, **attrs)
