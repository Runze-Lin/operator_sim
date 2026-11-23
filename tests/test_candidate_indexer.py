import unittest

from operator_model import estimate


class CandidateIndexerTests(unittest.TestCase):
    shape = (1, 9, 32, 128, 131081, 2048, 512)

    def test_flash_scan_accounts_for_all_five_inputs(self):
        result = estimate('inference::candidate_filtered_indexer', self.shape)
        self.assertEqual(result.status, 'unavailable')
        self.assertIsNone(result.latency_us)
        self.assertIsNone(result.work['cube_ops'])
        self.assertEqual(result.work['score_dot_ops'], 2 * 9 * 32 * 128 * 131081)
        self.assertEqual(result.work['logical_read_bytes'], 2*(9*32*128 + 131081*128 + 9*32) + 8*9 + 4*9*2048)
        self.assertEqual(result.work['logical_write_bytes'], 4*9*512)

    def test_unknown_and_known_distinct_key_counts(self):
        attrs = {'score_policy': 'candidate_first', 'block_size': 8}
        bare = estimate('candidate_filtered_indexer', self.shape, attrs).work
        self.assertIsNone(bare['score_dot_ops'])
        self.assertEqual(bare['score_dot_ops_upper_bound'], 2*9*32*128*2048*8)
        counts = [0, 8, 8, 15, 16, 30, 28, 17, 16]
        known = estimate('candidate_filtered_indexer', self.shape, {**attrs, 'scored_keys_per_query': counts}).work
        self.assertEqual(known['score_dot_ops'], 2*32*128*sum(counts))
        for bad in (True, [1, 2], 2048*8+1):
            self.assertEqual(estimate('candidate_filtered_indexer', self.shape, {**attrs, 'scored_keys_per_query': bad}).status,
                             'unsupported')
