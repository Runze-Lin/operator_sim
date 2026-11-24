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

    def test_partial_pinned_block_counts_match_enumerated_mask(self):
        # A partial newest block displaces a full older one. Identities of the
        # other C-1 blocks change scores and indices, but never this count.
        import random
        gen = random.Random(731)
        for length,slots in ((641,65),(1031,65),(518,65)):
            shape = (2,7,32,128,length,slots,512)
            rows = []
            for v in range(length-6,length+1):
                last = (v-1)//8
                keep = set(gen.sample(range(last), min(slots-1,last))) | {last}
                rows.append(sum(j//8 in keep for j in range(v)))
            for policy in ('full_scan','candidate_first'):
                result = estimate('candidate_filtered_indexer',shape,dict(implementation='li_pipeline_projection',
                                  block_size=8,score_policy=policy))
                self.assertEqual(result.status,'extrapolated')
                self.assertEqual(result.projection['candidate_keys_per_request'],tuple(rows))
                pairs = 2*(sum(rows) if policy=='candidate_first' else 7*length)
                self.assertEqual(result.work['cube_ops'],2*32*128*pairs)
                self.assertEqual(result.projection['candidate_key_pairs'],2*sum(rows))
                self.assertAlmostEqual(sum(result.projection['stages_us'].values()),result.latency_us)

    def test_geometry_and_output_position_order_use_separate_work(self):
        attrs = dict(implementation='li_pipeline_projection', block_size=8)
        tiny = (1,1,32,128,800,75,512)
        e = estimate('candidate_filtered_indexer',tiny,attrs)
        self.assertEqual(e.projection['mmad_m_per_request_group'],[64])  # observed H32: 5 MMADs @ M=64
        # Independent combinatorics of 512-position bitonic sort; one compare
        # and two selects per pair, two ops/element to offset and check output.
        comps = (512//2)*sum(range(1,10))
        self.assertEqual(comps,11520)
        self.assertAlmostEqual(e.projection['stages_us']['position_order_and_map'], (3*comps+2*512)/(27e6/64))
        other = estimate('candidate_filtered_indexer',self.shape,attrs)
        self.assertEqual(other.projection['mmad_m_per_request_group'],[128,128,64])
        self.assertEqual((other.active_aic,other.active_aiv), (3,5))
        self.assertEqual(other.projection['candidate_key_pairs'],147421)
        for sku in ('Ascend950DT_9582','Ascend950PR_957b','Ascend970DT'):
            self.assertIsNone(estimate('candidate_filtered_indexer',tiny,attrs,chip=sku,timing_mode='calibrated').latency_us)
        for bad in ((1,18,32,128,131081,2048,512),(1,1,32,128,800,60,512)):
            self.assertIsNone(estimate('candidate_filtered_indexer',bad,attrs).latency_us)
        self.assertIsNone(estimate('candidate_filtered_indexer',tiny,attrs,dtype='fp32').latency_us)
