from unittest.mock import patch
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
        for bad in ((1,18,32,128,17,2048,512),(1,1,32,128,800,60,512)):
            self.assertIsNone(estimate('candidate_filtered_indexer',bad,attrs).latency_us)
        self.assertIsNone(estimate('candidate_filtered_indexer',tiny,attrs,dtype='fp32').latency_us)


    def test_prefill_mask_and_padding_match_enumerated_rows(self):
        attrs = dict(implementation='li_pipeline_projection',block_size=8)
        for b,q,length,slots in ((8,4096,4096,2048),(2,4096,32768,2048),
                                 (1,613,800,64),(3,66,555,65)):
            result=estimate('candidate_filtered_indexer',(b,q,32,128,length,slots,512),attrs)
            self.assertEqual(result.status,'extrapolated')
            rows=[min(v,(slots-1)*8+1+(v-1)%8) for v in range(length-q+1,length+1)]
            self.assertEqual(result.projection['candidate_key_pairs'],b*sum(rows))
            self.assertEqual(result.projection['padded_output_slots'],b*sum(max(0,512-c) for c in rows))
            self.assertEqual(result.projection['scored_query_key_pairs'],b*q*length)
            self.assertEqual(result.work['cube_ops'],2*32*128*b*q*length)
            self.assertEqual(result.work['vec_ops'],(3*32-1)*b*q*length)
            self.assertEqual(result.projection['visibility_first_last'],(length-q+1,length))
            self.assertEqual(result.work['logical_write_bytes'],4*b*q*512)
            self.assertIsNone(estimate('candidate_filtered_indexer',(b,q,32,128,length,slots,512),
                                dict(attrs,score_policy='candidate_first')).latency_us)

    def test_prefill_closed_schedule_matches_explicit_round_robin(self):
        from operator_model import load_chip
        attrs=dict(implementation='li_pipeline_projection',block_size=8)
        for sku in ('Ascend950DT_9582','Ascend950PR_957b','Ascend970DT'):
            chip=load_chip(sku)
            for b,q in ((8,18),(19,126),(3,128),(70,157)):
                group_types=[min(4,q-j) for j in range(0,q,4)]
                group_times={}
                for nr in set(group_types):
                    plan=estimate('candidate_filtered_indexer',(1,nr,32,128,20991,2048,512),
                                  attrs,chip=chip).projection
                    group_times[nr]=sum(t for s,t in plan['stages_us'].items() if s!='additional_shared_service')
                lanes=[0.]*chip.cube_cores
                for i,nr in enumerate(b*group_types):
                    lanes[i%len(lanes)]+=group_times[nr]
                result=estimate('candidate_filtered_indexer',(b,q,32,128,20991,2048,512),attrs,chip=chip)
                local=sum(t for s,t in result.projection['stages_us'].items() if s!='additional_shared_service')
                self.assertAlmostEqual(local,max(lanes),places=8)
                self.assertAlmostEqual(sum(result.projection['stages_us'].values()),result.latency_us)

    def test_prefill_long_table_has_constant_number_of_geometry_evaluations(self):
        from operator_model.operators.attention import indexer
        q=(1<<20)-1
        with patch.object(indexer,'_sort_row',wraps=indexer._sort_row) as sort:
            r=estimate('candidate_filtered_indexer',(8,q,32,128,1<<20,2048,512),
                       dict(implementation='li_pipeline_projection',block_size=8))
        self.assertEqual(r.status,'extrapolated')
        self.assertLessEqual(sort.call_count,8)
        self.assertIsNone(r.projection['candidate_keys_per_request'])
        self.assertEqual(r.projection['mmads_m_group_runs_per_request'],[(128,q//4),(128,1)])
