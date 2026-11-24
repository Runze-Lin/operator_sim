import unittest

from operator_model import estimate, load_chip
from operator_model.operators.attention.flash_sparse import IMPLEMENTATION, price
from operator_model.workload import account_work


class FlashPackedCandidate(unittest.TestCase):
    def test_amount_with_causal_prefix_and_short_final_tile(self):
        for b,q,length in [(3,4,516), (2,2031,2061), (1,29,16403), (1,4096,1048576)]:
            s = (b,q,32,128,length,2048,512)
            visible = range(length-q+1,length+1)
            live = [min((v+7)//8,2048) for v in visible]
            valid = [min(v,8*c-(-v%8)) for c,v in zip(live,visible)]
            expected_tiles = b*sum((c+63)//64 for c in live)
            e = estimate("candidate_filtered_indexer",s,{"implementation":IMPLEMENTATION},dtype="mxfp4")
            self.assertEqual(e.status,"extrapolated")
            self.assertEqual(e.projection["candidate_tiles"],expected_tiles)
            self.assertEqual(e.projection["candidate_key_pairs"],b*sum(valid))
            self.assertEqual(e.projection["scored_query_key_pairs"],512*expected_tiles)
            self.assertEqual(e.work["cube_ops_by_dtype"],{"mxfp4":2*32*128*b*sum(valid)})
            self.assertEqual(e.work,account_work("candidate_filtered_indexer",s,{"implementation":IMPLEMENTATION,"dtype":"mxfp4"}))
            self.assertAlmostEqual(e.latency_us,sum(e.projection["stages_us"].values()))

    def test_splits_and_allocated_table(self):
        s = (1,1,32,128,16384,2048,512)
        dt = load_chip()
        fast,_,fp = price(s,dt,table_pages=2048)
        slow,_,sp = price(s,dt,table_pages=2049)
        self.assertEqual((fp["address_path"],sp["address_path"]),("UB","scalar"))
        self.assertGreater(slow,fast)
        self.assertEqual(fp["largest_row_shards"],32)
        self.assertEqual(fp["split_rows"],1)
        for kwargs in (dict(pa_block_size=13),dict(table_pages=3),dict(scalar_lookup_cycles=-1)):
            with self.assertRaises(ValueError):
                price(s,dt,**kwargs)
        opts = {"implementation":IMPLEMENTATION}
        self.assertEqual(estimate("candidate_filtered_indexer",s,opts,dtype="bf16").status,"unsupported")
        self.assertEqual(estimate("candidate_filtered_indexer",s,opts,dtype="mxfp4",timing_mode="calibrated").status,"unavailable")

    def test_source_ld_streaming_carry_boundaries(self):
        chip = load_chip()
        shape = (1,1,32,128,1 << 20,2048,512)
        for parts in (2,7,8,14,15,21,22,28,29,32):
            with self.subTest(parts=parts):
                _, _, plan = price(shape,chip,candidate_limit=parts*64)
                self.assertEqual(plan["largest_row_shards"],parts)
                passes = len(range(0,parts,7))
                self.assertEqual(plan["merge_passes"],passes)
                self.assertEqual(plan["merge_scored_entries"],512*(parts+passes-1))
                self.assertAlmostEqual(sum(plan["merge_service_cycles"].values())/1650,
                                       plan["stages_us"]["merge"])

    def test_compiled_metadata_remaining_work_and_no_ld(self):
        shape = (1,32,32,128,1 << 20,2048,512)
        _, _, pr = price(shape,load_chip("Ascend950PR_957b"))
        self.assertEqual((pr["split_rows"],pr["largest_row_shards"],pr["merge_passes"]),(27,2,1))
        _, _, dt = price(shape,load_chip())
        self.assertEqual(dt["split_rows"],0)
        self.assertEqual(sum(dt["merge_service_cycles"].values()),0)
        self.assertEqual(dt["merge_scored_entries"],0)

    def test_single_row_preserves_original_issue_model(self):
        chip = load_chip()
        for qlen in (1024,1048576):
            shape = (1,1,32,128,qlen,2048,512)
            low,_,plan = price(shape,chip,table_pages=8201,candidate_order_overlap=0)
            high,_,_ = price(shape,chip,table_pages=8201,candidate_order_overlap=1)
            self.assertFalse(plan["parallel_row_issue"])
            self.assertAlmostEqual(plan["interval_cycles"],109555/31)
            self.assertEqual(low,high)

    def test_parallel_issue_depends_on_rank_and_metadata_phase(self):
        for suffix,expected_phase in [("DT_9582",1),("PR_957b",1/27)]:
            chip = load_chip("Ascend950"+suffix)
            shape = (1,32,32,128,1<<20,2048,512)
            opts = {"implementation":IMPLEMENTATION, "table_pages":8201}
            values = []
            for overlap in (0,None,1):
                attrs = dict(opts)
                if overlap is not None:
                    attrs["candidate_order_overlap"] = overlap
                e = estimate("candidate_filtered_indexer",shape,attrs,dtype="mxfp4",chip=chip)
                self.assertTrue(e.projection["parallel_row_issue"])
                self.assertAlmostEqual(e.projection["team_phase_pair_fraction"],expected_phase)
                self.assertEqual(e.work,account_work("candidate_filtered_indexer",shape,attrs|{"dtype":"mxfp4"}))
                self.assertAlmostEqual(sum(e.projection["stages_us"].values()),e.latency_us)
                values.append(e.latency_us)
            self.assertLess(values[0],values[1])
            self.assertLess(values[1],values[2])
            self.assertAlmostEqual(values[1],(values[0]+values[2])/2)

    def test_manual_resource_and_short_table_ignore_candidate_order(self):
        for attrs in ({"table_pages":2048},{"table_pages":8201,"scalar_lookup_cycles":64}):
            shape = (2,64,32,128,65536,2048,512)
            prices = [price(shape,load_chip(),candidate_order_overlap=r,**attrs) for r in (0,1)]
            self.assertEqual(prices[0][0],prices[1][0])
            self.assertFalse(prices[0][2]["parallel_row_issue"])
        for bad in (-1,2,True,float("nan"),"shared"):
            with self.assertRaises(ValueError):
                price(shape,load_chip(),candidate_order_overlap=bad)
            with self.assertRaises(ValueError):
                account_work("candidate_filtered_indexer",shape,dict(implementation=IMPLEMENTATION,
                             dtype="mxfp4",candidate_order_overlap=bad))

    def test_same_rank_in_different_documents_does_not_collide(self):
        # Each team owns a different complete document with two query rows.
        shape = (32,2,32,128,1<<20,2048,512)
        low,_,p = price(shape,load_chip(),candidate_order_overlap=0)
        high,_,_ = price(shape,load_chip(),candidate_order_overlap=1)
        self.assertTrue(p["parallel_row_issue"])
        self.assertEqual(p["team_phase_pair_fraction"],0)
        self.assertEqual(low,high)

    def test_compressed_phase_sampling_matches_explicit_ragged_rows(self):
        from collections import Counter

        # Exercise multi-document owners, a partial last tile, and visible
        # length crossing the 512/1024/1536 tile boundaries inside a request.
        for b,q,length,limit in [(3,61,547,2048), (7,84,1086,193),
                                 (2,32,1057,2048), (9,80,1558,202), (17,6,2191,109)]:
            shape = (b,q,32,128,length,2048,512)
            with self.subTest(shape=shape):
                _,_,plan = price(shape,load_chip("Ascend950PR_957b"),table_pages=8201,
                                 candidate_limit=limit)
                work = []
                for req in range(b):
                    for visible in range(length-q+1,length+1):
                        count = min((visible+7)//8,limit)
                        work.extend((req,tile) for tile in range((count+63)//64))
                ranges = []
                cursor = 0
                live = plan["target_plan"]["active_aic"]
                for left in range(live,0,-1):
                    count = int((len(work)-cursor)/left+0.5)
                    ranges.append(work[cursor:cursor+count])
                    cursor += count
                self.assertEqual(cursor,len(work))
                depth = min(map(len,ranges))
                fractions = []
                for step in {0,(depth-1)//2,depth-1}:
                    counts = Counter(r[step] for r in ranges)
                    fractions.append(sum(v*(v-1) for v in counts.values())/(live*(live-1)))
                self.assertAlmostEqual(sum(fractions)/len(fractions),plan["team_phase_pair_fraction"])

    def test_locality_penalty_is_paid_only_on_the_first_document_fragment(self):
        # Same first fragment, then zero, one or 127 more fully prefetched rows.
        # More query rows do not multiply the first-fragment layout penalty.
        chip = load_chip()
        deltas = []
        for q in (32,64,4096):
            shape = (1,q,32,128,1<<20,2048,512)
            low,_,lo = price(shape,chip,candidate_order_overlap=0)
            high,_,hi = price(shape,chip,candidate_order_overlap=1)
            self.assertAlmostEqual(lo["carried_fragment_interval_cycles"],1343.6129032258063)
            self.assertEqual(hi["carried_fragment_interval_cycles"],lo["carried_fragment_interval_cycles"])
            self.assertGreater(hi["first_fragment_interval_cycles"],lo["first_fragment_interval_cycles"])
            deltas.append(high-low)
        self.assertAlmostEqual(deltas[0],deltas[1])
        self.assertAlmostEqual(deltas[0],deltas[2])

    def test_prefetched_partial_rows_respect_target_transfer_floor(self):
        from dataclasses import replace

        chip = load_chip("Ascend950PR_957b")
        slow_mte = replace(chip,details=dict(chip.details,mte2_bandwidth_gbs=12))
        shape = (1,32,32,128,1<<20,2048,512)
        for overlap in (0,1):
            _,_,p = price(shape,slow_mte,candidate_order_overlap=overlap)
            self.assertGreater(p["split_rows"],0)
            self.assertGreater(p["tile_service_cycles"]["cube_read"],1343.6129032258063)
            self.assertEqual(p["carried_fragment_interval_cycles"],p["tile_service_cycles"]["cube_read"])
