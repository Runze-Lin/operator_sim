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

    def test_loop_price_changes_only_between_tiles_in_a_fragment(self):
        chip = load_chip()
        opts = {"table_pages":8201}
        for shape,changes in [((1,1,32,128,1024,2048,512),False),
                              ((1,4096,32,128,1048576,2048,512),True)]:
            current,_,plan = price(shape,chip,**opts)
            frozen,_,_ = price(shape,chip,scalar_lookup_cycles=64.,**opts)
            self.assertAlmostEqual(plan["interval_cycles"],109555/31)
            if changes:
                extra_intervals = 128*31
                self.assertAlmostEqual(current-frozen,extra_intervals*(109555/31-plan["resource_interval_cycles"])/1650)
            else:
                self.assertEqual(current,frozen)
