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
