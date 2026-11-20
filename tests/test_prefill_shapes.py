import unittest

from examples.workloads.prefill import CHIPS, MODELS, Scenario, contracts, presets


def get(rows, op):
    return next(r for r in rows if r["op"] == op)


class PrefillShapeTest(unittest.TestCase):
    def test_packed_and_sharded_heads(self):
        scenario = Scenario("Kimi-K3", 131072, chunk=2048, step=511)
        doc = contracts(scenario)
        rows = doc["contracts"]
        self.assertEqual((doc["tokens_per_rank"], doc["q_per_request"], doc["kv_after_per_request"]),
                         (2048, 256, 131072))
        self.assertEqual(doc["cu_seqlens"], list(range(0,2049,256)))
        beta = get(rows, "kda.beta.sigmoid")
        self.assertEqual(beta["standalone_call"]["shape"], [2048,12])
        self.assertEqual(len(beta["layers"]),69)
        self.assertEqual(list(beta["standalone_prediction"]),list(CHIPS))
        mla = get(rows,"mla.causal_prefill")
        self.assertEqual(len(mla["layers"]),24)
        self.assertEqual(mla["tensors"]["q"],[2048,12,192])
        self.assertEqual(mla["tensors"]["cache_logical"],[8,131072,1,576])
        self.assertEqual(get(rows,"moe.routed_grouped_ffn")["tensors"]["w_down_logical"],[112,3584,3072])
        self.assertIn("lm_head.last_position", {r["op"] for r in rows})

    def test_flash_source_and_candidate_ownership(self):
        s = Scenario(MODELS[0],32768)
        rows = contracts(s)["contracts"]
        cast = get(rows,"indexer.project_weights.float")
        self.assertEqual(cast["standalone_call"]["shape"], [262144,32])
        self.assertEqual(cast["layers"],[2,8,14,20,24,28,32,36])
        self.assertEqual(get(rows,"indexer.candidate_blocks")["tensors"]["block_ids"],[262144,1,2048])
        a = next(i for i in rows if i["op"]=="dsa.sparse_attention" and 26 in i["layers"])
        self.assertEqual((a["kv_source"],a["index_source"]),(20,24))
        self.assertEqual(a["tensors"]["q"],[262144,8,512])

    def test_ep_histogram_conservation_and_chunk_tail(self):
        s = Scenario("DeepSeek-V4-Pro",1000,batch=8,tp=1,dp=8,chunk=2048,step=3)
        self.assertEqual((s.q,s.t,s.prefix),(232,1856,768))
        hist = [s.t*s.dp]*6 + [0]*378
        rows = contracts(s,hist)["contracts"]
        cast = get(rows,"indexer.prepare_dsa_weights.float")
        self.assertEqual(cast["standalone_call"]["shape"],[1856,64])
        self.assertEqual(len(cast["layers"]),30)
        gmm = get(rows,"moe.routed_grouped_ffn")
        self.assertEqual(gmm["local_expert_rows"],hist[:48])
        self.assertEqual(gmm["tensors"]["x"][0],s.t*s.dp*6)
        with self.assertRaises(ValueError):
            contracts(s,hist[:-1])
        with self.assertRaises(ValueError):
            Scenario(MODELS[0],1048577)
        all_scenarios = list(presets())
        self.assertEqual(len(all_scenarios),54)
        self.assertEqual(len(set(all_scenarios)),54)
