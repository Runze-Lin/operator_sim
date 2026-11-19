import unittest

from operator_model.predictor import VECTOR, CUBE, FUSION
from operator_model.workload import account_work, default_dtype


class TestWorkloadAccounting(unittest.TestCase):
    def test_each_reference_route_counts_its_tensor_contract(self):
        shapes = {
            "quant": (9, 128), "rmsnorm": (2, 6144), "matmul": (9, 128, 256),
            "addmm": (9, 128, 256), "fa": (9, 512, 64),
            "lightning_indexer": (1, 9, 1024, 8, 128), "mhc": (8, 4, 512),
            "mhc_pre": (8, 4, 512), "mhc_pre_sinkhorn": (8, 4, 4096),
            "mhc_post": (8, 4, 512), "mhc_sinkhorn": (8, 4),
            "kda_recurrent": (1, 8, 1, 1, 128, 256),
            "kda_chunk": (1, 64, 1, 1, 128, 128),
            "moe_gating_top_k": (2, 896, 16),
        }
        for op in ("grouped_matmul_swiglu_quant_v2", "moe_ffn", "gmm2"):
            shapes[op] = (32, 256, 128, 2)
        self.assertEqual(set(shapes), VECTOR | CUBE | FUSION)
        for op, shape in shapes.items():
            attrs = {"expert_counts": [32, 0]} if len(shape) == 4 else {}
            with self.subTest(op=op):
                work = account_work(op, shape, attrs)
                self.assertEqual(work["dtype"], default_dtype(op))
                self.assertGreater(work["cube_ops"] + work["vec_ops"], 0)
                self.assertEqual(work["vec_ops"], sum(work["vec_ops_by_class"].values()))
                self.assertGreater(work["logical_read_bytes"], 0)
                self.assertGreater(work["logical_write_bytes"], 0)
                self.assertIsNone(work["physical_hbm_bytes"])

    def test_fa_counts_stats_not_quadratic_score_storage(self):
        work = account_work("fa", (9, 512, 64, 2, 4))
        self.assertEqual(work["cube_ops"], 4*9*512*64*8)
        self.assertEqual(work["logical_write_bytes"], 8*9*(2*64+64))
        self.assertEqual(work["logical_read_bytes"], 8*2*64*(9+2*512))

    def test_mixed_precision_and_fused_bias(self):
        work = account_work("addmm", (9, 128, 256))
        self.assertEqual(work["cube_ops_by_dtype"], {"bf16": 2*9*128*256})
        self.assertEqual(work["vec_ops"], 0)
        self.assertEqual(work["fused_bias_additions"], 9*128)
        rms = account_work("rmsnorm", (2, 6144), {"resident_aiv": 64})
        self.assertEqual(rms["vec_ops_by_dtype"], {"fp32": 2*(4*6144+2)})
        self.assertEqual(rms["logical_write_bytes"], 2*6144*2+8)

    def test_composed_moe_includes_intermediate_and_active_weights(self):
        shape = (32, 256, 128, 2)
        attrs = {"expert_counts": [32, 0]}
        components = [account_work(op, shape, attrs) for op in ("grouped_matmul_swiglu_quant_v2", "gmm2")]
        chain = account_work("moe_ffn", shape, attrs)
        for field in ("cube_ops", "vec_ops", "logical_write_bytes", "logical_read_bytes"):
            self.assertEqual(chain[field], sum(w[field] for w in components))
        second = components[1]
        self.assertEqual(second["logical_read_bytes"], (32+256)*(128+4)+16)
        self.assertEqual(second["logical_write_bytes"], 32*256*2)
        skew = account_work("moe_ffn", shape, {"expert_counts": [1, 31]})
        self.assertEqual(skew["cube_ops"], chain["cube_ops"])
        self.assertGreater(skew["logical_read_bytes"], chain["logical_read_bytes"])

    def test_kda_partial_bytes_and_mhc_aux(self):
        shape = (1, 64, 1, 1, 128, 128)
        work = account_work("kda_chunk", shape, {"use_gate_in_kernel": True})
        self.assertIsNone(work["logical_read_bytes"])
        pre = account_work("mhc_pre", (1, 4, 256))
        self.assertEqual(pre["logical_write_bytes"], 512+4*(4+16+1+24+4))


    def test_split_mhc_keeps_legacy_large_token_contract(self):
        from operator_model.predictor import predict
        for tokens in (256, 257, 4096):
            for width in (16, 16384):
                shape = (tokens, 4, width)
                for op in ("mhc_pre", "mhc"):
                    with self.subTest(shape=shape, op=op):
                        self.assertGreater(predict(op, shape)["predicted_us"], 0)
                        work = account_work(op, shape)
                        self.assertEqual(work["cube_ops"], 2*tokens*4*width*24)
                        self.assertGreater(work["logical_write_bytes"], 0)

    def test_sinkhorn_one_iteration_or_two(self):
        for iters, expected in [(1, 120), (2, 184)]:
            work = account_work("mhc_sinkhorn", (1, 4), {"num_iters": iters})
            self.assertEqual(work["vec_ops_by_dtype"], {"fp32": expected})
            self.assertEqual(work["logical_read_bytes"], 64)
            self.assertEqual(work["logical_write_bytes"], 64)
