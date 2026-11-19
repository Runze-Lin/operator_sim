import unittest
from operator_model import estimate
from operator_model.workload import account_work


class HcPostTest(unittest.TestCase):
    def test_matches_pinned_deepseek_binding(self):
        attrs = {"implementation":"vllm_hc_post"}
        small = estimate("mhc_post", (1,3,4,5120), attrs)
        self.assertEqual((small.status, small.latency_us, small.active_aiv), ("unavailable", None, 3))
        self.assertEqual(small.work["logical_read_bytes"], 153840)
        self.assertEqual(small.work["logical_write_bytes"], 122880)
        self.assertEqual(small.work["vec_ops"], 552960)
        self.assertEqual(small.work, account_work("mhc_post",(1,3,4,5120),attrs))
        plan = small.projection["target_plan"]
        self.assertEqual((plan["full_d_tiles"],plan["tail_elements"], plan["output_writes"]), (1,1024,6))
        full = estimate("mhc_post", (1,8*32768,4,7168), attrs, chip="Ascend950PR_957b")
        self.assertEqual((full.active_aiv,full.projection["target_plan"]["max_rows_per_core"]), (56,4682))
        self.assertEqual(estimate("mhc_post",(3,4,5120),attrs).status,"unsupported")
        self.assertEqual(estimate("mhc_post",(1,3,4,5120),attrs,dtype="fp32").status,"unsupported")
