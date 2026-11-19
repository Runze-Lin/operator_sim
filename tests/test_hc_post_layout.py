import unittest
from operator_model import OperatorStream, estimate
from operator_model.workload import account_work


class HcPostTest(unittest.TestCase):
    def test_matches_pinned_deepseek_binding(self):
        attrs = {"implementation":"vllm_hc_post"}
        small = estimate("mhc_post", (1,3,4,5120), attrs)
        self.assertEqual((small.status, small.active_aiv), ("extrapolated", 3))
        self.assertAlmostEqual(small.latency_us, 5.08420795755)
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

    def test_vllm_binding_has_its_own_stream_history(self):
        attrs = {"implementation":"vllm_hc_post"}
        stream = OperatorStream()
        shape = (1,3,4,5120)
        event = stream.advance("mhc_post",shape,attrs)
        self.assertEqual(event, estimate("mhc_post",shape,attrs))
        # Switching to the built-in kernel cannot inherit the extension's lanes.
        self.assertEqual(stream.preview("mhc_post",(3,4,5120)),estimate("mhc_post",(3,4,5120)))
        self.assertIsNone(estimate("mhc_post",shape,attrs,timing_mode="calibrated").latency_us)
        self.assertIsNone(estimate("mhc_post",(1,7,4,4112),attrs).latency_us)
