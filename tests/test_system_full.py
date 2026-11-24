"""Distinguish device views, local expert routes, wire traffic and unknown operations."""
import unittest
from types import SimpleNamespace as NS

from operator_model import load_chip
from operator_model.system_routes import estimate_full_features
from operator_model.system_routes.communication import FabricScenario, price


class FullRoutes(unittest.TestCase):
    def test_metadata_and_new_name(self):
        attrs = dict(inputs=(),outputs=(),attributes=(),category="zero")
        known=estimate_full_features(NS(raw_name="aten.permute.default",**attrs),chip=load_chip())
        self.assertEqual((known.status,known.latency_us,known.work["logical_write_bytes"]),("structural",0,0))
        missing=estimate_full_features(NS(raw_name="new::unpriced",**attrs),chip=load_chip())
        self.assertIsNone(missing.latency_us)

    def test_tp_ring_scales_with_rank_group_and_link(self):
        t=NS(shape=(8192,64),dtype="torch.bfloat16")
        node=NS(raw_name="collective::all_reduce",inputs=(t,),outputs=(t,),
                communication=(("rank",0),("rank_groups",((0,2,4,6),)),("reduce_op","SUM")))
        def run(bw):
            return price(node,load_chip(),FabricScenario(bw,1))
        fast,slow=run(200),run(50)
        schedule=fast.work["schedule"]
        self.assertEqual((schedule["serial_steps"],schedule["remote_send_bytes"]), (6,1572864))
        self.assertGreater(slow.latency_us,fast.latency_us)
        self.assertEqual(fast.work["vec_ops"],393216)

    def test_moe_combine_rank_is_group_local_and_counts_only_valid_rows(self):
        output=NS(shape=(1,128),dtype="torch.bfloat16")
        facts=dict(phase="combine",codec="fp8_token",weighted_finalize_inside_boundary=True,
                   group=(4,6),ep_rank_id=1,send_payload_bytes=(256,128),recv_payload_bytes=(128,128),
                   send_scale_bytes=(8,4),recv_scale_bytes=(4,4),send_splits=(2,1),recv_splits=(1,1),
                   valid_rows=3,capacity_rows=400)
        def run(sem):
            node=NS(raw_name="npu::npu_moe_distribute_combine_v2",outputs=(output,),
                    attributes=(("semantic",tuple(sem.items())),))
            return price(node,load_chip(),FabricScenario())
        r=run(facts)
        self.assertEqual(r.work["communication_access"],264)
        self.assertEqual(r.work["schedule"]["remote_recv_bytes"],132)
        self.assertEqual(r.work["vec_ops"],3*2*128)
        self.assertEqual(run(dict(facts,capacity_rows=40000)).latency_us,r.latency_us)
        self.assertIsNone(run(dict(facts,ep_rank_id=2)).latency_us)
        with self.assertRaises(ValueError):
            FabricScenario(float("nan"),1)
