from unittest import TestCase

from operator_model.operators.kda.prefill import plan_kimi_kda_prefill


class KimiKdaDispatchTest(TestCase):
    def test_packed_boundaries_change_route_at_identical_payload(self):
        small = plan_kimi_kda_prefill([21824], attention_tp=8)
        split = plan_kimi_kda_prefill([1, 0, 21823], attention_tp=8)
        self.assertEqual(small['work_items'], 4092)
        self.assertEqual(small['aclnn'], 'aclnnChunkKdaFwd')
        self.assertEqual(len(small['main_kernels']), 4)
        self.assertIsNone(small['main_kernels_us'])
        self.assertEqual(split['work_items'], 4104)
        self.assertEqual(split['aclnn'], 'aclnnChunkKdaFwdV2')
        self.assertIsNone(split['main_kernels_us'])

    def test_transfers_and_explicit_time_scope(self):
        cfg = dict(attention_tp=8, initialized_sequences=3)
        pack = plan_kimi_kda_prefill([32768] * 8, **cfg)
        amount = 32768 * 8 * 12 * 128
        io = pack['graph_io']
        self.assertEqual(io['qk_l2norm_read_bytes'], 4*amount)
        self.assertEqual(io['state_gather_read_bytes_range'], (3*12*128*128*4,) * 2)
        self.assertEqual(io['state_scatter_write_bytes'], 8*12*128*128*4)
        self.assertEqual(pack['work_items'], 49152)
        self.assertAlmostEqual(pack['main_kernels_us'], sum(pack['main_stages_us'].values()))
        self.assertIsNone(pack['full_dispatch_us'])

    def test_capacity_and_empty_requests(self):
        full = plan_kimi_kda_prefill([524288]*8, attention_tp=8)
        self.assertEqual(full['status'], 'infeasible')
        self.assertGreater(full['minimum_payload_bytes'], full['capacity_bytes'])
        self.assertIsNone(full['main_kernels_us'])
        empty = plan_kimi_kda_prefill([0,0], attention_tp=8)
        self.assertIsNone(empty['aclnn'])
        self.assertEqual(empty['full_dispatch_us'], 0)
        with self.assertRaises(ValueError):
            plan_kimi_kda_prefill([5], attention_tp=5)
        with self.assertRaises(ValueError):
            plan_kimi_kda_prefill([512, 0], attention_tp=8, initialized_sequences=2)
