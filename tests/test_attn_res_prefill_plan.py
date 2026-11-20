import unittest
from operator_model import estimate
from operator_model.attn_res_prefill_plan import prefill_service_plan


class AttnResPrefillPlanTest(unittest.TestCase):
    def test_source_work_conservation(self):
        for chip in ('Ascend950DT_9582', 'Ascend950PR_957b', 'Ascend970DT'):
            shape = [512, 7168, 8, 1]
            p = prefill_service_plan(shape, chip)
            w = p['logical_work']
            self.assertEqual(w['dma_read_bytes'], 2*7168*(3*512+3*p['active_aiv']))
            self.assertEqual(w['dma_write_bytes'], 2*7168*2*512)
            self.assertGreaterEqual(p['conditional_schedule_us'], max(p['component_floor_us'].values()))
            forecast=estimate('attn_res_fwd',shape,{'addend':True,'optimize_prefill':True},chip=chip)
            self.assertEqual(forecast.status, 'extrapolated')
            self.assertGreaterEqual(forecast.latency_us, p['component_floor_us']['cold_input_hbm'])
            self.assertIsNone(p['qualified_latency_us'])

    def test_gate(self):
        with self.assertRaises(ValueError):
            prefill_service_plan([1,7168,8,1], 'Ascend950DT_9582')
