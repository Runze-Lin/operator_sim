import unittest
from dataclasses import replace

from operator_model.chips import load_chip
from operator_model.operators.vector.router import resource_router, router_work


class RouterScheduleTests(unittest.TestCase):
    def test_logical_tensors_and_lane_counts(self):
        shape = (57, 896, 16)
        account = router_work(shape)
        self.assertEqual(account['logical_read_bytes'], 58*896*4)
        self.assertEqual(account['logical_write_bytes'], 57*16*8)
        for sku, active, rows in [('Ascend950DT_9582', 57, 1),
                                  ('Ascend950PR_957b', 29, 2),
                                  ('Ascend970DT', 57, 1)]:
            us, cube, vector, diagnostics = resource_router(shape, load_chip(sku))
            self.assertEqual((cube, vector), (0, active))
            self.assertEqual(diagnostics['target_plan']['rows_per_busy_core'], rows)
            self.assertAlmostEqual(us, sum(diagnostics['stages_us'].values()))

    def test_resource_sensitivity(self):
        chip = load_chip('Ascend950DT_9582')
        shape = (4096, 896, 16)
        default = resource_router(shape, chip)[0]
        self.assertGreater(resource_router(shape, replace(chip,hbm_bandwidth_gbs=10))[0], default)
        self.assertGreater(resource_router(shape, replace(chip,vector_cores=1))[0], default)
        with self.assertRaises(ValueError):
            router_work([1,256,257])
