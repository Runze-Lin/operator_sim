import json
from dataclasses import replace
from pathlib import Path
from unittest import TestCase

from operator_model import estimate, load_chip
from operator_model.operators.matmul.resources import resource_matmul


ROOT = Path(__file__).resolve().parents[1]
DT = 'Ascend950DT_9582'
PR = 'Ascend950PR_957b'
FUTURE = 'Ascend970DT'


def read(path):
    return json.loads((ROOT / path).read_bytes())


class PublicRetiledMatmulTests(TestCase):

    def test_chip_resources_choose_the_grid_and_price_the_merge(self):
        shape = [3585, 32, 5120]
        dt, pr = load_chip(DT), load_chip(PR)
        clean = estimate('matmul', shape, chip=dt)
        renamed = estimate('matmul', shape, chip=replace(dt, name='scenario'))
        self.assertEqual(clean.projection, renamed.projection)
        self.assertGreater(clean.active_aiv, 0)
        self.assertEqual(estimate('matmul', shape, chip=pr).active_aiv, 0)
        self.assertEqual(estimate('matmul', shape, chip=FUTURE).status, 'extrapolated')
        self.assertEqual(estimate('matmul', shape, chip=FUTURE).active_aiv, 0)
        slow = replace(dt, vector_tops={**dt.vector_tops, '32': dt.vector_tops['32'] / 2})
        slow_result = estimate('matmul', shape, chip=slow)
        self.assertGreater(slow_result.projection['stages_us']['merge_issue'], clean.projection['stages_us']['merge_issue'])
        self.assertGreater(slow_result.latency_us, clean.latency_us)
        biased = estimate('addmm', shape, chip=dt)
        self.assertAlmostEqual(biased.latency_us, resource_matmul(shape, dt, column_bias=True)[0])
        self.assertNotIn('implementation_work', biased.work)
