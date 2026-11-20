import json
from dataclasses import replace
from pathlib import Path
from unittest import TestCase

from operator_model import estimate, load_chip
from operator_model.matmul_resource import resource_matmul


ROOT = Path(__file__).resolve().parents[1]
DT = 'Ascend950DT_9582'
PR = 'Ascend950PR_957b'
FUTURE = 'Ascend970DT'


def read(path):
    return json.loads((ROOT / path).read_bytes())


class PublicRetiledMatmulTests(TestCase):
    def test_physical_work_and_reduction_agree_with_archived_geometry(self):
        rows = read('results/v0.28/matmul/multi/score.json')['rows']
        cases = [(r['shape'], r['sku'], r['geometry']) for r in rows if r['role'] == 'used in development']
        for m, sku, path in [(2048, DT, 'ca-2048-geometry'), (2048, PR, 'ca-pr-2048-geometry'),
                             (4096, DT, 'chunk-4096-geometry')]:
            cases.append(([m, 32, 5120], sku, read(f'results/v0.29/matmul/{path}.json')['geometry']))
        for shape, sku, observed in cases:
            with self.subTest(shape=shape, sku=sku):
                result = estimate('matmul', shape, chip=sku)
                plan = result.projection['target_plan']
                self.assertEqual((result.active_aic, result.active_aiv), (observed['active_aic'], observed['active_aiv']))
                self.assertEqual(plan['productive_aic'], observed['productive_aic'])
                self.assertEqual(plan['output_tiles'], observed['groups'])
                self.assertEqual([plan['k_splits']], observed['splits_per_output'])
                self.assertEqual(plan['mmads_total'], observed['actual_mmad'])
                self.assertEqual((plan['k_splits'] - 1) * plan['max_segment_k'] + plan['tail_segment_k'], shape[2])
                self.assertEqual(result.work['logical_write_bytes'], observed['all_output_bytes'])
                work = result.work['implementation_work']
                self.assertEqual(work['fp32_partial_write_bytes'], sum(core['fixp_bytes'] for core in observed['cores']))
                self.assertEqual(work['fp32_partial_read_bytes'], work['fp32_partial_write_bytes'])
                self.assertAlmostEqual(result.latency_us, sum(result.projection['stages_us'].values()))

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

    def test_pending_shapes_reproduce_preexisting_frozen_predictions(self):
        for r in read('results/v0.29/matmul/timing-freeze.json')['rows']:
            for sku, price in r['predictions'].items():
                with self.subTest(shape=r['shape'], sku=sku):
                    result = estimate('matmul', r['shape'], chip=sku)
                    self.assertAlmostEqual(result.latency_us, price['latency_us'], places=9)
