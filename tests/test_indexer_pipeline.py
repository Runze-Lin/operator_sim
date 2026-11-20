"""Issue counts, resource sensitivity and supported routes for dense indexer."""
import unittest
from dataclasses import replace

from operator_model import estimate, load_chip


class IndexerPipeline(unittest.TestCase):
    def test_partial_query_group(self):
        x = estimate('lightning_indexer', [1, 9, 4096, 8, 128])
        plan = x.projection['target_plan']
        self.assertEqual((x.active_aic, x.active_aiv), (3, 5))
        self.assertEqual(plan['mmads_by_shape'], [[16, 128, 128, 32], [32, 128, 128, 64]])
        self.assertEqual(plan['modeled_histogram_instructions'], 1152)
        self.assertAlmostEqual(sum(x.projection['stages_us'].values()), x.latency_us)

    def test_causal_mask_and_physical_last_page(self):
        x = estimate('lightning_indexer', [1, 9, 1029, 8, 128])
        counts = x.projection['target_plan']['mmads_by_shape']
        self.assertEqual(counts, [[16, 5, 128, 1], [16, 128, 128, 8],
                                  [32, 5, 128, 1], [32, 128, 128, 16]])

    def test_scratch_carry_across_histogram_chunks(self):
        x = estimate('lightning_indexer', [1, 1, 8193, 8, 128])
        self.assertEqual(x.projection['target_plan']['modeled_histogram_instructions'], 8*(32+1+2))
        self.assertTrue(x.projection['timing_diagnostics']['unvalidated_multiple_histogram_chunks'])
        self.assertEqual(x.status, 'extrapolated')

    def test_target_resources(self):
        chip = load_chip()
        shape = [1, 9, 16384, 8, 128]
        base = estimate('lightning_indexer', shape, chip=chip)
        for changes in (dict(hbm_bandwidth_gbs=1), dict(details={**chip.details, 'mte2_bandwidth_gbs':1}),
                        dict(vector_tops={**chip.vector_tops, '32':0.2}),
                        dict(cube_tops={**chip.cube_tops, '16':1})):
            result = estimate('lightning_indexer', shape, chip=replace(chip, **changes))
            self.assertGreater(result.latency_us, base.latency_us)
            self.assertEqual(result.work, base.work)

    def test_contract_and_chip_scope(self):
        for shape, attrs in (([1, 129, 1024, 8, 128], {}), ([1, 2, 3072, 8, 128], {'sparse_count':1024})):
            x = estimate('lightning_indexer', shape, attrs)
            self.assertNotEqual((x.projection or {}).get('method'), 'query_group_kv_pipeline')
        self.assertEqual(estimate('lightning_indexer', [1, 9, 8192, 8, 128]).status, 'extrapolated')
