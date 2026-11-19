from dataclasses import replace
from unittest import TestCase

from operator_model.chips import load_chip
from operator_model.quant_fragments import fragment_costs
from operator_model.quant_resource import resource_quant
from operator_model.quant_scale import peak_scale_writers


class QuantFragmentWave(TestCase):
    def test_two_grains_and_target_independence(self):
        chip=load_chip()
        result=resource_quant((5,320),chip)
        self.assertEqual(result,resource_quant((5,320),replace(chip,name='future')))
        wave=result[-1]['fragment_wave']
        self.assertEqual((result[2],wave['shared_grains'],wave['peak_writers']),(10,2,7))
        crit=wave['critical']
        self.assertEqual(crit['extra_queue_cycles'],crit['scale_end_cycles']-crit['scale_ready_cycles'])
        self.assertGreater(crit['extra_queue_cycles'],0)

    def test_override_resource_and_unshared(self):
        chip=load_chip()
        self.assertIsNone(resource_quant((1,128),chip)[-1]['fragment_wave'])
        self.assertIsNone(resource_quant((1024,4096),chip)[-1]['fragment_wave'])
        params=fragment_costs()
        # The critical core changes with target retirement interval; aggregate
        # bytes and active cores remain determined by the kernel tiling.
        slow=replace(chip,details={**chip.details,'quant_fragment_grain_ii_cycles':3*params['grain_ii_cycles']})
        fast=resource_quant((5,320),chip); high=resource_quant((5,320),slow)
        self.assertGreater(high[0],fast[0])
        self.assertEqual(high[1:3],fast[1:3])

    def test_odd_scale_count_is_packed_in_pairs(self):
        chip=load_chip()
        d=resource_quant((4,96),chip)[-1]
        self.assertEqual(d['logical_write_bytes'],4*(96+4))
        self.assertEqual(d['assumed_offchip_write_bytes'],0)
        self.assertEqual(d['first_tile_cycles']['scale_bytes'],4)
        self.assertEqual(d['fragment_wave']['peak_writers'],4)
        self.assertEqual(peak_scale_writers(2_000_000_002,64,1_000_000_001,64),2)
