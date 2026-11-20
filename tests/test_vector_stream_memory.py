from dataclasses import replace
import unittest

from operator_model import load_chip
from operator_model.scheduling.memory import stream_memory


class MixedStreamMemoryTest(unittest.TestCase):
    def test_clean_and_dirty_residency_share_one_target_cache(self):
        chip=load_chip()
        capacity=chip.details['l2_bytes']
        # Cast reads half as many bytes as it writes. Once the working set
        # overflows, the final cache contains both clean and dirty lines.
        row=stream_memory(chip,64,capacity,2*capacity)
        self.assertAlmostEqual(row['l2_input_reserve_bytes'],capacity/3)
        self.assertAlmostEqual(row['assumed_retained_write_bytes'],2*capacity/3)
        self.assertAlmostEqual(row['modeled_external_bytes'],7*capacity/3)

    def test_request_issue_scales_with_participating_aivs_then_hits_off_chip_limit(self):
        c=replace(load_chip(),hbm_bandwidth_gbs=1500)
        one=stream_memory(c,1,2**20,2**20)
        two=stream_memory(c,2,2**20,2**20)
        all_cores=stream_memory(c,c.vector_cores,2**20,2**20)
        self.assertEqual(two['effective_external_gbs'],2*one['effective_external_gbs'])
        self.assertEqual(all_cores['effective_external_gbs'],1500)
        self.assertEqual(one['modeled_external_bytes'],two['modeled_external_bytes'])
