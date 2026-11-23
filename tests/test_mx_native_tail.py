"""Native V3 output ownership, fragment and buffer invariants at public entry points."""
import json
import unittest

from operator_model import estimate
from operator_model.resources import chip_file


class NativeWindowTests(unittest.TestCase):
    def native(self,shape,chip):
        return estimate('matmul',shape,dtype='mxfp4',chip=chip)

    def test_native_tail_matches_two_geometry_families(self):
        for m,n,expected_split in [(11,8192,7),(13,8448,5)]:
            with self.subTest(shape=[m,n]):
                result=self.native((m,n,1536),'Ascend950PR_957b')
                plan=result.projection['target_plan']
                work=plan['native_asw']
                self.assertEqual(work['mmad_k'],256)
                self.assertEqual(work['l1_k'],512)
                self.assertEqual(work['tail_split'],[1,expected_split])
                self.assertEqual(sum(h*w*count for h,w,count in work['rectangles']),m*n)
                self.assertEqual(result.work['implementation_work']['candidate_cube_k_fragments'],
                                 sum(row[2] for row in work['rectangles'])*6)
                self.assertEqual(result.projection['traffic_bytes']['l2_read'],(m*28+n)*(768+48))
                self.assertAlmostEqual(sum(result.projection['stages_us'].values()),result.latency_us)
        self.assertEqual(self.native((11,8192,1536),'Ascend950PR_957b').projection['target_plan']
                         ['native_asw']['tail_rectangles'],[(11,34,4),(11,37,24)])

    def test_target_resources_change_buffers_and_stream_prices(self):
        shape=(11,8192,1536)
        dt=self.native(shape,'Ascend950DT_9582')
        self.assertEqual(dt.projection['target_plan']['l1_page_k'],1024)
        self.assertFalse(dt.projection['target_plan']['native_asw']['tail_rectangles'])
        chip=json.loads(chip_file('Ascend950PR_957b').read_text())
        chip['details']['mte2_bandwidth_gbs']=33.0
        slow=self.native(shape,chip)
        self.assertAlmostEqual(slow.projection['prices']['mte2_bytes_per_cycle_core'],20.0)
        self.assertGreater(slow.latency_us,self.native(shape,'Ascend950PR_957b').latency_us)
        self.assertEqual(self.native(shape,'Ascend970DT').status,'extrapolated')

    def test_geometry_is_bounded_on_a_large_grid(self):
        shape=(2**24+7,2**21+2,2048)
        native=self.native(shape,'Ascend950PR_957b').projection['target_plan']['native_asw']
        self.assertLess(len(native['rectangles']),40)
        self.assertEqual(sum(h*w*count for h,w,count in native['rectangles']),shape[0]*shape[1])


if __name__=='__main__':
    unittest.main()
