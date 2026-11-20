import random
import unittest

from development.validation.kda29.audit import plan


class FinalizeWork(unittest.TestCase):
    def test_physical_fixture(self):
        p=plan([1,1],1,32)['finalize_work']
        self.assertEqual(p['logical_ops_per_aic'],[33024,33024]+[0]*30)
        self.assertEqual(p['logical_cube_ops'],66048)

    def test_strided_assignment_for_ragged_segments(self):
        rng=random.Random(4103)
        for _ in range(100):
            lengths=[rng.randrange(0,194) for _ in range(rng.randrange(2,12))]
            heads=rng.randrange(1,10);cores=rng.choice((1,3,28,32,120))
            result=plan(lengths,heads,cores)
            if 'finalize_work' not in result:continue
            expected=[0]*cores;idx=0
            for length in lengths:
                for offset in range(0,length,64):
                    valid=min(length-offset,64)
                    for head in range(heads):
                        expected[idx%cores]+=2*valid*128*(128+valid);idx+=1
            self.assertEqual(result['finalize_work']['logical_ops_per_aic'],expected)

    def test_selector_boundaries(self):
        self.assertEqual(plan([0,0],12,32)['automatic_route'],'no_launch')
        self.assertEqual(plan([64],12,32)['v1_phases'],['full'])
        self.assertEqual(plan([2688]*8,12,32)['automatic_route'],'v1')
        self.assertEqual(plan([2689]*8,12,32)['automatic_route'],'v2')
        self.assertNotIn('finalize_work',plan([2689]*8,12,32))


if __name__=='__main__':unittest.main()
