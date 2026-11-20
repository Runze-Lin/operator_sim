"""Mechanism invariants for the internal frozen candidate."""
from pathlib import Path
import math
import json,sys,unittest
HERE=Path(__file__).resolve().parent
sys.path[:0]=[str(HERE),str(HERE.parents[2])]
from schedule import schedule
D,P,F='Ascend950DT_9582','Ascend950PR_957b','Ascend970DT'

class PostScheduleTest(unittest.TestCase):
    def test_independent_shapes_and_chips_match_frozen(self):
        reg=json.loads((HERE.parents[2]/'results/v0.35/post/fresh-freeze.json').read_bytes())
        for chip in (D,P,F):
            for x in reg['rows'][chip]:
                q=schedule(x['shape'],chip)
                # The frozen Python and the system libm may round a float one ULP apart.
                self.assertAlmostEqual(q['us'], x['candidate_us'], delta=2 * math.ulp(x['candidate_us']))
                self.assertEqual(q['states'].keys(), x['conditional_us'].keys())
                for name, expected in x['conditional_us'].items():
                    self.assertAlmostEqual(q['states'][name], expected, delta=2 * math.ulp(expected))
    def test_conditional_front_end_is_charged_once(self):
        q=schedule([65,4,5120],D);long=schedule([1024,4,5120],D)
        self.assertEqual(q['fragments']['count'],2)
        self.assertEqual(len({v['later_writes'] for v in q['prices'].values()}),1)
        self.assertGreater(q['prices']['first']['first_write'],q['prices']['repeat']['first_write'])
        self.assertGreater(long['prices']['reuse']['later_writes'],q['prices']['reuse']['later_writes'])
        self.assertAlmostEqual(q['prices']['first']['first_write']-q['prices']['reuse']['first_write'],
                               long['prices']['first']['first_write']-long['prices']['reuse']['first_write'],delta=0.12)
    def test_point_does_not_see_prior(self):
        for chip in (D,P,F):
            for sh in ([69,4,176],[2,4,368],[8*131072,4,5120]):
                q=schedule(sh,chip);d=q['states']
                self.assertTrue(min(d.values())<=q['us']<=max(d.values()))
                if q['route']=='post_token':
                    self.assertAlmostEqual(q['us']**2,min(d.values())*max(d.values()),places=5)
if __name__=='__main__':unittest.main()
