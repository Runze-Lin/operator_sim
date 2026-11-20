import copy
from pathlib import Path
import tempfile
import unittest
from calibration.__main__ import HERE, freeze, ingest, read, score, seal
from calibration.msopprof import boundaries
from calibration.verify import verify

class ReceiptTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.receipt=read(HERE/'examples/matmul-ca.json')
        cls.frozen=freeze(read(HERE/'examples/cases.json'))

    def test_published_example(self):
        m=ingest(self.frozen,self.receipt)
        self.assertEqual(m['data_role'],'replay')
        self.assertAlmostEqual(m['ca_us'],15.5618181818)
        s=score(self.frozen,[m])
        self.assertEqual(s['pending'],[])
        self.assertAlmostEqual(s['summary'][0]['wape_pct'],.609156,places=4)
        with self.assertRaises(ValueError):score(self.frozen,[m,m])

    def test_reject_mismatch_or_partial(self):
        mutations=[
            lambda r:r['case'].update(chip='Ascend950PR_957b'),
            lambda r:r['correctness']['Y'].update(checked=1),
            lambda r:r['stages'][0]['launched_cores'].pop(),
            lambda r:r['stages'][0]['cores']['core0.cubecore0'].update(end_count=0),
            lambda r:r.update(timing='wrapper'),
            lambda r:r.update(complete=False),
        ]
        for mutate in mutations:
            r=copy.deepcopy(self.receipt); mutate(r)
            with self.subTest(mutate=mutate),self.assertRaises(ValueError):ingest(self.frozen,r)

    def test_fingerprint_and_chronology(self):
        fake=copy.deepcopy(self.frozen)
        fake['cases'][0]['shape'][0]+=1
        with self.assertRaises(ValueError):ingest(fake,self.receipt)
        reg=dict(self.frozen)
        reg.pop('sha256')
        reg['created_utc']='2020-01-01T00:00:00+00:00'
        reg=seal(reg)
        with self.assertRaises(ValueError):ingest(reg,self.receipt)
        receipt={**self.receipt,'freeze_sha256':reg['sha256']}
        self.assertEqual(ingest(reg,receipt)['data_role'],'prospective')

    def test_partial_trace_is_not_valid(self):
        p=Path(tempfile.gettempdir())
        with tempfile.NamedTemporaryFile(dir=p) as f:
            f.write(b'{"traceEvents":[{"dur":1,"name":"END","ph":"X"');f.flush()
            with self.assertRaises(ValueError):boundaries(Path(f.name),{'core0.cubecore0':1})

    def test_reference_summary(self):
        self.assertEqual(verify()['windows'],1980)

if __name__=='__main__':
    unittest.main()
