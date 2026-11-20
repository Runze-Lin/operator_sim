"""Guard the boundary between historical starts and independent device windows."""
from tempfile import TemporaryDirectory
from pathlib import Path
import json,sys,unittest
HERE=Path(__file__).resolve().parent
sys.path[:0]=[str(HERE),str(HERE.parents[2])]
from registration import ROOT,OUT,D,sha
from evaluate import observed
class Qualification(unittest.TestCase):
    def test_no_new_role_for_existing_start(self):
        with TemporaryDirectory(dir=OUT) as d:
            f=Path(d)
            owner=f/'owner.json';owner.write_text('{"shape":[2,4,368]}')
            receipt=f/'window.json'
            s=dict(ca_us=3.2,trace_sha256='1'*64,kernel='MhcPost_example_5',active_aiv=4,active_aic=0,trace='synthetic validator fixture')
            obj=dict(chip=D,freeze_sha256=sha(owner),cost=dict(started_utc='2026-11-20T03:00:00',seconds=1,exit_code=0,stop_reason=None),
                     rows=[dict(i=0,shape=[2,4,368],timing_qualified=True,stage=s)])
            receipt.write_text(json.dumps(obj))
            j=dict(sku=D,shape=[2,4,368],ordinal=0,group='fresh',forecast={'us':3.3},future970_us=3,
                   owner=str(owner.relative_to(ROOT)),owner_sha256=sha(owner),owner_field='freeze_sha256',receipt=str(receipt.relative_to(ROOT)))
            reg={'utc':'2026-11-20T02:00:00'}
            self.assertEqual(observed(j,reg,None)['role'],'frozen independent')
            self.assertEqual(observed(dict(j,already_started=True),reg,None)['role'],'historical replay')
            self.assertEqual(observed(j,{'utc':'2026-11-20T04:00:00'},None)['role'],'historical replay')
            with self.assertRaises(ValueError):observed(dict(j,owner_sha256='a'*64),reg,None)
if __name__=='__main__':unittest.main()
