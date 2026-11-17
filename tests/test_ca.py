import csv
import json
from pathlib import Path
import tempfile
import unittest

from operator_model.ca import extract


class ExtractTests(unittest.TestCase):
    def fixture(self,root):
        profile=root/'profile';report=profile/'OPPROF_fixture'
        sim=report/'simulator';core=sim/'core0.cubecore0';core.mkdir(parents=True)
        (report/'dump').mkdir();(report/'dump/aicore_binary.o').write_bytes(b'test-only')
        events=[{'ph':'M','pid':'core0.cubecore0','tid':'CUBE','name':'thread_sort_index'},
                {'ph':'X','pid':'core0.cubecore0','tid':'CUBE','name':'MMAD','ts':10.,'dur':2.,
                 'args':{'detail':json.dumps({'m':16,'n':16,'k':16})}},
                {'ph':'X','pid':'core0.cubecore0','tid':'CUBE','name':'WAIT_FLAG','ts':11.,'dur':3.,
                 'args':{'detail':json.dumps({'is_wait':1})}}]
        trace={'displayTimeUnit':'ns','traceEvents':events}
        (sim/'trace.json').write_text(json.dumps(trace))
        with (core/'core0.cubecore0_instr_exe.csv').open('w') as stream:
            w=csv.writer(stream);w.writerow(['instr','addr','pipe','call_count','cycles','running_time(us)','detail'])
            w.writerow(['MMAD','0','CUBE',1,3300,2,'']);w.writerow(['WAIT_FLAG','1','CUBE',1,4950,3,''])
        log=root/'run.log'
        log.write_text('runtime_soc=Ascend950DT_9582\nSMOKE_PASS\nAll task success.\nProfiling data parse finished.\nStart profiling on kernel: fixture\ncore0.cubecore0 4.00 4.00\n')
        return profile,log,sim/'trace.json'

    def test_units_scope_and_sync_exclusion(self):
        with tempfile.TemporaryDirectory() as folder:
            profile,log,_=self.fixture(Path(folder));r=extract(profile,log)
            self.assertEqual(r['reference_us'],4.)
            self.assertEqual(r['cores'][0]['busy_union_us_by_pipe']['CUBE'],2.)
            self.assertEqual(r['cores'][0]['executed_cube_flops'],8192)
            self.assertEqual(r['cores'][0]['dynamic_calls_by_pipe']['CUBE'],2)

    def test_incomplete_and_multikernel_fail_closed(self):
        with tempfile.TemporaryDirectory() as folder:
            profile,log,_=self.fixture(Path(folder));original=log.read_text()
            log.write_text(original.replace('SMOKE_PASS','SMOKE_FAIL'))
            with self.assertRaises(ValueError):extract(profile,log)
            log.write_text(original+'Start profiling on kernel: second\n')
            with self.assertRaises(ValueError):extract(profile,log)

    def test_timestamp_scaling_mismatch(self):
        with tempfile.TemporaryDirectory() as folder:
            profile,log,path=self.fixture(Path(folder));trace=json.loads(path.read_text())
            for e in trace['traceEvents']:
                if e['ph']=='X':e['ts']*=1000;e['dur']*=1000
            path.write_text(json.dumps(trace))
            with self.assertRaises(ValueError):extract(profile,log)


if __name__=='__main__':unittest.main()
