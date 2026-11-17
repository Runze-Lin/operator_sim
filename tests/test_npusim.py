import sqlite3
import tempfile
from pathlib import Path
import unittest
from operator_model.npusim import extract_compact

class CompactTests(unittest.TestCase):
 def setUp(self):
  self.tmp=tempfile.TemporaryDirectory();self.addCleanup(self.tmp.cleanup);self.root=Path(self.tmp.name)
  self.path=self.root/'npusim_test/record/kernel.db';self.path.parent.mkdir(parents=True)
  self.log=self.root/'run.log';self.log.write_text('runtime_soc=Ascend950DT_9582\nSMOKE_PASS\nSimulation SUCCESS\n')
  with sqlite3.connect(self.path) as d:
   d.execute('CREATE TABLE KernelInfo(KernelInfoAttr TEXT,KernelInfoVal)')
   d.executemany('INSERT INTO KernelInfo VALUES (?,?)',list({'helper_version':'0.8.0','contract_version':7,'frequency_ghz':1.65,'instrs_exec_num':3,'exec_time_ticks':165,'exec_time_ns':100.,'ai_core_active':3,'name':'test','block_dim':1}.items()))
   d.execute('CREATE TABLE ExecutedInstructions(ExecInstrCoreId,CoreTypeId,ExecInstrTickStart,ExecInstrTickEnd,ExecInstrName)')
   d.executemany('INSERT INTO ExecutedInstructions VALUES(?,?,?,?,?)',[(0,1,100,200,'MMAD'),(0,2,120,265,'VF'),(0,3,110,250,'VF')])
 def test_window_is_cross_core_and_cycles_convert(self):
  out=extract_compact(self.root,self.log);self.assertEqual(out['reference_us'],.1)
  self.assertEqual((out['active_aic'],out['active_aiv'],out['max_core_mmad']),(1,2,1))
 def test_reject_success_banner_without_correctness_or_with_wrong_soc(self):
  for log in ['runtime_soc=Ascend950DT_9582\nSimulation SUCCESS','runtime_soc=Ascend950PR_9589\nSimulation SUCCESS\nSMOKE_PASS']:
   self.log.write_text(log)
   with self.assertRaises(ValueError):extract_compact(self.root,self.log)
 def test_reject_truncated_or_inconsistent_database(self):
  with sqlite3.connect(self.path) as d:d.execute("UPDATE KernelInfo SET KernelInfoVal=166 WHERE KernelInfoAttr='exec_time_ticks'")
  with self.assertRaisesRegex(ValueError,'makespan'):extract_compact(self.root,self.log)
 def test_native_requires_explicit_successful_process_and_correctness(self):
  folder=self.root/'helper';folder.mkdir();self.path.rename(folder/'kernel.db')
  self.log.write_text('runtime_soc=Ascend950DT_9582\nSMOKE_PASS')
  self.assertEqual(extract_compact(self.root,self.log,native_returncode=0)['reference_us'],.1)
  with self.assertRaises(ValueError):extract_compact(self.root,self.log,native_returncode=124)
 def test_pipeline_union_does_not_sum_overlap_or_wait(self):
  with sqlite3.connect(self.path) as d:
   d.execute('ALTER TABLE ExecutedInstructions ADD COLUMN ExtendParams')
   d.execute('ALTER TABLE ExecutedInstructions ADD COLUMN ExecInstrIsDurationExcluded DEFAULT 0')
   d.execute('UPDATE ExecutedInstructions SET ExtendParams=\'{"queue_type":"SIMD"}\'')
   d.execute('UPDATE ExecutedInstructions SET ExecInstrCoreId=0,CoreTypeId=2')
   d.execute("UPDATE ExecutedInstructions SET ExecInstrIsDurationExcluded=1 WHERE ExecInstrName='MMAD'")
   d.execute("UPDATE KernelInfo SET KernelInfoVal=1 WHERE KernelInfoAttr='ai_core_active'")
  out=extract_compact(self.root,self.log)
  self.assertAlmostEqual(out['cores'][0]['nonexcluded_instruction_union_us_by_queue']['SIMD'],155/1650)
