"""Receipt schema/identity regression checks; no CA or repository reports are generated."""
from pathlib import Path
import json,sys,tempfile,unittest
from unittest.mock import patch
HERE=Path(__file__).resolve().parent;sys.path.append(str(HERE));import receipts
class ReadFixture(unittest.TestCase):
 def setUp(self):
  self.tmp=tempfile.TemporaryDirectory();self.addCleanup(self.tmp.cleanup)
  self.root=Path(self.tmp.name);self.out=self.root/'results/v0.32';p=self.out/'large';p.mkdir(parents=True);self.p=p
  freeze=self.out/'large/freeze.json';self.put(freeze,dict(utc='2026-11-20T06:00:00+00:00',predictions={receipts.DT:[dict(shape=[1024,4],iterations=20,baseline_public_us=None,unknown_history_mean_us=8.,baseline_known_us=10.,explicit_history={'us':9.})]}))
  self.digest=receipts.digest(freeze);base=self.root/'results/v0.31/recurrent/freeze.json';base.parent.mkdir(parents=True);self.put(base,dict(cases={s:dict(predictions={}) for s in ['6','128']}));self.put(p/'recurrent-freeze.json',dict(prices={}))
  ctx=patch.multiple(receipts,ROOT=self.root,OUT=self.out);ctx.start();self.addCleanup(ctx.stop)
 def put(self,p,x):p.write_text(json.dumps(x))
 def rec(self):
  r=dict(shape=[1024,4],h=0,iters=20,all_output_elements=16384,qualified=True,stage=dict(kernel='MhcSinkhorn_example',trace_sha256='a'*64,ca_us=7.,written_bytes=65536))
  return dict(chip=receipts.DT,verdict=r,freeze_sha256=self.digest,cost=dict(exit_code=0,stop_reason=None,started_utc='2026-11-20T06:01:00+00:00',ended_utc='2026-11-20T06:02:00+00:00'))
 def test_single_device_window_is_one_per_separate_protocol(self):
  self.put(self.p/('sink1024-'+receipts.DT+'.json'),self.rec());rows=list(receipts.early_registered())
  self.assertEqual(len(rows),2);self.assertEqual({r[7] for r in rows},{'a'*64});self.assertEqual({r[-2] for r in rows},{'independent'});self.assertEqual({r[10] for r in rows},{8.,9.})
 def test_failed_audit_and_bad_freeze(self):
  raw=self.rec();self.put(self.p/('sink1024-'+receipts.DT+'-audit.json'),raw);self.put(self.p/('sink1024-'+receipts.PR+'.json'),dict(verdict={'qualified':False}));self.assertEqual(list(receipts.early_registered()),[])
  raw['freeze_sha256']='0'*64;self.put(self.p/('sink1024-'+receipts.DT+'.json'),raw)
  with self.assertRaises(AssertionError):list(receipts.early_registered())
 def test_only_before_collection_is_new_independent(self):
  self.assertEqual(receipts.relationship('2026-11-20T06:01:00+00:00','2026-11-20T06:00:00+00:00','2026-11-20T06:03:00+00:00'),('replay','blinded_after_start'))
  self.assertEqual(receipts.relationship('2026-11-20T06:01:00+00:00',None,None)[0],'replay')
 def test_matmul_register_chunk_full_and_decode_with_acquisition_roles(self):
  dt,pr=receipts.DT,receipts.PR
  ts='2026-11-20T06:00:00+00:00'
  mat=self.out/'matmul';mat.mkdir()
  self.put(mat/'freeze.json',dict(utc=ts,rows=[]))
  def freeze(folder,**data):
   folder.mkdir(parents=True);p=folder/'freeze.json';self.put(p,dict(utc=ts,**data));return p
  prices={dt:dict(us=12),pr:dict(us=13)}
  freeze(mat/'kimi-head',shape=[11,64,512],predictions=prices)
  original=freeze(self.root/'results/v0.28/large')
  full=dict(shape=[262144,64,7168],source=str(original.relative_to(self.root)),predictions=prices)
  pro=freeze(mat/'pro-large',shape=[32768,64,7168],predictions=prices,reuse_full=full)
  decode=freeze(mat/'decode',jobs=[dict(shape=[81,1280,5120],predictions={dt:dict(us=16,old_us=17),pr:dict(us=18,old_us=19)})])
  def write(path,reg,shape,h):
   self.put(path,dict(qualified=True,chip=dt,shape=shape,ca_us=15,stages=[dict(kernel='MatMulV3',trace_sha256=h*64,ca_us=15)],cost=dict(started_utc='2026-11-20T06:01:00+00:00',finished_utc='2026-11-20T06:02:00+00:00'),freeze_sha256=receipts.digest(reg)))
  write(pro.parent/'dt.json',pro,[32768,64,7168],'a')
  write(original.parent/'DeepSeek-V4-Pro-dt.json',original,full['shape'],'b')
  write(decode.parent/'81-1280-5120-dt.json',decode,[81,1280,5120],'c')
  rows={r[7][0]:r for r in receipts.matmul_windows()}
  self.assertEqual({k:v[-2] for k,v in rows.items()},dict(a='independent',b='replay',c='independent'))
  self.assertEqual(rows['c'][9:11],(17,16))
  write(pro.parent/'dt.json',decode,[32768,64,7168],'a')
  with self.assertRaises(AssertionError):list(receipts.matmul_windows())
if __name__=='__main__':unittest.main()
