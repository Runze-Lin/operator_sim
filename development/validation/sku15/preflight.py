"""Compare native PR helper DB with official msopprof on an already measured PR control."""
import sqlite3,re
from campaign import *
def audit():
 d=ROOT/'runs/preflight15'; ref=next(r for r in load(ROOT/'results/v0.14/sku/holdout.json') if r['id']=='fa_9_4096_64_1_1')
 gate=load(OUT/'preflight-gate.json'); a,rel=gate['max_each_core_span_abs_or_relative'];log=(d/'run.log').read_text(); dbfile,=d.glob('helper/*.db')
 assert 'runtime_soc='+PR in log and 'soc version: Ascend950PR' in log and 'SIM_SKU_ADAPTER target='+PR+' physical_model=Ascend950PR' in log and 'lpddr5_6400M_bailu.ini' in log and 'bad=0' in log and 'SMOKE_PASS' in log
 with sqlite3.connect(dbfile.resolve().as_uri()+'?mode=ro',uri=True) as db:
  f=dict(db.execute('select * from KernelInfo')); assert db.execute('pragma quick_check').fetchone()[0]=='ok'
  cores=db.execute('select ExecInstrCoreId,CoreTypeId,min(ExecInstrTickStart),max(ExecInstrTickEnd),count(*) from ExecutedInstructions group by 1,2').fetchall()
  assert sum(x[-1] for x in cores)==int(f['instrs_exec_num'])
 freq=float(f['frequency_ghz']);assert freq==1.65
 base=min(x[2]for x in cores); us=(max(x[3]for x in cores)-base)/(1000*freq);assert int(f['exec_time_ticks'])==max(x[3]for x in cores)-base
 windows=[];refcores={x['core']:x for x in ref['pr']['core_windows']}
 for cid,t,s,e,n in cores:
  name=f'core{cid}.'+('cubecore0' if t==1 else 'veccore'+str(t-2));w=refcores[name]; span=(e-s)/(1000*freq);rs=w['end_us']-w['start_us'];windows.append(dict(core=name,msopprof_span_us=rs,native_span_us=span,native_start_us=(s-base)/(1000*freq),relative=span/rs-1,pass_gate=abs(span-rs)<=max(a,rel*rs)))
 assert set(refcores)=={w['core']for w in windows}
 ref_us=ref['pr_ca_us'];times=list(map(float,(d/'cpu.txt').read_text().split()));res=dict(shape=ref['shape'],official_us=ref_us,native_us=us,relative=us/ref_us-1,passed=all(w['pass_gate']for w in windows)and abs(us-ref_us)<=max(a,rel*ref_us),runtime_soc=PR,physical_model='Ascend950PR',memory_model='lpddr5_6400M_bailu.ini',correctness=re.findall('^correctness (.+)',log,re.M),windows=windows,cpu_seconds=sum(times[:2]),wall_seconds=times[2],raw_bytes=sum(p.stat().st_size for p in d.rglob('*') if p.is_file()),db_sha256=sha(dbfile),log_sha256=sha(d/'run.log'),msopprof_trace_sha256=ref['pr']['trace_sha256'],kernel_info=f)
 write(OUT/'preflight.json',res);print(us,ref_us,res['passed'])
if __name__=='__main__':audit()
