"""Auxiliary all-physical-instruction metric, consistently removing all synthetic lane labels."""
from collections import Counter
from pathlib import Path
import re,subprocess,sys
ROOT=Path(__file__).resolve().parents[3];sys.path.insert(0,str(ROOT))
from development.validation.post24.collect import LIB,checked,inputs,tables,text_bytes,sha
from development.validation.post24.physical import measure
from development.validation.post24.report import stats
from development.validation.post26.freeze import OUT as V26,OLD as V25
from development.validation.post27.issue import OUT
from development.validation.sku19.campaign import read,write,now

def one(f):
    o=read(f);reg=read(V26/'registration.json');sku=o['chip'];folder=ROOT/'runs/post26'/f.stem;log=(folder/'run.log').read_text()
    checked(o['freeze_sha256']==sha(V26/'registration.json') and o['cost']['started_utc']>reg['utc'] and o['cost']['exit_code']==0 and not o['cost']['stop_reason'],'launch/registration')
    checked(re.findall(r'^runtime_soc=(\S+)',log,re.M)==[sku] and ('lpddr5_6400M_bailu.ini' if 'PR' in sku else 'hbm3_8000M.ini') in log,'chip/DRAM config')
    checked('SMOKE_PASS' in log and 'All task success.' in log and 'Profiling data parse finished.' in log and not re.search(r'SMOKE_FAIL|\[ERROR\].*(?:deadlock|invalid instruction|execute_(?:set|wait)_flag)',log,re.I),'host/parser/backend')
    jobs=[j for j in reg['rows'] if j['chip']==sku and j['group']==o['group']]
    checked((folder/'batch.txt').read_text()==inputs(jobs),'shape order')
    names=re.findall(r'Start profiling on kernel: (\S+)',log);scopes=re.findall(r'^BATCH_BEGIN (\d+)\s*$\n(.*?)^BATCH_END \1\s*$',log,re.M|re.S);tab=tables(log)
    checked(len(jobs)==len(names)==len(scopes)==len(tab),'rosters')
    elf=text_bytes(LIB);syms=subprocess.check_output(['readelf','-Ws','--wide',str(LIB)],text=True);profile,=folder.glob('profile/OPPROF_*');idx=Counter();rr=[]
    for j,name,(i,scope),official in zip(jobs,names,scopes,tab):
        k=idx[name];idx[name]+=1;t,n,d=j['shape'];row=dict(version='v0.26',sku=sku,shape=j['shape'],role=j['role'],group=j['group'],i=j['i'],policy_us=j['us'],public_us=j['old_known_us'])
        try:
            checked(int(i)==j['i'] and re.findall(r'^correctness name=output count=(\d+) bad=(\d+)\b',scope,re.M)==[(str(t*n*d),'0')] and f'case=mhc_post api=aclnnMhcPost T={t} N={n} D={d} dtype=BF16' in scope and name.endswith('_6'),'correctness, API, ordinal or compiled token kernel')
            addr,=re.findall(r'\s([a-f\d]{16})\s+\d+\s+FUNC\s+GLOBAL[^\n]+\s'+re.escape(name)+r'$',syms,re.M)
            v=measure(folder,profile,name,k,official,j['shape'],elf,int(addr,16))
            row.update(qualified=True,**{q:v[q] for q in ('ca_us','active','complete_physical_end','missing_synthetic_label','trace')})
        except (AssertionError,ValueError,OSError,KeyError,IndexError) as e:row.update(qualified=False,reason=repr(e))
        rr.append(row)
    return rr
def main():
    out=OUT/'physical';out.mkdir(exist_ok=True)
    for f in (V26/'receipts').glob('*.json'):
        if (out/f.name).exists():continue
        try:v=dict(rows=one(f))
        except (AssertionError,ValueError,OSError,KeyError,IndexError) as e:v=dict(rows=[],reason=repr(e))
        write(out/f.name,v)
    allrows=[r for f in out.glob('*.json') for r in read(f)['rows']]
    for f in (V25/'physical').glob('*.json'):
        if not f.name.startswith(('width-','many-')):continue
        for v in read(f)['rows']:
            allrows.append(dict(version='v0.25',sku=v['sku'],shape=v['shape'],role=v['role'],group=v['group'],i=v['i'],policy_us=v['candidate_us'] if v['role']=='repeat' else v['public_us'],public_us=v['public_us'],qualified=v['qualified'],**{k:v[k]for k in ('ca_us','active','complete_physical_end','missing_synthetic_label','trace','reason') if k in v}))
    reg=read(V26/'registration.json');summary={}
    for s in reg['measured']:
        rr=[x for x in allrows if x['qualified'] and x['role']=='repeat' and x['sku']==s]
        summary[s]=dict(different_shapes=len({tuple(r['shape']) for r in rr}),policy=stats(rr,'policy_us'),old=stats(rr,'public_us'))
    write(OUT/'physical-summary.json',dict(utc=now(),scope='all launched AIVs earliest ELF entry through final real instruction. Synthetic END_LABEL excluded uniformly. Separate from preregistered inclusive policy gate.',summary=summary,rows=allrows))
if __name__=='__main__':main()
