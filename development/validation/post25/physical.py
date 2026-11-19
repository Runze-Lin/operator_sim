"""Re-evaluate all windows using immutable pre-registered ELF-entry/real-END timing."""
from collections import Counter
import re,subprocess,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[3];sys.path.insert(0,str(ROOT))
from development.validation.post25.collect import RAW,OUT
from development.validation.post24.collect import LIB,inputs,checked,tables,text_bytes,sha
from development.validation.post24.physical import measure
from development.validation.post24.report import stats
from development.validation.sku19.campaign import read,write,now

def score(file):
    obj=read(file);reg=read(OUT/'freeze.json');c=obj['cost'];s=obj['chip'];folder=RAW/file.stem;log=(folder/'run.log').read_text()
    checked(obj['freeze_sha256']==sha(OUT/'freeze.json') and c['exit_code']==0 and not c['stop_reason'] and c['started_utc']>reg['utc'],'host outcome / registration')
    checked('SMOKE_PASS' in log and 'All task success.' in log and 'Profiling data parse finished.' in log and not re.search(r'SMOKE_FAIL|\[ERROR\].*(?:deadlock|invalid instruction|execute_set_flag|execute_wait_flag)',log,re.I),'host window')
    checked(re.findall(r'^runtime_soc=(\S+)',log,re.M)==[s] and ('lpddr5_6400M_bailu.ini' if 'PR' in s else 'hbm3_8000M.ini') in log,'wrong SKU')
    rs=[r for r in reg['rows'] if (r['group'],r['chip'])==(obj['group'],s)]
    checked((folder/'batch.txt').read_text()==inputs(rs),'input identity')
    gs=re.findall(r'^BATCH_BEGIN (\d+)\s*$\n(.*?)^BATCH_END \1\s*$',log,re.M|re.S);names=re.findall(r'Start profiling on kernel: (\S+)',log);times=tables(log)
    checked(len(rs)==len(gs)==len(names)==len(times),'missing launch/host result')
    elf=text_bytes(LIB);syms=subprocess.check_output(['readelf','-Ws','--wide',str(LIB)],text=True);profile,=folder.glob('profile/OPPROF_*')
    counts=Counter();rows=[]
    for r,(ordinal,scope),name,tab in zip(rs,gs,names,times):
        t,n,d=r['shape'];i=counts[name];counts[name]+=1
        q=dict(sku=s,group=r['group'],i=r['i'],shape=r['shape'],role=r['role'],state=r['mode'],candidate_us=r['us'],public_us=r['old_known_us'],candidate_unknown_us=r['candidate_unknown_us'],public_unknown_us=r['old_unknown_us'])
        try:
            checked(int(ordinal)==r['i'] and re.findall(r'^correctness name=output count=(\d+) bad=(\d+)\b',scope,re.M)==[(str(t*n*d),'0')] and f'case=mhc_post api=aclnnMhcPost T={t} N={n} D={d} dtype=BF16' in scope,'incomplete golden')
            addr,=re.findall(r'\s([a-f\d]{16})\s+\d+\s+FUNC\s+GLOBAL[^\n]+\s'+re.escape(name)+r'$',syms,re.M)
            z=measure(folder,profile,name,i,tab,r['shape'],elf,int(addr,16));q.update(qualified=True,**z)
        except (AssertionError,ValueError,OSError,KeyError,IndexError) as exc:q.update(qualified=False,reason=repr(exc))
        rows.append(q)
    return rows

def main():
    target=OUT/'physical';target.mkdir(exist_ok=True)
    for p in sorted((OUT/'receipts').glob('*.json')):
        if (target/p.name).exists():continue
        try:out=dict(rows=score(p))
        except (AssertionError,ValueError,OSError,KeyError,IndexError) as exc:out=dict(rows=[],reason=repr(exc))
        write(target/p.name,out)
    allrows=[r for f in target.glob('*.json') for r in read(f)['rows'] if r['qualified']];skus=read(OUT/'freeze.json')['measured']
    summary={s:{k:stats([x for x in allrows if x['sku']==s],k) for k in ['candidate_us','public_us','candidate_unknown_us','public_unknown_us']} for s in skus}
    write(OUT/'physical-summary.json',dict(utc=now(),scope='all active lanes ELF entry to final physical instruction, synthetic labels excluded for every lane',windows=len(allrows),shapes={s:len({tuple(x['shape']) for x in allrows if x['sku']==s}) for s in skus},summary=summary,rows=[{k:v for k,v in r.items() if k!='per_core_physical_ticks'} for r in allrows]))
if __name__=='__main__':main()
