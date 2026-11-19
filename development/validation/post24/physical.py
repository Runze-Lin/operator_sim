"""Independent full physical-instruction measurement, excluding synthetic END_LABEL on every lane."""
from collections import Counter,defaultdict
import json,os,re,struct,subprocess,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[3];sys.path.insert(0,str(ROOT))
from development.validation.post24.collect import ROOT,OUT,RAW,LIB,inputs,tables,events,text_bytes,checked
from development.validation.post24.report import stats,fmt
from development.validation.sku19.campaign import read,write,now

def measure(folder,profile,kernel,idx,table,shape,elf,entry):
    path=profile/kernel/str(idx);tr=path/'simulator/trace.json'
    checked(text_bytes(path/'dump/aicore_binary.o')==elf,'vendor .text mismatch')
    base=int((path/'dump/pc_start_addr.txt').read_text(),16)
    by={};selected=defaultdict(list);ends=Counter();label={};writes=Counter()
    for ev in events(tr):
        k=ev['pid'];name=ev['name'];a=round(ev['ts']*1650);b=round((ev['ts']+ev['dur'])*1650)
        checked(b>=a and re.fullmatch(r'core\d+\.veccore\d+',k),'invalid time/core')
        if name=='END_LABEL':
            checked(int(ev['args']['pc_addr'],16)==0x11111111 and k not in label,'invalid synthesized label');label[k]=b;continue
        if k not in by:by[k]=[a,b]
        else:by[k]=[min(a,by[k][0]),max(b,by[k][1])]
        if name=='MOV_XD_SPR' and int(ev['args']['pc_addr'],16)==base+entry:selected[k].append(a)
        if name=='END':
            pc=int(ev['args']['pc_addr'],16)-base
            checked(0<=pc<=len(elf)-4 and struct.unpack_from('<I',elf,pc)[0]==0x41600000,'missing ELF physical END word');ends[k]+=1
        if name.startswith('MOV_SRC_TO_DST'):
            det=json.loads(ev['args']['detail'])
            if det['queue_type']=='MTE3' and not any(det.get(flag) for flag in ('is_wait','is_sync','is_duration_excluded')): writes[k]+=det['processed_bytes']
    checked(set(by)==set(table)==set(writes) and all(selected[k]==[b[0]] and ends[k]==1 for k,b in by.items()),'physical ELF-entry/END/official active sets')
    for k,(lo,hi) in by.items():
        checked(abs(float(table[k][0])-(max(hi,label.get(k,0))-lo)/1650)<.015,'rounded official duration outside tolerance')
    checked(sum(writes.values())==2*shape[0]*shape[1]*shape[2],'incorrect output footprint')
    return dict(ca_us=(max(r[1] for r in by.values())-min(r[0] for r in by.values()))/1650,
                active=len(by),complete_physical_end=len(ends),missing_synthetic_label=sorted(by.keys()-label.keys()),
                per_core_physical_ticks=by,write_bytes=sum(writes.values()),trace=str(tr.relative_to(ROOT)))

def generate(f):
    obj=read(f);chip=obj['chip'];folder=RAW/f.stem;log=(folder/'run.log').read_text()
    hostscopes=re.findall(r'^BATCH_BEGIN (\d+)\s*$\n(.*?)^BATCH_END \1\s*$',log,re.M|re.S)
    frozen=[r for r in read(OUT/'freeze.json')['rows'] if r['chip']==chip and r['group']==obj['group']]
    checked(obj['cost']['exit_code']==0 and not obj['cost']['stop_reason'] and 'SMOKE_PASS' in log and 'All task success.' in log and 'Profiling data parse finished.' in log,'host/collector outcome')
    checked(re.findall(r'^runtime_soc=(\S+)',log,re.M)==[chip] and ('lpddr5_6400M_bailu.ini' if 'PR' in chip else 'hbm3_8000M.ini') in log and not re.search(r'SMOKE_FAIL|\[ERROR\].*(?:deadlock|invalid instruction|execute_set_flag|execute_wait_flag)',log,re.I),'SKU/CA error')
    checked((folder/'batch.txt').read_text()==inputs(frozen),'host order');names=re.findall(r'Start profiling on kernel: (\S+)',log);ts=tables(log);checked(len(names)==len(ts)==len(frozen)==len(hostscopes),'missing launches or references')
    profile,=folder.glob('profile/OPPROF_*');elf=text_bytes(LIB);syms=subprocess.check_output(['readelf','-Ws','--wide',str(LIB)],text=True)
    counter=Counter();rows=[]
    for q,name,tab,(ordinal,scope) in zip(frozen,names,ts,hostscopes):
        i=counter[name];counter[name]+=1;t,n,d=q['shape'];r=dict(group=q['group'],ordinal=q['i'],chip=chip,shape=q['shape'],role=q['role'],state=q['state'],candidate_us=q['us'],baseline_us=q['old_known_us'],unknown_us=q['candidate_unknown_us'])
        try:
            checked(int(ordinal)==q['i'] and re.findall(r'^correctness name=output count=(\d+) bad=(\d+)\b',scope,re.M)==[(str(t*n*d),'0')] and f'case=mhc_post api=aclnnMhcPost T={t} N={n} D={d} dtype=BF16' in scope,'wrong host contract')
            address,=re.findall(r'\s([a-f\d]{16})\s+\d+\s+FUNC\s+GLOBAL[^\n]+\s'+re.escape(name)+r'$',syms,re.M)
            r.update(qualified=True,**measure(folder,profile,name,i,tab,q['shape'],elf,int(address,16)))
        except (OSError,ValueError,IndexError,KeyError,AssertionError) as err:r.update(qualified=False,error=repr(err))
        rows.append(r)
    return rows
def run(follow):
    dest=OUT/'physical';dest.mkdir(exist_ok=True)
    while True:
        for file in sorted((OUT/'receipts').glob('*.json')):
            output=dest/file.name
            if output.exists():continue
            try:rows=generate(file);out=dict(rows=rows)
            except (OSError,ValueError,IndexError,KeyError,AssertionError)as err:out=dict(rows=[],error=repr(err))
            write(output,out)
        rr=[r for p in sorted(dest.glob('*.json')) for r in read(p)['rows'] if r['qualified']]
        chips=read(OUT/'freeze.json')['measured'];summary={s:{k:stats([r for r in rr if r['chip']==s],k) for k in ('candidate_us','baseline_us','unknown_us')} for s in chips}
        write(OUT/'physical-summary.json',dict(utc=now(),scope='first physical ELF instruction across all actual AIVs to last physical instruction finish. Excludes generated END_LABEL on all lanes. Separate from registered inclusive-window score.',windows=len(rr),summary=summary,rows=[{k:v for k,v in r.items() if k!='per_core_physical_ticks'} for r in rr]))
        if not follow or len(list(dest.glob('*.json')))==8:break
        time.sleep(30)
if __name__=='__main__':run('follow' in sys.argv)
