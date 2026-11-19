"""Bind one actual ACL launch, full host golden and complete device trace to each prediction."""
from collections import Counter, defaultdict
from pathlib import Path
import json
import re
import subprocess
import sys

ROOT=Path(__file__).resolve().parents[3]
sys.path.insert(0,str(ROOT))
from development.validation.mixed_history25.campaign import OUT,ROOT,write
from development.validation.rms_ordered_verify import verify as v

ADD='MatMulV3_ND_ND_ND_ND_BF16_BF16_BF16_BF16_high_performance'
CUBE=v.VENDOR.parents[1]/'mat_mul_v3'/(ADD+'.o')


def physical(folder,kernel,shape,table,vector):
    elf=folder/'dump/aicore_binary.o'
    vendor=v.VENDOR if vector else CUBE
    v.checked(v.text_bytes(elf)==v.text_bytes(vendor),'captured .text disagrees with installed library')
    base=int((folder/'dump/pc_start_addr.txt').read_text(),16)
    symbols=subprocess.check_output(['readelf','-Ws','--wide',str(elf)],text=True)
    address,=re.findall(r'\s([a-f\d]{16})\s+\d+\s+FUNC\s+GLOBAL[^\n]+\s'+re.escape(kernel)+r'$',symbols,re.M)
    entry=base+int(address,16)
    cores={};rows=0;mmads=0;writes=Counter()
    shards={}
    for p in sorted((folder/'simulator').glob('core*/trace.json')):
        span=v.Span()
        first=1<<80; entry_ticks=[]
        for e in v.events(p):
            if e['name']=='END_LABEL':continue
            pc=int(e['args']['pc_addr'],16)
            if pc==entry and e['name']=='MOV_XD_SPR':entry_ticks.append(round(e['ts']*v.CLOCK))
            span.add(e,base)
            if e['name']=='MMAD':mmads+=1
        v.checked(entry_ticks==[span.first] and len(span.end)==1,'missing physical entry/END '+p.parent.name)
        c=p.parent.name
        kind='vec' if vector else 'cube'
        v.checked(re.fullmatch(r'core\d+\.'+kind+r'core\d+',c),'unexpected engine '+c)
        v.checked(c in table and abs((span.last-span.first)/v.CLOCK-float(table[c][0]))<=.01,'physical versus official duration '+c)
        if vector:
            rows+=span.yrows;writes.update(span.writes)
            v.checked(span.widths=={2*shape[1]} and set(span.writes)=={0x4f38,0x5128},'output write signature')
        cores[c]=span
        shards[c]=v.digest(p)
    v.checked(cores and set(cores)==set(table),'core roster mismatch')
    if vector:
        v.checked(rows==shape[0] and writes=={0x4f38:4*rows,0x5128:2*rows*shape[1]},'incomplete tensor writes')
    else:
        v.checked(mmads>0,'matmul has no MMAD')
    merged=defaultdict(v.Span)
    path=folder/'simulator/trace.json'
    for e in v.events(path):
        if e['name']!='END_LABEL':merged[e['pid']].add(e,base)
    v.checked(merged.keys()==cores.keys() and all(s.signature()==merged[k].signature() for k,s in cores.items()),'aggregate and shards differ')
    first=min(c.first for c in cores.values());last=max(c.last for c in cores.values())
    return dict(ca_us=(last-first)/v.CLOCK,core_windows={k:[s.first,s.last] for k,s in cores.items()},trace=v.rel(path),trace_sha256=v.digest(path),
                elf_text_sha256=v.sha256(v.text_bytes(vendor)).hexdigest(),shards=shards,kernel=kernel,
                first_tick=first,last_tick=last,observed_aic=0 if vector else len(cores),observed_aiv=len(cores) if vector else 0,mmads=mmads)


def qualify(role,chip,group):
    f=OUT/(role+'-freeze.json')
    plan=json.loads(f.read_text())
    stem=role+'-'+group+'-'+chip
    cost=json.loads((OUT/(stem+'-cost.json')).read_text());d=ROOT/cost['file'];log=(d/'run.log').read_text()
    jobs=[r for r in plan['rows'] if r['chip']==chip and r['group']==group]
    v.checked(cost['exit_code']==0 and not cost['stop'] and cost['started_utc']>plan['utc'] and cost['plan_sha256']==v.digest(f),'process or chronology')
    v.checked('SMOKE_PASS' in log and 'Profiling running finished. All task success.' in log and not re.search(r'SMOKE_FAIL|\[ERROR\].*(?:execute_[a-z]+_flag|deadlock|invalid instruction)',log),'host/device failure')
    v.checked(re.findall(r'^runtime_soc=(\S+)',log,re.M)==[chip] and ('lpddr5_6400M_bailu.ini' if 'PR' in chip else 'hbm3_8000M.ini') in log,'observed SKU/memory model')
    expected=''.join(r['op']+' '+' '.join(map(str,r['shape']+([0] if r['op']=='rmsnorm' else [])))+'\n' for r in jobs)
    v.checked((d/'inputs.txt').read_text()==expected,'input list binding')
    spans=re.findall(r'^BATCH_BEGIN (\d+)\s*$\n(.*?)^BATCH_END \1\s*$',log,re.M|re.S)
    tables=v.tables(log)
    v.checked(len(spans)==len(jobs)==len(tables),'host/launch/table count')
    last=-1;seq=Counter();out=[]
    for r,(i,b),table in zip(jobs,spans,tables):
        v.checked(int(i)==r['i'],'host order')
        vector=r['op']=='rmsnorm';a,z,*k=r['shape']
        needle=f'case=rmsnorm api=aclnnRmsNorm R={a} C={z}' if vector else f'case=addmm api=aclnnAddmm M={a} N={z} K={k[0]} self=bias'
        v.checked(needle in b,'incorrect actual shape')
        gold=re.findall(r'^correctness [^\n]+',b,re.M)
        v.checked([int(re.match(r'correctness count=(\d+) bad=0\b',s)[1]) for s in gold]==([a*z,a] if vector else [a*z]),'full host golden')
        kernel,=re.findall(r'Start profiling on kernel: (\S+)',b)
        v.checked(kernel==(v.KERNEL if vector else ADD+'_2'),'unrecognized actual implementation '+kernel)
        task_ids=set(re.findall(r'\[TASK_BEGIN\].* full_task_id:(\d+) BEGIN',b))
        v.checked(task_ids=={i} and '[TASK_DONE]' in b,'runtime ordinal')
        idx=seq[kernel];seq[kernel]+=1
        dir_,=d.glob('profile/OPPROF_*/'+kernel+'/'+str(idx))
        try:
            p=physical(dir_,kernel,r['shape'],table,vector)
            v.checked(p['first_tick']>last,'simulated launches overlap or reordered')
            if vector:v.checked(len(table)==r['active_aiv'],'RMS coverage disagreement')
            last=p['last_tick']
            out.append(dict(**r,**p,goldens=gold,qualified=True))
        except (ValueError,OSError,IndexError,KeyError,TypeError) as e:
            out.append(dict(**r,qualified=False,reason=str(e),goldens=gold,actual_roster=list(table)))
    return dict(cost=cost,role=role,chip=chip,group=group,rows=out)


def publish(role):
    plan=json.loads((OUT/(role+'-freeze.json')).read_text())
    for group in plan['groups']:
        for chip in plan['chips']:
            stem=role+'-'+group+'-'+chip
            f=OUT/(stem+'-windows.json')
            if f.exists() or not (OUT/(stem+'-cost.json')).is_file():continue
            receipt=qualify(role,chip,group)
            write(f,receipt)
            print(stem,[(r['i'],round(r['ca_us'],3) if r['qualified'] else r['reason']) for r in receipt['rows']])

if __name__=='__main__':publish(sys.argv[1])
