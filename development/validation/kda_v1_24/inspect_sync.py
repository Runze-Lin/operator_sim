"""Map runtime PC faults to exact wheel machine code and consecutive fence issues."""
from collections import Counter
import hashlib
import json
import re
import subprocess
from pathlib import Path

from investigate import RAW, OUT, ROOT, REG, read, save, probe


def code_map(binary, offsets):
    if not offsets:
        return {}
    args=['bash',str(ROOT/'scripts/cann-run.sh'),'llvm-objdump','-d','--start-address='+hex(min(offsets)),
          '--stop-address='+hex(max(offsets)+8),str(binary)]
    dis=subprocess.run(args,capture_output=True,text=True,check=True).stdout
    words={}
    for line in dis.splitlines():
        match=re.match(r'^\s*([0-9a-f]+):\s+([0-9a-f]+)\s+(\S+)',line)
        if match and int(match[1],16) in offsets:
            words[int(match[1],16)]={'word':match[2],'mnemonic':match[3]}
    return words

def tile_events(path,base,offsets):
    # Event names, flags and ticks are native CA data. Discard sync-inactive auxiliaries.
    events=json.loads(path.read_bytes())['traceEvents']
    flags=[];mm=[];end=[];label=[]
    for x in events:
        if x.get('ph')!='X':continue
        n=x['name'];a=round(x['ts']*1650);b=round((x['ts']+x['dur'])*1650)
        if n=='MMAD':mm.append([a,b])
        if n=='END':end.append([a,b])
        if n=='END_LABEL':label.append([a,b])
        if n not in ('SET_FLAG','WAIT_FLAG'):continue
        d=json.loads(x['args']['detail'])
        if d.get('target_pipe')=='CUBE' and d.get('consumer_pipe')=='MTE1':
            flags.append((a,n,d['flag_id'],int(x['args']['pc_addr'],16)-base))
    out={};prev={}
    for tick,name,flag,pc in sorted(flags):
        if pc in offsets and name=='SET_FLAG' and flag in prev and prev[flag][1]=='SET_FLAG' and str(flag) not in out:
            prior=prev[flag];out[str(flag)]=dict(prior_set_tick=prior[0],prior_pc_offset=hex(prior[2]),
                                                second_set_tick=tick,second_pc_offset=hex(pc))
        prev[flag]=(tick,name,pc)
    return dict(first_mmad=mm[0] if mm else None,mmad_count=len(mm),physical_end=end,terminal_label=label,
                first_duplicate_without_intervening_wait=out,shard_sha256=probe.digest(path))

def inspect(case):
    file=OUT/(case['id']+'.json')
    if not file.exists():return
    receipt=read(file)
    if receipt.get('skipped'):return
    main=[r for r in receipt['stages'] if r['kernel'].startswith('ChunkKdaFwd_')]
    if len(main)!=1 or not main[0].get('raw_path'):return
    row=main[0];path=ROOT/row['raw_path'];binary=path/'dump/aicore_binary.o'
    base=int((binary.parent/'pc_start_addr.txt').read_text(),16)
    offs={int(k,16)-base:v for k,v in receipt['sync_fault_pc_counts'].items()};words=code_map(binary,offs)
    old=read(probe.OUT/'faults.json')
    oldp={int(d['offset'],16) for d in old['instructions']}
    cores={}
    for p in sorted((path/'simulator').glob('core*.cubecore*/trace.json')):
        try:cores[p.parent.name]=tile_events(p,base,set(offs))
        except (KeyError,OSError,ValueError) as exc:cores[p.parent.name]=dict(error=str(exc))
    old_text=hashlib.sha256(probe.text_bytes(ROOT/old['captured_binary'])).hexdigest()
    same=old_text==row['device_text_sha256']
    evidence=dict(role='diagnostic only, device-local PC evidence',chip=case['sku'],case=case['id'],
        source_receipt_sha256=probe.digest(file),binary=probe.rel(binary),captured_text_sha256=row['device_text_sha256'],
        pc_base=hex(base),distinct_fault_offsets=len(offs),resolved_faults=len(words),total_duplicate_sets=sum(offs.values()),
        same_text_as_old_varlen_t64_h2=same,common_fault_offsets_with_old_varlen_t64_h2=len(set(offs)&oldp) if same else None,
        faults=[dict(offset=hex(pc),occurrences=n,**words.get(pc,{})) for pc,n in sorted(offs.items())],aic=cores)
    save(OUT/(case['id']+'-sync.json'),evidence)

if __name__=='__main__':
    for case in read(REG)['cases']:inspect(case)
