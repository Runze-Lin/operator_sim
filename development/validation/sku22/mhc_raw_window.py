"""Independently certify complete physical mixed-core windows from CA's retained logs."""
from hashlib import sha256
import json
import re
import struct
import subprocess

from development.validation.rms_ordered_verify.verify import checked, text_bytes

LINE = re.compile(br'^\[info\] \[(\d+)\] \(PC: (0x[\da-fA-F]+)\)\s+\w+\s+: \(Binary: (0x[\da-fA-F]+)\) \(ID: (\d+)\) (\w+)\s+',re.M)


def raw_log(path, base, text, entry, vector):
    data=path.read_bytes()
    checked(data.endswith(b'\n'),f'incomplete raw last record on {path.name}')
    pos=0
    previous=-1
    first=endmark=label=None
    events=0
    for m in LINE.finditer(data):
        tick=int(m[1]); name=m[5];events+=1
        checked(tick>=previous,f'raw timestamps reversed on {path.name}')
        previous=tick
        if first is None or name in (b'END',b'END_LABEL'):
            pc=int(m[2],16); word=int(m[3],16)
            landmark=(tick,pc,int(m[4]))
            if first is None:
                checked(pc==base+entry and name==b'MOV_XD_SPR','ELF entry in '+path.name)
                first=landmark
            if name==b'END_LABEL':
                checked(label is None and pc==0x11111111 and vector,'terminal LABEL on '+path.name)
                label=landmark
            else:
                offset=pc-base
                checked(0<=offset<len(text)-3 and struct.unpack_from('<I',text,offset)[0]==word,'ELF instruction word '+path.name)
            if name==b'END':
                checked(word==0x41600000 and endmark is None,'physical END on '+path.name)
                endmark=landmark
        pos=m.end()
    checked(events==data.count(b'\n') and pos and first and endmark,'raw record coverage or END missing '+path.name)
    checked(bool(label)==vector and (not vector or label[0]==previous),'raw terminal-label contract on '+path.name)
    return dict(first=first,physical_end=endmark,label=label,last_tick=previous,count=events,sha256=sha256(data).hexdigest())


def raw_windows(profile, name, official, alias_windows, elf_text, dump_root, control=False):
    device_dump,=profile.glob('device0/'+name+'/0/dump')
    def matches_roster(directory):
        for suffix in ('.instr_log.dump','.instr_popped_log.dump'):
            if {f.name.removesuffix(suffix) for f in directory.glob('core*'+suffix)} != set(official):
                return False
        return True
    dump=next((p for p in (device_dump,profile/'dump') if matches_roster(p)),None)
    checked(dump is not None,'retained raw file roster disagrees with executed and exported cores')
    base=int((dump/'pc_start_addr.txt').read_text(),16)
    elf=dump/'aicore_binary.o'
    checked(text_bytes(elf)==elf_text,'device0 captured ELF mismatch')
    syms=subprocess.check_output(['readelf','-Ws','--wide',str(elf)],text=True)
    entries={}
    for key in ('aic','aiv'):
        symbol=name.removesuffix('_aic')+'_'+key
        addr,=re.findall(r'^\s*\d+:\s+([a-f\d]{16})\s+\d+\s+FUNC\s+GLOBAL\s+\S+\s+\d+\s+'+re.escape(symbol)+r'\s*$',syms,re.M)
        entries[key]=int(addr,16)
    cores={f.name.removesuffix('.instr_log.dump') for f in dump.glob('core*.instr_log.dump')}
    popped={f.name.removesuffix('.instr_popped_log.dump') for f in dump.glob('core*.instr_popped_log.dump')}
    checked(cores==popped==set(official)==set(alias_windows),'roster raw/official/export differs')
    bounds={};proof={};different=[]
    for core in sorted(cores):
        vector='veccore' in core
        a=raw_log(dump/(core+'.instr_popped_log.dump'),base,elf_text,entries['aiv' if vector else 'aic'],vector)
        b=raw_log(dump/(core+'.instr_log.dump'),base,elf_text,entries['aiv' if vector else 'aic'],vector)
        checked(a['first'][1:]==b['first'][1:] and a['first'][0]<=b['first'][0],'entry PC / ID differs')
        checked(a['physical_end'][1:]==b['physical_end'][1:] and a['physical_end'][0]<=b['physical_end'][0],'END PC / ID differs')
        checked(not vector or a['label'][1:]==b['label'][1:] and a['label'][0]<=b['label'][0],'endpoint ID mismatch')
        lo=a['first'][0];hi=b['last_tick']
        reported=alias_windows[core]
        # Export rounding on affected cores truncates their duration tables as well.
        # Intact control cores must match exact raw clocks; affected projections
        # must be prefixes with identical origins, never larger than the raw window.
        checked(lo==reported[0] and hi>=reported[1], 'raw versus export clocks '+core+' '+str((lo,hi,reported)))
        checked(abs(float(official[core][0])-(reported[1]-reported[0])/1650)<.015,'vendor does not describe exported prefix '+core)
        if hi!=reported[1]:different.append(core)
        bounds[core]=[lo,hi]
        proof[core]=dict(first=lo,last=hi,entry_pc=hex(a['first'][1]),physical_end_tick=b['physical_end'][0],
                        physical_end_pc=hex(b['physical_end'][1]),kernel_end_event_id=b['physical_end'][2],
                        label_tick=b['label'][0] if vector else None,popped_sha256=a['sha256'],executed_sha256=b['sha256'],
                        popped_records=a['count'],executed_records=b['count'],original_trace_bounds=reported,
                        official_duration_us=float(official[core][0]))
    if control:checked(not different,'control full windows differ '+str(different))
    return bounds,dict(dump=str(dump.relative_to(dump_root)),entry_offsets=entries,roster_count=len(cores),
                       shortened_trace_windows=different,cores=proof)


def repair(root, profile, name, duration, bounds, matched_text, out, observed_ends):
    folder=out/'mhc_pre_sinkhorn_56_4_4096_Ascend950PR_957b.json'
    prior=json.loads(folder.read_text())
    checked(prior['qualified'] and prior['observation']['kernel']==name,'missing verified complete comparator')
    retained=prior['observation']
    checked(retained['elf_text_sha256']==sha256(matched_text).hexdigest(),'control text differs')
    control_profile=next((root/'runs/fusion21'/folder.stem).glob('profile/OPPROF_*'))
    from development.validation.rms_ordered_verify.verify import tables
    table,=tables((control_profile.parent.parent/'run.log').read_text())
    ctl,cp=raw_windows(control_profile,name,table,retained['windows_ticks'],matched_text,root,control=True)
    r,proof=raw_windows(profile,name,duration,bounds,matched_text,root)
    cp.update(control_receipt=str(folder.relative_to(out.parent.parent.parent)),first_tick=min(a for a,b in ctl.values()),
              last_tick=max(b for a,b in ctl.values()),ca_us=retained['ca_us'])
    proof.update(prior_comparison=cp,missing_physical_END_in_export=[c for c in bounds if observed_ends[c]!=1],
                 policy='Earliest ELF-entry pop through maximum actual completion record. Per-core END IDs/PC and binary instruction words verified. Equal raw/merged ticks on the entire independent control roster. Official duration table describes the clipped JSON on damaged cores and is not substituted for the raw endpoint.')
    return r,proof
