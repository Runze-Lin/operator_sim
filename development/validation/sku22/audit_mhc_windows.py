"""Check each recovered window against an independent per-core timing or terminal-label proof."""
import argparse
from collections import defaultdict
import json
from pathlib import Path
import re
import struct
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
from development.validation.rms_ordered_verify.verify import checked, events, tables, text_bytes
from development.validation.sku19.campaign import write
from development.validation.sku22.mhc_raw_window import raw_windows


def post_ends(profile, trace, observation):
    """Check first/last ticks in the measured Post kernel, including the drain after END."""
    elf = profile/'dump/aicore_binary.o'
    text = text_bytes(elf)
    base = int((profile/'dump/pc_start_addr.txt').read_text(), 16)
    symbols = subprocess.check_output(['readelf','-Ws','--wide',str(elf)],text=True)
    address,size = re.findall(r'^\s*\d+:\s+([a-f\d]{16})\s+(\d+)\s+FUNC\s+GLOBAL\s+\S+\s+\d+\s+'
                              +re.escape(observation['kernel'])+r'\s*$',symbols,re.M)[0]
    entry = int(address,16)
    marks = defaultdict(lambda: defaultdict(list))
    bounds = observation['windows_ticks']
    for event in events(trace):
        name = event['name']
        if name not in ('MOV_XD_SPR','END','END_LABEL'):
            continue
        pc = int(event['args']['pc_addr'],16)
        start = round(event['ts']*1650)
        end = round((event['ts']+event['dur'])*1650)
        core = event['pid']
        if name == 'MOV_XD_SPR' and pc != base+entry:
            continue
        if name == 'END':
            offset=pc-base
            checked(entry<=offset<entry+int(size) and struct.unpack_from('<I',text,offset)[0]==0x41600000,
                    'physical END word/ELF symbol on '+core)
        if name == 'END_LABEL':
            checked(pc==0x11111111,'unexpected terminal label PC on '+core)
        marks[core][name].append([start,end])
    checked(set(marks)==set(bounds),'Post final-label roster differs')
    for core,(start,end) in bounds.items():
        m=marks[core]
        for name in ('MOV_XD_SPR','END','END_LABEL'):
            checked(len(m[name])==1,'expected one '+name+' on '+core)
        checked(m['MOV_XD_SPR'][0][0]==start and m['END'][0][1]<=m['END_LABEL'][0][1]==end,
                'ELF entry or terminal label disagrees with archived core window on '+core)
    return dict(method='All vector ELF-entry instructions identified by symbol and physical END instruction word followed by processor END_LABEL at the exact last tick.',
                entry_offset=hex(entry),cores=dict(marks))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('archive',type=Path)
    args=parser.parse_args()
    archive=args.archive.resolve()
    directory=ROOT/'results/v0.22/recovered'
    for receipt in sorted(directory.glob('mhc*.json')):
        info=json.loads(receipt.read_text())
        if not info['qualified']:
            continue
        obs=info['observation']
        dest=directory/'integrity'/receipt.name
        if dest.is_file():
            previous=json.loads(dest.read_text())
            if previous.get('qualified') and previous.get('trace_sha256') == obs['trace_sha256']:
                continue
        root=ROOT if info.get('new_ca_seconds') else archive
        trace=root/obs['trace']
        profile=next(p for p in trace.parents if p.name.startswith('OPPROF_'))
        audit={'measurement':receipt.name,'trace_sha256':obs['trace_sha256'],'ca_us':obs['ca_us']}
        try:
            if obs.get('raw_proof'):
                audit.update(qualified=True,method='Full primary/control raw proof embedded in receipt',roster_count=obs['physical_end_cores'])
            elif info['op']=='mhc_post':
                audit.update(post_ends(profile,trace,obs),qualified=True)
            else:
                official,=tables((profile.parent.parent/'run.log').read_text())
                _,proof=raw_windows(profile,obs['kernel'],official,obs['windows_ticks'],text_bytes(profile/'dump/aicore_binary.o'),root,control=True)
                audit.update(qualified=True,method='All first popped and final retired ticks exactly equal CA trace on every physical core',proof=proof)
        except (OSError,ValueError,IndexError,StopIteration) as exc:
            audit.update(qualified=False,reason=repr(exc))
        dest.parent.mkdir(exist_ok=True)
        write(dest,audit)
        print(receipt.stem,audit['qualified'],audit.get('reason',''),flush=True)

if __name__=='__main__':main()
