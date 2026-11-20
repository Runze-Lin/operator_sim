"""Locate all executed SQLite instruction encodings at one symbol of a profiled installed ELF."""
from pathlib import Path
import re
import sqlite3
import struct
from subprocess import check_output
from watch import ROOT,OUT,digest,read,save

def identify(native,paired):
    p=ROOT/native['source']
    image,=(ROOT/paired['source']).parent.glob('profile/*/dump/aicore_binary.o')
    symbol=paired['kernel']
    names = check_output(['readelf','--wide','--symbols',str(image)],text=True)
    line,=[r for r in names.splitlines() if r.split() and r.split()[-1]==symbol and 'FUNC' in r.split()]
    addr,size=int(line.split()[1],16),int(line.split()[2])
    sections=check_output(['readelf','--wide','--section-headers',str(image)],text=True)
    m=re.search(r'\[\s*\d+\]\s+\.text\s+PROGBITS\s+([0-9a-f]+)\s+([0-9a-f]+)\s+([0-9a-f]+)',sections)
    virtual,text_start,text_size=(int(v,16) for v in m.groups())
    text=image.read_bytes()[text_start:text_start+text_size]
    with sqlite3.connect(f'file:{p}?mode=ro&immutable=1',uri=True) as db:
        codes=db.execute('SELECT DISTINCT SourceInstrAddr, SourceInstrEncoding FROM ExecutedInstructions ORDER BY SourceInstrAddr').fetchall()
        entry_pc=int(db.execute("SELECT KernelInfoVal FROM KernelInfo WHERE KernelInfoAttr='start_pc'").fetchone()[0])
    assert [v[0] for v in codes[:6]]==list(range(0,24,4))
    assert all(a%4==0 and 0<=a<=size-4 for a,c in codes)
    words=[(a,struct.pack('<I',int(c,16))) for a,c in codes]
    signature=b''.join(v for _,v in words[:6]);pos=0;matches=[]
    while (candidate:=text.find(signature,pos))!=-1:
        pos=candidate+1
        if candidate%4==0 and all(text[candidate+a:candidate+a+4]==word for a,word in words):
            matches.append(candidate+virtual)
    assert matches==[addr]
    return dict(shape=native['shape'],native_sku=native['sku'],comparison_sku=paired['sku'],
                symbol=symbol,text_address_hex=hex(addr),symbol_size_B=size,
                native_start_pc_hex=hex(entry_pc),unique_executed_pc_encodings=len(words),elf_text_matches=len(matches),
                reference_executable=str(image.relative_to(ROOT)),reference_sha256=digest(image),
                native_sqlite=native['source'],native_sha256=native['source_sha256'])

if __name__=='__main__':
    data=read(OUT/'compact-db.json');matches={tuple(r['shape']):r for r in data['exact_geometry_msopprof_matches']}
    rows=[identify(r,matches[tuple(r['shape'])]) for r in data['rows']]
    save('sqlite-kernel-identity.json',dict(rows=rows,scope='All observed PC encodings uniquely match the named function within the captured official ELF. Does not authenticate unexecuted bytes in a native launch or equate the 950DT and 950PR memory model.'))
