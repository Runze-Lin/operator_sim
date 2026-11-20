"""Identify the code observed on each mixed engine, preserving their shared native PC origin."""
from hashlib import sha256
import json
from pathlib import Path
import re
import sqlite3
import struct
from subprocess import check_output

ROOT=Path(__file__).resolve().parents[3]
OUT=ROOT/'results/v0.34/indexer'

def digest(path):
    h=sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda:stream.read(1024*1024), b''):
            h.update(chunk)
    return h.hexdigest()

def read_image(path):
    sections=check_output(['readelf','-W','-S',str(path)],text=True)
    m=re.search(r'\[\s*\d+\]\s+\.text\s+PROGBITS\s+([0-9a-f]+)\s+([0-9a-f]+)\s+([0-9a-f]+)',sections)
    virtual,start,size=(int(v,16) for v in m.groups())
    text=path.read_bytes()[start:start+size]
    symbols={}
    for line in check_output(['readelf','-W','-s',str(path)],text=True).splitlines():
        row=line.split()
        if len(row)>=8 and row[3]=='FUNC' and int(row[2])>0:
            symbols[row[-1]]=(int(row[1],16),int(row[2]))
    return virtual,text,symbols

def match_engine(virtual,text,raw):
    # Do not rebase the AIV separately when intersecting matches with the AIC.
    first=raw[0][0]
    assert [r[0] for r in raw[:6]]==list(range(first,first+24,4))
    words=[(pc-first,struct.pack('<I',int(code,16))) for pc,code,count in raw]
    signature=b''.join(word for _,word in words[:6])
    candidates=[];best=[];best_count=-1;pos=0
    while (offset:=text.find(signature,pos))!=-1:
        pos=offset+1
        if offset%4:continue
        matches=sum(text[offset+pc:offset+pc+4]==word for pc,word in words)
        if matches==len(words):candidates.append(virtual+offset-first)
        if matches==best_count:best.append(virtual+offset)
        if matches>best_count:best_count=matches;best=[virtual+offset]
    return candidates,dict(raw_pc_start=first,raw_pc_end=raw[-1][0],distinct_pc_encodings=len(words),
                          distinct_pcs=len(set(pc for pc,_,_ in raw)),executed_records=sum(c for _,_,c in raw),
                          best_agreeing_pc_encodings=best_count,best_entry_matches_hex=list(map(hex,best)),
                          exact_common_origin_candidates_hex=list(map(hex,candidates)))

def inspect(db_path,row,image,refs):
    virtual,text,symbols=image
    assert digest(db_path)==row['sha']
    sets=[];engines={};pc_by_kind={}
    with sqlite3.connect(f'file:{db_path}?mode=ro&immutable=1',uri=True) as db:
        assert db.execute('pragma quick_check').fetchone()==('ok',)
        info=dict(db.execute('select * from KernelInfo'))
        for kind, in db.execute('select distinct CoreTypeId from ExecutedInstructions order by 1'):
            raw=db.execute('''select SourceInstrAddr,SourceInstrEncoding,count(*) from ExecutedInstructions where CoreTypeId=?
                              group by SourceInstrAddr,SourceInstrEncoding order by SourceInstrAddr''',(kind,)).fetchall()
            candidates,stats=match_engine(virtual,text,raw)
            sets.append(set(candidates));engines[kind]=stats;pc_by_kind[kind]=raw
    assert set(engines)=={1,2,3}
    assert sum(v['executed_records'] for v in engines.values())==info['instrs_exec_num']
    common=sorted(set.intersection(*sets))
    mapped=False;runtime_base=None
    if len(common)==1:
        base,=common
        kernel,={r['kernel'] for r in refs}
        for kind,name in [(1,kernel),(2,kernel.replace('_mix_aic','_mix_aiv')),(3,kernel.replace('_mix_aic','_mix_aiv'))]:
            lo,size=symbols[name];hi=lo+size
            entry=base+engines[kind]['raw_pc_start']
            engines[kind].update(expected_symbol=name,matched_entry_hex=hex(entry),symbol_start_hex=hex(lo),
                symbol_size_B=size,symbol_entry_matches=entry==lo,
                unique_pc_encodings_outside_entry_body=sum(not lo<=base+pc<hi for pc,code,n in pc_by_kind[kind]))
        mapped=all(e['symbol_entry_matches'] for e in engines.values())
        runtime_base=hex(int(info['start_pc'])-base)
    same_geometry=[r for r in refs if r['shape']==row['shape']]
    same_target=[r for r in same_geometry if r['chip']==row['chip']]
    native_ticks=int(info['exec_time_ticks'])
    assert native_ticks==row['duration_ticks']
    return dict(shape=row['shape'],native_chip=row['chip'],native_source=row['source'],native_sha256=row['sha'],
        native_kernel_label=row['kernel'],native_start_pc_hex=hex(int(info['start_pc'])),native_object_path=info['object_path'],
        engines=engines,common_text_origin_candidates_hex=list(map(hex,common)),runtime_image_base_hex=runtime_base,
        exact_observed_mixed_program=mapped,official_refs_same_geometry=[r['source'] for r in same_geometry],
        matched_same_chip_timing=[dict(source=r['source'],native_ticks=native_ticks,official_ticks=r['duration_ticks'],
            native_over_official_signed_pct=100*(native_ticks/r['duration_ticks']-1),native_mmad=sum(row['mmads'].values()),
            official_mmad=sum(r['mmads'].values()),native_histograms=row['total_histogram_instructions'],
            official_histograms=r['total_histogram_instructions']) for r in same_target])

def main():
    manifest=OUT/'physical.json'
    data=json.loads(manifest.read_bytes());native=[r for r in data if r['protocol']=='npusim']
    profiles=[r for r in data if r['protocol']=='msopprof']
    refs=[];images={}
    for r in profiles:
        folder=(ROOT/r['source']).parents[1]/'dump'
        obj=folder/'aicore_binary.o';h=digest(obj)
        reported=(folder/'object_dump.txt').read_text().splitlines()[-1]
        assert reported==r['kernel']
        images[h]=obj
        refs.append(dict(shape=r['shape'],chip=r['chip'],source=r['source'],trace_sha256=r['sha'],duration_ticks=r['duration_ticks'],
            kernel=r['kernel'],mmads=r['mmads'],total_histogram_instructions=r['total_histogram_instructions'],
            allocated_base_hex=(folder/'pc_start_addr.txt').read_text().strip(),elf_sha256=h,elf=str(obj.relative_to(ROOT))))
    # All supplied targets must agree before using the shared image for a different workload.
    assert len(images)==1 and len(native)==13
    image=read_image(next(iter(images.values())))
    rows=[inspect(ROOT/r['source'],r,image,refs) for r in native]
    for r in rows:
        if r['exact_observed_mixed_program']:
            assert {r['runtime_image_base_hex']}=={v['allocated_base_hex'] for v in refs}
    result=dict(scope='Observed instructions on all three core types must agree with a single load-base offset in the official ELF .text. AIV aliases resolved only by the shared relative PC origin and the AIC. Executed helper positions outside the entry body are checked in .text too. No claim about unexecuted bytes or calibration of a different shape, chip, timing protocol, or kernel.',
                physical_manifest_sha256=digest(manifest),official_elf_sha256=next(iter(images)),
                official_receipts=refs,verified_compact=len([r for r in rows if r['exact_observed_mixed_program']]),total_compact=len(rows),rows=rows)
    (OUT/'identity.json').write_text(json.dumps(result,indent=2,ensure_ascii=False)+'\n')
if __name__=='__main__':main()
