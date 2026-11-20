"""Separate fully observed Cube partitioning from incomplete PR final-store exports."""
from collections import Counter, defaultdict
import csv
import hashlib
import json
from pathlib import Path
import struct
import sys
import time

ROOT=Path(__file__).resolve().parents[4]
sys.path.insert(0,str(ROOT))
from development.validation.matmul31.device.campaign import ca, REG, OUT
from development.validation.matmul31.device.traffic import flattened
from development.validation.matmul31.device.work_audit import expected,describe


def observed(d):
    return not any(d.get(f) for f in ('is_wait','is_sync','is_duration_excluded'))


def transfers(events):
    rows=[]
    for e in events:
        if e.get('ph')!='X':continue
        d=json.loads(e['args']['detail'])
        if not observed(d):continue
        # Test every mnemonic that advertises a UB-to-GM write. Don't whitelist MOV.
        if d.get('src_mem')=='UB' and d.get('dst_mem')=='OUT':
            rows.append((e['name'],e['args']['pc_addr'],d['dst_addr'],d['burst_number'],d['burst_len'],d['dst_stride']))
    return rows


def audit(leaf):
    p=OUT/leaf;receipt=ca.read(p);m,n,k=receipt['shape'];s,=receipt['stages'];base=(ROOT/s['trace']).parent
    pred=next(v for v in ca.read(REG)['rows'] if v['shape']==receipt['shape'])['predictions'][receipt['chip']]['geometry']
    whole=json.loads((base/'trace.json').read_bytes())['traceEvents'];by_core=defaultdict(list)
    for e in whole:
        if e.get('ph')=='X':by_core[e['pid']].append(e)
    blob=(base/'visualize_data.bin').read_bytes();sz,tag=struct.unpack_from('<QI',blob);assert blob[12:12+sz].rstrip(b'\0\n')==(base/'trace.json').read_bytes().rstrip(b'\0\n');off=12+sz;sz2,tag2=struct.unpack_from('<QI',blob,off)
    table=json.loads(blob[off+12:off+12+sz2].rstrip(b'\0\n'));cores=table['Cores'];osrows=[x for x in table['Instructions'] if x['Pipe']=='MTE3' and '"dst_mem":"OUT"' in x['Source'] and '"src_mem":"UB"' in x['Source']]; assert osrows
    bufs={}; raw_cube={}; coverage={}; destinations=[]; observed_stores=[];hist=Counter()
    for c in sorted(s['windows']):
        path=base/c/'trace.json';rs=path.read_bytes();xs=[e for e in json.loads(rs)['traceEvents'] if e.get('ph')=='X'];events=[];nonsync=[]
        for e in xs:
            x=json.loads(e['args']['detail']);
            if observed(x):events.append((e,x))
        if '.cube' in c:
            moves=[x['src_addr'] for e,x in events if e['name']=='MOV_OUT_TO_L1_MULTI_ND2NZ']; mm=[(x['m'],x['n'],x['k']) for e,x in events if e['name']=='MMAD'];fix=[(x['dst_addr'],x['processed_bytes']) for e,x in events if e['name']=='FIX_L0C_TO_DST'];
            if mm:raw_cube[c]=(moves,mm,fix);hist.update(mm)
            continue
        stores=transfers(xs); same=transfers(by_core[c]);assert sorted(same)==sorted(stores)
        iv=[]
        for nm,pc,addr,sn,width,stride in stores:
            iv.extend((addr+i*stride,addr+i*stride+width) for i in range(sn));observed_stores.append(pc)
        csvfile=base/c/(c+'_instr_exe.csv');csv_counts=0
        # Only the four conventional columns are parsed using CSV; PC is checked in binary perf table.
        pcs={x['Address'] for x in osrows}
        for e in csv.reader(csvfile.read_text().splitlines()):
            if len(e)>3 and e[1].isdigit() and hex(int(e[1])) in pcs:csv_counts+=int(e[3])
        count_table=sum(x['Instructions Executed'][cores.index(c)] for x in osrows);assert csv_counts==count_table==len(stores)
        partial=[x for e,x in events if x.get('src_mem')=='OUT' and x.get('dst_mem')=='UB' and x['burst_number']>1]
        endings=sorted((e['ts']+e['dur'],e['name']) for e in xs)
        coverage[c]=dict(partial_reads=len(partial),write_desc=len(stores),exported_bytes=sum(b-a for a,b in iv),last_exported_instruction=endings[-1][1],all_formats_agree=True,sha256=hashlib.sha256(rs).hexdigest());destinations.extend(iv)
    a0=min(a for moves,_,_ in raw_cube.values() for a in moves);b0=min(a for moves,_,_ in raw_cube.values() for a in moves if a>=a0+2*m*k);rect=defaultdict(list);workers={}
    for c,(moves,mm,fix) in raw_cube.items():
        mi,ki=divmod((min(a for a in moves if a<b0)-a0)//2,k); kb,ni=divmod((min(a for a in moves if a>=b0)-b0)//2,n);assert ki==kb
        dims={(a,b) for a,b,_ in mm};assert len(dims)==1;dm,dn=dims.pop();kend=ki+sum(kk for _,_,kk in mm);rect[mi,ni,dm,dn].append((ki,kend));assert len(fix)==1 and fix[0][1]==4*dm*dn;workers[c]=dict(m0=mi,n0=ni,m=dm,n=dn,k0=ki,k1=kend,mmads=len(mm),fixp=fix[0])
    expected_slices=[(i*pred['max_segment_k'],i*pred['max_segment_k']+(pred['tail_segment_k'] if i==pred['k_splits']-1 else pred['max_segment_k'])) for i in range(pred['k_splits'])];assert all(sorted(z)==expected_slices for z in rect.values())
    for i,(mi,ni,dm,dn) in enumerate(rect):
        assert mi>=0 and ni>=0 and mi+dm<=m and ni+dn<=n and all(max(mi,mj)>=min(mi+dm,mj+em) or max(ni,nj)>=min(ni+dn,nj+en) for mj,nj,em,en in list(rect)[:i])
    assert s['active_aic']==pred['active_aic'] and s['active_aiv']==pred['active_aiv']
    assert sum(dm*dn for _,_,dm,dn in rect)==m*n and len(rect)==pred['output_tiles'] and len(raw_cube)==pred['productive_aic'] and sum(hist.values())==s['mmad']==pred['mmads_total'] and hist==expected((m,n,k),pred)
    union,over=flattened(destinations);assert not over;origin=min(lo for lo,hi in union);missing=[];cursor=origin
    for lo,hi in union:
        if cursor<lo:missing.append([cursor-origin,lo-origin]);cursor=hi
        else:cursor=hi
    if cursor<origin+2*m*n:missing.append([cursor-origin,2*m*n]);cursor=origin+2*m*n
    assert cursor==origin+2*m*n;readers=[c for c,v in coverage.items() if v['partial_reads']];unfinished=[c for c in readers if not coverage[c]['write_desc']];unseen=sum(hi-lo for lo,hi in missing);writes=sum(v['exported_bytes'] for v in coverage.values())
    assert writes+unseen==2*m*n
    return dict(receipt=leaf,receipt_sha256=ca.digest(p),trace_sha256=s['trace_sha256'],freeze_sha256=ca.digest(REG),shape=[m,n,k],chip=receipt['chip'],kernel=s['kernel'],time_qualified=receipt['qualified'],ca_us=receipt['ca_us'],all_gold=receipt['correctness'],export= 'all descriptor names with src_mem=UB and dst_mem=OUT independently searched in every physical AIV shard and all-chip JSON. CSV counts at all GM write PCs and visualize_data.bin per-PC per-core execution counts independently crosschecked; binary timeline is byte-identical to global JSON.',cube_geometry=dict(matches_expected=True,groups=len(rect),slices_per_group=sorted({len(v) for v in rect.values()}),productive_aic=len(raw_cube),active_aic=s['active_aic'],active_aiv=s['active_aiv'],rectangles=[dict(m0=mi,n0=ni,m=dm,n=dn,k_spans=sorted(v)) for (mi,ni,dm,dn),v in rect.items()],mmad=describe(hist),cores=workers),end_writes=dict(fully_observed=not unseen,visible_bytes=writes,expected_bytes=2*m*n,unseen_bytes=unseen,missing_output_relative_byte_ranges=missing,partial_readers=len(readers),store_writers=sum(v['write_desc']>0 for v in coverage.values()),partial_readers_without_write=unfinished,store_pcs=sorted(set(observed_stores)),per_aiv=coverage))

def supplement(leaf):
    receipt_path=OUT/leaf; strict=OUT/(leaf[:-5]+'-strict.json'); original=ca.read(receipt_path)
    # Immutable snapshots; the score's active receipt may gain diagnostics but
    # neither the original time qualification nor a missing store is upgraded.
    if original.get('mechanism_checks') is None and not strict.exists():
        with strict.open('xb') as f:f.write(receipt_path.read_bytes())
    source=strict.name if strict.exists() else leaf
    out=audit(source);coverage_path=OUT/(leaf[:-5]+'-coverage.json');ca.save(coverage_path,out)
    if strict.exists():
        d=ca.read(strict);g=out['cube_geometry'];dest=out['end_writes'];without=dest['partial_readers_without_write']
        d['actual_geometry']=dict(groups=g['groups'],splits_per_output=g['slices_per_group'],productive_aic=g['productive_aic'],active_aic=g['active_aic'],active_aiv=g['active_aiv'],actual_mmad=g['mmad']['mmads'],mmads=g['mmad']['histogram'],output_rectangles=g['rectangles']);d['mechanism_checks']=dict(k_spans=True,groups=True,productive_aic=True,active_aic=True,active_aiv=True,mmads=True,final_output_descriptors_complete=dest['fully_observed']);d['secondary_evidence']=dict(original_strict_receipt=strict.name,original_sha256=ca.digest(strict),coverage=coverage_path.name,coverage_sha256=ca.digest(coverage_path),note=f'Cube address map and MMAD histogram passed. Exported stores account for {dest["visible_bytes"]}/{dest["expected_bytes"]} output bytes; {len(without)} partial readers lack a final write in all saved trace views. Full numerical correctness and official time still qualified; incomplete instruction work is not extrapolated.');ca.save(receipt_path,d)
    return out

def follow():
    # Passive raw-data audits only. Preserve failed audits rather than promoting them.
    reg=ca.read(REG)
    for j in reg['rows']:
        for chip,tag in ((ca.DT,'dt'),(ca.PR,'pr')):
            stem='-'.join(map(str,j['shape']))+'-'+tag;leaf=stem+'.json';completed=OUT/(stem+'-coverage.json');failure=OUT/(stem+'-coverage-failure.json')
            if completed.exists() or failure.exists():continue
            while not (OUT/leaf).exists():time.sleep(15)
            d=ca.read(OUT/leaf)
            if not d['qualified']:
                ca.save(failure,dict(qualified=False,reason=d.get('reason'),receipt=leaf));continue
            try:supplement(leaf)
            except (KeyError,ValueError,AssertionError,OSError) as exc:ca.save(failure,dict(receipt=leaf,reason=repr(exc),evidence='Original receipt preserved; supplemental output-coverage audit not qualified.'))
    from development.validation.matmul31.device.report import render
    from development.validation.matmul31.device.work_audit import build
    render();build()

if __name__=='__main__':
    if '--watch' in sys.argv:follow()
    else:
        for leaf in sys.argv[1:]:
            out=supplement(leaf);print(out['chip'],out['cube_geometry']['groups'],out['end_writes']['unseen_bytes'])
