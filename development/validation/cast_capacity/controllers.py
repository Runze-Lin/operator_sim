"""Decode cumulative command-size histograms in archived v0.20 DT controller logs."""
from collections import defaultdict
from hashlib import sha256
import json
from pathlib import Path
import re

ROOT=Path(__file__).resolve().parents[3]
OUT=ROOT/'results/cast-capacity'
LARGE=ROOT/'results/v0.20/large'
PERIOD=re.compile(rb'Current time: (\d+), tDFI: [\d.]+ ns, DMC Total bytes: (\d+), DDR Total bytes: (\d+)')
HIST=re.compile(rb'Total (32|64|128|256)B (Read|Write)\s*:\s*(\d+)')


def summarize(folder,reference):
    files=sorted(folder.glob('lpddr_state*.log'))
    if len(files)!=128:
        raise ValueError('need all 128 physical controller dumps')
    periods=defaultdict(lambda:[0,0,0,0])
    fingerprints={}
    for file in files:
        raw=file.read_bytes()
        fingerprints[file.name]=sha256(raw).hexdigest()
        prev=[0,0];stamps=list(PERIOD.finditer(raw))
        for i,match in enumerate(stamps):
            stop=stamps[i+1].start() if i+1<len(stamps) else len(raw)
            # One histogram per report block; cumulative, unlike interval DMC bytes.
            bins=HIST.findall(raw,match.end(),stop)
            assert len(bins)==8,(str(file),i,len(bins))
            now=[sum(int(size)*int(count) for size,kind,count in bins if kind==q) for q in (b'Read',b'Write')]
            rd,wr=now[0]-prev[0],now[1]-prev[1]
            assert min(rd,wr)>=0
            stamp=int(match[1]);dst=periods[stamp]
            dst[0]+=rd;dst[1]+=wr;dst[2]+=int(match[2]);dst[3]+=1
            prev=now
    bins=[dict(end_ns=t,read_bytes=r,write_bytes=w,dmc_bytes=d) for t,(r,w,d,n) in sorted(periods.items()) if n==128]
    assert len(bins)==len(periods)
    read=sum(b['read_bytes'] for b in bins);write=sum(b['write_bytes'] for b in bins)
    assert read+write==reference['dmc_bytes']==sum(b['dmc_bytes'] for b in bins)
    return dict(shape=reference['shape'],read_bytes=read,write_bytes=write,intervals=bins,
                sources=folder.relative_to(ROOT).as_posix(),file_sha256=fingerprints)


if __name__=='__main__':
    all_data=[]
    for p in sorted(LARGE.glob('cast_*_Ascend950DT_9582-official-dram.json')):
        ref=json.loads(p.read_text())
        f=ROOT/'runs/prefill20-memory'/p.name.removesuffix('-official-dram.json')/'log'
        all_data.append(summarize(f,ref))
    (OUT/'controller-periods.json').write_text(json.dumps(all_data,indent=2)+'\n')
