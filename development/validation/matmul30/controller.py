"""Use independent Cast/quant controller periods to price read/write traffic mix, without kernel times."""
from collections import defaultdict
from hashlib import sha256
import json
from pathlib import Path
from statistics import median
import sys

ROOT=Path(__file__).resolve().parents[3]
sys.path.insert(0,str(ROOT/'development/validation/kda27'))
from quant_service import PANEL,HEADER,FIELDS,NAMES

OUT=ROOT/'results/v0.30/matmul'
SOURCE=ROOT/'results/v0.27/quant-channel-service.json'


def observe():
    source=json.loads(SOURCE.read_bytes());rows=[]
    for s in source['rows']:
        bins=defaultdict(lambda:[0,0,0]);files=sorted((ROOT/s['raw']).glob('lpddr_state*.log'))
        assert len(files)==128
        for path in files:
            raw=path.read_bytes();assert sha256(raw).hexdigest()==s['channels'][path.name]['sha256']
            previous=0
            for page in raw.split(PANEL)[1:]:
                end,dmc,ddr=map(int,HEADER.search(page,0,400).groups())
                dt=end-previous;previous=end
                if dt!=1000:continue
                index=page.index(b' DMC Pressure Statistics ')
                line=page[page.index(b'\nCycle  |',index)+1:].splitlines()[0]
                hist=[int(x) for x in line.split(b'|')[1:] if x.strip()]
                if hist[0]*20>sum(hist):continue
                left=page.index(b' Request Statistics ');right=page.index(b' Latency Statistics ',left)
                fields={n:int(a) for n,a,_ in FIELDS.findall(page,left,right)}
                r=sum(fields[n] for n in NAMES[:2]);w=sum(fields[n] for n in NAMES[2:6]);beats=r+w
                assert beats*32==ddr
                if not beats:continue
                # Integer rounding to the nearest decile; ties go upward.
                idx=min(10,(20*w+beats)//(2*beats))
                b=bins[idx];b[0]+=r;b[1]+=w;b[2]+=dt
        rows.append(dict(op=s['case'],shape=s['shape'],bins={k:dict(read_beats=v[0],write_beats=v[1],slots=v[2],eta=(v[0]+v[1])/v[2]) for k,v in sorted(bins.items())}))
    consensus={}
    for idx in range(11):
        vals=[r['bins'][idx]['eta'] for r in rows if idx in r['bins'] and r['bins'][idx]['slots']>=128000]
        if vals:consensus[idx]=dict(min_eta=min(vals),median_eta=median(vals),max_eta=max(vals),operator_count=len(vals))
    out=dict(protocol='Independent Cast and Quant controller counts, 128 950DT channels, 32 B beats, 1000-cycle DFI periods with >=95% occupied DMC queue. Report by nearest decile of physical write fraction. Each source with >=128 channel-periods contributes one mean per bin; median across sources. No Matmul time, no operator total time used.',source_sha256=sha256(SOURCE.read_bytes()).hexdigest(),rows=rows,consensus=consensus)
    p=OUT/'external-service.json';tmp=p.with_suffix('.tmp');tmp.write_text(json.dumps(out,indent=2)+'\n');tmp.replace(p)

if __name__=='__main__':observe()
