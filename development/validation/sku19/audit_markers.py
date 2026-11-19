"""Separate vendor duration eligibility from recorded per-core epilogues and stores."""
import mmap
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[3]
sys.path.insert(0,str(ROOT))
from development.validation.sku19.campaign import OUT,read,write

def audit(r):
    trace=ROOT/r['trace'][0]
    absent={k:[] for k in ('END_LABEL','MOV_SRC_TO_DST_ALIGNv2')}
    for core in r['stages'][0]['windows']:
        shard=trace.parent/core/'trace.json'
        if not shard.is_file():return dict(id=r['case_id'],chip=r['chip'],unavailable=True)
        with shard.open('rb') as f, mmap.mmap(f.fileno(),0,access=mmap.ACCESS_READ) as stream:
            for token,missing in absent.items():
                if stream.find(('"name":"'+token+'"').encode())==-1:missing.append(core)
    return dict(id=r['case_id'],op=r['op'],shape=r['shape'],chip=r['chip'],active_aiv=r['active_aiv'],
                missing=absent,trace_sha256=r['trace_sha256'][0])
def main():
    path=OUT/'marker-audit.json'
    existing=read(path) if path.exists() else []
    by_sha={r.get('trace_sha256'):r for r in existing}
    audits=[]
    confirmations=OUT.parent/'rms-confirmation/records.json'
    for r in read(OUT/'final-records.json')+(read(confirmations) if confirmations.exists() else []):
        result=by_sha.get(r['trace_sha256'][0]) or audit(r)
        audits.append(result)
    write(path,audits)
    print(len(audits),'qualified records inspected',sum(any(v for v in a.get('missing',{}).values()) for a in audits),'with absent labels')
if __name__=='__main__':main()
