"""Confirm per-core physical END, shard windows and previously observed released binary."""
import json,sys
from pathlib import Path
from audit import ROOT, OUT, read
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'sku17'))
from campaign_collect import events,sha


def verify(path,known):
    r=read(path);s,=r['stages'];assert r['qualified']
    trace=ROOT/s['trace'];assert sha(trace)==s['trace_sha256']
    binary=trace.parents[1]/'dump/aicore_binary.o'
    assert sha(binary)==s['binary_sha256'] and (s['kernel'],s['binary_sha256']) in known
    cores={}
    for shard in sorted(trace.parent.glob('core*/trace.json')):
        ecount=0;labels=[];first=ends=float('inf');last=-first
        for e in events(shard):
            if e['ph']!='X':continue
            t=e['ts'];z=t+e['dur'];first=min(first,t);last=max(last,z)
            if e['name']=='END':
                ends=z;ecount+=1
            if e['name']=='END_LABEL':labels.append(z)
        if ecount!=1 or len(labels)>1 or any(abs(x-last)>.002 for x in labels):
            raise ValueError((r['case_id'],shard,ecount,labels,last))
        cores[shard.parent.name]=dict(first=first,last=last,physical_end=ends,end_label=labels,
                                     shard_sha256=sha(shard))
    assert set(cores)==set(s['windows'])
    origin=min(v['first'] for v in cores.values())
    for core,v in cores.items():
        lo,hi=s['windows'][core]
        assert abs(v['first']-origin-lo)<.002 and abs(v['last']-origin-hi)<.002
    assert abs(max(v['last'] for v in cores.values())-origin-r['ca_us'])<.002
    return dict(id=r['case_id'],receipt=path.name,qualified=True,source_sha256=sha(path),device_elf_sha256=s['binary_sha256'],
                kernel=s['kernel'],ca_us=r['ca_us'],earliest_global_us=origin,cores=cores)

if __name__=='__main__':
    archive=read('results/v0.21/coverage/audit.json')['rows']
    known={(x['kernel'],x['binary_sha256']) for x in archive if x['op']=='fa' and isinstance(x.get('binary_sha256'),str)}
    records=[verify(p,known) for p in sorted(OUT.glob('fa-Ascend*.json'))]
    (OUT/'fa-physical-windows.json').write_text(json.dumps(records,indent=2)+'\n')
