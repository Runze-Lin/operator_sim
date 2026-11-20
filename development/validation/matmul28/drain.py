"""Verify asynchronous engine drains in mixed Matmul from already captured traces."""
import hashlib
import math
from pathlib import Path
import re
import sys

ROOT=Path(__file__).resolve().parents[3]
sys.path.insert(0,str(ROOT))
from development.validation.matmul28.campaign import endpoint, stages
from campaign_collect import events


def recover(job, receipt, folder, *, acquisition=None):
    # Only the original strict terminal-label comparison can take this path.
    if receipt.get('reason')!="ValueError('end marker disagrees with all-core window')":
        return None
    chip=receipt['chip']; m,n,k=job['shape']
    log=(folder/'run.log').read_text(errors='replace')
    assert f'case=matmul api=aclnnMatmul M={m} N={n} K={k} dtype=BF16 layout=ND_ND' in log
    names=re.findall(r'Start profiling on kernel: (\S+)',log)
    ks,gold=stages(dict(kernels=len(names),nchecks=1),chip,folder)
    if gold!=[f'correctness count={m*n} bad=0'] or len(ks)!=1 or 'matmul' not in names[0].lower():
        raise ValueError('wrong numerical or kernel contract')
    stage=ks[0];base=(ROOT/stage['trace']).parent
    cores={p.parent.name:p for p in base.glob('core*/trace.json')}
    if set(cores)!=set(stage['windows']):
        raise ValueError('core inventory mismatch')
    ends={}; offsets=[]
    for core,shard in cores.items():
        marks=endpoint(shard)  # actual physical END mandatory on every participating engine
        physical=marks['END']['tick']
        first=math.inf; last=-math.inf; final=None
        for e in events(shard):
            if e['ph']!='X':continue
            lo=round(e['ts']*1650); hi=round((e['ts']+e['dur'])*1650)
            first=min(first,lo)
            if hi>last:
                last=hi; final=e
        if first>physical or physical>last or final is None:
            raise ValueError('physical END outside shard window')
        x,y=stage['windows'][core]
        offsets.extend([first-round(x*1650),last-round(y*1650)])
        ends[core]=dict(physical_end=physical,first_tick=first,retire_tick=last,
                        last_descriptor=final['name'],issue_tick=round(final['ts']*1650))
    if max(offsets)-min(offsets)>3:
        raise ValueError(f'shard and global clocks disagree ({min(offsets)}, {max(offsets)})')
    ca=stage['ca_us']; pred=job['predictions'][chip]['us']
    return dict(qualified=True,ca_us=ca,model_us=pred,signed_error_pct=100*(pred/ca-1),
                stages=ks,correctness=gold,terminal=ends,origin_ticks_range=[min(offsets),max(offsets)],
                basis='Supplemental physical END + independent min/max of all X descriptors on each shard + official aggregate and core table. Engine drains can retire later than scalar END.',
                acquisition_receipt=receipt['case']+'.json',acquisition_sha256=hashlib.sha256((acquisition or (ROOT/'results/v0.28/matmul'/(receipt['case']+'.json'))).read_bytes()).hexdigest())
