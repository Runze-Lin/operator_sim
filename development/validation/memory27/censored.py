"""Use only complete lanes of a failed PR capture to bound read demand; no full time."""
from collections import Counter
from math import floor,ceil
from pathlib import Path
import json,sys

from retention import ROOT, OUT, extract, read
from read_windows import peak_interval, earliest_deadline_missed

sys.path.insert(0,str(ROOT/'development/validation/cache25'))
from development.validation.cache25 import campaign
from descriptors import endpoint


def main():
    j=read(campaign.REG)['jobs'][0]
    folder=campaign.RAW/j['id']
    failure=read(campaign.OUT/(j['id']+'-result.json'))
    assert failure['qualified'] is False and j['op']=='cast_bf16_fp32'
    stages,correct=campaign.stages(j,j['sku'],folder)
    s,=stages
    assert correct==['correctness count='+str(j['expected_elements'])+' bad=0']
    assert s['kernel']==j['kernel'] and s['binary_sha256']==j['elf']
    base=(ROOT/s['trace']).parent
    # All timed reads of the payload are cold candidates; identify the input
    # from its complete, contiguous address range independently of END markers.
    from descriptors import tally
    events={};missing={};origins={}
    for core,window in s['windows'].items():
        path=base/core/'trace.json'
        events[core]=[row for kind,row in extract(path) if kind=='read']
        try:
            ticks=endpoint(path)
        except ValueError as problem:
            missing[core]=str(problem).split(' in ')[0]
            continue
        origins[core]=max(e['retire'] for e in ticks.values())/1650.-window[1]
    assert origins and missing
    assert max(origins.values())-min(origins.values())<3/1650
    region,other=tally([row for rows in events.values() for row in rows],2*j['expected_elements'])
    jobs=[]
    for core,origin in origins.items():
        for addr,n,issue,retire in events[core]:
            if region['gm_base']<=addr and addr+n<=region['gm_end']:
                jobs.append((floor((issue-origin)*1650)-1,ceil((retire-origin)*1650)+1,n))
    rate,excess=peak_interval(jobs,1650.,1400.)
    missed=earliest_deadline_missed(jobs,1400*1000/1650)
    out=dict(role='Censored lower bound. Original full-window job failed; no independent latency or all-core read-window verdict.',
             valid_lanes=len(origins),censored_lanes=missing,accounted_read_bytes=sum(n for _,_,n in jobs),
             logical_read_bytes=region['bytes'],kernel=s['kernel'],elf=s['binary_sha256'],
             trace_sha256=s['trace_sha256'],source=j['id'],ca_us=None,peak_rate=rate,max_excess=excess,
             missed_bytes_subset=missed,above_preregistered_threshold_in_subset=rate['rate_GBps']>1414.,
             relation='Adding omitted reads cannot reduce interval demand. All included lanes passed endpoint checks.')
    (OUT/'pr156-partial.json').write_text(json.dumps(out,indent=2)+'\n')


if __name__=='__main__':
    main()
