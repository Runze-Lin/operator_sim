"""Read completed physical events after a host deadline, without changing the acquisition receipt."""
import hashlib
import json
from pathlib import Path
import re
import sys
from collections import Counter

ROOT=Path(__file__).resolve().parents[3]
sys.path.insert(0,str(ROOT))
from development.validation.matmul28.campaign import endpoint
from campaign_collect import events

OUT=ROOT/'results/v0.30/host'


def scan(path):
    bounds={}; counts=Counter()
    for e in events(path):
        if e['ph']!='X':continue
        core=e['pid']; t=round(e['ts']*1650); end=round((e['ts']+e['dur'])*1650)
        if end<t:raise ValueError('negative duration')
        old=bounds.get(core,(t,end));bounds[core]=(min(t,old[0]),max(end,old[1]));counts[e['name']]+=1
    return bounds,counts


def audit():
    raw=ROOT/'runs/host30/control-pr'
    log=(raw/'run.log').read_text(errors='replace')
    measurement=json.loads((OUT/'pr.json').read_bytes())
    ref=json.loads((OUT/'freeze.json').read_bytes())['controls'][measurement['chip']]
    assert measurement['cost']['stop_reason']=='wall cap' and not measurement['validated_host_control']
    assert re.findall(r'^runtime_soc=(\S+)$',log,re.M)==[measurement['chip']] and 'lpddr5_6400M_bailu.ini' in log
    assert re.findall(r'^correctness [^\r\n]+',log,re.M)==ref['gold']
    kernel,=re.findall(r'Start profiling on kernel: (\S+)',log)
    assert kernel==ref['kernel']
    trace,=raw.glob('profile/OPPROF_*/device0/*/0/simulator/trace.json')
    bounds,counts=scan(trace); rows={}
    roster=re.findall(r'^(core\d+\.(?:cube|vec)core\d+)\s+[.\d]+\s+[.\d]+\s*$',log,re.M)
    assert sorted(bounds)==sorted(roster)==ref['roster']
    offsets=[]
    for shard in trace.parent.glob('core*/trace.json'):
        core=shard.parent.name
        queues=scan(shard)[0].values()
        first=min(v[0] for v in queues)
        last=max(v[1] for v in queues)
        end=endpoint(shard)['END']['tick']
        assert first<=end<=last
        offsets.extend((first-bounds[core][0],last-bounds[core][1]))
        rows[core]=dict(first=first,physical_end=end,retired=last)
    assert sorted(rows)==ref['roster'] and max(offsets)-min(offsets)<=3
    duration=(max(x[1] for x in bounds.values())-min(x[0] for x in bounds.values()))/1650
    mmads=sum(v for k,v in counts.items() if k.startswith('MMAD'))
    assert mmads==ref['mmad']
    result=dict(chip=measurement['chip'],shape=measurement['shape'],physical_event_window_us=duration,
                original_qualified_us=ref['baseline_us'],relative_delta_pct=100*(duration/ref['baseline_us']-1),
                correctness=ref['gold'],active_aic=len(rows),active_aiv=0,mmad=mmads,raw=str(trace.relative_to(ROOT)),
                trace_sha256=hashlib.sha256(trace.read_bytes()).hexdigest(),per_core=rows,clock_offset_span=max(offsets)-min(offsets),
                acquisition_qualified=False,host_control_validated=False,
                scope='Supplemental event reconstruction only. Official process hit 480 s cutoff. Frozen host-control acceptance and original measurement are preserved.')
    (OUT/'pr-retirement.json').write_text(json.dumps(result,indent=2)+'\n')


if __name__=='__main__':audit()
