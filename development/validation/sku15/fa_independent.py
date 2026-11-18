"""Post-freeze 32k K/V issue-cadence audit from official PR JSON + landmarks DB."""
from collections import defaultdict
import json
from pathlib import Path
import sqlite3
from statistics import mean, median
import sys

ROOT=Path(__file__).resolve().parents[3]
sys.path.insert(0,str(ROOT))
from operator_model.ca import sha256

OUT=ROOT/'results/v0.15'


def events(path):
    with path.open() as fd:
        chunk=fd.read(1<<20).split('"traceEvents":[',1)[1]
        decode=json.JSONDecoder().raw_decode
        i=0
        while True:
            if len(chunk)-i<100000:
                chunk=chunk[i:]+fd.read(1<<20)
                i=0
            if chunk[i]==']':
                break
            obj,i=decode(chunk,i)
            yield obj
            if chunk[i]==',':
                i+=1


def slopes(times):
    def slope(start,stop):
        x=range(start,stop)
        xm=mean(x)
        return sum((i-xm)*times[i] for i in x)/sum((i-xm)**2 for i in x)
    return {'slope_8_120':slope(8,120),'slope_blocks':[slope(a,b) for a,b in [(8,40),(40,80),(80,120)]],
            'middle_median_delta':median(times[i]-times[i-1] for i in range(9,120))}


def phases(r):
    marks=[d['MMAD'] for d in r['pr']['phase_landmarks'].values() if 'MMAD' in d]
    start=min(d['first_start_us'] for d in marks)
    end=max(d['last_end_us'] for d in marks)
    return dict(prefix=start,window=end-start,drain=r['pr_ca_us']-end,mmads=sum(d['count'] for d in marks))


def main():
    hold=[r for r in json.loads((OUT/'sku/holdout.json').read_text()) if r['op']=='fa']
    assert len(hold)==10
    r=next(r for r in hold if r['shape']==[9,32768,64,1,1])
    trace=ROOT/r['pr']['trace']
    assert sha256(trace)==r['pr']['trace_sha256']
    streams=defaultdict(list)
    for e in events(trace):
        if e['ph']=='X' and e['name']=='MOV_OUT_TO_L1_MULTI_ND2NZ':
            d=json.loads(e['args']['detail'])
            if d['n']==256 and d['d']==64:
                streams[e['args']['pc_addr']].append((e['ts'],d.get('src_addr')))
    seq=sorted(streams.items(),key=lambda item:min(t for t,_ in item[1]))
    assert [len(steps) for _,steps in seq]==[128,128]
    cadence={}
    for label,(pc,steps) in zip(('k','v'),seq):
        steps.sort()
        cadence[label]=dict(pc=pc,count=len(steps),source_stride_bytes=32768,
                            issue_since_first_us=[round(t-steps[0][0],8) for t,_ in steps],
                            **slopes([t for t,_ in steps]))
        src=[s for _,s in steps]
        assert all(y-x==32768 for x,y in zip(src,src[1:]))
    path=ROOT/'runs/sku15'/r['id']/'landmarks.db'
    with sqlite3.connect(path.resolve().as_uri()+'?mode=ro',uri=True) as db:
        mmads,start,end=db.execute("select count(*),min(ts_us),max(ts_us+duration_us) from landmarks where opcode='MMAD'").fetchone()
    obs=phases(r)
    assert obs['mmads']==mmads==512 and abs(obs['window']-(end-start))<1e-7
    short=max((r for r in hold if r['shape'][1]<=2048),key=lambda r:abs(r['pr_us']/r['pr_ca_us']-1))
    issue=max(d['slope_8_120'] for d in cadence.values())
    prior=json.loads((OUT/'fa-phases.json').read_text())
    info={'sku':'Ascend950PR_957b','shape':r['shape'],'trace_sha256':r['pr']['trace_sha256'],'landmarks_sha256':sha256(path),
          'frozen_us':r['pr_us'],'ca_us':r['pr_ca_us'],'pr32k_phases_us':obs,'pr32k_streams_us':cadence,
          'pr32k_derived_fill_us':obs['window']-127*issue,'pr32k_effective_issue_us':issue,
          'pr32k_forecast':r['projection']['timing_diagnostics'],'development_long_phases':prior['long'],
          'ordinary_max_error':dict(shape=short['shape'],dt_us=short['dt_us'],pr_us=short['pr_us'],
                     ca_us=short['pr_ca_us'],observed=phases(short),forecast=short['projection']['timing_diagnostics'],
                     actual_cores=[short['active_aic'],short['active_aiv']],estimated_cores=[short['pr_aic'],short['pr_aiv']],
                     trace_sha256=short['pr']['trace_sha256'])}
    (OUT/'fa-independent.json').write_text(json.dumps(info,indent=2)+'\n')


if __name__=='__main__':
    main()
