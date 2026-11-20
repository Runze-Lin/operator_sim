"""Score frozen predictions only when comparable endpoint evidence is qualified."""
from pathlib import Path
from statistics import median
from time import perf_counter_ns

from study import DT,PR,FUT,FROZEN,OUT,ROOT,OLD,read,store,digest,load_chip,estimate,scenario


def metrics(rows,key):
    vals=[(v[key],v['us']) for v in rows if v.get('us') is not None and v.get(key) is not None]
    if not vals:return None
    return dict(n=len(vals),wape=100*sum(abs(x-y) for x,y in vals)/sum(y for _,y in vals),
                maximum=100*max(abs(x/y-1) for x,y in vals))


def controllers(path,target):
    """Require one complete simultaneous final horizon and a known independent roster."""
    import re
    from collections import Counter
    from hashlib import sha256
    entries=sorted(path.glob('lpddr_state*.log'))
    if not entries:return dict(qualified=False,reason='No physical controller files archived')
    # DT physical 128-controller identity established by earlier complete native runs.
    # PR needs its own independently fixed identifiers, not a count inferred from output.
    known={p.name for p in (ROOT/'runs/kimi-full-quant20/log').glob('lpddr_state*.log')} if target==DT else None
    info=[];n=Counter();last=set();hist={};fingerprints={}
    hdr=re.compile(rb'Current time: (\d+), tDFI: ([.\d]+) ns, DMC Total bytes: (\d+), DDR Total bytes: (\d+)')
    count=re.compile(rb'Total (\d+)B (Read|Write)\s*:\s*(\d+)')
    try:
        for p in entries:
            raw=p.read_bytes()
            fingerprints[p.name]=sha256(raw).hexdigest()
            marks=list(hdr.finditer(raw));assert marks and b' END ' in raw[marks[-1].end():]
            for m in marks:
                n['DMC']+=int(m[3]);n['DDR']+=int(m[4])
            hist={q:sum(int(size)*int(cnt) for size,k,cnt in count.findall(raw,marks[-1].end()) if k==q) for q in (b'Read',b'Write')}
            assert sum(hist.values())==sum(int(m[3]) for m in marks)
            n['read']+=hist[b'Read'];n['write']+=hist[b'Write'];last.add((int(marks[-1][1]),float(marks[-1][2])))
        assert len(last)==1 and n['read']+n['write']==n['DMC']==n['DDR']
    except (AssertionError,ValueError,IndexError) as exc:return dict(qualified=False,reason='incomplete period or cumulative closure: '+repr(exc))
    qualified=known is not None and known==set(fingerprints) and len(known)==128
    return dict(qualified=qualified,observed_channels=len(entries),known_physical_roster=known is not None,final_header=list(last.pop()),counts=dict(n),
                sha256=fingerprints,scope='full CA process, including metadata; PR observed-channel subset only until independently mapped' if not qualified else 'entire 128-channel CA process, including metadata')


def snapshot():
    reg=read(FROZEN);src=ROOT/'results/v0.28/replay.json';old=read(src)
    eta=reg['predictions']['pair'][DT]['mixed_bus_efficiency']
    small=[];measured=[]
    for r in old['rows']:
        if r['op']!='quant':continue
        sku=r['chip'];c=load_chip(sku);base=estimate('quant',r['shape'],chip=c).latency_us
        assert abs(base-r['current_us'])<1e-9
        q=scenario(r['shape'],c,base,mixed_bus_efficiency=eta,wire_to_configured_bandwidth=1.024 if sku==DT else 1.)
        small.append(dict(chip=sku,shape=r['shape'],old_us=base,candidate_us=q['latency_us'],us=r['ca_us'],
                          trace_sha256=r['trace_sha256'],kernel=r['kernel'],qualified_metric='published all-core instruction window; development replay'))
    b8=read(ROOT/'results/v0.20/full-quant/native.json')
    assert b8['qualified_core_envelope'] and b8['shape']==reg['shapes']['historical_b8'] and b8['instruction_ca_us'] is None and b8['active_aiv']==64
    archive=read(ROOT/'results/v0.27/quant-envelope.json')
    assert archive['wrapper']['wrapper_us']==b8['core_envelope_us']
    old_mix=read(OLD/'freeze.json')
    costs={}
    for role,key,sku in [('historical native B8','historical_b8',DT),('previously running native B4','existing_b4',DT),('official forward pair','pair',DT),('official forward pair','pair',PR)]:
        p=reg['predictions'][key][sku];trace=receipt=None;backend='native wrapper envelope' if key!='pair' else 'all-core instruction window'
        ca=None;traffic=None;reason='pending'
        if key=='historical_b8':
            ca=b8['core_envelope_us'];receipt=ROOT/'results/v0.20/full-quant/native.json';traffic=archive['controller'];costs[key]=b8['cost']
        elif key=='existing_b4' and (OLD/'native.json').exists():
            receipt=OLD/'native.json';x=read(receipt)
            new_native=x.get('core_envelope_us') if x.get('qualified_core_envelope') else None
            if new_native is not None:
                assert x['freeze_sha256']==digest(OLD/'freeze.json') and x['capture']['cost']['finished_utc']>reg['utc']
                assert x['capture']['output_counts']==[131072*7168,131072*7168//32] and x['capture']['active_aiv']==64
                ca=new_native
            reason=str(x.get('failed_checks'));costs[key]=x.get('capture',{}).get('cost')
        elif key=='pair' and (OUT/(sku+'.json')).exists():
            receipt=OUT/(sku+'.json');x=read(receipt);assert x['freeze_sha256']==digest(FROZEN)
            if x['qualified']:ca=x['instruction_us'];trace=x['trace_sha256']
            reason=x.get('reason');f=OUT/f'{sku}-controllers.json'
            if not f.exists():store(f,controllers(ROOT/x['controller_raw'],sku))
            traffic=read(f);costs[key+'-'+sku]=x['cost']
        receipt=str(receipt.relative_to(ROOT)) if receipt else None
        measured.append(dict(role=role,key=key,chip=sku,shape=reg['shapes'][key],old_us=p['base_us'],resource_us=p['old_resource_us'],candidate_us=p['latency_us'],
                previous_transfer_us=old_mix.get('retrospective_b8' if key=='historical_b8' else 'independent_b4',{}).get('eta_case_us',{}).get('mid') if key!='pair' else None,
                ideal_mixed_service_us=p['ideal_mixed_service_us'],us=ca,metric=backend,source=receipt,trace_sha256=trace,
                official_instruction_us=ca if key=='pair' else None,native_wrapper_us=ca if key!='pair' else None,
                received='qualified' if ca is not None else reason,traffic=traffic))
    aggregate={sku:{k:metrics([r for r in small if r['chip']==sku],k) for k in ('old_us','candidate_us')} for sku in (DT,PR)}
    forward=[v for v in measured if v['key']=='pair']
    speed=None
    if all(v['us'] is not None for v in forward):
        a,b=forward;speed={k:a[k]/b[k] for k in ('us','old_us','candidate_us')}
    return dict(freeze_sha256=digest(FROZEN),scalar_role='changed new-price regime vs historical replay; wrapper and instructions disjoint',
                old_scored_inputs_sha256=digest(src),history=small,history_metrics=aggregate,measurements=measured,ca_cost=costs,
                forward_dt_over_pr=speed,forward_instruction_metrics={v['chip']:{k:metrics([v],k) for k in ('old_us','candidate_us')} for v in forward},
                queue=read(OUT/'state.json') if (OUT/'state.json').exists() else None)


def publish():
    d=snapshot();store(OUT/'score.json',d)
    from documentation import render
    render(d)


if __name__=='__main__':publish()
