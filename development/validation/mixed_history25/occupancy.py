"""Verify the engine coverage of an untimed warm-up from actual distinct output addresses."""
from collections import defaultdict
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[3]
sys.path.insert(0,str(ROOT))
from development.validation.mixed_history25.campaign import OUT,write
from development.validation.rms_ordered_verify import verify as v
from development.validation.mixed_history25.score import score


def mask(folder, shape, tables):
    t,width=shape
    elf=folder/'dump/aicore_binary.o'
    v.checked(v.text_bytes(elf)==v.text_bytes(v.VENDOR),'unexpected RMS kernel binary')
    base=int((folder/'dump/pc_start_addr.txt').read_text(),16)
    actual={};cores={};addresses=defaultdict(list)
    for path in sorted((folder/'simulator').glob('core*/trace.json')):
        c=path.parent.name
        v.checked('.veccore' in c,'unexpected engine')
        s=v.Span()
        for e in v.events(path):
            if e['name']=='END_LABEL':
                continue
            s.add(e,base)
            if e['name']=='MOV_SRC_TO_DST_ALIGNv2':
                d=json.loads(e['args']['detail'])
                if d.get('instr_type')=='MTE3':
                    pc=int(e['args']['pc_addr'],16)-base
                    v.checked(pc in (0x4f38,0x5128) and d['burst_number']==1,'expected single-row warmup')
                    add,n=int(d['dst_addr']),d['processed_bytes']
                    addresses[pc].append((add,add+n))
        v.checked(s.entry==[s.first] and s.yrows==1 and s.writes=={0x5128:2*width,0x4f38:4},'incomplete lane coverage')
        actual[c]=s
        cores[c]=dict(entry_tick=s.first,last_observed_tick=s.last,has_end=len(s.end)==1,trace_sha256=v.digest(path))
    v.checked(len(cores)==t and set(cores)==set(tables),'mask differs from official roster')
    for pc,n in ((0x5128,width*2),(0x4f38,4)):
        spans=sorted(addresses[pc])
        v.checked(len(spans)==t and all(b-a==n for a,b in spans) and all(b==c for (_,b),(c,_) in zip(spans,spans[1:])),
                  'gaps or duplicates in output writes')
    trace=folder/'simulator/trace.json'
    aggregate=defaultdict(v.Span)
    for e in v.events(trace):
        if e['name']!='END_LABEL':
            aggregate[e['pid']].add(e,base)
    v.checked(set(aggregate)==set(actual) and all(aggregate[c].signature()==actual[c].signature() for c in actual),'shard/aggregate mismatch')
    return dict(cores=cores,trace=v.rel(trace),trace_sha256=v.digest(trace),confirmed_aiv=len(cores),ca_us=None,
                missing_end=[c for c in cores if not cores[c]['has_end']],
                writes=[dict(pc=hex(pc),interval=[min(a for a,b in spans),max(b for a,b in spans)],distinct_rows=len(spans)) for pc,spans in addresses.items()])


def main():
    results={}
    for path in sorted(OUT.glob('independent-*-windows.json')):
        proof=json.loads(path.read_text())
        rows=proof['rows'];chip=proof['chip'];pre=rows[0]
        if pre['qualified'] or pre['op']!='rmsnorm' or pre['target'] or not pre['reason'].startswith('missing physical entry/END'):
            continue
        if not all(s['qualified'] for s in rows[1:]):
            continue
        folder=ROOT/proof['cost']['file']
        ref,=folder.glob('profile/OPPROF_*/'+v.KERNEL+'/0')
        evidence=mask(ref,pre['shape'],v.tables((folder/'run.log').read_text())[0])
        v.checked(max(x['last_observed_tick'] for x in evidence['cores'].values())<rows[1]['first_tick'],'overlapping predecessor')
        v.checked(evidence['confirmed_aiv']==pre['active_aiv'],'predicted versus executed warmup engine count')
        evidence.update(goldens=pre['goldens'],source=path.name,shape=pre['shape'])
        write(OUT/f"{proof['group']}-{chip}-warmup-mask.json",evidence)
        for row in rows:
            if row['target']:
                results[chip,tuple(row['shape'])]=dict(chip=chip,shape=row['shape'],previous_aiv=row['previous_aiv'],ca_us=row['ca_us'],
                    preserve_us=row['preserve_us'],public_us=row['public_us'],evidence='complete target, predecessor output/occupancy verified; preceding latency unknown')
    report=json.loads((OUT/'comparison.json').read_text())
    for row in report['rows']:
        if not row['source'].startswith('independent-'):
            continue
        results[row['chip'],tuple(row['shape'])]=dict(chip=row['chip'],shape=row['shape'],previous_aiv=row['previous_aiv'],ca_us=row['ca_us'],preserve_us=row['preserve_us'],public_us=row['public_us'],evidence='complete target and timed ordered prefix')
    metrics={}
    for chip,_ in results:
        if chip not in metrics:
            rows=[v for (c,s),v in results.items() if chip==c]
            metrics[chip]={k:score(rows,k) for k in ('public_us','preserve_us')}
    write(OUT/'occupancy-comparison.json',dict(definition='All scored RMS calls and preceding Addmm timed in full. Warmup AIV occupancy for an untimed prefix requires every row address, execution entry and completed outputs. Separate from original full-prefix test.',metrics=metrics,rows=list(results.values())))


if __name__=='__main__':
    main()
