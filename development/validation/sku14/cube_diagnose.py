"""Recover exposed phase landmarks from old hash-bound MatMulV3 / FA captures only."""
from collections import defaultdict, Counter
from hashlib import sha256
import json
from math import log2
from pathlib import Path
from statistics import median
import sys
ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0,str(ROOT))
from operator_model.model import fit_nonnegative_line


def dissect(path, digest, op):
    data = path.read_bytes()
    assert sha256(data).hexdigest() == digest, path
    es = [e for e in json.loads(data)['traceEvents'] if e['ph']=='X']
    end = lambda e: e['ts']+e['dur']
    start = min(e['ts'] for e in es)
    cores = defaultdict(list)
    for e in es: cores[e['pid']].append(e)
    critical = max(cores,key=lambda c:max(map(end,cores[c])))
    last = max(map(end,cores[critical]))
    info = dict(trace=str(path.relative_to(ROOT)),sha256=digest,critical_core=critical,
                ca_us=last-start,core_count=len(cores),events=len(es))
    if op=='fa':
        mmads=sorted((e for e in es if e['name']=='MMAD'),key=lambda x:x['ts'])
        info.update(mmad_start_us=[e['ts']-start for e in mmads],mmad_duration_us=[e['dur'] for e in mmads],
                    queues=dict(Counter(e['tid'] for e in es)))
        return info
    tail=cores[critical]; mmads=sorted((e for e in tail if e['name']=='MMAD'),key=lambda e:e['ts'])
    reads=sorted((e for e in tail if e['name']=='MOV_OUT_TO_L1_MULTI_ND2NZ'),key=lambda e:e['ts'])
    fixes=[e for e in tail if e['name']=='FIX_L0C_TO_DST']; assert reads and mmads and fixes
    detail=lambda e:json.loads(e['args']['detail'])
    ref=detail(mmads[0]); max_pair_bytes=max(detail(e)['processed_bytes'] for e in reads[:2]); pair_end=max(map(end,reads[:2]));first=reads[0]['ts'];lend=max(map(end,mmads));tiles=[]
    for i in range(0,len(mmads),2):
        sub=mmads[i:i+2];work=sum(2*detail(e)['m']*detail(e)['n']*detail(e)['k'] for e in sub)
        tiles.append(dict(work_128kiop=work/131072,body_us=max(map(end,sub))-sub[0]['ts']))
    info.update(arrival_us=min(e['ts'] for e in tail)-start,prefix_us=first-start,first_pair_us=pair_end-first,
                max_first_descriptor_kib=max_pair_bytes/1024,l0_lead_us=mmads[0]['ts']-pair_end,
                l0_kib=2*(ref['m']+ref['n'])*ref['k']/1024,compute_groups=tiles,
                next_page_intervals_us=[mmads[i]['ts']-mmads[i-2]['ts'] for i in range(2,len(mmads),2)],
                total_mmad_span_us=lend-mmads[0]['ts'],drain_us=last-lend,fixp_raw_us=max(e['dur'] for e in fixes),
                output_kib=2*ref['m']*ref['n']/1024,mmad_count=len(mmads),mmad_mnk=[ref[k] for k in ('m','n','k')])
    assert abs(last-start-sum(info[k] for k in ['prefix_us','first_pair_us','l0_lead_us','total_mmad_span_us','drain_us'])) < 1e-6
    return info


def main():
    out=ROOT/'results/v0.14'; rows=[]; seen=set(); pairs=[]
    # Keep original eligibility for records selected as calibration. Deduplicate imported traces.
    for name in ['dataset','imported-calibration']:
        for r in json.loads((ROOT/f'results/v0.2/{name}.json').read_text())['records']:
            c=r['case']; m=r.get('measurement',{}); digest=m.get('trace_sha256')
            if c['op']!='matmul' or c['split']!='calibration' or r['status']!='ok' or digest in seen:continue
            trace=next(Path(next(v[9:] for v in r['command'] if v.startswith('--output='))).glob('*/simulator/trace.json'))
            rows.append(dict(shape=c['shape'],sku='dt',**dissect(trace,digest,'matmul')));seen.add(digest)
    for name in ['pairs','holdout']:
        for r in json.loads((ROOT/f'results/v0.13/sku/{name}.json').read_text()):
            if r['op'] not in ('matmul','fa'):continue
            pair=dict(shape=r['shape'],op=r['op']);pairs.append(pair)
            for sku in ('dt','pr'):
                d=r[sku]; value=dissect(ROOT/d['trace'],d['trace_sha256'],r['op']);pair[sku]=value
                assert abs(value['ca_us']-d['ca_us'])<1e-6
                if r['op']=='matmul' and value['sha256'] not in seen:
                    rows.append(dict(shape=r['shape'],sku=sku,**value));seen.add(value['sha256'])
    def reg(x,y):return fit_nonnegative_line([x(r) for r in rows],[y(r) for r in rows])
    costs={'prefix':reg(lambda r:log2(1+r['core_count']),lambda r:r['prefix_us']),
           'pair':reg(lambda r:r['max_first_descriptor_kib'],lambda r:r['first_pair_us']),
           'l0':reg(lambda r:r['l0_kib'],lambda r:r['l0_lead_us']),
           'drain':{'intercept_us':median(r['drain_us'] for r in rows),'slope_us_per_unit':0.}}
    groups=[g for r in rows for g in r['compute_groups']];costs['compute']=fit_nonnegative_line([g['work_128kiop'] for g in groups],[g['body_us'] for g in groups]);
    # Intervals are never visible in single-page traces; infer a byte price only from observed page pairs.
    intervals=[v/r['max_first_descriptor_kib'] for r in rows for v in r['next_page_intervals_us']];assert intervals
    costs['repeat_us_per_max_descriptor_kib']=median(intervals)
    (out/'cube-phases.json').write_text(json.dumps(costs,indent=2)+'\n');(out/'cube-diagnosis.json').write_text(json.dumps(dict(development=rows,paired=pairs),indent=2)+'\n')

if __name__=='__main__':main()
