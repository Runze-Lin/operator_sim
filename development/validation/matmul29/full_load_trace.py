"""Extract instruction and data descriptors on the last physical AIC of three training windows."""
from collections import Counter, defaultdict
from hashlib import sha256
from pathlib import Path
import json
from statistics import median

ROOT = Path(__file__).resolve().parents[3]
OUT = ROOT/'results/v0.29/matmul/full-load'
CA = ROOT/'results/v0.28/matmul'
TRAIN = ('32768-128-512-dt','32768-128-512-pr','262144-128-512-dt')


def med(values):
    return median(values) if values else None


def extract(slug):
    p=CA/(slug+'-drain.json')
    record=json.loads(p.read_bytes())
    original=CA/(slug+'.json')
    assert record['acquisition_sha256']==sha256(original.read_bytes()).hexdigest() and record['qualified']
    stage,=record['stages']
    assert stage['kernel'].endswith('_131074') and not stage['active_aiv']
    core=max(stage['windows'],key=lambda c:stage['windows'][c][1])
    path=ROOT/Path(stage['trace']).parent/core/'trace.json';blob=path.read_bytes()
    trace=json.loads(blob)['traceEvents']
    origin=record['terminal'][core]['first_tick']-round(stage['windows'][core][0]*1650)
    ins=defaultdict(list)
    for e in trace:
        if e.get('ph')!='X':continue
        name=e['name']
        if name not in ('MOV_OUT_TO_L1_MULTI_ND2NZ','LOAD_2Dv2','FIX_L0C_TO_DST','MMAD'):continue
        d=json.loads(e['args']['detail'])
        if any(d.get(f) for f in ('is_wait','is_sync','is_duration_excluded')):continue
        a,b=round(e['ts']*1650)-origin,round((e['ts']+e['dur'])*1650)-origin
        row={k:d[k] for k in ('processed_bytes','src_addr','dst_addr','m','n','k','d') if k in d}
        ins[name].append(dict(begin=a,end=b,pc=e['args']['pc_addr'],**row))
    for v in ins.values():v.sort(key=lambda r:r['begin'])
    page=ins['MOV_OUT_TO_L1_MULTI_ND2NZ']; a=[r for r in page if r['n']==256];b=[r for r in page if r['n']==512]
    mma=ins['MMAD'];l0=ins['LOAD_2Dv2'];fix=ins['FIX_L0C_TO_DST'];n=len(fix)
    assert len(b)==1 and b[0]['d']==128 and len(a)==4*n and len(mma)==16*n and len(l0)==32*n
    assert all((r['m'],r['n'],r['k'])==(256,128,32) for r in mma)
    tiles=[]
    for i in range(n):
        ap=a[4*i:4*(i+1)];mm=mma[16*i:16*(i+1)];ld=l0[32*i:32*(i+1)]
        assert [r['src_addr']-ap[0]['src_addr'] for r in ap]==[0,256,512,768]
        assert sum(r['processed_bytes'] for r in ap)==256*512*2
        assert sum(r['processed_bytes'] for r in ld)==16*(256+128)*32*2
        assert fix[i]['processed_bytes']==256*128*4
        tiles.append(dict(index=i,a_pages=ap,l0=ld,mmad=mm,fix=fix[i]))
    steps={}
    for label,key in [('a_issue','a_pages'),('cube_issue','mmad'),('l0_issue','l0')]:
        ticks=[w[key][0]['begin'] for w in tiles]
        steps[label]=[y-x for x,y in zip(ticks,ticks[1:])]
    for label,edge in [('fix_issue','begin'),('fix_retire','end')]:
        ticks=[w['fix'][edge] for w in tiles];steps[label]=[y-x for x,y in zip(ticks,ticks[1:])]
    actuals=dict(first_b_issue=b[0]['begin'],first_a_issue=a[0]['begin'],first_a_retire=a[0]['end'],first_l0_issue=l0[0]['begin'],
                 first_mmad_issue=mma[0]['begin'],last_mmad_retire=mma[-1]['end'],last_fix_issue=fix[-1]['begin'],last_fix_retire=fix[-1]['end'],
                 mte2_page_response_median=med([r['end']-r['begin'] for r in a]),mmad_response_median=med([r['end']-r['begin'] for r in mma]),
                 within_tile_mmad_ii_median=med([y['begin']-x['begin'] for w in tiles for x,y in zip(w['mmad'],w['mmad'][1:])]),
                 fix_descriptor_median=med([r['end']-r['begin'] for r in fix]),steps_cycles=steps,step_medians={k:med(v) for k,v in steps.items()})
    return dict(case=slug,shape=list(map(int,slug.split('-')[:3])),core=core,ca_us=stage['ca_us'],chip='Ascend950PR_957b' if slug.endswith('pr') else 'Ascend950DT_9582',
                active=stage['active_aic'],origin_tick=origin,all_core_window=stage['windows'][core],tile_count=n,
                core_shard=str(path.relative_to(ROOT)),core_sha256=sha256(blob).hexdigest(),aggregate_sha256=stage['trace_sha256'],
                acquired_receipt=str(original.relative_to(ROOT)),original_sha256=record['acquisition_sha256'],
                drain_receipt=str(p.relative_to(ROOT)),drain_sha256=sha256(p.read_bytes()).hexdigest(),b=b[0],tiles=tiles,metrics=actuals)


if __name__=='__main__':
    OUT.mkdir(parents=True,exist_ok=True)
    rows=[extract(n) for n in TRAIN];f=OUT/'development-traces.json'
    f.write_text(json.dumps(rows,indent=1)+'\n')
    for r in rows:
        m=r['metrics']
        print(r['case'],{k:v for k,v in m.items() if k!='steps_cycles'})
