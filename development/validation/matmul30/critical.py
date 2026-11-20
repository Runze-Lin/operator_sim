"""Locate read, operand, MMAD and FIXP timing for every tile on development critical AICs."""
from collections import Counter,defaultdict
from math import ceil
from pathlib import Path
from statistics import median,mean
import json
import sys

from freeze import ROOT,OUT,read,digest
sys.path.insert(0,str(ROOT/'development/validation/sku17'))
from campaign_collect import events
DT='Ascend950DT_9582'
PR='Ascend950PR_957b'


def audit(receipt,shape,chip):
    root=read(receipt);stage,=root['stages'];assert root['qualified']
    core=max(stage['windows'],key=lambda k:stage['windows'][k][1])
    path=(ROOT/stage['trace']).parent/core/'trace.json';series=defaultdict(list)
    origin=root['terminal'][core]['first_tick']-round(stage['windows'][core][0]*1650)
    for e in events(path):
        name=e['name']
        if e.get('ph')!='X' or name not in ('MMAD','MOV_OUT_TO_L1_MULTI_ND2NZ','FIX_L0C_TO_DST','LOAD_2Dv2'):continue
        d=json.loads(e['args']['detail'])
        if any(d.get(k) for k in ('is_wait','is_sync','is_duration_excluded')):continue
        if name=='MMAD':assert (d['m'],d['n'],d['k'])==(256,128,32);key='cube'
        elif name=='MOV_OUT_TO_L1_MULTI_ND2NZ':key={(256,128):'a',(512,128):'b'}[d['n'],d['d']]
        elif name=='LOAD_2Dv2':key={16384:'l0a',8192:'l0b'}[d['processed_bytes']]
        else:key='write';assert d['pre_quant_mode'].startswith('F322BF16')
        series[key].append([round(e['ts']*1650)-origin,round((e['ts']+e['dur'])*1650)-origin,d.get('src_addr'),d.get('dst_addr')])
    for seq in series.values():seq.sort()
    cub=series['cube'];writes=series['write'];a=series['a'];n=len(writes)
    assert len(cub)==16*n and len(a)==4*n and len(series['b'])==1
    assert len(series['l0a'])==len(series['l0b'])==len(cub)
    tiles=[]
    for t in range(n):
        span=cub[16*t:16*t+16];pages=a[4*t:4*t+4];fix=writes[t]
        assert [p[2]-pages[0][2] for p in pages]==[0,256,512,768]
        e=[]
        for k,mm in enumerate(span):
            g=16*t+k
            if not g:continue
            expected=cub[g-1][0]+257
            stall=max(0,mm[0]-expected)
            pg=pages[k//4]
            l0a,l0b=series['l0a'][g],series['l0b'][g]
            # Under whole-operand readiness, clip the independent observed lateness to
            # the real issue delay. Violation counts expose any mismatch of that interpretation.
            e.append(dict(k=k,delay=stall,a_page_late=min(stall,max(0,pg[1]-expected)),
                          l0_late=min(stall,max(0,max(l0a[1],l0b[1])-expected)),
                          page_end_after_issue=pg[1]>mm[0],l0_end_after_issue=max(l0a[1],l0b[1])>mm[0]))
        tiles.append(dict(idx=t,mmads=[[r[0],r[1]] for r in span],a_pages=[[r[0],r[1]] for r in pages],
                          write=fix[:2],first_output_addr=fix[3],a_gm_base=pages[0][2],gaps=e))
    quartiles=[]
    for i in range(4):
        batch=tiles[i*n//4:(i+1)*n//4]
        gs=[g for x in batch for g in x['gaps']]
        spans=[tiles[t+1]['mmads'][0][0]-tiles[t]['mmads'][0][0] for t in range(i*n//4,min(n-1,(i+1)*n//4))]
        quartiles.append(dict(tile_range=[batch[0]['idx'],batch[-1]['idx']],a_response_median=median(b-a for t in batch for a,b in t['a_pages']),
            fix_duration_median=median(t['write'][1]-t['write'][0] for t in batch),tile_interval_mean=mean(spans),
            tile_interval_median=median(spans),extra_issue_cycles=sum(g['delay'] for g in gs),
            extra_at_page_boundary_cycles=sum(g['delay'] for g in gs if g['k']%4==0),
            a_retirement_late_cycles=sum(g['a_page_late'] for g in gs),l0_retirement_late_cycles=sum(g['l0_late'] for g in gs),
            page_end_after_issue_count=sum(g['page_end_after_issue'] for g in gs),l0_end_after_issue_count=sum(g['l0_end_after_issue'] for g in gs)))
    return dict(shape=shape,chip=chip,qualified_ca_us=stage['ca_us'],actual_core=core,terminal_receipt=str(receipt.relative_to(ROOT)),
                receipt_sha256=digest(receipt),critical_shard=str(path.relative_to(ROOT)),shard_sha256=digest(path),
                stage_elf_sha256=stage['binary_sha256'],global_cyc_per_us=1650,origin=origin,b=series['b'][0][:2],
                quartiles=quartiles,tiles=tiles)


def main():
    prefix=ROOT/'results/v0.28/matmul';cases=[(prefix/f'262144-128-512-{tag}-drain.json',[262144,128,512],sku) for tag,sku in [('dt',DT),('pr',PR)]]
    cases.append((prefix/'long-prefill/131072-dt.json',[1048576,128,512],DT))
    for case in cases:
        result=audit(*case);filename=f"critical-{case[1][0]}-{'dt' if case[2]==DT else 'pr'}.json"
        (OUT/filename).write_text(json.dumps(result,separators=(',',':'))+'\n')
        print(filename,result['quartiles'][-1],flush=True)

if __name__=='__main__':main()
