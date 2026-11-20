"""Validate per-AIC tile multiplicity and global GM destination tiling, from the original shards."""
from collections import Counter
from hashlib import sha256
import json
from pathlib import Path

from full_load_trace import OUT, ROOT


def audit(case):
    receipt=json.loads((ROOT/case['drain_receipt']).read_bytes())
    stage,=receipt['stages']
    base=(ROOT/stage['trace']).parent
    rows=[]; weights=set(); writes=[]; astarts=[]
    m,n,k=case['shape']
    for core in sorted(stage['windows']):
        blob=(base/core/'trace.json').read_bytes()
        mm=Counter();loads=Counter();b=[];a=[];c=[]
        for event in json.loads(blob)['traceEvents']:
            name=event.get('name')
            if event.get('ph')!='X' or name not in ('MMAD','LOAD_2Dv2','MOV_OUT_TO_L1_MULTI_ND2NZ','FIX_L0C_TO_DST'):
                continue
            d=json.loads(event['args']['detail'])
            if any(d.get(key) for key in ('is_sync','is_wait','is_duration_excluded')):
                continue
            if name=='MMAD':
                mm[d['m'],d['n'],d['k']]+=1
            elif name=='LOAD_2Dv2':
                loads[d['processed_bytes']]+=1
            elif name=='MOV_OUT_TO_L1_MULTI_ND2NZ':
                assert d['d']==128 and d['n'] in (256,512)
                (b if d['n']==512 else a).append(d['src_addr'])
            else:
                assert (d['m'],d['n'],d['processed_bytes'])==(256,128,131072)
                assert d['pre_quant_mode'].startswith('F322BF16')
                c.append(d['dst_addr'])
        tiles=len(c)
        assert len(b)==1 and len(a)==4*tiles and mm=={(256,128,32):16*tiles}
        assert loads=={8192:16*tiles,16384:16*tiles}
        for addr in sorted(a)[::4]:
            assert {addr+x for x in (0,256,512,768)}.issubset(a)
            astarts.append(addr)
        weights.update(b);writes.extend(c)
        rows.append(dict(core=core,tiles=tiles,mmad=16*tiles,source_sha256=sha256(blob).hexdigest(),
                         b_gm_base=b[0],a_page_count=len(a),fix_destinations=c,physical_end=receipt['terminal'][core]['physical_end']))
    assert len(rows)==stage['active_aic'] and len(weights)==1 and len(writes)==m//256
    assert sorted(writes)==[min(writes)+i*2*256*n for i in range(m//256)]
    assert sorted(astarts)==[min(astarts)+i*2*256*k for i in range(m//256)]
    return dict(case=case['case'],active_aic=len(rows),tiles_per_core=dict(Counter(r['tiles'] for r in rows)),
                cube_instructions=sum(r['mmad'] for r in rows),weight_gm_addresses=list(weights),per_core=rows)


if __name__=='__main__':
    rows=json.loads((OUT/'development-traces.json').read_bytes())
    rosters=[audit(row) for row in rows]
    p=OUT/'rosters.json'
    tmp=p.with_suffix('.tmp')
    tmp.write_text(json.dumps(rosters,indent=1)+'\n');tmp.replace(p)
