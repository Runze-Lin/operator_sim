"""Post-freeze global geometry audit on the blinded B8 PR trace; count real vs waits separately."""
import json
from collections import Counter,defaultdict
from pathlib import Path
from full_load_freeze import ROOT, OUT, CA, read,digest

def audit():
    receipt=CA/'262144-128-512-pr-drain.json'
    r=read(receipt)
    assert r['qualified']
    s,=r['stages'];base=(ROOT/s['trace']).parent
    cores=[];weights=Counter();dc=Counter();a_heads=[];c_heads=[];dims=Counter();classified=Counter()
    for c in sorted(s['windows']):
        source=base/c/'trace.json'
        raw=source.read_bytes();events=json.loads(raw)['traceEvents']
        allque=Counter();ops=Counter();counts=Counter();out=[];a=[];b=[];loc=Counter();wdim=Counter()
        for e in events:
            if e.get('ph')!='X':continue
            name=e['name'];d=json.loads(e['args']['detail'] or '{}');q=d.get('queue_type')
            if q not in ('MTE2','FIXP','MTE1','CUBE'):continue
            allque[q]+=1
            flags=tuple(k for k in ('is_wait','is_sync','is_duration_excluded') if d.get(k))
            ops[(q,name,','.join(flags) or 'payload')]+=1
            if flags:continue
            if name=='MOV_OUT_TO_L1_MULTI_ND2NZ':
                key=(d['n'],d['d'],d['processed_bytes'])
                if key==(512,128,131072):b.append(d['src_addr']);weights[d['src_addr']]+=1
                elif key==(256,128,65536):a.append(d['src_addr'])
                else:loc[str(key)]+=1
            if name=='FIX_L0C_TO_DST':
                out.append(d['dst_addr']);wdim[(d['m'],d['n'],d['processed_bytes'],d['pre_quant_mode'])]+=1
            if name=='MMAD':dims[(d['m'],d['n'],d['k'])]+=1;counts[(d['m'],d['n'],d['k'])]+=1
        if loc:raise ValueError(loc)
        assert set(wdim)=={(256,128,131072,'F322BF16(roundtothenearest,tietoeven)')}
        assert len(b)==1 and len(a)==4*len(out) and counts=={(256,128,32):16*len(out)}
        sort_a=sorted(a)
        groups=[sort_a[j:j+4] for j in range(0,len(a),4)]
        assert all([x-v[0] for x in v]==[0,256,512,768] for v in groups)
        a_heads.extend(v[0] for v in groups);c_heads.extend(out)
        classified.update(ops);dc.update(allque)
        cores.append(dict(core=c,shard_sha256=digest(source),tiles=len(out),mmad_geometry={str(k):v for k,v in counts.items()},b_addr=b[0],
                          output_destinations=out,first_a_page_per_output=[x[0] for x in groups]))
    assert len(set(weights))==1 and len(cores)==28 and len(a_heads)==1024 and len(c_heads)==1024
    assert sorted(a_heads)==[min(a_heads)+j*2*256*512 for j in range(1024)]
    assert sorted(c_heads)==[min(c_heads)+j*2*256*128 for j in range(1024)]
    result=dict(receipt=receipt.name,receipt_sha256=digest(receipt),kernel=s['kernel'],elf=s['binary_sha256'],mmad_geometry={str(k):v for k,v in dims.items()},
                tiles_per_core=Counter(x['tiles'] for x in cores),issued_counts={str(k):v for k,v in classified.items()},queue_all_x=dc,
                b_read_counts=weights,a_source_base=min(a_heads),c_output_base=min(c_heads),a_spacing=262144,c_spacing=65536,per_core=cores)
    (OUT/'pr32-geometry.json').write_text(json.dumps(result,indent=2)+'\n')
    print(result['tiles_per_core'],result['mmad_geometry'],result['queue_all_x'])
    print({k:v for k,v in classified.items() if k[0] in ('MTE2','FIXP')})
if __name__=='__main__':audit()
