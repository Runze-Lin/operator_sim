"""Read only the three needed real instructions from each retained 128Ki shard."""
from collections import Counter
import hashlib,json,mmap,re
from pathlib import Path
from freeze import ROOT, OUT, DT, PR
from operator_model import load_chip
from operator_model.matmul_supply32 import resident_schedule

def read(path):return json.loads(path.read_bytes())
def compact(counter):return [[*k,v] for k,v in sorted(counter.items())]

def scan(path):
    mm=Counter(); fixes=[];reads=[];sha=None
    with path.open('rb') as f, mmap.mmap(f.fileno(),0,access=mmap.ACCESS_READ) as b:
        sha=hashlib.sha256(b).hexdigest()
        for name in ('MMAD','FIX_L0C_TO_DST','MOV_OUT_TO_L1_MULTI_ND2NZ'):
            key=f'"name":"{name}","ph":"X"'.encode();p=0
            while (p:=b.find(key,p))>=0:
                begin=b.rfind(b'{"args":',max(0,p-8000),p);end=b.find(b'}',p)
                if begin<0:raise ValueError('Unsupported trace record')
                ev=json.loads(b[begin:end+1]);d=json.loads(ev['args']['detail']);p=end+1
                if any(d.get(s) for s in ('is_wait','is_sync','is_duration_excluded')):continue
                if name=='MMAD':mm[d['m'],d['n'],d['k']]+=1
                elif name=='FIX_L0C_TO_DST':fixes.append((d['dst_addr'],d['m'],d['n']))
                else:reads.append((d['src_addr'],d['n'],d['d'],d['processed_bytes']))
    return sha,mm,fixes,reads

def check(sku,tag):
    shape=(1048576,128,512);m,n,k=shape
    receipt=ROOT/f'results/v0.28/matmul/long-prefill/131072-{tag}.json';d=read(receipt);stage,=d['stages'];base=(ROOT/stage['trace']).parent;cores=[];hist=Counter();dest=[];base_b=[];abytes=0
    p=resident_schedule(shape,load_chip(sku))[-1]['target_plan']
    for c in stage['windows']:
        assert '.cubecore' in c
        sha,mm,fix,reads=scan(base/c/'trace.json');hist.update(mm)
        # B is an exact K×N ND load, repeated at the same address by every AIC.
        b=[r for r in reads if r[1:]==(k,n,2*k*n)];assert len(b)==1;base_b.append(b[0][0])
        avol=sum(v for a,_,_,v in reads if a!=b[0][0]);abytes+=avol
        dest.extend((a,a+2*dm*dn) for a,dm,dn in fix)
        cores.append(dict(core=c,sha256=sha,b_gm_addr=b[0][0],a_gm_read_bytes=avol,mmads=compact(mm),output_pieces=len(fix)))
    dest.sort();assert len(set(base_b))==1 and abytes==2*m*k and all(l==t for (_,t),(l,_) in zip(dest,dest[1:])) and dest[-1][1]-dest[0][0]==2*m*n
    predicted=Counter({(p['tile'][0],n,p['tile'][2]):(p['waves']-1)*p['active_aic']*p['iterations']})
    for dm,dn,count in p['tail_wave_histogram']:predicted[dm,dn,p['tile'][2]]+=count*p['iterations']
    assert predicted==hist and sum(hist.values())==stage['mmad']
    return dict(shape=shape,chip=sku,ca_us=d['ca_us'],receipt=str(receipt.relative_to(ROOT)),elf=stage['binary_sha256'],trace_sha256=stage['trace_sha256'],trace_core_count=len(cores),actual_mmad=compact(hist),predicted_mmad=compact(predicted),b_loads=len(base_b),b_source_addresses=list(set(base_b)),a_bytes=abytes,output_bytes=dest[-1][1]-dest[0][0],number_output_pieces=len(dest),no_overlapping_or_missing_output_bytes=True,all_cores=cores)

if __name__=='__main__':
    for sku,tag in ((PR,'pr'),(DT,'dt')):
        row=check(sku,tag);(OUT/f'128ki-{tag}-geometry.json').write_text(json.dumps(row,indent=2)+'\n');print(tag,row['actual_mmad'],flush=True)
