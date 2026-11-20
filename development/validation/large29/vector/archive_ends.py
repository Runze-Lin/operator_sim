"""Independently locate physical END and its label in each archived per-core event stream."""
import json
import mmap
import re
from common import *

def inspect(stage):
    trace=ROOT/stage['trace'];cores={}
    for f in sorted(trace.parent.glob('core*/trace.json')):
        with f.open('rb') as fd, mmap.mmap(fd.fileno(),0,access=mmap.ACCESS_READ) as raw:
            ts={}
            for name in ('END','END_LABEL'):
                needle=b'"name":"'+name.encode()+b'"';p=raw.find(needle)
                if p<0:ts[name]=None;continue
                if raw.find(needle,p+len(needle))>=0:raise ValueError('duplicate '+str(f)+' '+name)
                start=raw.rfind(b'{"args":',max(0,p-16384),p);end=raw.find(b'}',p)
                if start<0 or end<0:raise ValueError('missing valid X record')
                e=json.loads(raw[start:end+1]);assert e['name']==name and e['ph']=='X'
                ts[name]=e['ts']+e['dur']
            cores[f.parent.name]=ts
    if set(cores)!=set(stage['windows']):raise ValueError('shard/core roster mismatch')
    exposed={c:v['END_LABEL'] for c,v in cores.items() if v['END_LABEL'] is not None}
    offsets=[end-stage['windows'][c][1] for c,end in exposed.items()]
    good=bool(cores) and all(v['END'] is not None and v['END_LABEL'] is not None and v['END']<=v['END_LABEL']+.001 for v in cores.values())
    good=good and max(offsets)-min(offsets)<.002
    return dict(physical_end_all_cores=good,verified_core_count=len(cores),missing_end=[c for c,v in cores.items() if v['END'] is None],
                missing_label=[c for c,v in cores.items() if v['END_LABEL'] is None],end_offset_spread_us=(max(offsets)-min(offsets)) if offsets else None,
                trace_sha256=stage['trace_sha256'],kernel=stage['kernel'],source=stage['trace'],cores=cores)

def main():
    results=[]
    for chip,tag in [(DT,'dt'),(PR,'pr')]:
        d=read(ROOT/'results/v0.19/sku/final-records.json')
        stages=[next(x for x in d if x['chip']==chip and x['shape']==[2048,5120] and x['op']=='rmsnorm')['stages'][0],
                read(ROOT/f'results/v0.21/fusion/receipts/attn_res_fwd_3_7168_8_1_{chip}.json')['kernel'],
                next(x for x in read(ROOT/'results/v0.19/workloads/router-qualified.json') if x['sku']==chip and x['shape']==[24,896,16])['kernel'],
                next(x for x in read(ROOT/'results/v0.20/newop/holdout.json') if x['sku']==chip and x['shape']==[2097152,12])['kernel'],
                read(ROOT/f'results/v0.20/large/cast_4096_7168_{chip}.json')['stage']]
        for st in stages:
            r=inspect(st);r['chip']=chip;results.append(r)
            print(chip,st['kernel'],r['physical_end_all_cores'],r['missing_end'],flush=True)
        save(OUT/'archive-ends.json',results)
if __name__=='__main__':main()
