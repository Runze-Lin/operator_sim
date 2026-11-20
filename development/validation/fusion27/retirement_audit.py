"""Compare the frozen epilogue marker list with every actual late descriptor on one held-out raw."""
import hashlib
import sys
import re
from pathlib import Path
from resume import work
from observe import OUT,ROOT,save,read

def audit(case):
    job=next(j for j in read(OUT/'register.json')['jobs'] if j['id']==case)
    cost=read(OUT/'raw'/case/'cost.json')
    [s],gold=work.orig.stages(job,job['sku'],OUT/'raw'/case)
    t,n,d=job['shape']
    assert s['kernel']==job['kernel'] and s['binary_sha256']==job['elf']
    counts=[(m[1],int(m[2])) for line in gold if (m:=re.search(r'name=(\S+) count=(\d+) bad=0',line))]
    assert counts==[('hin',t*d),('hpost',t*n),('hres',t*n*n)]
    cores={}; full=[]; old=[]
    for name,w in s['windows'].items():
        file=(ROOT/s['trace']).parent/name/'trace.json'
        marks=work.ep.terminal(file); original=max(x['retire'] for x in marks.values())
        events=work.orig.stages.__globals__['events']
        end=0;last=[]
        for e in events(file):
            if e['ph']!='X':continue
            tick=round((e['ts']+e['dur'])*1650)
            if tick>end:last=[]
            if tick>=end:
                end=tick;last.append(dict(name=e['name'],pc=e['args']['pc_addr'],issue=round(e['ts']*1650),end=tick))
        old.append(original-w[1]*1650);full.append(end-w[1]*1650)
        if end>original:
            cores[name]=dict(window_us=w,end_markers=marks,latest=last,additional_cycles=end-original,
                     shard=str(file.relative_to(ROOT)),sha256=hashlib.sha256(file.read_bytes()).hexdigest())
    save(OUT/(case+'-physical.json'),dict(case=case,shape=job['shape'],sku=job['sku'],source_receipt='receipts/'+case+'.json',
            registration_sha256=work.orig.sha(OUT/'register.json'),raw_sha256=s['trace_sha256'],cost=cost,
            device_elf_sha256=s['binary_sha256'],kernel=s['kernel'],trace=s['trace'],golden=gold,observed_us=s['ca_us'],
            active_aic=s['active_aic'],active_aiv=s['active_aiv'],offset_spread_frozen_cycles=max(old)-min(old),
            offset_spread_all_raw_cycles=max(full)-min(full),descriptors_beyond_original_epilogue=cores,
            role='Read-only diagnostic. The original independent physical-retirement failure is preserved.'))
if __name__=='__main__':audit(sys.argv[1])
