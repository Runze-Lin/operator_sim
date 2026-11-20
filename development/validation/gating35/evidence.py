"""Extract only physical queue instructions and addresses from existing qualified router shards."""
from collections import Counter
from pathlib import Path
from statistics import median
from math import ceil
import hashlib,json,mmap,re

ROOT=Path(__file__).resolve().parents[3];OUT=ROOT/'results/v0.35/gating';DT='Ascend950DT_9582';PR='Ascend950PR_957b';FUT='Ascend970DT'
OPS=ROOT/'toolchains/cann/9.2.1/cann-9.2.1/opp/built-in/op_impl/ai_core/tbe';BODY=OPS/'impl/ops_transformer/ascendc/moe_gating_top_k/op_kernel';VENDOR=OPS/'kernel/ascend950/ops_transformer/moe_gating_top_k_apt/MoeGatingTopK_4ca391a7fc5a94c6f1e7d38c9bb0fa20.o';SORT=ROOT/'toolchains/cann/9.2.1/cann-9.2.1/aarch64-linux/asc/impl/basic_api/dav_3510/kernel_operator_proposal_impl.h'

def sha(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        while b:=f.read(2**20):h.update(b)
    return h.hexdigest()
def rel(p):return str(p.relative_to(ROOT))
def read(p):return json.loads(p.read_bytes())
def save(p,v):p.write_text(json.dumps(v,indent=2,allow_nan=False)+'\n')

def merge_ops(experts):
    runs=[32]*ceil(experts/32);ops=[];copies=1
    while len(runs)>1:
        width=max(runs);used=0;normal=[]
        while len(runs)-used>=4 and runs[used:used+4]==[width]*4:
            normal.append(4*width);used+=4
        if normal:ops.append((sum(normal),len(normal)))
        tail=runs[used:];assert len(tail)<=4
        if len(tail)>=2:ops.append((sum(tail),1))
        runs=normal+([sum(tail)] if tail else []);copies+=1
    return ops,copies

def selected(p):
    items=[];names=('VF','VBS32','VMS4V2','MOV_UB_TO_UB','MOV_SRC_TO_DST_ALIGNv2','END','MOV_XD_SPR')
    with p.open('rb') as stream,mmap.mmap(stream.fileno(),0,access=mmap.ACCESS_READ) as buf:
        for op in names:
            k=f'"name":"{op}","ph":"X"'.encode();pos=0
            while (hit:=buf.find(k,pos))>=0:
                start=buf.rfind(b'{"args":',max(0,hit-8000),hit);end=buf.find(b'}',hit);assert start>=0
                x=json.loads(buf[start:end+1]);pos=end+1;d=json.loads(x['args']['detail'] or '{}');
                if d.get('is_sync') or d.get('is_wait') or d.get('is_duration_excluded'):continue
                items.append((op,round(x['ts']*1650),round((x['ts']+x['dur'])*1650),d))
    return sorted(items,key=lambda x:x[1])

def lane(p,shape):
    t,e,k=shape;data=selected(p);align=32*ceil(e/32);moves=[x for x in data if x[0]=='MOV_SRC_TO_DST_ALIGNv2'];writes=[x for x in moves if x[3].get('dst_mem')=='OUT'];reads=[x for x in moves if x[3].get('src_mem')=='OUT'];body=[x for x in data if x[0] in ('VF','VBS32','VMS4V2','MOV_UB_TO_UB')];sort=[x for x in body if x[0]=='VBS32'];r=len(sort);assert r and len(writes)==2 and len(reads)==3;burst=reads[-1][3]['burst_number'];assert burst==r and all(x[3]['burst_number']==r and x[3]['burst_len']==4*k for x in writes);bias=reads[1];xread=reads[2];assert bias[3]['processed_bytes']==4*e and xread[3]['processed_bytes']==4*e*r
    vs=[x for x in body if x[0]=='VF'];ms=[x for x in body if x[0]=='VMS4V2'];cp=[x for x in body if x[0]=='MOV_UB_TO_UB'];model,percopy=merge_ops(e)
    assert len(vs)==1+3*r and len(ms)==r*len(model) and len(cp)==r*percopy and all(x[3]['processed_bytes']==8*align for x in cp),(p,len(ms),len(cp),r,e)
    start=min(x[1] for x in data);end=next(x[2] for x in data if x[0]=='END');cap=min(x[3]['src_addr'] for x in writes)//(8*align)
    return dict(core=p.parent.name,rows=r,ub_batch_rows=cap,input_dma_descriptors=len(reads)-2,bias_dma_descriptors=1,output_dma_descriptors=len(writes),
        sort32_durations=[x[2]-x[1] for x in sort],merge_runs=[[n,repeat,x[2]-x[1]] for x,(n,repeat) in zip(ms,model*r)],ub_copy_cycles=[x[2]-x[1] for x in cp],
        init_to_bias_issue=bias[1]-start,read_ready=max(xread[2],bias[2])-bias[1],x_response=xread[2]-xread[1],launch_abs=start,exit_abs=max(end,max(x[2] for x in writes)),
        norm_to_sort=[sort[i][1]-vs[1+3*i][1] for i in range(r)],select_cycles=[vs[3+3*i][2]-vs[3+3*i][1] for i in range(r)],first_norm_after_ready=vs[1][1]-max(xread[2],bias[2]),
        select_to_next_norm=[vs[1+3*(i+1)][1]-vs[3+3*i][1] for i in range(r-1)],row_issue_interval=[b[1]-a[1] for a,b in zip(sort,sort[1:])],
        write_latency=max(x[2] for x in writes)-min(x[1] for x in writes),write_issue_from_lastselect=min(x[1] for x in writes)-vs[-1][2])

def main():
    src=ROOT/'results/v0.19/workloads';items=[];seen=set()
    for source in ('router-calibration-ipc.json','router-qualified.json'):
        for r in read(src/source):
            if not r.get('qualified'):continue
            st=r['kernel'];key=st['trace_sha256'];
            if key in seen:continue
            seen.add(key);tr=ROOT/st['trace'];raw=tr.parent.parent/'dump/aicore_binary.o';assert sha(tr)==key and sha(raw)==sha(VENDOR) and st['kernel'].endswith('_10005');c=[]
            for p in sorted(tr.parent.glob('core*/trace.json')):c.append(lane(p,r['shape']))
            assert len(c)==st['active_aiv'] and sum(x['rows'] for x in c)==r['shape'][0]
            items.append(dict(shape=r['shape'],chip=r['sku'],source=rel(src/source),trace=rel(tr),trace_sha256=key,ca_us=st['ca_us'],aiv=len(c),cores=c,
                              golden=r['correctness'],role='primitive-price development' if r['sku']==DT and (r['shape'][0]==1 and r['shape'][1] in (128,896) or r['shape']==[65,128,8]) else 'archived replay'))
    assert len(items)==22
    save(OUT/'physical.json',dict(elf_sha256=sha(VENDOR),source_sha256={rel(p):sha(p) for p in [VENDOR,BODY/'moe_gating_top_k_apt.cpp',BODY/'arch35/moe_gating_top_k_without_group_regbase.h',SORT]},rows=items))
if __name__=='__main__':main()
