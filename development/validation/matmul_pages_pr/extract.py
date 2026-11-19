"""Recover each output rectangle from its FIX destination and each K page from ND2NZ."""
from collections import Counter, defaultdict
from hashlib import sha256
from math import isclose
from pathlib import Path
import json

ENGINES={'MMAD':'MMAD','MOV_OUT_TO_L1_MULTI_ND2NZ':'MTE2','LOAD_2Dv2':'MTE1','FIX_L0C_TO_DST':'FIXP'}


def require(cond,message):
    if not cond:raise ValueError(message)


def extract(path,shape,expected_sha,expected_us):
    data=path.read_bytes()
    require(sha256(data).hexdigest()==expected_sha,'merged trace SHA mismatch')
    events=[e for e in json.loads(data)['traceEvents'] if e['ph']=='X']
    origin=min(e['ts'] for e in events)
    terminal=max(events,key=lambda e:e['ts']+e['dur'])
    ca=terminal['ts']+terminal['dur']-origin
    require(isclose(ca,expected_us,abs_tol=1e-6),'all-core start/finish mismatch')
    groups=defaultdict(lambda:defaultdict(list))
    for e in events:
        q=ENGINES.get(e['name'])
        if q:
            groups[e['pid']][q].append((e['ts']-origin,e['ts']+e['dur']-origin,
                        e['args']['pc_addr'],json.loads(e['args']['detail'])))
    counts=Counter(e['tid'] for e in events)
    for group in groups.values():
        for queue in group.values():queue.sort(key=lambda e:e[0])
    return ca,terminal['pid'],counts,groups


def rectangles(shape, groups, terminal):
    M,N,K=shape
    core_count=len(groups);all_writes=[e for q in groups.values() for e in q['FIXP']]
    require(bool(all_writes),'no output FIXP descriptor')
    base_y=min(e[3]['dst_addr'] for e in all_writes)
    aa=[v[3]['src_addr'] for q in groups.values() for v in q['MTE2'][::2]]
    bb=[v[3]['src_addr'] for q in groups.values() for v in q['MTE2'][1::2]]
    require(aa and bb,'no GM→L1 input descriptor pairs')
    base_a=min(aa);base_b=min(bb)
    coverage=defaultdict(list);per_core={};critical=None;bench=Counter();MMADs=0;ops=0;tilehist=Counter()
    for core,q in groups.items():
        # Each accumulator begins on the observed init bit and ends on the final MMAD unit flag.
        accum=[]
        for x in q['MMAD']:
            if x[3]['matrix_c_init_val']==1:accum.append([])
            require(bool(accum),'first MMAD omitted accumulator initialization')
            accum[-1].append(x)
        require(len(q['FIXP'])==len(accum),'FIX count does not match accumulator initializations')
        iread=iload=0;tiles=[];labels=[]
        for tile_idx,(fix,macs) in enumerate(zip(q['FIXP'],accum)):
            detail=fix[3];m,n=detail['m'],detail['n']
            require(detail['src_mem']=='L0C' and detail['dst_mem']=='OUT','unsupported output buffer')
            offset=detail['dst_addr']-base_y;require(offset%2==0,'unaligned output')
            y,x=divmod(offset//2,N)
            require(y+m<=M and x+n<=N,'output rectangle exceeds tensor extent')
            for row in range(y,y+m):coverage[row].append((x,x+n))
            require([v[3]['hw_sets_unit_flag'] for v in macs]==[0]*(len(macs)-1)+[1],'MMAD end flag mismatch')
            require(all((v[3]['m'],v[3]['n'])==(m,n) for v in macs),'MMAD rows/cols and FIXP extent differ')
            require(sum(v[3]['k'] for v in macs)==K,'K coverage by MMAD fragments is incomplete')
            bench['read_operand_bytes']+=2*(m+n)*K;bench['output_bytes']+=2*m*n
            ops+=sum(2*m*n*v[3]['k'] for v in macs);MMADs+=len(macs);tilehist[m,n]+=1
            depth=used=0;pages=[];pgdepths=[]
            while depth<K:
                gm=q['MTE2'][iread:iread+2];iread+=2
                require(len(gm)==2,'unmatched page read descriptor')
                a,b=[v[3] for v in gm];k_page=a['d']
                require((a['n'],b['n'],b['d'])==(m,k_page,n),'rectangular A/B read descriptor extents disagree')
                require((a['src_addr'],b['src_addr'])==(base_a+2*(y*K+depth),base_b+2*(depth*N+x)),
                        'page addresses do not match FIXP origin and full input strides')
                require((a['processed_bytes'],b['processed_bytes'])==(2*m*k_page,2*n*k_page),'logical A/B page bytes mismatch')
                require(k_page>0 and depth+k_page<=K,'bad K page depth')
                page_macs=[];page_loads=[];z=0
                while z<k_page:
                    require(used<len(macs),'page exhausts MMAD sequence')
                    v=macs[used];used+=1;step=v[3]['k'];require(step>0 and z+step<=k_page,'MMAD crosses page boundary')
                    page_macs.append(v)
                    loads=q['MTE1'][iload:iload+2];iload+=2
                    require(len(loads)==2,'MTE1 operand pair missing')
                    for ld,dst,desc,length in zip(loads,['L0A','L0B'],[a,b],[m,n]):
                        ldv=ld[3]
                        padded_edge=16*((length+15)//16)
                        require(ldv['dst_mem']==dst and ldv['processed_bytes']==2*step*padded_edge,
                                'wrong rectangular LOAD_2D size or operand')
                        padded_l1_bytes=2*(16*((desc['n']+15)//16))*(16*((desc['d']+15)//16))
                        require(desc['dst_addr']<=ldv['src_addr']<desc['dst_addr']+padded_l1_bytes,
                                'LOAD_2D source lies outside parent L1 page')
                    page_loads.extend(loads);z+=step
                depth+=k_page;pgdepths.append(k_page)
                pages.append({k:[list(e[:2]) for e in v] for k,v in [('MMAD',page_macs),('MTE1',page_loads),('MTE2',gm)]})
            require(used==len(macs),'unassigned MMAD')
            tiles.append(dict(rect=[m,n],origin_mn=[y,x],pages=pages,write=list(fix[:2])))
            labels.append(dict(rect=[m,n],origin_mn=[y,x],pages_k=pgdepths,mmads=used,
                               fix_destination=detail['dst_addr']))
        require(iread==len(q['MTE2']) and iload==len(q['MTE1']),'unassigned input descriptors')
        per_core[core]=labels
        if core==terminal:critical=tiles
    require(len(coverage)==M,'missing output rows')
    for intervals in coverage.values():
        end=0
        for lo,hi in sorted(intervals):require(lo==end,'overlap or hole between output rectangles');end=hi
        require(end==N,'output row not completely covered')
    require(ops==2*M*N*K and bench['output_bytes']==2*M*N,'all-core useful ops/bytes disagree')
    require(critical is not None,'terminal core is not the page-level compute core')
    return dict(core_count=core_count,output_tiles=sum(tilehist.values()),mmad=MMADs,cube_ops=ops,
                tile_histogram=[[*key,v] for key,v in sorted(tilehist.items())],
                logical_operands_across_tiles=dict(bench),ca_tiles=critical,per_core_tiles=per_core,
                observed_input_bases=[base_a,base_b],observed_output_base=base_y)


def summarize(path,shape,expected_sha,expected_us):
    ca,core,counts,groups=extract(path,shape,expected_sha,expected_us)
    return dict(ca_us=ca,core=core,counts=dict(counts),**rectangles(shape,groups,core))
