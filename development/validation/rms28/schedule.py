"""RMS-specific cold/partition/warm endpoint DAG. Units are target core cycles until return."""
from collections import Counter,defaultdict
from math import ceil
from operator_model.rms_resource import READ_PER_KIB,WRITE_PER_KIB,FIRST_PAYLOAD_RELEASE
from operator_model.rms_dispatch import price_rms
from operator_model.rms_lane_schedule import _launch_class,parameters as lane_prices

def feature(width,rows):
    return rows,rows*ceil(width/128),rows*(ceil(width/64)-1).bit_length()

def schedule(shape,chip,h,exact_repeat,params,diag,disabled=None,inspect=False):
    t,w=shape;plan=diag['target_plan'];a=plan['active_aiv'];m=plan['rows_per_core'];cap=plan['full_tile_rows']
    clock=1000*chip.frequency_ghz;vec=diag['vector_price_cycles_per_reference_cycle'];p=params['prices'];stages=[];link=0.
    if plan['iterations']>2:
        base=price_rms(shape,chip,{'resident_aiv':h})[0]
        return dict(us=base,mode='multi-UB steady path',stages=None)
    lanes=ceil(chip.vector_cores/chip.cube_cores)
    rp=diag['read_cycles_per_kib'];wp=diag['write_cycles_per_kib'];rs=rp/READ_PER_KIB;ws=wp/WRITE_PER_KIB
    def val(k,*f):return sum(x*y for x,y in zip(p[k],[1,*f]))
    def types(b):
        return 'first' if not h else 'repeat' if exact_repeat else _launch_class(b,a,h,lanes)
    counts=Counter(types(b)for b in range(a));ranks=Counter();sectors=defaultdict(list);y_end=0.;max_meta=None
    block_volume=[sum(max(0,min(cap,t-b*m-j*cap,m-j*cap)) for b in range(a))*2*w for j in range(plan['iterations'])]
    for b in range(a):
        st=types(b);rank=ranks[st];ranks[st]+=1;n=counts[st];rows=min(m,t-b*m)
        skew=val('gamma_spread_'+st,a-1)*((rank+.5)/n-.5) if n>1 else 0.
        g=val('gamma_'+st,a-1)+skew
        if st=='first':g+=(chip.details.get('rms_first_payload_cycles') or FIRST_PAYLOAD_RELEASE)-FIRST_PAYLOAD_RELEASE
        g_end=g+max(val('gamma_response_'+st,2*w/1024*rs),2*w/1024*rp)
        x_issue=g+val('x_issue');prev_normalize_end=0.;row0=b*m
        for j in range(ceil(rows/cap)):
            local=min(cap,rows-j*cap);work=feature(w,local);kib=2*w*local/1024
            if j:
                key='second_x_'+st
                if key not in p:key='second_x_repeat' if h else 'second_x_first'
                x_issue+=max(val(key,2*w*cap/1024*rs),val('x_issue'))
            global_cycles=0. if exact_repeat else block_volume[j]*chip.frequency_ghz/chip.hbm_bandwidth_gbs
            local_x=val('x_response_'+st,kib*rs,global_cycles)
            if disabled=='shared first payload':global_cycles=0.;local_x=val('x_response_'+st,kib*rs,0.)
            xc=x_issue+max(kib*rp,local_x)
            ready=max(xc,g_end)+val('read_to_vf')
            v=max(ready,prev_normalize_end-val('vector_reissue')) if j else ready
            rstd_vf=v+val('reduce_to_rstd_vf',*(z*vec for z in work))
            normalize=rstd_vf+val('rstd_to_normalize_issue',local)
            rstd_end=rstd_vf+val('rstd_vf_response',local)
            if disabled=='VF issue overlap':normalize=max(normalize,rstd_end)
            norm_end=normalize+val('normalize_response',*(z*vec for z in work))
            r=rstd_end+val('issue_rstd');y=norm_end+val('issue_y')
            ye=y+max(val('y_service_'+st,kib*ws),kib*wp)
            if ye>y_end:y_end=ye;max_meta=(b,j,st)
            first=row0//32;last=(row0+local-1)//32
            for sector in {first,last,min(first+1,last)}:
                sectors[sector].append((r,b,j,st))
            if inspect:stages.append(dict(block=b,tile=j,rows=local,row0=row0,state=st,
                gamma=[g,g_end],x=[x_issue,xc],reduce_issue=v,rstd_vf=[rstd_vf,rstd_end],normalize=[normalize,norm_end],rstd_issue=r,y=[y,ye]))
            if not exact_repeat and j==0 and disabled!='shared first payload':
                # At least one first-tile x response follows completion of all compulsory first-tile ingress;
                # every full first tile still owes its own vector chain and y retirement.
                # Account for the possibly short last lane using only work common to every participating block.
                min_rows=min(cap,t-(a-1)*m);tri=feature(w,min_rows)
                min_dep=val('read_to_vf')+val('reduce_to_rstd_vf',*(z*vec for z in tri))+val('rstd_to_normalize_issue',min_rows)+val('normalize_response',*(z*vec for z in tri))+val('issue_y')
                if b==0:first_issued=x_issue
                first_issued=min(first_issued,x_issue)
                link=first_issued+global_cycles+min_dep
            prev_normalize_end=norm_end;row0+=local
    r_end=0.
    for group in sectors.values():
        prev=0.
        for issued,b,j,st in sorted(group):
            service=val('rstd_solo_'+st)
            if disabled=='sector arbitration':prev=issued+service
            else:prev=max(issued+service,prev+lane_prices()['sector_interval'])
            r_end=max(r_end,prev)
    drain=lane_prices()['drain']
    floors=[diag['local_read_port_floor_cycles'],m*(2*w+4)*wp/1024]
    if not exact_repeat:floors.append(min(val('gamma_'+state,a-1) for state in counts)+diag['shared_io_floor_cycles'])
    else:
        working=4*w*t+2*w+4*t
        l2=chip.details.get('l2_bytes') or 0.
        floors.append(max(0.,working-l2)*chip.frequency_ghz/chip.hbm_bandwidth_gbs)
        bw=chip.details.get('l2_bandwidth_gbs')
        if bw:floors.append(working*chip.frequency_ghz/bw)
    us=max(y_end+drain,r_end+drain,link+drain,*floors)/clock
    return dict(us=us,mode='per-tile VF issue and sector retirement',final_y_cycles=y_end,final_rstd_cycles=r_end,
                critical_y=max_meta,first_ingress_afterwork=link,minimum_cycles=max(floors),stages=stages if inspect else None)

def forecast(shape,chip,prices,disabled=None):
    old,_,a,d=price_rms(shape,chip,{})
    h1=schedule(shape,chip,min(1,a),False,prices,d,disabled)['us'] if a>1 else None
    first=schedule(shape,chip,0,False,prices,d,disabled)['us']
    hot=schedule(shape,chip,a,True,prices,d,disabled)['us']
    hi=max(first,h1 or first,hot);lo=min(first,h1 or first,hot)
    if disabled=='partition wake':hi=max(first,hot)
    if disabled=='RMS-specific warm':lo=min(first,price_rms(shape,chip,{'resident_aiv':a})[0])
    point=2*lo*hi/(lo+hi)
    return dict(us=point,old_us=old,first_us=first,partition_us=h1,hot_us=hot,lower_us=lo,upper_us=hi,active_aiv=a,plan=d['target_plan'])
