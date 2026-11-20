"""Predict unobserved history from independent cold/partition and warm descriptor schedules."""
from math import ceil
from statistics import median
from operator_model import rms_lane_schedule
from operator_model.rms_resource import READ_PER_KIB,WRITE_PER_KIB
from operator_model.rms_dispatch import price_rms


def curve(points):
    slopes=[(v-u)/(b-a) for i,(a,u) in enumerate(points) for b,v in points[i+1:] if a!=b]
    slope=max(0.,median(slopes)) if slopes else 0.
    return (max(0.,median(v-slope*a for a,v in points)),slope)


def fit(rows):
    """One observation per launch. Fit descriptor issue, response and VF service; never the device end."""
    records={k:[] for k in ('gamma_issue','issue_spread','x_response','gamma_response','reduce','normalize','y_response')}
    x_gaps=[]; solos=[]
    for r in rows:
        if r['role']!='immediate_repeat':continue
        t,w=r['shape'];active=r['A'];lanes=r['lanes'];values={k:[] for k in records}
        for lane in lanes:
            v=lane['ops'];read=v['x']['bytes']/1024;rows_on_lane=v['rstd']['bytes']//4;groups=rows_on_lane*ceil(w/128)
            x_gaps.append(v['x']['b']-v['gamma']['b'])
            values['gamma_issue'].append(v['gamma']['b']);values['x_response'].append((read,v['x']['e']-v['x']['b']))
            values['gamma_response'].append((v['gamma']['bytes']/1024,v['gamma']['e']-v['gamma']['b']))
            ready=max(v['gamma']['e'],v['x']['e'])
            values['reduce'].append((groups,v['rstd']['b']-ready));values['normalize'].append((groups,v['y']['b']-v['rstd']['b']))
            values['y_response'].append((v['y']['bytes']/1024,v['y']['e']-v['y']['b']))
        records['gamma_issue'].append((active-1,median(values['gamma_issue'])))
        records['issue_spread'].append((active-1,max(values['gamma_issue'])-min(values['gamma_issue'])))
        for k in ('reduce','normalize','x_response','gamma_response','y_response'):
            # Fastest y observes earliest uncontended drain on the same store port.
            # Reads/transforms use the median service over same-sized AIV blocks.
            vals=values[k];dominant=max(set(x for x,_ in vals),key=lambda x:sum(a==x for a,_ in vals))
            fn=min if k=='y_response' else median
            records[k].append((dominant,fn(y for x,y in vals if x==dominant)))
        solos.append(min(l['ops']['rstd']['e']-l['ops']['rstd']['b'] for l in lanes))
    return dict(curves={k:curve(v) for k,v in records.items()},x_lead=median(x_gaps),rstd_solo=median(solos),
                shared_retire_ii=rms_lane_schedule.parameters()['sector_interval'],drain=rms_lane_schedule.parameters()['drain'],
                counts={k:len(v) for k,v in records.items()})


def forecast(shape,chip,p):
    old,_,active,d=price_rms(shape,chip,{})
    first=d['conditional_us']['first'];plan=d['target_plan']
    full=price_rms(shape,chip,{'resident_aiv':active})[0];next_partition=price_rms(shape,chip,{'resident_aiv':1})[0] if active>1 else first
    top=max(first,next_partition);floor=d['shared_io_floor_cycles'];clk=1000*chip.frequency_ghz
    states=dict(first_us=first,one_aiv_prefix_us=next_partition,fully_covered_us=full)
    if plan['iterations']>1:
        # Multi-UB hot-state pipeline has its own block overlap and local port floors.
        bottom=full;details={'route':'multi-UB'}
    else:
        t,w=shape;stride=plan['rows_per_core'];curves=p['curves'];vec=d['vector_price_cycles_per_reference_cycle']
        def cycles(key,quantity,scale=1.):
            base,service=curves[key];return base+quantity*service*scale
        # Configured port speeds affect the variable descriptor service and enforce a strict byte-volume floor.
        read_price=d['read_cycles_per_kib'];write_price=d['write_cycles_per_kib']
        read_scale=read_price/READ_PER_KIB if chip.details.get('mte2_bandwidth_gbs') else 1.
        write_scale=write_price/WRITE_PER_KIB if chip.details.get('mte3_bandwidth_gbs') else 1.
        gamma=cycles('gamma_issue',active-1);spread=cycles('issue_spread',active-1) if active>1 else 0.
        sector={};max_y=0.
        for block in range(active):
            count=min(stride,t-block*stride);wk=count*ceil(w/128);gamma_kib=2*w/1024;x_kib=2*w*count/1024
            payload=max(cycles('gamma_response',gamma_kib,read_scale),
                        p['x_lead']+cycles('x_response',x_kib,read_scale),(gamma_kib+x_kib)*read_price)
            issue=gamma+payload+cycles('reduce',wk,vec)
            y=cycles('normalize',wk,vec)+max(cycles('y_response',x_kib,write_scale),x_kib*write_price)
            start=block*stride//32;last=(block*stride+count-1)//32
            for sec in {start,last,min(start+1,last)}:sector.setdefault(sec,[]).append((issue,y))
        max_r=0.
        for group in sector.values():
            retired=0.
            for i,(ready,y) in enumerate(sorted(group)):
                at=ready+spread*((i+.5)/len(group)-.5)
                retired=max(at+p['rstd_solo'],retired+p['shared_retire_ii'])
                max_y=max(max_y,at+y)
            max_r=max(max_r,retired)
        floor=max(d['local_read_port_floor_cycles'],stride*(2*w+4)/1024*write_price,
                  gamma+floor)
        bottom=max(max_y+p['drain'],max_r+p['drain'],floor)/clk
        details=dict(route='single-UB',gamma_issue_cycles=gamma,issue_spread_cycles=spread,
                     final_y_cycles=max_y,final_rstd_cycles=max_r,lower_bound_cycles=floor)
    # First entry and newly opened second dispatch partition are physically different endpoints.
    # Extremely slow port/link hardware may also make the covered state the upper endpoint.
    lo,hi=sorted((bottom,top));point=2*lo*hi/(lo+hi)
    return dict(us=point,lower_us=lo,upper_us=hi,old_us=old,conditional=states,covered_us=bottom,
                active_aiv=active,detail=details)
