"""Shape-only RMSNorm estimate from branch endpoints and resource-priced single-tile lane landmarks."""
from collections import defaultdict
from math import ceil
from statistics import median
from operator_model.rms_dispatch import price_rms
from operator_model.rms_resource import FIRST_PAYLOAD_RELEASE,FIRST_SCALAR
from operator_model.rms_v12 import parameters as vector_parameters
from operator_model.rms_lane_schedule import parameters as lane_parameters,_launch_class

STATES=('first','resident','core','sibling','partition')


def fit(records):
    """Each launch contributes one median per physical lane class, using only instruction landmarks."""
    v=vector_parameters();data=defaultdict(list)
    for row in records:
        active=row['active'];width=row['shape'][1];classes=defaultdict(list)
        for lane in row['lanes']:
            classes[lane['state']].append(lane)
        for state,lanes in classes.items():
            x=active-1
            for metric in ('arrival','front','g_latency','x_latency','x_gap','vf_gate','stat_work','y_work'):
                values=[]
                for lane in lanes:
                    value=lane[metric];groups=lane['rows']*ceil(width/128)
                    if metric=='stat_work':value-=groups*v['reduce'][1]
                    if metric=='y_work':value-=groups*(v['reduce'][1]+v['normalize'][1])
                    values.append(value)
                data[metric,state].append((x,median(values)))
    result={}
    for (metric,state),points in data.items():
        # A fitted slope requires replication at distinct occupancies; the
        # three observations of a newly activated partition have only two A.
        slope=0.; unique=set(x for x,_ in points)
        if metric in ('arrival','front','x_latency','g_latency') and len(unique)>=4:
            slopes=[(y-y0)/(x-x0) for i,(x,y) in enumerate(points) for x0,y0 in points[:i] if x!=x0];slope=max(0.,median(slopes))
        offset=median(y-slope*x for x,y in points)
        result.setdefault(state,{})[metric]=[max(0.,offset),slope,len(points)]
    return result


def scenarios(shape,chip,p):
    version,_,active,diag=price_rms(shape,chip,{});plan=diag['target_plan']; clock=1000*chip.frequency_ghz
    current=diag['conditional_us'];cache_floor=diag['shared_io_floor_cycles'];forecast=lane_parameters();v=vector_parameters();tokens,width=shape
    if plan['iterations']>1:
        return [current['first']]+[current['expand']]*(active-1)+[current['reuse']],version,active
    stride=plan['rows_per_core'];c=ceil(chip.vector_cores/chip.cube_cores);pvec=diag['vector_price_cycles_per_reference_cycle']; reduced=v['reduce'][1]*pvec; normalized=v['normalize'][1]*pvec
    read_price=diag['read_cycles_per_kib']; write_price=diag['write_cycles_per_kib'];has_write=chip.details.get('mte3_bandwidth_gbs');yslope=write_price if has_write else forecast['y_cycles'][1];money={}
    for state,st in p.items():
        money[state]={metric:(base+slope*(active-1)) for metric,(base,slope,n) in st.items()}
    if active==1:
        for q in money.values():q['arrival']=0.
    # A first-payload launch-resource difference per target is already known
    # from DT/PR phase pairs; an unconfigured target assumes the reference.
    money['first']['front'] += (chip.details.get('rms_first_payload_cycles') or FIRST_PAYLOAD_RELEASE)-FIRST_PAYLOAD_RELEASE
    outputs=[]
    for high in range(active+1):
        sectors=defaultdict(lambda:defaultdict(list));finish=0
        for block in range(active):
            state=_launch_class(block,active,high,c) if high else 'first';s=money[state]
            rows=min(stride,tokens-block*stride);work=rows*ceil(width/128)
            gamma_latency=max(2*width*read_price/1024,s['g_latency']);x_latency=max(2*width*rows*read_price/1024,s['x_latency']);
            ready=s['arrival']+s['front']+max(gamma_latency,s['x_gap']+x_latency,2*width*(1+rows)*read_price/1024)+s['vf_gate']
            issue=ready+s['stat_work']+work*reduced
            ydelay=s['y_work']+work*(reduced+normalized)+forecast['y_cycles'][0]+yslope*2*width*rows/1024-(s['stat_work']+work*reduced)
            lo=(block*stride)//32;hi=(block*stride+rows-1)//32
            for sect in {lo,hi,min(lo+1,hi)}:sectors[sect][state].append((issue,ydelay))
        for states in sectors.values():
            issues=[]
            for state,ls in states.items():
                for idx,(ready,ydelay) in enumerate(sorted(ls),1):
                    time=ready+forecast['spread'].get(state,0.)*(idx/(len(ls)+1)-.5)
                    issues.append(time);finish=max(finish,time+ydelay)
            retire=0
            for ready in sorted(issues):retire=max(ready+forecast['solo'],retire+forecast['sector_interval'])
            finish=max(finish,retire)
        phase='first' if not high else 'reuse' if high==active else 'expand'
        if high==0:release=chip.details.get('rms_first_payload_cycles') or FIRST_PAYLOAD_RELEASE
        else:release=diag['conditional_stages_us'][phase]['SCALAR']*clock*FIRST_PAYLOAD_RELEASE/FIRST_SCALAR
        outputs.append(max(finish+forecast['drain'],cache_floor+release,diag['local_read_port_floor_cycles'],stride*(2*width+4)*write_price/1024)/clock)
    return outputs,version,active
