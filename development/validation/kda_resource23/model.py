"""Development-only FLA 26.9.1 KDA V2 model, uniform packed BSND sequences."""
from collections import Counter
from functools import lru_cache
from math import ceil
from statistics import median
import sys

from extract import FAMILIES, OUT, ROOT, read

sys.path.insert(0, str(ROOT))
from operator_model.chips import load_chip

DT = 'Ascend950DT_9582'
D = 128


def div(a, b):
    return (a+b-1)//b


@lru_cache(maxsize=1)
def prices():
    samples = read(OUT/'dt-primitive-services.json')
    cost = read(ROOT/'results/v0.16/matmul.json')['parameters']
    tiny = samples['2']['stages']
    rates = {'OUT->UB':cost['mte2_bytes_per_cycle'], 'OUT->L1':cost['mte2_bytes_per_cycle'],
             'L1->L0A':cost['mte1_bytes_per_cycle'], 'L1->L0B':cost['mte1_bytes_per_cycle'],
             'UB->OUT':cost['write_bytes_per_cycle'], 'L0C->UB':cost['write_bytes_per_cycle'],
             'L0C->OUT':cost['write_bytes_per_cycle'], 'L0C->L1':cost['write_bytes_per_cycle'],
             'UB->L1':cost['write_bytes_per_cycle']}
    latency = {}
    for route,rate in rates.items():
        residuals = [max(0,e['end']-e['begin']-e['size']/rate) for st in tiny.values()
                     for e in st['transfers'] if e['route']==route]
        latency[route] = median(residuals) if residuals else latency['UB->OUT' if route=='UB->L1' else 'L0C->UB']
    cube_residuals = []
    dt = load_chip(DT)
    peak = dt.cube_tops['16']*1000/(dt.frequency_ghz*dt.cube_cores)
    for st in tiny.values():
        for m in st['mmads']:
            x,y,k=m['mnk']
            # Matrix hardware allocates 16-element blocks for service.
            work=2*(16*div(x,16))*(16*div(y,16))*(16*div(k,16))
            cube_residuals.append(max(0,m['end']-m['begin']-work/peak))
    return samples,rates,latency,median(cube_residuals)


def vf_weight(i,n,family):
    # V1 does a growing prefix on each active 16-row band, then pads fixed banks.
    if family==FAMILIES[0] and i==1:
        return n+sum(min(n,j*16) for j in range(1,div(n,16)+1))
    return n


def vf_estimate(family, i, n, chip):
    samples=prices()[0]
    base=samples['2']['stages'][family]['vfs'][i]
    full=samples['64']['stages'][family]['vfs'][i]
    weight=lambda t:vf_weight(i,t,family)
    fraction=(weight(n)-weight(2))/(weight(64)-weight(2))
    counts={q:max(0,round(v+(full['counts'].get(q,0)-v)*fraction)) for q,v in base['counts'].items()}
    dt=load_chip(DT)
    per_aiv=lambda c:c.vector_tops['32']/(c.frequency_ghz*c.vector_cores)
    # Four RV execution records per clock is an issue-width proxy. Critical VF
    # dependency/dispatch cost stays on the control clock; price extra work separately.
    ratio=per_aiv(dt)/per_aiv(chip)
    base_cycles=base['end']-base['begin']
    observed_service=min(base_cycles,base['counts']['RVECEX']/4)
    cycles=(base_cycles-observed_service)+max(0,observed_service+(counts['RVECEX']-base['counts']['RVECEX'])/4)*ratio
    return cycles,counts


class Tile:
    def __init__(self,chip,family,active):
        self.chip=chip
        self.family=family
        self.active=active
        self.legs=[]
        self.bytes=Counter()
        self.issues=Counter()
        self.mm=Counter()
        self.vf=Counter()

    def dma(self,route,*sizes):
        _,rates,lat,_=prices()
        self.bytes[route]+=sum(sizes)
        self.issues[route]+=len(sizes)
        d=self.chip.details
        ghz=self.chip.frequency_ghz
        local='mte2' if route.startswith('OUT') else 'mte1' if route.startswith('L1') else 'mte3'
        bandwidth=(d.get(local+'_bandwidth_gbs') or rates[route]*ghz)/ghz
        if 'OUT' in route:
            # Cache-payload pipe shares aggregate fabric; off-chip spill is a separate floor.
            l2=(d.get('l2_bandwidth_gbs') or 100*self.chip.cube_cores*ghz)/ghz
            bandwidth=min(bandwidth,l2/self.active)
        return lat[route]+sum(sizes)/bandwidth+max(0,len(sizes)-1)*2

    def calc(self, shapes):
        clocks=0
        cfg=self.chip
        peak=cfg.cube_tops['16']*1000/(cfg.frequency_ghz*cfg.cube_cores)
        for m,n,k,dtype in shapes:
            key=f'{m}x{n}x{k}:{dtype}'
            self.mm[key]+=1
            work=2*(16*div(m,16))*(16*div(n,16))*(16*div(k,16))
            # C4/C5 use FP32/TF32. Strict TF32 throughput is absent on all profiles.
            factor=0.5 if dtype=='f32_tf32' else 1.
            clocks+=prices()[3]+work/(peak*factor)
        return clocks

    def vec(self,i,n):
        t,counts=vf_estimate(self.family,i,n,self.chip)
        self.vf.update(counts)
        return t

    def leg(self,name,*steps):
        self.legs.append((name,sum(steps)))


def prepare(n,chip,a):
    j=div(n,16); m=32 if n<=32 else 64; v=256*n
    z=Tile(chip,FAMILIES[0],a)
    z.leg('V0 gate and prefix',z.dma('OUT->UB',v,v,v,4*n,512,4),z.vec(0,n),z.dma('UB->OUT',v,v,2*v))
    z.leg('V1 score packing',z.vec(1,n),z.dma('UB->OUT',32768,40960))
    read=z.dma('OUT->L1',16384,16384,4096,8192,12288,16384)
    c2=[]
    for t in range(1,j+1):
        ld=max(z.dma('L1->L0A',4096,4096),z.dma('L1->L0B',4096*t))
        c2.append((ld,z.calc([(32,16*t,128,'bf16')]),z.dma('L0C->UB',2048*t)))
    z.leg('C2 triangular bands',read,pipeline(c2))
    z.leg('V3 invert two 32-row blocks',z.vec(2,n),z.dma('UB->OUT',128*n,4096,4096,4096,8192,128*n))
    rows=[8192]+([4096]*3 if n>32 else [])
    c4=z.dma('OUT->L1',*rows)
    if n>32:
        for i in range(2):
            c4+=max(z.dma('L1->L0A',4096),z.dma('L1->L0B',4096))
            c4+=z.calc([(32,32,32,'f32_tf32')])
            if i==0:
                c4+=z.dma('L0C->OUT',4096)+z.dma('OUT->L1',4096)
            else:
                c4+=max(z.dma('L0C->L1',4096),z.dma('L0C->OUT',128*(n-32)))
    vm=z.dma('OUT->UB',v,v,2*v,v)+z.vec(3,n)+z.dma('UB->OUT',v,v,256*m,256*m)
    z.leg('C4/C5 quadrant and V6 RHS',max(c4,vm))
    z.leg('C7 W and U',z.dma('OUT->L1',256*m,256*m),
          max(z.dma('L1->L0A',2*m*m),z.dma('L1->L0B',256*m,256*m)),
          z.calc([(m,128,m,'bf16')]*2),z.dma('L0C->OUT',2*v,2*v))
    return z


def recurrence(n,chip,a,first,last):
    z=Tile(chip,FAMILIES[1],a); v=256*n; p=16*div(n,16)
    if first:
        z.leg('FP32 initial state to BF16 H',z.dma('OUT->UB',65536),z.vec(0,n),z.dma('UB->OUT',32768))
    z.leg('S0 W times H',z.dma('OUT->L1',v,32768),max(z.dma('L1->L0A',256*p),z.dma('L1->L0B',32768)),
          z.calc([(p,128,128,'bf16')]),z.dma('L0C->UB',2*v))
    z.leg('S1 corrected value',z.dma('OUT->UB',v),z.vec(1,n),z.dma('UB->OUT',v,v))
    c=z.dma('OUT->L1',v,v)+max(z.dma('L1->L0A',256*p),z.dma('L1->L0B',256*p))
    c+=z.calc([(128,128,p,'bf16')])+z.dma('L0C->UB',65536)
    # Stage3 state and gate reads can overlap the S2 K^T times right product.
    state=z.dma('OUT->UB',*([65536,512] if first else [512]))
    z.leg('S2 outer product',max(c,state))
    z.leg('S3 update recurrent state',z.vec(2,n),z.dma('UB->OUT',65536 if last else 32768))
    return z


def final(n,chip,a,mover):
    p=16*div(n,16);v=256*n
    z=Tile(chip,FAMILIES[2],a)
    if mover:
        # AIV MTE2 gathers padded NZ bands. Eight separate MTE3 moves pack
        # a 128-column Q/H/V tile into L1, one move per 16-column stripe.
        z.leg('AIV Q/H to UB and L1',z.dma('OUT->UB',v,32768),
              z.dma('UB->L1',*([32*n]*8+[4096]*8)))
        z.leg('AIV Aqk/V to UB and L1',z.dma('OUT->UB',2*n*p,v),
              z.dma('UB->L1',*([32*n]*(p//16)+[32*n]*8)))
    else:
        z.leg('Q, H, Aqk and value',z.dma('OUT->L1',v,32768,2*n*n,v))
    pairs=[(max(z.dma('L1->L0A',256*p),z.dma('L1->L0B',32768)),z.calc([(n,128,128,'bf16')]),0),
           (max(z.dma('L1->L0A',2*p*p),z.dma('L1->L0B',256*p)),z.calc([(n,128,n,'bf16')]),0)]
    z.leg('QH plus AqkV',pipeline(pairs))
    z.leg('output',z.dma('L0C->OUT',2*v))
    return z


def pipeline(jobs):
    ends=[0,0,0]
    for job in jobs:
        for i,t in enumerate(job):
            ends[i]=max(ends[i],ends[i-1] if i else 0)+t
    return ends[-1]


def weighted(hist, Tiles, attr):
    out=Counter()
    for key,count in hist.items():
        for k,v in getattr(Tiles[key],attr).items():out[k]+=count*v
    return dict(sorted(out.items()))


def predict(shape, chip=DT):
    chip=load_chip(chip)
    if len(shape)!=6 or any(type(x) is not int or x<1 for x in shape):
        raise ValueError('Expected positive integer B,S,Hq,Hv,K,V')
    b,s,hq,h,k,v=shape
    if h!=hq or (k,v)!=(128,128):
        raise ValueError('Prototype covers equal Kimi heads, K=V=128 only')
    if not chip.cube_tops['16'] or not chip.vector_tops['32'] or chip.vector_cores<2:
        raise ValueError('BF16 Cube, FP32 Vector and a two-AIV group are required')
    chunks=div(s,64); num=b*chunks
    last=s-64*(chunks-1)
    mixed_capacity=min(chip.cube_cores,chip.vector_cores//2)
    stages=[]
    for i,f in enumerate(FAMILIES):
        cap=chip.cube_cores if i==2 else mixed_capacity
        group=h if num>=cap else 1
        if i==1:
            act=div(b*h,div(b*h,cap))
        else:
            act=min(cap,num*h//group)
        mover=i==2 and group==h and num//act>=8
        if mover:
            # 950 arch35 host permits 2 AIV to gather+pack into L1. A custom 1:1
            # SKU needs a core mapping, explicit rather than silently scheduling two AIVs.
            cap=mixed_capacity
            if cap<act:
                group=h if num>=cap else 1;act=min(cap,num*h//group)
        histograms=[]
        for core in range(act):
            if i==1:
                stride=div(b*h,act)
                heads=min(stride,b*h-core*stride)
                seq=Counter()
                if chunks==1: seq[(last,True,True)]=heads
                else:
                    seq[(64,True,False)]=heads
                    if chunks>2:seq[(64,False,False)]=(chunks-2)*heads
                    seq[(last,False,True)]=heads
                histograms.append(seq)
            else:
                parts=num*h//group
                begin=parts*core//act*group;end=parts*(core+1)//act*group
                def tails(x):
                    period=chunks*h
                    return (x//period)*h+max(0,x%period-(chunks-1)*h)
                tail=tails(end)-tails(begin)
                hist=Counter()
                if tail:hist[last]+=tail
                if end-begin>tail:hist[64]+=end-begin-tail
                histograms.append(hist)
        total=Counter()
        for hist in histograms:total.update(hist)
        phase={}
        for key in total:
            if i==0: phase[key]=prepare(key,chip,act)
            elif i==1: phase[key]=recurrence(key[0],chip,act,key[1],key[2])
            else: phase[key]=final(key,chip,act,mover)
        # Retired small-case front end up to first data-bearing MTE2 command; metadata
        # <=48 bytes is included here. Charge once per concurrently launched core group.
        anchor=prices()[0]['2']['stages'][f]
        startup=min(t['begin'] for t in anchor['transfers'] if t['size']>48)
        cycles=[startup+sum(count*sum(t for _,t in phase[key].legs) for key,count in hist.items()) for hist in histograms]
        critical=max(range(act),key=cycles.__getitem__)
        dma=weighted(total,phase,'bytes'); desc=weighted(total,phase,'issues')
        if i==0:
            dma['OUT->UB']+=96*act;desc['OUT->UB']+=2*act
        mm=weighted(total,phase,'mm'); vec=weighted(total,phase,'vf')
        # Scenario lower bound: only spill beyond aggregate L2 for now. Local rates
        # already charge all workspace traffic once at the shared on-chip boundary.
        issued_ops=0
        for signature,count in mm.items():
            m,n,kdim=map(int,signature.split(':')[0].split('x'))
            issued_ops+=2*m*n*kdim*count
        external_read_spill=max(0,sum(v for k,v in dma.items() if k.startswith('OUT'))-(chip.details.get('l2_bytes') or 0))
        # OUT descriptors include workspace reads from previous stages. The cold
        # miss model is a scenario; churn does not prove those bytes left the chip.
        floor=external_read_spill/(chip.hbm_bandwidth_gbs*1000)
        est=max(max(cycles)/(chip.frequency_ghz*1000),floor)
        stages.append(dict(family=f,active_aic=act,launched_aiv=act*2 if i<2 or mover else 0,
            aiv_mover=mover,issued_mmad_ops=issued_ops,timing_evidence='uncalibrated AIV-mover extrapolation' if mover else 'resource candidate',chip_group_assumption='two physical AIV per mixed workgroup',mmads=mm,
            dma_bytes=dma,dma_descriptors=desc,
            estimated_rvec_subqueue_records=vec,first_payload_cycles=startup,
            max_core_head_chunks=max(sum(x.values()) for x in histograms),device_us=est,
            hbm_spill_scenario_us=floor,
            critical_core=critical,critical_core_head_chunks={str(k):v for k,v in histograms[critical].items()},
            representative_tile_phase_cycles={str(k):dict(v.legs) for k,v in phase.items()}))
    times=[s['device_us'] for s in stages]
    return dict(shape=list(shape),chip=chip.name,dtype='bf16',stages=stages,
                serialized_device_us=sum(times) if all(x is not None for x in times) else None,
                scope='three main V2 kernels only, no auxiliary launches or inter-launch intervals',
                uncalibrated=['cross-head overlap','TF32 throughput','HBM spill and cache reuse','AIV input-mover finalization','970 physical mapping'])
