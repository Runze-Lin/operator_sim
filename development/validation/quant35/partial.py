"""Extract only complete common 1-us DDR intervals from a terminated 950DT native run."""
from collections import Counter
from hashlib import sha256
import json
from pathlib import Path
import re
from time import perf_counter

ROOT = Path(__file__).resolve().parents[3]
OUT = ROOT/'results/v0.35/quant'
SOURCE = ROOT/'results/v0.27/quant-transfer'
START = b'======================================== START ========================================'
END = b'========================================= END ========================================='
HEADER = re.compile(rb'Current time: (\d+), tDFI: 1\.0000 ns, DMC Total bytes: (\d+), DDR Total bytes: (\d+)')
COMMANDS = re.compile(rb'^(Read cnt|Read with AP|Write|Write with AP|Mask write|Mask write with AP) +: (\d+) +\| Total [^:\n]+: (\d+) *$',re.M)
QUEUE = re.compile(rb'^Cycle  \|([^\n]+)',re.M)
READS = (b'Read cnt',b'Read with AP')
WRITES = (b'Write',b'Write with AP',b'Mask write',b'Mask write with AP')


def digest(path):
    return sha256(path.read_bytes()).hexdigest()


def extract(channel):
    raw = channel.read_bytes()
    panels = raw.split(START)
    previous = Counter()
    bins = []
    for index,panel in enumerate(panels[1:],1):
        if END not in panel:
            assert index==len(panels)-1
            break
        assert panel.count(END)==1 and b'hbm3 8000Mbps, x32, DMC Data Width: 256' in panel
        tick,dmc_bytes,ddr_bytes=map(int,HEADER.search(panel).groups())
        assert tick==index*1000
        cmd={name:(int(n),int(total)) for name,n,total in COMMANDS.findall(panel)}
        assert set(cmd)==set(READS+WRITES)
        for key,(count,total) in cmd.items():
            assert previous[key]+count==total
            previous[key]=total
        q=[int(x) for x in QUEUE.search(panel)[1].split(b'|') if x.strip()]
        assert len(q)==65 and abs(sum(q)-1000)<=1
        read=sum(cmd[k][0] for k in READS)
        write=sum(cmd[k][0] for k in WRITES)
        assert (read+write)*32==ddr_bytes
        bins.append((read,write,dmc_bytes,q[0]))
    return bins,sha256(raw).hexdigest()


def main():
    started=perf_counter()
    get=lambda p:json.loads(p.read_bytes())
    failure=get(SOURCE/'native.json')
    log=SOURCE/'raw/run.log'
    # The terminated process appended the EINTR message six seconds after the
    # acquisition supervisor sealed the receipt. Authenticate the exact prefix.
    raw_log=log.read_bytes()
    suffix=b'Interrupted system call ; error code 4\n'
    prefix=raw_log.removesuffix(suffix)
    assert sha256(prefix).hexdigest()==failure['run_log_sha256'] and not failure['qualified_core_envelope']
    capture=failure['capture']
    assert capture['shape']==[131072,7168] and capture['chip']=='Ascend950DT_9582'
    assert capture['cost']['stop_reason']=='time cap' and b'hbm3_8000M.ini' in log.read_bytes()
    raw=SOURCE/'raw/log'
    names={f'lpddr_state{die+i}.log' for die in (0,131072) for i in range(64)}
    assert names=={p.name for p in raw.glob('lpddr_state*.log')}
    channels={}
    hashes={}
    for name in sorted(names):
        channels[name],hashes[name]=extract(raw/name)
    total_bins=min(map(len,channels.values()))
    periods=[]
    saturated=[0,0,0]
    for i in range(total_bins):
        bins=[rows[i] for rows in channels.values()]
        r,w,dmc,empty=map(sum,zip(*bins))
        for read,write,_,q0 in bins:
            if read and write and q0<=50:
                saturated[0]+=read
                saturated[1]+=write
                saturated[2]+=1
        periods.append(dict(end_us=i+1,read_bytes=32*r,write_bytes=32*w,dmc_bytes=dmc,
                            idle_queue_cycles=empty,dfi_payload_duty=(r+w)/128000))
    b8=get(ROOT/'results/v0.29/quant/b8-service.json')
    aligned=[]
    for b in b8['controller_periods']:
        l,h=b['start_us'],b['end_us']
        if h>total_bins:break
        sample=periods[l:h]
        aligned.append(dict(start_us=l,end_us=h,b4_read=sum(v['read_bytes'] for v in sample),
                            b4_write=sum(v['write_bytes'] for v in sample),b8_read=32*b['read_beats'],b8_write=32*b['write_beats']))
    assumptions=get(ROOT/'results/v0.29/quant/freeze.json')['predictions']['existing_b4']['Ascend950DT_9582']
    def first_write(prefix):
        return next(v for v in aligned if v[prefix+'_write']>0)
    # Endpoint registration defined the mixed phase as an equivalent service
    # budget, not the physical time of the first individual dirty eviction.
    steady=next((v['end_us'] for v in periods if v['write_bytes']>=1024*1024),None)
    measured=dict(common_us=total_bins,last_complete_us_by_channel={k:len(v) for k,v in channels.items()},
                  read_bytes=sum(v['read_bytes'] for v in periods),write_bytes=sum(v['write_bytes'] for v in periods),
                  busy_mixed_samples=saturated[2],busy_mixed_efficiency=sum(saturated[:2])/(saturated[2]*1000),
                  first_write_8us_bin={k:first_write(k)['start_us'] for k in ('b4','b8')},
                  first_bin_end_writing_at_least_mib=steady)
    result=dict(shape=capture['shape'],chip=capture['chip'],per_channel_sha256=hashes,
                failure_receipt_sha256=digest(SOURCE/'native.json'),run_log_sha256=digest(log),sealed_log_prefix_bytes=len(prefix),whole_kernel_us=None,terminal_golden=False,
                role='partial process-wide DFI observation; only fully closed common periods; no aligned instruction clock',
                elapsed_s=perf_counter()-started,observed=measured,frozen_cast_efficiency=assumptions['mixed_bus_efficiency'],
                b4_frozen_prefix_us=assumptions['prefix_us'],b8_source_sha256=digest(ROOT/'results/v0.29/quant/b8-service.json'),
                comparable_8us_bins=aligned,periods=periods)
    (OUT/'b4-partial.json').write_text(json.dumps(result,indent=2)+'\n')
    print({k:v for k,v in measured.items() if k!='last_complete_us_by_channel'},flush=True)


if __name__=='__main__':main()
