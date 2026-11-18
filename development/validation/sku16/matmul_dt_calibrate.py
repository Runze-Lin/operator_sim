"""Only enumerates old, explicitly 950DT Matmul captures. No target labels/estimates."""
from collections import defaultdict
from hashlib import sha256
import json
from math import ceil, log2
from pathlib import Path
import sqlite3
from statistics import median

ROOT=Path(__file__).resolve().parents[3]
O=ROOT/'results/v0.16'
chip_file=ROOT/'configs/chips/Ascend950DT_9582.json'
chip=json.loads(chip_file.read_text()); GHZ=chip['frequency_ghz']
DT_PEAK=chip['cube_tops']['16']*1000/chip['cube_cores']/GHZ
align16=lambda n:16*ceil(n/16)
def line(xs,ys):
    # Nonnegative 2-term least squares for stage duration, never for total.
    n=len(xs);mx=sum(xs)/n;my=sum(ys)/n
    var=sum((x-mx)**2 for x in xs)
    slope=sum((x-mx)*(y-my) for x,y in zip(xs,ys))/var if var else 0
    options=[(max(0,my),0.),(0.,max(0,sum(x*y for x,y in zip(xs,ys))/sum(x*x for x in xs)))]
    if slope>=0 and my>=slope*mx:options.append((my-slope*mx,slope))
    return min(options,key=lambda q:sum((a*q[1]+q[0]-b)**2 for a,b in zip(xs,ys)))

def files():
    seen=set()
    for file in ('dataset.json','imported-calibration.json'):
        for r in json.loads((ROOT/'results/v0.2'/file).read_text())['records']:
            if r['case']['op']!='matmul' or r['status']!='ok':continue
            assert r['identity']['soc_version']==chip['name']
            d=r['measurement'];p=Path(d['report_path'])/'simulator/trace.json';key=d['trace_sha256']
            if key not in seen:
                yield p,key,r['case']['shape'];seen.add(key)
    for p in sorted((ROOT/'runs/v8cube').glob('matmul_*/result.json')):
        r=json.loads(p.read_text())
        if r['status']!='ok':continue
        assert 'OPERATOR_SIM_TARGET_SOC='+chip['name'] in r['command'] and len(r['kernels'])==1
        db,=p.parent.rglob('*.db')
        yield db,r['kernels'][0]['db_sha256'],r['shape']

def get_events(p):
    if p.suffix=='.db':
        db=sqlite3.connect(p.as_uri()+'?mode=ro&immutable=1',uri=True)
        try:
            for core,start,end,op,param in db.execute('SELECT ExecInstrCoreId,ExecInstrTickStart,ExecInstrTickEnd,ExecInstrName,ExtendParams FROM ExecutedInstructions'):
                yield core,start,end,op,json.loads(param or '{}')
        finally:db.close()
    else:
        for e in json.loads(p.read_bytes())['traceEvents']:
            if e['ph']=='X':
                yield e['pid'],round(e['ts']*1000*GHZ),round((e['ts']+e['dur'])*1000*GHZ),e['name'],json.loads(e.get('args',{}).get('detail','{}'))

def main():
    evidence=[]
    for p,key,shape in files():
        assert sha256(p.read_bytes()).hexdigest()==key
        cores=defaultdict(list)
        for e in get_events(p):cores[e[0]].append(e)
        start=min(e[1] for g in cores.values() for e in g)
        crit=max(cores.values(),key=lambda g:max(e[2] for e in g))
        read=sorted((e for e in crit if e[3]=='MOV_OUT_TO_L1_MULTI_ND2NZ'),key=lambda e:e[1])
        mm=sorted((e for e in crit if e[3]=='MMAD'),key=lambda e:e[1])
        a,b=read[:2];first=mm[0];s=first[4]
        frag=lambda e:2*align16(e[4]['m'])*align16(e[4]['n'])*align16(e[4]['k'])
        ideal=frag(first)/DT_PEAK
        fg=sorted((e for e in crit if e[3]=='LOAD_2Dv2'),key=lambda e:e[1])
        assert s['m'] and s['n'] and s['k'] and fg
        # Read service: two independent descriptors, one critical pair and
        # exposed pair-to-MMAD. Stage evidence preserves cycles and bytes.
        page_end=max(a[2],b[2]); maxbytes=max(a[4]['processed_bytes'],b[4]['processed_bytes'])
        residual_ii=(mm[1][1]-mm[0][1]-ideal) if len(mm)>=2 and len(mm)>len(read)//2 else None
        # Next DMA page: compare the second pair's last completion to first pair's.
        page_period=(max(e[2] for e in read[2:4])-page_end) if len(read)>=4 else None
        evidence.append(dict(shape=shape,reference_ticks=max(e[2] for e in crit)-start,trace=str(p.relative_to(ROOT)),sha256=key,active=len(cores),prefix=a[1]-start,
            first_read=max(a[2],b[2])-a[1],first_read_bytes=maxbytes,read_issue_pair=b[1]-a[1],
            l0_lead=first[1]-page_end,l0_bytes=2*s['k']*max(s['m'],s['n']),
            first_mmad=first[2]-first[1],first_mmad_ops=frag(first),mmad_ii_residual=residual_ii,
            drain=max(e[2] for e in crit)-max(e[2] for e in mm),out_bytes=2*s['m']*s['n'],
            measured_page_period=page_period))
    col=lambda k:[r[k] for r in evidence]
    launch,arrival=line([log2(v) for v in col('active')],col('prefix'))
    read,read_slope=line(col('first_read_bytes'),col('first_read'))
    l0,l0_slope=line(col('l0_bytes'),col('l0_lead'))
    drain,drain_slope=line(col('out_bytes'),col('drain'))
    # Issue floor vs shape-dependent throughput for successive DMA pages.
    # Repeated A/B descriptor issue gaps are directly measured; no end-to-end fit.
    cube=median(max(0,r['first_mmad']-r['first_mmad_ops']/DT_PEAK) for r in evidence)
    cube_ii=median(max(0,r['mmad_ii_residual']) for r in evidence if r['mmad_ii_residual'] is not None)
    params=dict(launch_cycles=launch,arrival_log2_cycles=arrival,read_latency_cycles=read,
                read_ii_cycles=median(col('read_issue_pair')),mte2_bytes_per_cycle=1/read_slope,
                l0_latency_cycles=l0,mte1_bytes_per_cycle=1/l0_slope,mmad_latency_cycles=cube,
                mmad_ii_cycles=cube_ii,drain_cycles=drain,write_bytes_per_cycle=1/drain_slope,
                default_bf16_ops_per_cycle=DT_PEAK)
    (O/'matmul.json').write_text(json.dumps(dict(schema=1,training_soc=chip['name'],parameters=params),indent=2)+'\n')
    (O/'matmul-dt-provenance.json').write_text(json.dumps(dict(units='ticks at DT frequency, bytes, ops',clock_ghz=GHZ,reference_chip_sha256=sha256(chip_file.read_bytes()).hexdigest(),script_sha256=sha256(Path(__file__).read_bytes()).hexdigest(),traces=evidence),indent=2)+'\n')
if __name__=='__main__':main()
