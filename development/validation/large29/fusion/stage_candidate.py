"""Preregister the split-Pre and conditional Sinkhorn large-geometry schedules."""
import hashlib
import json
import math
from pathlib import Path
from statistics import median
import sqlite3
import subprocess
import sys
from datetime import datetime, timezone

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT))
from operator_model import load_chip
from operator_model.sink_history import _epochs, _prices, price

OUT = ROOT / 'results/v0.29/large/fusion'
DEV = ('Ascend950DT_9582', 'Ascend950PR_957b')
SKUS = DEV + ('Ascend970DT',)
INSTALL = ROOT / 'toolchains/cann/9.2.1/cann-9.2.1/opp/built-in/op_impl/ai_core/tbe/impl/ops_transformer/ascendc'
def read(p):
    return json.loads(Path(p).read_bytes())
def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def align(n, k):
    return k * math.ceil(n / k)
def rel(p):
    return str(Path(p).relative_to(ROOT))


def extract_prices():
    """Select only M-K launches; one physical core and indexed epochs per launch."""
    rows = []
    seen = set()
    for x in read(ROOT/'results/v0.9/mhc/rows.json'):
        if x['op'] != 'mhc_pre' or x['kernel'] != 'entry_kernel_2':
            continue
        k = read(ROOT/x['source'])['measurement']['kernels'][0]
        if k['db_sha256'] in seen:
            continue
        seen.add(k['db_sha256'])
        db = sqlite3.connect('file:'+k['db_path']+'?mode=ro', uri=True)
        try:
            entries = db.execute("SELECT ExecInstrName,ExecInstrTickStart,ExecInstrTickEnd,ExtendParams,SourceInstrAddr FROM ExecutedInstructions WHERE ExecInstrCoreId=0 AND ((CoreTypeId=1 AND ExecInstrName='MMAD') OR (CoreTypeId=2 AND ExecInstrName='VF')) ORDER BY ExecInstrTickStart").fetchall()
            zero = min(c['start_ticks'] for c in k['cores'])
            zero = db.execute('SELECT MIN(ExecInstrTickStart) FROM ExecutedInstructions').fetchone()[0] + zero
        finally:
            db.close()
        record = event_prices(entries, zero, x['shape'])
        if record:
            rows.append(dict(record, shape=x['shape'], chip=DEV[0], db=rel(k['db_path']), sha256=k['db_sha256']))
    for x in read(ROOT/'results/v0.17/sku/pairs.json'):
        if x['op'] != 'mhc_pre' or not x['pr']['kernel'].endswith('_2_mix_aic'):
            continue
        k = x['pr']['stages'][0]
        parent = (ROOT/k['trace']).parent
        entries, shards = [], []
        for core,op in [('core0.cubecore0','MMAD'),('core0.veccore0','VF')]:
            p = parent/core/'trace.json'
            es = [e for e in read(p)['traceEvents'] if e.get('ph') == 'X']
            zero = min(e['ts'] for e in es)-k['windows'][core][0]
            shards.append(dict(path=rel(p),sha256=sha(p)))
            entries.extend((op,round((e['ts']-zero)*1650),round((e['ts']+e['dur']-zero)*1650),
                            e['args']['detail'],int(e['args']['pc_addr'],16)) for e in es if e['name']==op)
        record = event_prices(sorted(entries,key=lambda r:r[1]),0,x['shape'])
        if record:
            rows.append(dict(record, shape=x['shape'], chip=DEV[1], shards=shards, sha256=k['trace_sha256']))
    price_ref = [r for r in rows if r['chip']==DEV[0]]
    periods = {b:median(r['mmad_ticks'] for r in price_ref if r['m_blocks']==b) for b in (1,2)}
    cube_step = periods[2]-periods[1]
    # Two row occupancies of the same 256-wide local CAST, verified 7 ordered PCs.
    def cast(r):
        return math.ceil(r['shape'][0]/2) * r['local_k']
    shapes = [r for r in price_ref if r['local_k']==256]
    slopes = [(b['cast_ticks']-a['cast_ticks'])/(cast(b)-cast(a)) for a in shapes for b in shapes if cast(b)>cast(a)]
    cast_rate = median(slopes)
    ptr = dict(mmad_fixed_ticks=periods[1]-cube_step,mmad_m16_ticks=cube_step,
               cast_ticks_per_element=cast_rate,cast_fixed_ticks=median(r['cast_ticks']-cast_rate*cast(r) for r in shapes),
               first_cast_issue_ticks=median(r['first_vf'] for r in price_ref),
               from_last_mmad_to_first_stage2_ticks=median(max(0,r['first_stage2']-r['last_mmad_end']) for r in price_ref),
               stage2_coeff_group_ticks=median(r['vf_before_hin_span'] for r in price_ref),
               hin_ticks_per_output=median(r['hin_ticks']/(r['stage2_rows']*r['shape'][2]) for r in price_ref if r['shape'][2]>=512),
               residual_drain_ticks=median(r['drain_ticks'] for r in price_ref))
    return rows,ptr


def event_prices(entries,zero,shape):
    mm = [e for e in entries if e[0]=='MMAD']
    vf = [e for e in entries if e[0]=='VF']
    if not mm or len(vf)!=7:
        return None
    geom = [json.loads(e[3]) for e in mm]
    if any(d['k']!=32 or d['n']!=24 or d['matrix_a_dtype']!='f32' for d in geom):
        return None
    # The six stage-2 calls share one row group in the small development windows.
    t = shape[0]
    return dict(local_k=sum(d['k'] for d in geom),m_blocks=math.ceil(geom[0]['m']/16),
                mmads=len(mm),mmad_ticks=median(e[2]-e[1] for e in mm),cast_ticks=vf[0][2]-vf[0][1],
                first_vf=vf[0][1]-zero,first_stage2=vf[1][1]-zero,last_mmad_end=mm[-1][2]-zero,
                vf_before_hin_span=max(e[2] for e in vf[1:-1])-vf[1][1],hin_ticks=vf[-1][2]-vf[-1][1],
                stage2_rows=min(2,t),drain_ticks=max(0,vf[-1][1]-max(e[2] for e in vf[1:-1])),
                pcs_hex=[hex(e[4]) for e in vf])


def pre(t,chip,p):
    # No 1:1 binary with the same launch contract has been observed.
    if chip.vector_cores!=2*chip.cube_cores:
        return dict(us=None,reason='Installed 950 Pre is MIX_AIC_1_2; no verified 1:1 handoff or inferred stage2 implementation')
    h=5120; k=4*h; o=24
    md0=math.ceil(t/256);kd=chip.cube_cores//md0
    if kd<1:
        return dict(us=None,reason='Outside reference M-K grid')
    m=align(math.ceil(t/(chip.cube_cores//kd)),16)
    md=math.ceil(t/m)
    k0=align(math.ceil(k/kd),256)
    sk=align(math.ceil(k/kd/1024),1)*1024 if k0>=2048 else k0
    nk=math.ceil(k/sk)
    # FP32 K+=32 for MMAD even if a wider L0 page is reused.
    l1=128*(min(32768//m,32768//24,1024)//128)
    if sk>=2048:
        while 1024%l1:
            l1-=128
    pages=math.ceil(sk/l1);r=math.ceil(t/chip.vector_cores)
    clock=1000*chip.frequency_ghz
    ref=load_chip()
    vr=(ref.vector_tops['32']/ref.vector_cores)/(chip.vector_tops['32']/chip.vector_cores)*chip.frequency_ghz/ref.frequency_ghz
    cr=(ref.cube_tops['16']/ref.cube_cores)/(chip.cube_tops['16']/chip.cube_cores)*chip.frequency_ghz/ref.frequency_ghz
    # Chunked strip-mining + one full-size per-slot producer/consumer pipe.
    ready=0.;finish=0.;backend=[0.,0.];page_rows=math.ceil(m/2)
    for page in range(pages):
        w=min(l1,sk-page*l1)
        idx=page%2
        ready=max(ready,backend[idx])+p['cast_fixed_ticks']+p['cast_ticks_per_element']*page_rows*w*vr
        finish=max(ready,finish)+math.ceil(w/32)*(p['mmad_fixed_ticks']+math.ceil(m/16)*p['mmad_m16_ticks']*cr)
        backend[idx]=finish
    kg=math.ceil(k/1024) if sk>=2048 else nk
    remaining=r
    chunks=[]
    while remaining:
        rows=min(32,remaining)
        if rows>8:
            rows=rows//8*8
        while kg*align(rows*o,8)+nk*align(rows,8)>20480:
            rows-=8 if rows>8 else 1
        chunks.append(rows);remaining-=rows
    # Stage-2 HIn buffer is 80 KiB X and 20 KiB output; exactly two D5120 rows fit.
    batch=min(4,81920//(2*4*h),20480//(2*h))
    def span(rows):
        # Coefficient-vector unit is one observed 2-row group; no independently
        # identifiable large-row or sequential-partial reduction issue price.
        return math.ceil(rows/2)*p['stage2_coeff_group_ticks']*vr + rows*h*p['hin_ticks_per_output']*vr+math.ceil(rows/batch)*p['residual_drain_ticks']
    target=p['first_cast_issue_ticks']+finish+p['from_last_mmad_to_first_stage2_ticks']+sum(span(x) for x in chunks)
    logical=2*t*k+4*k*o+108+2*t*h+4*t*(4+16+1+24+4)
    # Unique I/O cold-service scenario; on-chip workspace traffic may retain in L2.
    hbm=logical/(chip.hbm_bandwidth_gbs*1000)
    return dict(us=max(target/clock,hbm),m_groups=md,k_groups=nk,m_local=m,k_local=sk,active_projection_aic=md*nk,
                rows_stage2=r,stage2_reduce_chunks=chunks,hin_rows_per_batch=batch,l1_k=l1,pages=pages,
                mmads_critical=math.ceil(sk/32),mmad_k=32,aligned_m16=math.ceil(m/16),unique_io_B=logical,
                cold_hbm_us=hbm,projection_page_sweep_us=finish/clock,handoff_us=p['from_last_mmad_to_first_stage2_ticks']/clock,
                first_issue_us=p['first_cast_issue_ticks']/clock,stage2_us=sum(span(x) for x in chunks)/clock,
                unpriced='L1-staging shuffle/GM fallback, K-group reduction-rate change, L1→L0/refills and HBM contention, final epilogue omitted')


def sink(t,chip):
    a=chip.vector_cores; local=math.ceil(t/a); active=math.ceil(t/local)
    reg=_prices()
    # Conservative existing cap. Relaxed cap uses actual inference buffer allocations
    # (two 16+16+32+8-element FP32 queues per row, 832 B fixed).
    ub=chip.details['ub_bytes']; caps=[(ub-832)//896,(ub-832)//576]
    data={}
    for label,h in [('first',0),('expand',1),('resident',active)]:
        us,_,_,diag=price((t,4),chip,dict(resident_aiv=h,num_iters=20))
        cases=[]
        if us is None:
            for cap in caps:
                remaining=local;segments=[]
                while remaining:
                    nr=min(cap,remaining)
                    plan=dict(rows_per_core=nr,active_aiv=active,register_passes=math.ceil(nr/reg['aligned_rows_per_vreg']),
                              logical_read_bytes=64*min(t,active*nr))
                    sc=_epochs(chip,reg,reg['phase_prices'][label],plan,20)
                    segments.append((nr,sc));remaining-=nr
                # Per-block payload depends on row/column iteration. All per-block input to write
                # epochs in sequence; mask + launch occur once. Inter-block reissue remains a hypothesis.
                repeated=sum(x['write_end']-x['setup_vf'] for _,x in segments[1:])
                cases.append(dict(cap=cap,blocks=len(segments),tail=segments[-1][0],us=segments[0][1]['end']+repeated,
                                  local_payload_vfs=41*len(segments),setup_mask_vfs=1))
        data[label]=dict(explicit_one_ub_us=us,qualified_status=diag.get('history_timing_status'),block_scenarios=cases)
    return dict(rows=local,active_aiv=active,iterations=20,scenarios=data)


def register():
    out=OUT/'stage-freeze.json'
    if out.exists():
        raise SystemExit('Stage registration is immutable')
    events,p=extract_prices()
    jobs=read(OUT/'freeze.json')['jobs']
    inputs={t:next(o['model'] for j in jobs for o in j['outputs'] if o['op']=='mhc_pre' and o['shape'][0]==t) for t in (512,4096)}
    sk={t:next(o['model'] for j in jobs for o in j['outputs'] if o['op']=='mhc_sinkhorn' and o['shape'][0]==t) for t in (512,4096,262144)}
    proof=[]
    for j in jobs:
        for sku in DEV:
            raw=OUT/'raw'/(j['id']+'-'+sku)
            proof.append(dict(job=j['id'],chip=sku,raw=rel(raw),started=raw.exists()))
    docs=['mhc_pre/op_kernel/arch35/mhc_pre_m_k_split.h','mhc_pre/op_kernel/arch35/mhc_pre_cube_compute.h',
          'mhc_pre/op_kernel/arch35/mhc_pre_vector_compute.h','mhc_sinkhorn/op_kernel/arch35/mhc_sinkhorn_arch35.h']
    frozen={'utc':datetime.now(timezone.utc).isoformat(),'parent_sha256':sha(OUT/'freeze.json'),'source_sha256':sha(__file__),
            'host_reference':'14d022b1a4c3df263bf697534e3c25c7c992ff5d:mhc/mhc_pre/op_host/op_tiling/arch35/mhc_pre_tiling.cpp',
            'installed_sources':{d:sha(INSTALL/d) for d in docs},'old_traces':events,'prices_ticks':p,'roles_before_large_CA':proof,
            'pre':[{ 't':t,'chip':name,'public_us':inputs[t][name]['us'],'candidate':pre(t,load_chip(name),p)} for t in inputs for name in SKUS],
            'sink':[{'t':t,'chip':name,'public_us':sk[t][name]['us'],'conditional':sink(t,load_chip(name))} for t in sk for name in SKUS]}
    with out.open('x') as f:
        json.dump(frozen,f,indent=2,allow_nan=False);f.write('\n')


if __name__=='__main__':
    register()
