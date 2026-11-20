"""Re-audit registered RMS chunk windows on both 950 SKUs. Never launches CA."""
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
import argparse
import hashlib
import json
import math
import mmap
import os
import re
import sys
import time

ROOT = Path(__file__).resolve().parents[3]
OUT = ROOT / 'results/v0.33/large'
sys.path.append(str(ROOT))
sys.path.append(str(ROOT/'development/validation/pro_prefill23'))
from development.validation.rms_ordered_verify.verify import digest, events, tables, text_bytes, VENDOR
from development.validation.pro_prefill23.evidence import endpoint

DT, PR, FUT = 'Ascend950DT_9582', 'Ascend950PR_957b', 'Ascend970DT'
CHIPS = DT, PR
CLOCK = 1650
KERNEL = 'RmsNorm_normal_all_bf16_high_performance_5000'
REG = OUT / 'freeze.json'
SHAPES = [[8192,5120],[32768,1280]]
HIST = ROOT/'results/v0.20/rms'

def read(path):
    return json.loads(path.read_bytes())

def save(path, obj, *, new=False):
    if new and path.exists():
        raise FileExistsError(path)
    tmp = path.with_suffix('.'+str(os.getpid())+'.tmp')
    tmp.write_text(json.dumps(obj, indent=2, ensure_ascii=False, allow_nan=False)+'\n')
    tmp.replace(path)

def rel(path):
    return str(path.relative_to(ROOT))

def check(condition, reason):
    if not condition:
        raise ValueError(reason)

def moves(shard):
    """Decode physical descriptors individually. The instruction CSV loses tail payloads."""
    needle=b'"name":"MOV_SRC_TO_DST_ALIGNv2"'
    with shard.open('rb') as file, mmap.mmap(file.fileno(),0,access=mmap.ACCESS_READ) as data:
        offset=0
        while True:
            pos=data.find(needle,offset)
            if pos<0:break
            start=data.rfind(b'{"args":{',max(0,pos-16384),pos)
            check(start>=0,'DMA JSON object')
            event=json.JSONDecoder().raw_decode(data[start:pos+1024].decode())[0]
            check(event['name']=='MOV_SRC_TO_DST_ALIGNv2','unexpected DMA')
            if event['ph']=='X':yield event
            offset=pos+len(needle)

def freeze():
    previous = read(ROOT/'results/v0.29/large/vector/freeze.json')
    registrations = [r for r in previous['jobs'] if r['op']=='rmsnorm' and r['protocol']=='instruction' and r['shape']==SHAPES[0]]
    check({r['chip'] for r in registrations}==set(CHIPS),'original paired large target')
    replay = read(ROOT/'results/v0.32/replay.json')['rows']
    jobs = []
    for shape in SHAPES:
        for chip in CHIPS:
            row, = [r for r in replay if r['op']=='rmsnorm · msopprof' and r['chip']==chip and r['shape']==shape]
            folder=ROOT/'runs/sku20-rms'/('rmsnorm_'+'_'.join(map(str,shape))+'_'+chip)
            jobs.append(dict(shape=shape,chip=chip,raw=rel(folder),old_receipt_sha256=digest(folder/'qualified.json'),
                archived_v032_price_us=row['us'],archived_future_970dt_us=row['extrapolated_970dt_us']))
    files=[ROOT/'results/v0.29/large/vector/freeze.json',ROOT/'results/v0.32/model.json',HIST/'freeze.json',HIST/'records.json',
           ROOT/'smoke/large_vector.cpp',ROOT/'build/smoke/large_vector',VENDOR]
    cfg=ROOT/'configs/models/DeepSeek-V4.1-Flash.json'
    check(read(cfg)['text_config']['hidden_size']==5120,'Flash hidden width')
    save(REG,dict(utc=datetime.now(timezone.utc).isoformat(),cases=jobs,hashes={rel(f):digest(f) for f in [*files,cfg]},
        historical_role='Previously measured, visible endpoints; archive replay, never independent for v0.33.',
        large_criterion='Exact match to the pre-existing v0.29 instruction-mode large case [8192,5120], actual Flash hidden, B4×2048 scheduled per request (or B8×1024 scheduled). Exact same v0.20 runner, dtype, epsilon and both tensors. [32768,1280] has the same elements and is a shape-sensitivity control, never substituted as a full-hidden or additional deployment validation.',
        production_target=dict(batch=8,per_request_sequence=[32768,131072,262144],nonchunked_rows=[262144,1048576,2097152],width=5120),
        audit_budget='No new CA. Reuse complete original launch records, merged SHA and official windows; recheck every physical shard END, core duration and both output byte volumes. Read bounded-memory from disk. Existing numbers were visible.'),new=True)

def audit(job):
    t,d=job['shape'];chip=job['chip'];folder=ROOT/job['raw'];orig=folder/'qualified.json'
    check(digest(orig)==job['old_receipt_sha256'],'original receipt changed')
    receipt=read(orig);stage,=receipt['stages'];cost=read(folder/'cost.json');log=(folder/'run.log').read_text()
    check(receipt['api']=='aclnnRmsNorm' and receipt['kernel']==stage['kernel']==KERNEL and receipt['shape']==[t,d] and receipt['dtype']=='bf16' and receipt['chip']==chip,'operator contract')
    check(cost['exit_code']==0 and not cost['stop_reason'] and 'SMOKE_PASS' in log and 'Profiling running finished. All task success.' in log,'terminal status')
    check(re.findall(r'^runtime_soc=(\S+)',log,re.M)==[chip] and
          ('lpddr5_6400M_bailu.ini' if chip==PR else 'hbm3_8000M.ini') in log,'actual SKU and memory')
    check(f'case=rmsnorm api=aclnnRmsNorm R={t} C={d} dtype=BF16 epsilon=1e-6' in log and
          re.findall(r'Start profiling on kernel: (\S+)',log)==[KERNEL],'actual launch')
    correct = re.findall(r'^correctness [^\n]+',log,re.M)
    check(correct==receipt['correctness']==[f'correctness count={t*d} bad=0',f'correctness count={t} bad=0'],'incomplete Y or stats')
    physical=(folder/'stars_log0_0_task.dump').read_text()
    b,=re.findall(r'\[(\d+)\] \[TASK_BEGIN\].*stream_id:(\d+), task_id:(\d+) full_task_id:(\d+) BEGIN',physical)
    e,=re.findall(r'\[(\d+)\] \[TASK_DONE\].*stream_id:(\d+), task_id:(\d+) TASK_DONE',physical)
    check(b[1:3]==e[1:3] and int(e[0])>int(b[0]),'hardware retirement')
    trace=ROOT/stage['trace'];sim=trace.parent;export=sim.parent/'dump/aicore_binary.o'
    published_text=text_bytes(VENDOR)
    check(text_bytes(export)==published_text and digest(export)==stage['binary_sha256'],'actual installed kernel')
    table,=tables(log);wins=stage['windows'];shards={p.parent.name:p for p in sim.glob('core*/trace.json')}
    check(set(table)==set(wins)==set(shards) and len(table)==(64 if chip==DT else 56),'physical roster')
    check([digest(trace)]==[stage['trace_sha256']]==receipt['trace_sha256'],'merged raw fingerprint')
    base=int((sim.parent/'dump/pc_start_addr.txt').read_text(),16)
    by_core={};origins=[];actual=Counter()
    for name, p in sorted(shards.items()):
        marks=endpoint(p)
        check('END' in marks,'missing unique physical retirement '+name)
        lo,hi=wins[name];retire=max(x['retire'] for x in marks.values())
        if 'END_LABEL' not in marks:
            # END_LABEL is a profiler annotation. Where absent, independently
            # scan every X event to find the true first/last device descriptors.
            lo_tick=math.inf;hi_tick=-math.inf
            for event in events(p):
                lo_tick=min(lo_tick,round(event['ts']*CLOCK))
                hi_tick=max(hi_tick,round((event['ts']+event['dur'])*CLOCK))
            check(lo_tick<=marks['END']['tick']<=hi_tick,'physical END outside completed shard')
            check(abs((hi_tick-lo_tick)/CLOCK-float(table[name][0]))<.025,'full-shard versus vendor duration')
            origins.append(lo_tick-round(lo*CLOCK));retire=hi_tick
        check(abs((hi-lo)-float(table[name][0]))<.025,'vendor duration '+name)
        origins.append(retire-round(hi*CLOCK))
        writes=Counter();tiles=Counter()
        for event in moves(p):
            detail=json.loads(event['args']['detail'])
            if detail['instr_type']!='MTE3':continue
            if any(detail.get(k) for k in ('is_wait','is_sync','is_duration_excluded')):continue
            check(detail['src_mem']=='UB' and detail['dst_mem']=='OUT' and detail['processed_bytes']==detail['burst_number']*detail['burst_len'],'write descriptor')
            off=int(event['args']['pc_addr'],16)-base
            # PCs from released high-performance BF16 ELF, pc_start_addr relocates each capture.
            key={0x5128:'y',0x4f38:'rstd'}[off]
            writes[key]+=detail['processed_bytes']
            if key=='y':
                check(detail['burst_len']==2*d,'RMS output row width')
                tiles[detail['burst_number']]+=1
        check(writes['y']==sum(k*v for k,v in tiles.items())*2*d and writes['rstd']==4*sum(k*v for k,v in tiles.items()),'per-core tile output and stats')
        actual.update(writes);by_core[name]=dict(physical_end=marks['END'],end_label=marks.get('END_LABEL'),farthest_retire=retire,
            vendor_duration_us=table[name][0],merged_window_us=[lo,hi],rows=sum(k*v for k,v in tiles.items()),y_tiles_by_row_count=tiles,output_bytes=dict(writes))
    check(max(origins)-min(origins)<=3,'core terminal ticks disagree with aggregate')
    check(actual==dict(y=2*t*d,rstd=4*t),'complete physical output bytes')
    ca=max(hi for lo,hi in wins.values())-min(lo for lo,hi in wins.values())
    check(math.isclose(ca,receipt['ca_us'],abs_tol=.001),'complete device window')
    estimate=job['archived_v032_price_us']
    return dict(qualified=True,role='revalidated archive replay',chip=chip,shape=[t,d],ca_us=ca,public_v032_us=estimate,
        signed_v032_pct=100*(estimate/ca-1),future_970dt_us=job['archived_future_970dt_us'],active_aiv=len(table),
        minmax_y_tiles=[min(sum(c['y_tiles_by_row_count'].values()) for c in by_core.values()),max(sum(c['y_tiles_by_row_count'].values()) for c in by_core.values())],
        minmax_rows_per_aiv=[min(c['rows'] for c in by_core.values()),max(c['rows'] for c in by_core.values())],cores=by_core,
        logical_input_bytes=2*t*d+2*d,logical_output_bytes=dict(actual),stage=KERNEL,installed_object=rel(VENDOR),api='aclnnRmsNorm',
        installed_text_sha256=hashlib.sha256(published_text).hexdigest(),dump_sha256=digest(export),trace=rel(trace),trace_sha256=stage['trace_sha256'],
        log_sha256=digest(folder/'run.log'),physical_task=[b,e],golden=correct,origin_tick_range=[min(origins),max(origins)],
        previous_receipt=rel(orig),cost=cost,registration_sha256=digest(REG))

def run():
    registration=read(REG)
    for p,h in registration['hashes'].items():check(digest(ROOT/p)==h,'changed '+p)
    for job in registration['cases']:
        t,d=job['shape'];tag='dt' if job['chip']==DT else 'pr'
        dest=OUT/f'rms-{t}-{d}-{tag}.json'
        if dest.exists():continue
        started=time.monotonic();r=audit(job);r['archive_audit_wall_s']=time.monotonic()-started
        save(dest,r,new=True);print(dest.name,r['ca_us'],r['minmax_rows_per_aiv'],r['minmax_y_tiles'],flush=True)
if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('action',choices=['freeze','audit']);arg=parser.parse_args()
    if arg.action=='freeze':freeze()
    else:run()
