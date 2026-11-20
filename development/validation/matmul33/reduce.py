"""Offline all-core timing certificate and STARS proxy audit. No simulator launch."""
import hashlib,json,re,sys,time
from pathlib import Path
from datetime import datetime,timezone
from watch import ROOT,OUT,DT,save,read,digest

WORD=rb'([-+0-9.eE]+)'
EVENT=re.compile(rb'"dur":'+WORD+rb',"name":"([^"]+)","ph":"X","pid":[^,]+,"tid":[^,]+,"ts":'+WORD+rb'}')
TASK=re.compile(r'\[info\]\s*\[(\d+)\]\s*\[(TASK_BEGIN|TASK_DONE)\].*stream_id:(\d+), task_id:(\d+)')
CORE=re.compile(r'^(core\d+\.(?:cube|vec)core\d+)\s+([0-9.]+)\s+([0-9.]+)\s*$',re.M)
CASES=[(f'results/v0.28/matmul/{m}-128-512-{tag}-drain.json',f'runs/matmul28/{m}-128-512-{tag}') for m in (32768,262144) for tag in ('dt','pr')]+[(f'results/v0.28/matmul/long-prefill/131072-{tag}.json',f'runs/matmul28/long-prefill/131072-{tag}') for tag in ('dt','pr')]+[(f'results/v0.31/matmul/device/99-64-7168-{tag}.json',f'runs/matmul31/device/99-64-7168-{tag}') for tag in ('dt','pr')]
def run(reference,raw):
    cost=read(raw/'cost.json');d=read(reference);assert d['qualified'] and cost['exit_code']==0 and not cost['stop_reason']
    s,=d['stages'];base=(ROOT/s['trace']).parent;assert raw in base.parents;log=(raw/'run.log').read_text(errors='replace')
    m,n,k=map(int,re.search(r'case=matmul api=aclnnMatmul M=(\d+) N=(\d+) K=(\d+) dtype=BF16',log).groups())
    assert f'correctness count={m*n} bad=0' in log and 'SMOKE_PASS' in log
    sku=re.search(r'runtime_soc=(\S+)',log).group(1);table={core:float(duration) for core,duration,_ in CORE.findall(log)}
    assert set(table)==set(s['windows']);elapsed=time.perf_counter();cores={};tot=0;source_bytes=0
    for path in sorted(base.glob('core*/trace.json')):
        name=path.parent.name;b=path.read_bytes();lo=float('inf');end=-1;mark=None;ends=cnt=mma=0
        for match in EVENT.finditer(b):
            dur,op,t=match.groups();t=float(t);t1=t+float(dur);cnt+=1
            lo=min(lo,t);end=max(end,t1)
            if op==b'END':ends+=1;mark=t
            mma+=op==b'MMAD'
        assert cnt and cnt==b.count(b'"ph":"X"') and ends==1 and lo<=mark<=end and abs(end-lo-table[name])<=.011
        cores[name]=dict(count=cnt,first_us=lo,retire_us=end,end_marker_us=mark,mmad=mma,sha256=hashlib.sha256(b).hexdigest())
        source_bytes+=len(b);tot+=mma
    assert set(cores)==set(table);origin=min(c['first_us'] for c in cores.values());window=max(c['retire_us'] for c in cores.values())-origin
    assert abs(window-s['ca_us'])<=2/1650 and tot==s['mmad']
    max_edge_cycles=max(round(1650*abs(v-(c[q]-origin))) for name,c in cores.items() for v,q in zip(s['windows'][name],('first_us','retire_us')))
    assert max_edge_cycles<=3  # original all-core drain contract compares rounded 1.65 GHz ticks
    elapsed=time.perf_counter()-elapsed
    src_file=raw/'stars_log0_0_task.dump';st=TASK.findall(src_file.read_text())
    assert len(st)==2 and st[0][1]=='TASK_BEGIN' and st[1][1]=='TASK_DONE' and st[0][2:]==st[1][2:]
    task=(int(st[1][0])-int(st[0][0]))/1000
    st_start=datetime.fromisoformat(cost['started_utc']).timestamp();st_end=datetime.fromisoformat(cost['finished_utc']).timestamp();st_mid=src_file.stat().st_mtime
    assert st_start<=st_mid<=st_end
    prefix=base.parent/'dump'/'aicore_binary.o';actual=digest(prefix);assert actual==s['binary_sha256']
    peak=cost.get('peak_bytes',cost.get('raw_bytes'));assert peak
    return dict(shape=[m,n,k],sku=sku,reference=str(reference.relative_to(ROOT)),reference_sha256=digest(reference),trace_sha256=s['trace_sha256'],kernel=s['kernel'],elf=actual,actual_golden_elements=m*n,timing_scope='all X records, every active core physical END, earliest X start to latest X retire, firmware table matched to rounding',source_log_sha256=digest(raw/'run.log'),compact_us=window,full_us=s['ca_us'],all_mmad=tot,cores=cores,max_core_edge_cycles=max_edge_cycles,
        task_proxy_us_if_1ns=task,task_ticks=[int(st[0][0]),int(st[1][0])],task_log_sha256=digest(src_file),longest_table_duration_us=max(table.values()),
        acquisition_wall_s=cost['seconds'],wall_to_stars_dump_mtime_s=st_mid-st_start,wall_after_stars_dump_mtime_s=st_end-st_mid,peak_bytes=peak,shard_input_bytes=source_bytes,
        global_file_bytes=(base/'trace.json').stat().st_size,visualize_file_bytes=(base/'visualize_data.bin').stat().st_size,
        observed_offline_reduce_s=elapsed,provider_exit_to_original_receipt_s=reference.stat().st_mtime-st_end)
def main():
    for ref,raw in CASES:
        p=ROOT/ref;r=ROOT/raw;name=r.name+'-certificate.json';dest=OUT/name
        if dest.exists():continue
        data=run(p,r);save(name,data);print(name,data['compact_us'],data['observed_offline_reduce_s'],flush=True)
if __name__=='__main__':main()
