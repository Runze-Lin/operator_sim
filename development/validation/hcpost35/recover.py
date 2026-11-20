"""Recover the original per-launch HcPost export. No new acquisition, no edits to old receipts."""
from concurrent.futures import ProcessPoolExecutor
from hashlib import sha256
import json
import math
from pathlib import Path
import re
import struct
import sys
from time import perf_counter

ROOT=Path(__file__).resolve().parents[3]
OUT=ROOT/'results/v0.35/hcpost'
sys.path.insert(0,str(ROOT))
from development.validation.release32.recover_hcpost import scan, digest, text_bytes

CHIP='Ascend950PR_957b'
OP='vllm_hc_post'
SHAPE=[1,1024,4,5120]

def load(path):return json.loads(path.read_bytes())

def packed(path,global_sha):
    with path.open('rb') as f:
        extent,_=struct.unpack('<QI',f.read(12))
        raw_size=(path.parent/'trace.json').stat().st_size
        assert extent==(raw_size+3)//4*4 or extent==raw_size
        h=sha256()
        for start in range(0,raw_size,4*1024**2):
            n=min(raw_size-start,4*1024**2);data=f.read(n);assert len(data)==n;h.update(data)
        assert h.hexdigest()==global_sha and f.read(extent-raw_size)==bytes(extent-raw_size)
        extent,_=struct.unpack('<QI',f.read(12));secondary=f.read(extent);assert extent==len(secondary)
        data=json.loads(secondary.rstrip(b'\0\n'))
        counters={}
        for op in ('END','END_LABEL'):
            row,=[v for v in data['Instructions'] if v['Source'].partition(' ')[0]==op]
            assert len(row['Instructions Executed'])==len(data['Cores'])
            counters[op]=dict(zip(data['Cores'],row['Instructions Executed']))
        return counters,sha256(secondary).hexdigest()

def main():
    t=perf_counter()
    regfile=ROOT/'results/v0.31/post/freeze.json';reg=load(regfile)
    job,=[v for v in reg['jobs'] if v['op']==OP]
    audited=(job['runner'],'results/v0.29/large/fusion/freeze.json','development/validation/large29/fusion/fusion_campaign.py','development/validation/specialized31/large_post.py')
    assert all(digest(ROOT/n)==reg['dependencies'][n] for n in audited)
    initial=regfile.parent/(OP+'-'+CHIP+'.json');d=load(initial)
    assert d['shape']==job['shape']==SHAPE and d['chip']==CHIP and d['freeze_sha256']==digest(regfile)
    assert not d['qualified'] and d['cost']['stop_reason']=='raw cap' and d['cost']['exit_code']==-15
    raw=ROOT/'runs/specialized31/post'/(OP+'-'+CHIP)
    assert load(raw/'cost.json')==d['cost']
    log=(raw/'run.log').read_text();assert re.findall(r'^runtime_soc=(\S+)',log,re.M)==[CHIP]
    assert 'lpddr5_6400M_bailu.ini' in log and 'SMOKE_PASS' in log and 'SMOKE_FAIL' not in log
    assert 'Profiling running finished. All task success.' in log and 'Model stopped successfully.' in log
    assert 'api=aclnnHcPost library='+str(ROOT/job['runner']) in log
    assert len(re.findall(r'\[TASK_DONE\]',(raw/'stars_log0_0_task.dump').read_text()))==1
    assert not re.search(r'\[ERROR\].*(?:execute_set_flag|execute_wait_flag|deadlock|invalid instruction)',log)
    gold,=re.findall(r'^correctness name=output count=(\d+) bad=0 comparison=bf16_rne_exact$',log,re.M)
    gold=int(gold);assert gold==job['expected_numerics']['output']==math.prod(SHAPE)
    kernel,=re.findall(r'Start profiling on kernel: (\S+)',log)
    base,=raw.glob(f'profile/OPPROF_*/device0/{kernel}/0')
    elf=base/'dump/aicore_binary.o';elf_hash=digest(elf);text_hash=sha256(text_bytes(elf)).hexdigest()
    golden,=[r for r in load(ROOT/'results/v0.29/large/fusion/freeze.json')['provider'][OP] if r['text_sha256']==text_hash]
    assert digest(ROOT/golden['file'])==golden['sha256']
    assert sha256(text_bytes(ROOT/golden['file'])).hexdigest()==text_hash
    merged_file=base/'simulator/trace.json';files=sorted(base.glob('simulator/core*/trace.json'))
    table={c:float(x) for c,x,_ in re.findall(r'^(core\d+\.veccore[01])\s+([0-9.]+)\s+([0-9.]+)\s*$',log,re.M)}
    assert len(files)==len(table)==56 and set(table)=={f.parent.name for f in files}
    diagnostics={};stores=[]
    # The explicit optional scratch cache is bound to immutable raw file metadata and the acquisition SHA.
    # It lets diagnostic rechecks reuse the expensive independent JSON parses if a secondary format gate fails.
    from tempfile import gettempdir
    scratch=Path(gettempdir())/('operator-sim-hcpost35-'+digest(initial)+'.json')
    stamps={str(p.relative_to(ROOT)):[p.stat().st_size,p.stat().st_mtime_ns,p.stat().st_ino] for p in [merged_file,*files]}
    cache=load(scratch) if scratch.exists() else {}
    if cache.get('stamps')==stamps and cache.get('parser')==digest(ROOT/'development/validation/release32/recover_hcpost.py'):
        merged=cache['merged'];global_hash=cache['global_sha'];stores=cache['stores'];diagnostics=cache['diagnostics']
        assert digest(merged_file)==global_hash and all(digest(p)==diagnostics[p.parent.name]['shard_sha256'] for p in files)
        assert all(digest(next(p.parent.glob('*instr_exe*.csv')))==diagnostics[p.parent.name]['csv_sha256'] for p in files)
    else:
        with ProcessPoolExecutor(max_workers=3) as pool:
            merged_future=pool.submit(scan,merged_file,merged=True)
            for p,(spans,writes,sig) in zip(files,pool.map(scan,files)):
                c=p.parent.name;w,=spans.values();assert set(spans)=={c};stores+=writes
                all_csv,=p.parent.glob('*instr_exe*.csv')
                counts={name:sum(int(v.split(b',',4)[3]) for v in all_csv.read_bytes().splitlines() if v.startswith(name.encode()+b',')) for name in ('END','END_LABEL')}
                diagnostics[c]=dict(shard_ticks=w,shard_sha256=sig,output_ranges=len(writes),output_bytes=sum(b-a for a,b in writes),
                                    csv_counts=counts,csv_sha256=digest(all_csv),table_us=table[c])
            merged,_,global_hash=merged_future.result()
        scratch.write_text(json.dumps(dict(stamps=stamps,parser=digest(ROOT/'development/validation/release32/recover_hcpost.py'),merged=merged,global_sha=global_hash,stores=stores,diagnostics=diagnostics)))
    assert set(merged)==set(diagnostics)
    compact,packed_hash=packed(merged_file.parent/'visualize_data.bin',global_hash)
    origins=[]
    for c,r in diagnostics.items():
        m=merged[c];w=r['shard_ticks'];r['global_ticks']=m
        assert len(w[2])==len(m[2])==1
        assert w[0]<=w[2][0]<=w[1] and m[0]<=m[2][0]<=m[1] and w[0]<w[1]
        assert all(compact[k][c]==r['csv_counts'][k]==1 for k in ('END','END_LABEL'))
        assert abs(table[c]-(w[1]-w[0])/1650)<=.011
        origins.extend([w[0]-m[0],w[1]-m[1],w[2][0]-m[2][0]])
    assert max(origins)-min(origins)<=3
    stores.sort();assert stores and all(b==next_a for (_,b),(next_a,_) in zip(stores,stores[1:]))
    output=stores[-1][1]-stores[0][0]; assert output==2*gold
    low=min(v[0] for v in merged.values());high=max(v[1] for v in merged.values())
    us=(high-low)/1650
    freeze=ROOT/'results/v0.32/post/hc-pr-freeze.json'
    price,=[r for r in load(freeze)['results'] if r['sku']==CHIP];assert price['shape']==SHAPE
    result=dict(acquisition_receipt=str(initial.relative_to(ROOT)),original_qualified=False,original_reason=d['reason'],
        acquisition_sha256=digest(initial),acquisition_cost=d['cost'],original_budget=dict(wall_s=job['wall_s'],raw_mib=job['raw_mib']),
        chip=CHIP,shape=SHAPE,op=OP,api='aclnnHcPost',kernel=kernel,runtime_log_sha256=digest(raw/'run.log'),kernel_text_sha256=text_hash,
        captured_elf_sha256=elf_hash,installed_kernel=golden,qualified=True,scope='Independent recovery of single-launch closed device data under exact numerics/ELF, full-roster END and retire, global-to-shard clocks, official core durations and contiguous all-output writes; original top-level collection stopped on raw budget.',
        active_aiv=len(merged),critical_core=max(merged,key=lambda c:merged[c][1]),physical_outputs=gold,output_payload_bytes=output,
        store_runs=len(stores),global_ticks=[low,high],cycles_per_us=1650,ca_us=us,clock_offsets=[min(origins),max(origins)],
        candidate_us=price['candidate_us'],public_baseline_us=price['public_baseline_us'],candidate_signed_pct=100*(price['candidate_us']/us-1),
        public_signed_pct=100*(price['public_baseline_us']/us-1),price_freeze_sha256=digest(freeze),causal_role='CA already running when v0.32 DT-only candidate froze; PR outcome unseen at freeze.',
        global_trace_path=str(merged_file.relative_to(ROOT)),global_sha256=global_hash,packed_counters_sha256=packed_hash,
        identical_packed_global=True,all_shards=diagnostics,offline_validation_wall_s=perf_counter()-t)
    (OUT/'pr-full.json').write_text(json.dumps(result,indent=2)+'\n')
    print('qualified',us,len(merged),output,result['candidate_signed_pct'],flush=True)
if __name__=='__main__':main()
