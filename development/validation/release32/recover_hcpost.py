"""Verify the closed per-launch export when msopprof was stopped copying its top-level duplicate."""
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor
from hashlib import sha256
import json
import math
from pathlib import Path
import re
import sys

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
from development.validation.rms_ordered_verify.verify import events, text_bytes


def digest(path):
    h = sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda:f.read(4*2**20), b''):
            h.update(chunk)
    return h.hexdigest()


def scan(path, *, merged=False):
    cores = {}
    writes = []
    for e in events(path):
        c = e['pid'] if merged else path.parent.name
        assert re.fullmatch(r'core\d+\.veccore[01]',c),(path,c)
        lo = round(e['ts']*1650)
        hi = round((e['ts']+e['dur'])*1650)
        assert lo <= hi
        v = cores.setdefault(c,[lo,hi,[],0])
        v[0],v[1] = min(v[0],lo),max(v[1],hi)
        v[3] += 1
        if e['name']=='END':
            v[2].append(lo)
        if not merged and e['name']=='MOV_SRC_TO_DST_ALIGNv2':
            d = json.loads(e['args']['detail'])
            if d.get('dst_mem')=='OUT' and d.get('src_mem')=='UB' and not any(d.get(k) for k in ('is_wait','is_duration_excluded','is_sync')):
                assert d['burst_number']*d['burst_len']==d['processed_bytes'],(path,d)
                for i in range(d['burst_number']):
                    a = d['dst_addr']+i*d['dst_stride']
                    writes.append((a,a+d['burst_len']))
    assert cores, str(path)
    return cores,writes,digest(path)


def verify(chip):
    name = 'vllm_hc_post-'+chip
    regfile = ROOT/'results/v0.31/post/freeze.json'
    strict = ROOT/'results/v0.31/post'/(name+'.json')
    original = json.loads(strict.read_bytes())
    j, = [v for v in json.loads(regfile.read_bytes())['jobs'] if v['op']=='vllm_hc_post']
    assert not original['qualified'] and original['cost']['stop_reason']=='raw cap'
    assert original['freeze_sha256']==digest(regfile) and j['shape']==original['shape']
    raw = ROOT/'runs/specialized31/post'/name
    log = (raw/'run.log').read_text()
    assert re.findall(r'^runtime_soc=(\S+)',log,re.M)==[chip] and 'SMOKE_PASS' in log and 'SMOKE_FAIL' not in log
    assert ('lpddr5_6400M_bailu.ini' if 'PR' in chip else 'hbm3_8000M.ini') in log
    assert not re.search(r'\[ERROR\].*(?:execute_set_flag|execute_wait_flag|deadlock|invalid instruction)',log)
    gold = re.findall(r'^correctness name=output count=(\d+) bad=0 comparison=bf16_rne_exact$',log,re.M)
    assert list(map(int,gold)) == [j['expected_numerics']['output']]
    kernel, = re.findall(r'Start profiling on kernel: (\S+)',log)
    assert kernel.startswith('HcPost_')
    base, = raw.glob(f'profile/OPPROF_*/device0/{kernel}/0')
    profile = json.loads((ROOT/'results/v0.29/large/fusion/freeze.json').read_bytes())
    binary = base/'dump/aicore_binary.o'
    text_id = sha256(text_bytes(binary)).hexdigest()
    providers = [r for r in profile['provider']['vllm_hc_post'] if r['text_sha256']==text_id]
    assert providers
    folders = {p.parent.name:p for p in base.glob('simulator/core*/trace.json')}
    physical = {f'core{c//2}.veccore{c%2}' for c in map(int,re.findall(r'\[block_end\].*?core_id=(\d+)',log))}
    assert set(folders)==physical and len(folders)==j['predictions'][chip]['active_aiv']
    trace = base/'simulator/trace.json'
    spans, fingerprint, stores = {},{},[]
    with ProcessPoolExecutor(max_workers=3) as pool:
        global_window = pool.submit(scan,trace,merged=True)
        for core,vals in zip(folders,pool.map(scan,folders.values())):
            per_core, writes, sign = vals
            assert set(per_core)=={core}
            spans[core],fingerprint[core]=per_core[core],sign
            stores.extend(writes)
        merged,_,global_sha = global_window.result()
    assert set(merged)==set(spans)
    offsets = [spans[c][i]-merged[c][i] for c in spans for i in (0,1)]
    end_counts = {c:(len(spans[c][2]),len(merged[c][2])) for c in spans}
    diagnostic = dict(chip=chip,terminal_counts=end_counts,merged_windows=merged,shard_windows=spans,
                      elapsed_bounds_ticks=[min(x[0] for x in merged.values()),max(x[1] for x in merged.values())],
                      trace_sha256=global_sha,shard_sha256=fingerprint,output_writes=len(stores),
                      transferred_bytes=sum(b-a for a,b in stores),output_elements=int(gold[0]),
                      physical_scheduler_ends=len(physical),kernel_text_sha256=text_id)
    (ROOT/'results/v0.32/hcpost'/(chip+'-audit.json')).write_text(json.dumps(diagnostic,indent=2)+'\n')
    assert all(v==(1,1) for v in end_counts.values()),end_counts
    for c in spans:
        for v in (spans[c],merged[c]):
            assert v[0]<=v[2][0]<=v[1] and v[0]<v[1],(c,v)
        offsets.append(spans[c][2][0]-merged[c][2][0])
    assert max(offsets)-min(offsets)<=3, (min(offsets),max(offsets))
    stores.sort()
    assert stores and all(a[1]==b[0] for a,b in zip(stores,stores[1:]))
    output_bytes = stores[-1][1]-stores[0][0]
    assert output_bytes==2*int(gold[0])
    us = (max(v[1] for v in merged.values())-min(v[0] for v in merged.values()))/1650
    return dict(chip=chip,shape=j['shape'],op=j['op'],kernel=kernel,ca_us=us,physical_complete=True,
                official_summary_complete=False,elapsed_minutes=original['cost']['seconds']/60,
                clock_ticks_per_us=1650,critical_core=max(merged,key=lambda c:merged[c][1]),active_aiv=len(merged),
                good_outputs=int(gold[0]),output_bytes=output_bytes,output_ranges=len(stores),
                chip_and_golden_log_sha256=digest(raw/'run.log'),trace=str(trace.relative_to(ROOT)),trace_sha256=global_sha,
                strict_receipt=str(strict.relative_to(ROOT)),strict_sha256=digest(strict),kernel_text_sha256=text_id,
                matching_installed=providers,raw_clock_offsets=[min(offsets),max(offsets)],
                shard_sha256=fingerprint,merged_windows_ticks=merged,shard_windows_ticks=spans)


if __name__=='__main__':
    result = verify(sys.argv[1])
    (ROOT/'results/v0.32/hcpost'/(sys.argv[1]+'.json')).write_text(json.dumps(result,indent=2)+'\n')
    print(result['ca_us'],result['output_bytes'])
