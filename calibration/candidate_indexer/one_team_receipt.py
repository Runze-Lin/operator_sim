"""Verify the 950DT single-owner AOT row; optionally replay its observed issue pace on the last chunk."""
import argparse
from dataclasses import replace
from datetime import datetime
from hashlib import sha256
import json
from pathlib import Path
import re
from statistics import median
from zipfile import ZipFile

from calibration.__main__ import require
from calibration.candidate_indexer.fused_qsli_report import WHEEL_KEY
from calibration.msopprof import CORE, sha
from calibration.mx_matmul.collect import core_windows, trace_events

HERE=Path(__file__).parent
CLOCK=1650
def us(ticks):
    return round(ticks/CLOCK,9)


def receipt(raw,wheel):
    freeze=json.loads((HERE/'qsli-one-team-freeze.json').read_text())
    log=(raw/'run.log').read_text()
    require((raw/'rc.txt').read_text().strip()=='0' and 'All task success.' in log
            and 'hbm3_8000M.ini' in log and 'STATIC_SINGLE_OWNER' in log, 'Invalid CA run')
    fixture=json.loads(next(s.removeprefix('FIXTURE ') for s in log.splitlines() if s.startswith('FIXTURE ')))
    require(fixture['metadata_rows']==freeze['metadata_nonzero'] and fixture['workers']==32 and
            f"FUSED_GOLDEN_PASS 512 {fixture['expected_ids_sha256']}" in log, 'Metadata or golden mismatch')
    prof,=(raw/'profile').glob('OPPROF_*')
    device=(prof/'dump/aicore_binary.o').read_bytes()
    report=json.loads((raw/'provider.json').read_text())
    require(report['counts']=={'installed_hit':1} and report['events'][0]['key']==WHEEL_KEY, 'Unqualified provider')
    key=f'ops/_native/operators/qsli/variants/{WHEEL_KEY}/provider.so'
    with ZipFile(wheel) as z:
        expected=z.read(key)
        require(sha256(expected).hexdigest()==json.loads(z.read(key+'.meta.json'))['binary_sha256']
                and device in expected,'Not the wheel device object')
    roster=CORE.findall(log);require(len(roster)==len(dict(roster))==96,'Expected all 32+64 workers in the physical launch')
    cores,hashes=core_windows(prof/'simulator',{k:float(v) for k,v in roster})
    origin=min(v['first_tick'] for v in cores.values())
    end=max(v['retire_tick'] for v in cores.values())-origin
    mm=[]; rv={}; active=set(); hist=[]; writes=[]
    for shard in sorted((prof/'simulator').glob('core*.*core*/trace.json')):
        core=shard.parent.name
        for e in trace_events(shard):
            if e.get('ph')!='X' or e['dur']<=0:continue
            begin=round(e['ts']*CLOCK)-origin
            done=round((e['ts']+e['dur'])*CLOCK)-origin
            name=e['name']
            if name=='MMAD_MX':
                attrs=json.loads(e['args']['detail'])
                require([attrs[k] for k in ('m','n','k','matrix_a_dtype','matrix_b_dtype')]==[32,512,128,'fp4_e2m1','fp4_e2m1'],'Wrong dot')
                mm.append((begin,done,core))
            if core.endswith(('veccore0','veccore1')):
                if name.startswith('RV_'):
                    rv[core]=max(rv.get(core,0),done)
                if name=='RV_VMAXS':active.add(core)
                if name=='RV_CHISTv2':hist.append((begin,done,core))
                if e['pid']==70:writes.append((begin,done,core,name))
    require(len(mm)==32 and {e[2] for e in mm}=={'core0.cubecore0'}
            and active=={'core0.veccore0','core0.veccore1'} and hist, 'Unverified useful team')
    mm.sort();hist.sort();gaps=[b[0]-a[0] for a,b in zip(mm,mm[1:])]
    last_mm=mm[-1][1]
    tail_hist=[(s,e,core) for s,e,core in hist if s>=last_mm]
    require(tail_hist, 'No terminal local histogram identified')
    tag=log.split('Start profiling on kernel: ')[1].split()[0]
    require('QuantSparseLightningIndexer' in tag,'Wrong kernel symbol')
    def intervals_summary(g):
        return dict(n=len(g),min_cycles=min(g),median_cycles=median(g),mean_cycles=sum(g)/len(g),max_cycles=max(g))
    return dict(sku='Ascend950DT_9582',role='forced-metadata mechanism diagnostic; no independent accuracy claim',
        shape=freeze['shape'],table_pages=8201,cmp_ratio=1,metadata=fixture['metadata_rows'],fixture=fixture,
        native_kernel=tag,installed_aot_key=WHEEL_KEY,provider_sha256=sha256(expected).hexdigest(),device_sha256=sha256(device).hexdigest(),
        log_sha256=sha(raw/'run.log'),freeze_sha256=sha(HERE/'qsli-one-team-freeze.json'),
        launched_aic=32,launched_aiv=64,useful_score_aic=1,useful_score_aiv=2,split_or_ld=False,
        global_window_us=us(end),global_window_cycles=end,slowest_retiring_core=max(cores,key=lambda c:cores[c]['retire_tick']),
        first_mmad_issue_to_last_device_retire_us=us(end-mm[0][0]),first_mmad_issue_to_last_mmad_retire_us=us(last_mm-mm[0][0]),
        frequency_ghz=1.65,origin='earliest first instruction across all physical core traces',mmad_issue_us=[us(s) for s,e,c in mm],
        mmads_retire_us=[us(e) for s,e,c in mm],mmad_issue_cycles=[s for s,e,c in mm],mmad_retire_cycles=[e for s,e,c in mm],
        issue_gaps_cycles=gaps,issue_interval_segments={'first_8':intervals_summary(gaps[:8]),
            'middle_7':intervals_summary(gaps[8:15]),'last_16':intervals_summary(gaps[15:])},
        median_issue_interval_cycles=median(gaps),median_issue_interval_us=us(median(gaps)),
        mean_issue_interval_cycles=sum(gaps)/len(gaps),median_mmad_duration_cycles=median(e-s for s,e,c in mm),
        predicted_relative_window_error_pct=100*(freeze['model_us']/us(end)-1),
        local_histogram_instruction_envelope_us=us(max(e for s,e,c in hist)-hist[0][0]),
        terminal_tail_after_last_mmad_us=us(end-last_mm),
        last_register_vector_instruction_retire_us={k:us(rv[k]) for k in sorted(active)},
        local_histogram=dict(instruction='RV_CHISTv2',total_count=len(hist),count_starting_after_last_mmad=len(tail_hist),
              first_us=us(hist[0][0]),last_retire_us=us(max(e for s,e,c in hist)),
              first_after_last_mmad_us=us(tail_hist[0][0]),from_first_terminal_hist_to_end_us=us(end-tail_hist[0][0]),
              note='Landmark interval includes local selection, index remapping, writes and retirement. Not isolated TopK latency.'),
        last_vec_write=dict(zip(('issue_cycle','retire_cycle','core','instruction'),max(writes,key=lambda v:v[1]))),
        physical_roster_trace_sha256=sha256(json.dumps(hashes,sort_keys=True).encode()).hexdigest(),
        ca_wall_s=(datetime.fromisoformat((raw/'finish.txt').read_text().strip())-datetime.fromisoformat((raw/'start.txt').read_text().strip())).total_seconds(),
        prediction_us=freeze['model_us'],frozen_predicted_ii_cycles=freeze['plan']['interval_cycles'])


def chunk_scenario(r,scratch,cache):
    from calibration.candidate_indexer.context_sensitivity import bundle, caps, case_id
    from sim_platform.backend.l1 import simulate_graph
    import csv,gzip,pickle
    spec=('v41-flash',1048576,1,32,'prefill',1044480,4096)
    workload,o,_,_=bundle(scratch,spec)
    graph,=workload.graphs
    with gzip.open(cache/(case_id(spec)+'.gz'),'rb') as f:stored,=pickle.load(f)
    existing=[v for v in csv.DictReader((HERE/'context-sensitivity-scalar.csv').open())
              if v['scalar_lookup_cycles']=='64']
    by_id={v['node']:v for v in existing}
    require(len(by_id)==4,'Missing four neutral candidate prices')
    costs={};base={};changed=[]
    # Change the 31 start-to-start intervals only, retaining the row head/tail
    # of each of 128 complete rows per team; this ignores contention changes.
    step=r['median_issue_interval_cycles']-r['frozen_predicted_ii_cycles']
    for node in graph.nodes:
        cost=stored[node.ref.node_id][0]
        if node.ref.node_id in by_id:
            old=float(by_id[node.ref.node_id]['packed_candidate_us'])
            cost=replace(cost,duration_us=old)
            new=old+128*31*step/CLOCK
            require(new>0,'Invalid substituted duration')
            costs[node.ref]=replace(cost,duration_us=new)
            changed.append(new)
        else:costs[node.ref]=cost
        base[node.ref]=cost
    baseline=simulate_graph(graph,base,caps(o)).duration_us
    require(abs(baseline-float(existing[0]['packed_stage_us']))<1e-6,'Wrong baseline')
    return dict(case=case_id(spec),rows_per_team=128,adjusted_intervals_per_row=31,
          frozen_packed_stage_us=baseline,with_single_team_ii_stage_us=simulate_graph(graph,costs,caps(o)).duration_us,
          neutral_node_us=float(existing[0]['packed_candidate_us']),with_single_team_ii_node_us=changed[0],
          caveat='Frozen single-team median inserted only in adjacent tile-to-tile slots. No full-window calibration; concurrent teams, memory reuse, row transitions and selection tails in the 4096-row block remain unmeasured.')


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('raw','wheel','scratch','cache','output'):p.add_argument(name,type=Path)
    p.add_argument('--failed-attempt',type=Path,help='record wall time of excluded IPC failure, without using its timing')
    a=p.parse_args();r=receipt(a.raw,a.wheel)
    if a.failed_attempt:
        failed=a.failed_attempt; log=(failed/'run.log').read_text()
        require('connect server failed' in log and 'Profiling results are empty' in log,'Unexpected failed attempt')
        r['excluded_attempt']=dict(reason='Sandbox denied profiler Unix socket; no physical trace emitted',
              passed_golden='FUSED_GOLDEN_PASS 512' in log, raw_log_sha256=sha(failed/'run.log'),
              wall_seconds=(datetime.fromisoformat((failed/'finish.txt').read_text().strip())-
                            datetime.fromisoformat((failed/'start.txt').read_text().strip())).total_seconds())
    r['conditional_last_chunk']=chunk_scenario(r,a.scratch,a.cache)
    a.output.write_text(json.dumps(r,indent=2,allow_nan=False)+'\n')
