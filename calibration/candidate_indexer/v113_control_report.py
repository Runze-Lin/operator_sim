"""Audit full-wave timing, ordered folded source addresses and actual time coincidence."""
import argparse
from collections import Counter, defaultdict
from datetime import datetime
from hashlib import sha256
import json
from pathlib import Path
import re
from statistics import median
from struct import pack
import zipfile

from calibration.__main__ import require
from calibration.candidate_indexer.fused_qsli_report import WHEEL_KEY
from calibration.candidate_indexer.v112_merge_trace import DEVICE
from calibration.candidate_indexer.v112_wave_report import FREQ, intervals_peak, scan
from calibration.msopprof import CORE, sha
from calibration.mx_matmul.collect import MEMORY_CONFIG, trace_events

SOURCE = Path(__file__).parent
SKU = 'Ascend950DT_9582'
BLOCK = 544
FROZEN = 'cad6aecd937a30fcda1ad6339fd1f34a09fd7abcd1e1ea7c11c6f08a216dc2b5'


def folded_instructions(shard):
    """Return each MTE2 source block in issue order, including its descriptor window."""
    descriptors = []
    for seq,event in enumerate(trace_events(shard)):
        if event.get('ph')!='X' or event.get('name')!='MOV_SRC_TO_DST_ALIGNv2' or event['dur']<=0:
            continue
        args = json.loads(event['args']['detail'])
        if (args['src_mem'],args['dst_mem'],args['burst_len']) != ('OUT','UB',BLOCK):
            continue
        n = args['burst_number']
        require(n in (1,2) and args['processed_bytes']==BLOCK*n,'Unexpected folded descriptor')
        start = round(event['ts']*FREQ)
        end = round((event['ts']+event['dur'])*FREQ)
        addresses = [args['src_addr']+j*args['src_stride'] for j in range(n)]
        descriptors.append((start,seq,end,addresses))
    return sorted(descriptors)


def coincidence(issues,origin,width):
    """Cross-team operand-block issue coincidence, with bins anchored at first device event."""
    by_bin = defaultdict(Counter)
    by_bin_addr = defaultdict(Counter)
    for t,team,addr in issues:
        key = (t-origin)//width
        by_bin[key][team] += 1
        by_bin_addr[key,addr][team] += 1
    def cross_pairs(counts):
        return (sum(counts.values())**2 - sum(n*n for n in counts.values()))//2
    denom = sum(cross_pairs(teams) for teams in by_bin.values())
    numer = sum(cross_pairs(teams) for teams in by_bin_addr.values())
    has_peer = sum(sum(teams.values()) for teams in by_bin_addr.values() if len(teams)>1)
    return dict(bin_cycles=width,cross_team_pairs_in_same_bin=denom,same_address_cross_team_pairs=numer,
        same_address_pair_fraction=numer/denom if denom else None,
        request_fraction_with_other_team_same_address_in_bin=has_peer/len(issues))


def summarize(raw,wheel):
    cost = json.loads((raw/'cost.json').read_text())
    arm = cost['case']
    log_path = raw/'run.log'
    log = log_path.read_text()
    require(cost['sku']==SKU and cost['rc']==0 and not cost['watchdog'] and
            'use simulator Ascend950DT 9582 by --soc-version' in log and MEMORY_CONFIG[SKU] in log and
            'All task success.' in log and 'Profiling data parse finished.' in log and 'Traceback' not in log,
            'Incomplete capture, invalid SKU or missing vendor export')
    facts_path = SOURCE/f'v113_control_{arm}_fixture.json'
    facts = json.loads(facts_path.read_text())
    freeze_path = SOURCE/'v113_frozen.json'
    require(sha(freeze_path)==FROZEN, 'Frozen predictions have changed')
    freeze = json.loads(freeze_path.read_text())
    expected_files = freeze['input_sha256']
    require(datetime.fromisoformat(cost['start_utc'])>datetime.fromisoformat(freeze['frozen_utc']) and
            cost['probe_sha256']==expected_files['v113_control_probe.py'] and
            cost['cpu_control_sha256']==expected_files[facts_path.name], 'Capture does not match pre-CA freeze')
    frozen_point, = (row for row in freeze['prices'] if row['case']=='control_'+arm and row['sku']==SKU)
    reported = re.search(r'^V113_FIXTURE (\{.*\})$',log,re.M)
    require(reported and facts==json.loads(reported[1]) and sha(facts_path)==cost['cpu_control_sha256'], 'Fixture mismatch')
    golden = re.search(r'^V113_CONTROL_GOLDEN_PASS (\d+) ([a-f0-9]{64})$',log,re.M)
    require(golden and int(golden[1])==32768 and golden[2]==facts['input_sha256']['golden'], 'Incorrect top-k')
    prof, = (raw/'profile').glob('OPPROF_*')
    kernel, = re.findall(r'Start profiling on kernel: (\S+)',log)
    hit=json.loads((raw/'provider.json').read_text())
    require('kernel_QuantSparseLightningIndexer' in kernel and hit['counts']=={'installed_hit':1} and
            hit['events'][0]['key']==WHEEL_KEY, 'Unexpected kernel/provider')
    elf = (prof/'dump/aicore_binary.o').read_bytes()
    with zipfile.ZipFile(wheel) as archive:
        entry=f'ops/_native/operators/qsli/variants/{WHEEL_KEY}/provider.so'
        provider=archive.read(entry)
        provider_sha=json.loads(archive.read(entry+'.meta.json'))['binary_sha256']
    require(sha256(provider).hexdigest()==provider_sha and sha256(elf).hexdigest()==DEVICE and provider.find(elf)>=0,
            'Device ELF does not match released provider')
    roster = CORE.findall(log)
    shards = {path.parent.name:path for path in (prof/'simulator').glob('core*.*core*/trace.json')}
    require(len(roster)==96 and len(set(n for n,_ in roster))==96 and set(dict(roster))==set(shards), 'Missing physical cores')
    physical = {n:scan(shards[n],float(t)) for n,t in roster}
    start,end = min(r['first'] for r in physical.values()),max(r['last'] for r in physical.values())
    mmads = [r['mmads'] for n,r in physical.items() if '.cubecore' in n]
    require(len(mmads)==32 and all(len(r)==64 for r in mmads),'MMAD volume differs from frozen metadata')
    gathers = {n:folded_instructions(path) for n,path in shards.items() if '.veccore' in n}
    base = min(a for events in gathers.values() for _,_,_,addrs in events for a in addrs)-facts['min_physical_block']*BLOCK
    ids = {}
    issues = []
    for n, events in gathers.items():
        seq = []
        team = int(re.match(r'core(\d+)\.',n)[1])
        for tick,_,_,addresses in events:
            for a in addresses:
                block, rem = divmod(a-base,BLOCK)
                require(rem==0 and 0<=block<=facts['max_physical_block'],'Bad folded address')
                seq.append(block)
                issues.append((tick,team,block))
        require(len(seq)==2048,'Missing per-lane operands')
        ids[n] = sha256(b''.join(pack('<I',b) for b in seq)).hexdigest()
    expected = json.loads((SOURCE/'v113_control_order.json').read_text())[arm]
    require(expected['candidate_sha256']==facts['input_sha256']['ids'],'Source rank reference mismatch')
    actual = Counter((ids[f'core{i}.veccore0'],ids[f'core{i}.veccore1']) for i in range(32))
    matched = actual==Counter(tuple(pair) for pair in expected['ordered_lane_sha256_by_logical_team'])
    require(matched and len(set(b for _,_,b in issues))*BLOCK==facts['folded_distinct_bytes'], 'Actual folded source order mismatch')
    gaps = sorted(b[0]-a[0] for stream in mmads for a,b in zip(stream,stream[1:]))
    handoff = [stream[32][0]-stream[31][0] for stream in mmads]
    intervals = [(a,end_) for events in gathers.values() for a,_,end_,_ in events]
    bursts = Counter(len(addresses) for events in gathers.values() for _,_,_,addresses in events)
    first_mac,last_mac=min(row[0][0] for row in mmads),max(row[-1][1] for row in mmads)
    device_us=(end-start)/FREQ
    estimates = {kind:{'device_us':frozen_point[kind]['ca_window_us'],
                       'signed_error_pct':100*(frozen_point[kind]['ca_window_us']/device_us-1)}
                 for kind in ('shape_only','known_layout')}
    return dict(arm=arm,sku=SKU,shape=facts['shape'],status='qualified full device window',
        freeze_sha256=FROZEN,frozen_predictions=estimates,
        first_tick=start,last_tick=end,elapsed_cycles=end-start,device_us=(end-start)/FREQ,frequency_cycles_per_us=FREQ,
        golden_positions=int(golden[1]),golden_sha256=golden[2],aic=32,aiv=64,qualified_end_events=96,
        native_aot_key=WHEEL_KEY,mmad_count=sum(map(len,mmads)),mmad_geometry=[32,512,128],mmad_types=['E2M1','E2M1','FP32'],
        physical_team_ordered_lane_sha256={n:ids[n] for n in sorted(ids)},all_ordered_lane_streams_match=matched,
        operand_gather_bytes=len(issues)*BLOCK,distinct_operand_bytes=len(set(b for _,_,b in issues))*BLOCK,
        logical_all_row_pair_same_rank=facts['all_unordered_row_pairs_same_rank_fraction'],
        profile_time_coincidence=[coincidence(issues,start,width) for width in (128,512,2048)],
        gather_descriptors_by_burst_count=dict(bursts),source_expected_descriptors_by_burst_count=expected['source_expected_descriptors_by_burst_count'],
        burst_counts_match_copy_pa_pair=(dict(bursts)=={int(k):v for k,v in expected['source_expected_descriptors_by_burst_count'].items()}),
        descriptors_per_operand_record=sum(bursts.values())/len(issues),first_gather_us=(min(a for a,b in intervals)-start)/FREQ,
        last_gather_issue_us=(max(a for a,b in intervals)-start)/FREQ,
        folded_records_per_chip_cycle_between_first_and_last_issue=len(issues)/(max(a for a,b in intervals)-min(a for a,b in intervals)),
        final_gather_us=(max(b for a,b in intervals)-start)/FREQ,peak_gather_aiv=intervals_peak([v['reads'] for v in physical.values()]),
        peak_mmad_aic=intervals_peak(mmads),first_mmad_us=(first_mac-start)/FREQ,last_mmad_retire_us=(last_mac-start)/FREQ,
        last_mmad_to_end_us=(end-last_mac)/FREQ,typical_issue_gaps_cycles_p10_p50_p90=[gaps[len(gaps)//10],median(gaps),gaps[len(gaps)*9//10]],
        median_interrow_issue_gap_cycles=median(handoff),
        first_row_median_mmad_window_us=median((stream[31][1]-stream[0][0])/FREQ for stream in mmads),
        second_row_median_mmad_window_us=median((stream[63][1]-stream[32][0])/FREQ for stream in mmads),
        last_finishing_physical=max(physical,key=lambda n:physical[n]['last']),device_sha256=DEVICE,provider_sha256=provider_sha,
        simulator_memory_model=MEMORY_CONFIG[SKU],simulator_exit_code=cost['rc'],wall_s=cost['wall_s'],golden_observed_s=cost['golden_observed_s'],
        post_golden_export_s=cost['wall_s']-cost['golden_observed_s'],host_annotation_warning='HOST_RANGE_POP_ERROR_AFTER_ENQUEUE' in log,
        probe_sha256=cost['probe_sha256'],cpu_control_sha256=sha(facts_path),log_sha256=sha(log_path),
        trace_set_sha256=sha256(json.dumps({n:r['sha256'] for n,r in physical.items()},sort_keys=True).encode()).hexdigest())


if __name__ == '__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('raw',type=Path)
    parser.add_argument('--wheel',type=Path,required=True)
    parser.add_argument('--out',type=Path,required=True)
    args=parser.parse_args()
    args.out.write_text(json.dumps(summarize(args.raw,args.wheel),indent=2)+'\n')
