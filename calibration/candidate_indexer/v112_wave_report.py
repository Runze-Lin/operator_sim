"""Verify frozen paired-QSLI wave measurements and audit source-page accesses."""
import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
import re
from statistics import median
import struct
import zipfile

from calibration.__main__ import require
from calibration.candidate_indexer.fused_qsli_report import WHEEL_KEY
from calibration.candidate_indexer.v112_merge_trace import DEVICE, MERGE_PART_READ, MERGE_CARRY_READ, MERGE_OUT
from calibration.msopprof import CORE, sha
from calibration.mx_matmul.collect import MEMORY_CONFIG, trace_events

SOURCE = Path(__file__).parent
FREQ = 1650
SCENARIO = {'pr':'shared_wave','dt':'varied_two_waves'}


def intervals_peak(intervals):
    """Peak simultaneous physical workers, merging overlapping activity within each worker."""
    events=Counter()
    for worker_intervals in intervals:
        merged=[]
        for start,end in sorted(worker_intervals):
            if not merged or start>merged[-1][1]:
                merged.append([start,end])
            else:
                merged[-1][1]=max(merged[-1][1],end)
        for start,end in merged:
            events[start]+=1
            events[end]-=1
    level=peak=0
    for tick in sorted(events):
        level+=events[tick]
        peak=max(peak,level)
    return peak


def fold_digest(source):
    packed=bytearray()
    for block,count in sorted(source.items()):
        packed.extend(struct.pack('<II',block,count))
    return hashlib.sha256(packed).hexdigest()


def scan(path, expected_us):
    first=float('inf')
    last=0
    retire=0
    mmads=[]
    reads=[]
    blocks=Counter()
    transfer=Counter()
    merge=defaultdict(list)
    for event in trace_events(path):
        if event['ph']!='X':
            continue
        start,duration=event['ts'],event['dur']
        require(start>=0 and duration>=0,'Bad instruction timestamp')
        first=min(first,start)
        last=max(last,start+duration)
        name=event['name']
        if name=='END':
            retire+=1
        if name not in ('MMAD_MX','MOV_SRC_TO_DST_ALIGNv2') or duration<=0:
            continue
        arg=json.loads(event['args']['detail'])
        time=(round(start*FREQ),round((start+duration)*FREQ))
        if name=='MMAD_MX':
            require(tuple(arg[n] for n in ('m','n','k'))==(32,512,128) and
                    tuple(arg[n] for n in ('matrix_a_dtype','matrix_b_dtype','matrix_c_dtype'))==('fp4_e2m1','fp4_e2m1','f32'), 'Unexpected score geometry/precision')
            mmads.append(time)
        else:
            key=(arg['src_mem'],arg['dst_mem'])
            transfer[' → '.join(key)]+=arg['processed_bytes']
            pc=int(event['args']['pc_addr'],16)%(1<<20)
            if pc in (*MERGE_PART_READ,*MERGE_CARRY_READ,*MERGE_OUT):
                merge[pc].append(time)
            if key==('OUT','UB') and arg['burst_len']==544:
                count=arg['burst_number']
                require(arg['processed_bytes']==544*count and count in (1,2),'Unexpected folded source descriptor')
                # The reported stride is the byte distance between burst starts.
                for step in range(count):
                    blocks[arg['src_addr']+step*arg['src_stride']]+=1
                reads.append(time)
    require(retire==1 and abs(last-first-expected_us)<.011,'END/physical duration mismatch: '+str(path))
    return dict(first=round(first*FREQ),last=round(last*FREQ),mmads=sorted(mmads),reads=reads,blocks=blocks,
                transfer=transfer,merge=merge,sha256=sha(path))


def price(filename,case,sku):
    rows=json.loads((SOURCE/filename).read_text())['prices']
    (point,)=[r for r in rows if r['case']==SCENARIO[case] and r['sku']==sku]
    return point


def summarize(raw, wheel):
    cost=json.loads((raw/'cost.json').read_text())
    case=cost['case']
    sku=cost['sku']
    frozen=price('v112_frozen.json',case,sku)
    old=price('v112_baseline.json',case,sku)
    log=raw/'run.log'
    text=log.read_text()
    expected_simulator='use simulator '+sku.replace('_',' ')+' by --soc-version'
    require(expected_simulator in text, 'Unexpected simulator launch configuration')
    complete_adopted=(cost['rc'] is None and cost.get('profiler_final_log_seen') is True and
                      'Profiling data parse finished.' in text and 'V112_WAVE_GOLDEN_PASS' in text)
    require((cost['rc']==0 or complete_adopted) and not cost['watchdog'] and MEMORY_CONFIG[sku] in text and
            'All task success.' in text and 'Traceback' not in text,'Incomplete CA or unexpected SKU')
    facts=json.loads(re.search(r'^V112_FIXTURE (\{.*\})$',text,re.M)[1])
    cpu_path=SOURCE/f'v112_wave_{case}_fixture.json'
    cpu_facts=json.loads(cpu_path.read_text())
    require(cpu_facts==facts,'Device run did not use the CPU-verified fixture')
    rows=facts['rows']
    golden=re.search(r'^V112_WAVE_GOLDEN_PASS (\d+) ([0-9a-f]{64})$',text,re.M)
    require(golden and int(golden[1])==rows*512 and golden[2]==facts['input_sha256']['golden'], 'Missing golden')
    prof,=(raw/'profile').glob('OPPROF_*')
    kernel,=re.findall(r'Start profiling on kernel: (\S+)',text)
    require('kernel_QuantSparseLightningIndexer' in kernel, 'Unexpected launch')
    hit=json.loads((raw/'provider.json').read_text())
    require(hit['counts']=={'installed_hit':1} and hit['events'][0]['key']==WHEEL_KEY,'AOT key mismatch')
    elf=(prof/'dump/aicore_binary.o').read_bytes()
    binary_hash=hashlib.sha256(elf).hexdigest()
    require(binary_hash==DEVICE,'Unsupported code for source-PC audit')
    with zipfile.ZipFile(wheel) as archive:
        entry=f'ops/_native/operators/qsli/variants/{WHEEL_KEY}/provider.so'
        provider=archive.read(entry)
        expected_hash=json.loads(archive.read(entry+'.meta.json'))['binary_sha256']
    require(hashlib.sha256(provider).hexdigest()==expected_hash and provider.find(elf)>=0,'Provider/device ELF differs')
    roster=CORE.findall(text)
    shards={p.parent.name:p for p in (prof/'simulator').glob('core*.*core*/trace.json')}
    require(shards and len(roster)==len(dict(roster)) and set(shards)==set(dict(roster)), 'Incomplete core roster')
    physical={name:scan(shards[name],float(us)) for name,us in roster}
    origin=min(r['first'] for r in physical.values())
    retire=max(r['last'] for r in physical.values())
    measured=(retire-origin)/FREQ
    workers={name:r for name,r in physical.items() if r['mmads']}
    mmads=sum(len(r['mmads']) for r in workers.values())
    require(mmads==rows*32 and len(workers)==facts['metadata_workers_arg'],'Incorrect MX schedule')
    prefills=[r['mmads'][0][0]-origin for r in workers.values()]
    intervals=[right[0]-left[0] for r in workers.values() for left,right in zip(r['mmads'],r['mmads'][1:])]
    last_mac=max(r['mmads'][-1][1] for r in workers.values())-origin
    transfers=Counter()
    address_sets=defaultdict(Counter)
    phases=[t for r in physical.values() for t in r['merge'].get(MERGE_PART_READ[0],[]) ]
    merge_ends=[t for r in physical.values() for t in r['merge'].get(MERGE_OUT[-1],[]) ]
    for name,r in physical.items():
        address_sets[name.split('.')[0]].update(r['blocks'])
        transfers.update(r['transfer'])
    all_source=Counter()
    for address_set in address_sets.values():
        all_source.update(address_set)
    addr_ref=json.loads((SOURCE/'v112_wave_addresses.json').read_text())[case]
    base=min(all_source)-addr_ref['min_block']*544
    source_total=sum(all_source.values())*544
    exact_address_union=(len(all_source)==addr_ref['distinct_source_blocks'] and
                         max(all_source)-base==addr_ref['max_block']*544)
    matching=defaultdict(list)
    for logical in addr_ref['expected']:
        matching[logical['folded_physical_block_multiset_sha256']].append(logical['logical_team'])
    hits=Counter()
    teams=[]
    for name,r in sorted(workers.items()):
        core=name.split('.')[0]
        histogram=address_sets[core]
        aligned=all((addr-base)%544==0 for addr in histogram)
        normalized={(addr-base)//544:n for addr,n in histogram.items()}
        digest=fold_digest(normalized) if aligned and min(normalized,default=0)>=0 else None
        hits[digest]+=1
        teams.append(dict(physical=name,tiles=len(r['mmads']),first_mmad_cycles=r['mmads'][0][0]-origin,
                          last_mmad_cycles=r['mmads'][-1][1]-origin, candidate_reads=sum(histogram.values()),
                          source_address_multiset_sha256=digest,matching_logical_teams=matching.get(digest,[])))
    matched=(hits==Counter({d:len(ids) for d,ids in matching.items()}))
    predicted=frozen['latency_us']
    previous=old['latency_us']
    return dict(sku=sku, shape=frozen['shape'], attrs=frozen['attrs'],role='independent holdout for v1.12 and v1.11',
        status='qualified device window', device_us=measured, elapsed_cycles=retire-origin, first_tick=origin,last_retire_tick=retire,
        v112_us=predicted,v111_us=previous,v112_signed_error_pct=100*(predicted/measured-1),v111_signed_error_pct=100*(previous/measured-1),
        golden_positions=int(golden[1]),golden_sha256=golden[2], memory_model=MEMORY_CONFIG[sku],
        fixture_file=cpu_path.name,fixture_sha256=sha(cpu_path),metadata_sha256=facts['metadata_sha256'],
        metadata_workers_arg=facts['metadata_workers_arg'],logical_rows=rows,physical_aic_roster=sum('.cubecore' in n for n in shards),
        physical_aiv_roster=sum('.veccore' in n for n in shards),mmad_issuing_aic=len(workers),mmad_count=mmads,
        metadata_fd_records=facts['fd_records'], fd_partial_pair_read_counts=len(phases),fd_output_pair_count=len(merge_ends),
        fd_first_partial_read_cycles=min((a for a,b in phases),default=origin)-origin if phases else None,
        fd_last_output_retire_cycles=max((b for a,b in merge_ends),default=origin)-origin if merge_ends else None,
        active_mmad_peak_cores=intervals_peak([r['mmads'] for r in workers.values()]),
        peak_concurrent_gather_aivs=intervals_peak([r['reads'] for r in physical.values() if r['reads']]),
        first_mmad_cycles=min(prefills),last_mmad_cycles=last_mac,median_same_core_issue_gap_cycles=median(intervals),
        last_finishing_core=max(physical,key=lambda n:physical[n]['last']),
        folded_candidate_source=dict(descriptor='OUT → UB MOV_SRC_TO_DST_ALIGNv2, burst_len=544',
            requested_bytes=facts['candidate_gather_demand_bytes'],observed_bytes=source_total,
            expected_unique_bytes=facts['distinct_folded_source_bytes'],observed_unique_bytes=len(all_source)*544,
            unique_address_extent_matches=exact_address_union, every_logical_team_multiset_matches=matched,
            note='GM operand bytes from physical instruction descriptors. HBM miss traffic is not directly observed.'),
        all_mov_src_to_dst_alignv2_bytes=dict(transfers),physical_teams=teams,
        physical_shard_count=len(shards),end_count=len(shards),
        trace_set_sha256=hashlib.sha256(json.dumps({n:r['sha256'] for n,r in physical.items()},sort_keys=True).encode()).hexdigest(),
        kernel=kernel,native_aot_key=WHEEL_KEY,provider_sha256=expected_hash,device_sha256=binary_hash,
        device_elf_inside_provider_at=provider.find(elf),wall_s=cost['wall_s'],golden_observed_s=cost['golden_observed_s'],
        trace_export_after_golden_s=(cost['wall_s']-cost['golden_observed_s']) if cost['golden_observed_s'] is not None else None,
        simulator_exit_code=cost['rc'],cost_record_source=cost.get('source','Original process watcher'),
        probe_sha256=cost['probe_sha256'],log_sha256=sha(log),host_annotation_warning='HOST_RANGE_POP_ERROR_AFTER_ENQUEUE' in text)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('raw',type=Path)
    p.add_argument('--wheel',type=Path,required=True)
    p.add_argument('--out',type=Path,required=True)
    args=p.parse_args()
    args.out.write_text(json.dumps(summarize(args.raw,args.wheel),indent=2)+'\n')
