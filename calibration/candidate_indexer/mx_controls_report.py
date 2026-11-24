"""Verify all physical CA shards for the frozen three compact MX controls per SKU."""
import argparse
from collections import Counter
from datetime import datetime
import hashlib
import json
from pathlib import Path
import re
from statistics import median

from calibration.__main__ import require
from calibration.msopprof import CORE, sha
from calibration.mx_matmul.collect import MEMORY_CONFIG, core_windows, trace_events
from calibration.candidate_indexer.receipt import KERNEL

ROUTE = ['GatherV2', 'GatherV2', 'GatherV2', 'QuantLightningIndexerV2', 'GatherV2', 'Sort']


def instructions(path, origin):
    mmads, vectors = [], Counter()
    for shard in sorted(path.glob('core*.*core*/trace.json')):
        for event in trace_events(shard):
            if event.get('ph') != 'X':
                continue
            name = event['name']
            if name == 'MMAD_MX' and event['dur'] > 0:
                f = json.loads(event['args']['detail'])
                mmads.append((round(event['ts'] * 1650), round(event['dur'] * 1650), shard.parent.name, f['m'], f['n'], f['k']))
            if '.veccore' in shard.parent.name and name.startswith('RV_'):
                vectors[name] += 1
    require(mmads and vectors['RV_VMULA'], 'Missing weighted reduction or MX score instructions')
    mmads.sort()
    require(len({x[2] for x in mmads}) == 1, 'Expected the explicitly requested one score shard')
    gaps = [b[0]-a[0] for a,b in zip(mmads,mmads[1:])]
    return dict(mmad_mnk_counts=[[*mnk,n] for mnk,n in sorted(Counter(x[3:] for x in mmads).items())],
                issuing_aic=mmads[0][2], first_issue_cycles=mmads[0][0]-origin,
                last_retire_cycles=mmads[-1][0]+mmads[-1][1]-origin,
                issue_intervals_cycles=gaps, median_issue_interval_cycles=median(gaps),
                vector_instructions=dict(sorted((n,c) for n,c in vectors.items() if n in
                    ('RV_VMULA','RV_VMAX','RV_VSELR','RV_VSORT32','RV_VADD','RV_VMRGSORT4','RV_VLD'))))


def verify(base, sku, opp, freeze):
    log = base / (sku+'.log')
    text = log.read_text()
    require('All task success.' in text and 'SMOKE_PASS' in text and 'SMOKE_FAIL' not in text
            and 'runtime_soc='+sku in text and MEMORY_CONFIG[sku] in text, 'Unqualified simulator or probe')
    shapes = [(int(q),int(l),int(c)) for q,l,c in re.findall(r'^mx_case Q=(\d+) L=(\d+) C=(\d+) H=32 D=128 K=512 S=8$',text,re.M)]
    cases = freeze['scenarios']
    require(shapes == [(s['Q'],s['L'],s['C']) for s in cases], 'Mismatch in frozen shapes')
    good = [int(v) for v in re.findall(r'^mx_correctness ordered_global_positions=(\d+) bad=0$',text,re.M)]
    require(good == [q*512 for q,l,c in shapes], 'Golden mismatch')
    names = KERNEL.findall(text)
    require([n.split('_')[0] for n in names] == ROUTE * len(cases), 'Unexpected dispatch order')
    tables = re.split(r'^core_name\s+duration_time\(us\)\s+running_time\(us\)[ \t]*$',text,flags=re.M)[1:]
    profile, = (base/sku/'profile').glob('OPPROF_*')
    require(len(list(profile.glob('*/*/dump/aicore_binary.o')))==len(names), 'Unexpected ELF count')
    seen, rows = Counter(), []
    for i,(name,table) in enumerate(zip(names,tables,strict=True)):
        scenario = cases[i//len(ROUTE)]
        if i%len(ROUTE) == 0:
            rows.append(dict(Q=scenario['Q'], L=scenario['L'], C=scenario['C'], correct_ordered_ids=good[i//6], stages=[]))
        folder = profile/name/str(seen[name]); seen[name]+=1
        checksum=sha(folder/'dump/aicore_binary.o')
        file=re.sub(r'_\d+(?:_mix_ai[cv])?$', '', name)+'.o'
        (obj,)=[obj for obj in opp.glob('*/*/'+file) if sha(obj)==checksum]
        roster = CORE.findall(table)
        require(roster and len(roster)==len(dict(roster)), 'Incomplete core table')
        cores, hashes = core_windows(folder/'simulator', {core:float(us) for core,us in roster})
        origin=min(r['first_tick'] for r in cores.values())
        cycles=max(r['retire_tick'] for r in cores.values())-origin
        stage=dict(api='aclnn'+ROUTE[i%6], object=str(obj.relative_to(opp)), sha256=checksum,
                   trace_set_digest=hashlib.sha256(json.dumps(hashes,sort_keys=True).encode()).hexdigest(),
                   launched_aic=sum('.cubecore' in c for c in cores), launched_aiv=sum('.veccore' in c for c in cores),
                   cycles=cycles, device_us=round(cycles/1650,6))
        if i%6 == 3:
            stage['score_instructions']=instructions(folder/'simulator', origin)
            pred=scenario['predictions'][sku]['ordinary_qli_proxy_us']
            rows[-1].update(qli_predicted_us=pred, qli_ca_us=stage['device_us'], qli_relative_error_pct=round(100*(pred/stage['device_us']-1),3))
        rows[-1]['stages'].append(stage)
    for row in rows:
        row['serial_stage_sum_us']=round(sum(s['cycles'] for s in row['stages'])/1650,6)
    stamps=re.findall(r'\[(\d{4}-\d\d-\d\d \d\d:\d\d:\d\d)\]',text)
    return dict(sku=sku, log_sha256=sha(log), memory_config=MEMORY_CONFIG[sku], log_timestamp_span_s=(datetime.fromisoformat(stamps[-1])-datetime.fromisoformat(stamps[0])).total_seconds(), cases=rows)

if __name__ == '__main__':
    args=argparse.ArgumentParser(description=__doc__)
    args.add_argument('base',type=Path);args.add_argument('--opp',type=Path,required=True);args.add_argument('--out',type=Path,required=True)
    args=args.parse_args(); freeze=json.loads(Path(__file__).with_name('mx-controls-freeze.json').read_text()); args.out.write_text(json.dumps([verify(args.base,sku,args.opp,freeze) for sku in MEMORY_CONFIG],indent=2)+'\n')
