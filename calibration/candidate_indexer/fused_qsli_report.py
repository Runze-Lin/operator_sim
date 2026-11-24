"""Qualify installed QSLI AOT CA windows and the paired lookup-width experiment."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import re
import zipfile

from calibration.__main__ import require
from calibration.msopprof import CORE, sha
from calibration.mx_matmul.collect import MEMORY_CONFIG, core_windows, trace_events

SOURCE = Path(__file__).parent
WHEEL_KEY = '400228d98408fad5510dc0ff2580bb878c30b3f8dd4457e2ee349b2300d4e3de'


def evidence(simulator, origin):
    calls, counters = [], {}
    for shard in sorted(simulator.glob('core*.*core*/trace.json')):
        counts=Counter()
        for event in trace_events(shard):
            if event['ph'] != 'X':
                continue
            name=event['name']; counts[name]+=1
            if name=='MMAD_MX' and event['dur']>0:
                detail=json.loads(event['args']['detail'])
                require([detail[x] for x in ('m','n','k')] == [32,512,128] and
                        [detail[x] for x in ('matrix_a_dtype','matrix_b_dtype','matrix_c_dtype')] == ['fp4_e2m1','fp4_e2m1','f32'], 'Unexpected native MX dot product')
                calls.append(dict(core=shard.parent.name, issue_cycles=round(event['ts']*1650)-origin,
                                  retire_cycles=round((event['ts']+event['dur'])*1650)-origin,mnk=[32,512,128]))
        if '.veccore' in shard.parent.name and counts['RV_VMAXS']:
            counters[shard.parent.name] = {n:counts[n] for n in ('RV_VLDI','RV_VMAXS','RV_VMUL','RV_VMULS','LD_XD_XN_IMM','LD_XD_XN','LDP_XI_XJ_XN')}
    require(len(calls)==2 and len(counters)==4, 'Missing a score team or Vector worker')
    return calls,counters


def verify(base, sku, variant, wheel):
    case=sku+'-'+variant
    freeze=json.loads((SOURCE/('qsli-freeze.json' if variant=='static' else 'qsli-wide-table-freeze.json')).read_text())
    log=base/(case+'.log'); text=log.read_text()
    require('FUSED_GOLDEN_PASS 512' in text and 'All task success.' in text and
            MEMORY_CONFIG[sku] in text and 'Traceback' not in text,'Failed complete device-ROI eligibility')
    if variant=='wide': require('METADATA_HOST_REFERENCE_PASS 1024' in text,'Metadata was not compared with the real host interpreter')
    schedule=json.loads(next(line for line in text.splitlines() if line.startswith('{"0": [1,')))
    require(schedule=={'0':[1,0,0,0,0,0,1,0],'1':[1,0,0,1,1,0,0,1],'36':[1,0,0,0,2,0,1,0]},'Unexpected LI/FD boundary')
    native=json.loads((base/(case+'-provider.json')).read_text()); require(native['counts']=={'installed_hit':1},'Expected exactly one installed AOT hit')
    require(native['events'][0]['key']==WHEEL_KEY,'AOT mismatch')
    entry=f'ops/_native/operators/qsli/variants/{WHEEL_KEY}/provider.so'
    host=wheel.read(entry)
    require(hashlib.sha256(host).hexdigest()==json.loads(wheel.read(entry+'.meta.json'))['binary_sha256'],'Provider ELF digest mismatch')
    name,=re.findall(r'Start profiling on kernel: (\S+)',text); require('kernel_QuantSparseLightningIndexer' in name,'Not QSLI')
    profile,=(base/case/'profile').glob('OPPROF_*'); device=(profile/'dump/aicore_binary.o').read_bytes();offset=host.find(device)
    require(offset>=0,'Dumped device ELF differs from the published DSO')
    rows=CORE.findall(text); roster={core:float(us) for core,us in rows}; require(roster and len(roster)==len(rows),'Incomplete core table')
    cores,hashes=core_windows(profile/'simulator',roster)
    origin=min(c['first_tick'] for c in cores.values()); end=max(c['retire_tick'] for c in cores.values()); us=(end-origin)/1650
    calls, counters=evidence(profile/'simulator',origin)
    cost=json.loads((SOURCE/'mx-acquisition-cost.json').read_text()).get(base.name+'/'+log.name)
    wall_s=cost['wall_s_from_log_open_to_last_write'] if cost and cost['sha256']==sha(log) else None
    return dict(sku=sku, memory_model=MEMORY_CONFIG[sku], table_columns=freeze['attrs'].get('table_pages',8),
                numerical_check='512/512 distinct CPU reference top-scoring global positions, compared after position sort; score ties within winners',
                source_metadata=schedule, metadata_kind='host-replayed two LI rows + one split-row Vector merge',
                exact_native_kernel_window_us=us, precollection_prediction_us=freeze['predictions'][sku]['us'],
                signed_error_pct=100*(freeze['predictions'][sku]['us']/us-1), window_cycles=end-origin,
                trace_roi='Global minimum first to latest retirement over all profiler physical cores, simulator ticks @1.65GHz',
                kernel=name, wheel_key=WHEEL_KEY, provider_sha256=hashlib.sha256(host).hexdigest(), device_sha256=hashlib.sha256(device).hexdigest(),
                elf_inside_provider_at=offset, launched_aic=sum('.cubecore' in c for c in cores), launched_aiv=sum('.veccore' in c for c in cores),
                cube_issues=calls, vector_instruction_counts=counters, last_finishing_core=max(cores,key=lambda c:cores[c]['retire_tick']),
                trace_roster_digest=hashlib.sha256(json.dumps(hashes,sort_keys=True).encode()).hexdigest(), log_sha256=sha(log),
                acquisition_wall_s=wall_s, host_range_annotation_warning='HOST_RANGE_POP_ERROR_AFTER_ENQUEUE' in text)

if __name__ == '__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('raw',type=Path);p.add_argument('--wheel',type=Path,required=True);p.add_argument('--out',type=Path,required=True);p.add_argument('--variants',nargs='+',choices=('static','wide'),default=('static','wide'));a=p.parse_args()
    with zipfile.ZipFile(a.wheel) as wheel:
        data=[verify(a.raw,soc,v,wheel) for v in a.variants for soc in MEMORY_CONFIG]
    a.out.write_text(json.dumps(data,indent=2)+'\n')
