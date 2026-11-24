"""Verify and summarize the individual native kernel windows for a candidate-indexer fixture."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import re

from calibration.msopprof import CORE, sha
from calibration.mx_matmul.collect import MEMORY_CONFIG, core_windows, trace_events
from calibration.__main__ import require

KERNEL = re.compile(r'Start profiling on kernel: (\S+)')
COMPACT = ['GatherV2','GatherV2','LightningIndexer','GatherV2','Sort']
DENSE = ['aclnnCast','aclnnCast','aclnnCast','aclnnMatmul','aclnnInplaceRelu','aclnnInplaceMul',
         'aclnnReduceSum','aclnnNeTensor','aclnnAll','aclnnInplaceMaskedFillScalar','aclnnTopk',
         'aclnnSort','aclnnCast']

def ordinary_li_instructions(simulator):
    # The ordinary kernel launches all AICs. Count actual MMADs on every
    # shard independently of the launch roster and old H8 tiling template.
    groups = []
    for path in sorted(simulator.glob('core*.cubecore*/trace.json')):
        operations = []
        for event in trace_events(path):
            if event.get('ph')!='X' or event.get('name')!='MMAD' or event['dur']<=0:
                continue
            fields = json.loads(event['args']['detail'])
            operations.append((round(event['ts']*1650), fields['m'], fields['n'], fields['k']))
        if operations:
            operations.sort()
            counts = Counter((m,n,k) for _,m,n,k in operations)
            groups.append({'core':path.parent.name,'m_n_k_count':[[*mnk,count] for mnk,count in sorted(counts.items())],
                           'issue_intervals_cycles':[r[0]-l[0] for l,r in zip(operations,operations[1:])]})
    require(groups, 'Missing ordinary LI MMAD events')
    return groups


def verify(raw, opp):
    cost = json.loads((raw/'cost.json').read_text())
    text = (raw/'run.log').read_text()
    require(cost['exit_code'] == 0 and 'All task success.' in text and 'Profiling data parse finished.' in text
            and 'SMOKE_PASS' in text and 'SMOKE_FAIL' not in text, 'Incomplete run')
    sku, = re.findall(r'^runtime_soc=(\S+)$', text, re.M)
    require('--soc-version='+sku in cost['command'] and MEMORY_CONFIG[sku] in text, 'SKU/HBM mismatch')
    check, = re.findall(r'^correctness ordered_global_positions=(\d+) bad=0$',text,re.M)
    require(int(check)==512, 'Incorrect output')
    fixture, = re.findall(r'^case=(compact|dense) blocks=(\d+) selected=(\d+) heads=(\d+) block_size=8 D=128 Q=1 top=512$', text, re.M)
    mode, blocks, selected, heads = fixture
    launch = list(KERNEL.finditer(text))
    require(launch, 'Missing kernels')
    if mode == 'compact':
        require([m[1].split('_')[0] for m in launch] == COMPACT, 'Unexpected compact graph')
    else:
        require(re.findall(r'^launch=(\w+)$',text,re.M) == DENSE, 'Unexpected dense graph')
    base, = (raw/'profile').glob('OPPROF_*')
    require(len(list(base.glob('*/*/dump/aicore_binary.o'))) == len(launch), 'Mismatch between profiles and dispatches')
    seen = Counter()
    stages = []
    tables = re.split(r"^core_name\s+duration_time\(us\)\s+running_time\(us\)[ \t]*$",text,flags=re.M)[1:]
    require(len(tables)==len(launch), "Unexpected number of kernel roster tables")
    for i,m in enumerate(launch):
        kernel = m[1]
        num = seen[kernel]
        seen[kernel] += 1
        directory = base/kernel/str(num)
        elf = directory/'dump/aicore_binary.o'
        digest = sha(elf)
        filename = re.sub(r'_\d+(?:_mix_ai[cv])?$', '', kernel) + '.o'
        candidates = list(opp.glob('*/*/'+filename))
        matched = [f for f in candidates if sha(f)==digest]
        require(len(matched)==1,f'Cannot match independent OPP object for {kernel}')
        # Collection of the following launch can start before the exporter
        # prints this launch's table; the serialized profiler tables stay ordered.
        rows = CORE.findall(tables[i])
        roster = {n:float(v) for n,v in rows}
        require(roster and len(roster)==len(rows),'Missing or duplicate profiler cores')
        cores, hashes = core_windows(directory/'simulator',roster)
        cycles = max(c['retire_tick'] for c in cores.values()) - min(c['first_tick'] for c in cores.values())
        api = ('aclnn'+COMPACT[i]) if mode=='compact' else re.findall(r'^launch=(\w+)$',text[:m.start()],re.M)[-1]
        stages.append(dict(api=api,kernel=kernel,object=str(matched[0].relative_to(opp)),elf_sha256=digest,
                           physical_shards_sha256=hashlib.sha256(json.dumps(hashes,sort_keys=True).encode()).hexdigest(),
                           aic=sum('.cubecore' in n for n in roster), aiv=sum('.veccore' in n for n in roster),
                           cycles=cycles,window_us=round(cycles/1650,6)))
        if api == 'aclnnLightningIndexer':
            stages[-1]['mmad'] = ordinary_li_instructions(directory/'simulator')
    return dict(mode=mode,chip=sku,memory_model=MEMORY_CONFIG[sku],blocks=int(blocks),selected=int(selected),
                compressed_keys=int(blocks)*8,candidate_keys=int(selected)*8,heads=int(heads),
                query_rows=1,head_dim=128,correct_ordered_global_indices=int(check),wrong=0,
                stages=stages,sum_kernel_windows_us=round(sum(x['cycles'] for x in stages)/1650,6),
                log_sha256=sha(raw/'run.log'),probe_sha256=cost['probe_sha256'],acquisition_wall_s=round(cost['wall_seconds'],2))

if __name__ == '__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('raw',type=Path,nargs='+')
    p.add_argument('--opp',type=Path,required=True,help='Installed ascend950 kernel root containing ops_math/ and siblings')
    p.add_argument('--out',type=Path,required=True)
    a=p.parse_args()
    data=[verify(d,a.opp) for d in a.raw]
    a.out.write_text(json.dumps(data,indent=2)+'\n')
