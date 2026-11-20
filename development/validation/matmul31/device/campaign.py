"""Independent single-slot verification of retiled BF16 Matmul decode projections."""
import fcntl
import json
import os
from pathlib import Path
import re
import shutil
import sys
import time

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT))
from development.validation.matmul28 import campaign as ca
from development.validation.matmul28.drain import recover
from development.validation.matmul28.multi.verify import geometry
from operator_model.matmul_multisplit import estimate_matmul_retile
from operator_model.matmul_placement import matmul_placement
from operator_model.matmul_splitk import estimate_matmul_splitk

OUT = ROOT / 'results/v0.31/matmul/device'
RAW = ROOT / 'runs/matmul31/device'
REG = OUT / 'freeze.json'
HOST = ROOT / 'results/v0.30/matmul-dispatch/freeze.json'
PREVIOUS = ROOT / 'results/v0.29/matmul/decode/freeze.json'
SHAPES = [(99,64,7168), (11,1280,5120)]
CHIPS = [(ca.DT,'dt'), (ca.PR,'pr')]


def triples(obj):
    if isinstance(obj,dict):
        if isinstance(obj.get('shape'),list):
            yield tuple(obj['shape'])
        for v in obj.values():yield from triples(v)
    elif isinstance(obj,list):
        for v in obj:yield from triples(v)


def freeze():
    paths = [ROOT / 'results/v0.28/matmul/overview-prices.json', ROOT / 'results/v0.29/matmul/score.json',
             ROOT / 'results/v0.30/matmul-dispatch/device/freeze.json', PREVIOUS,
             *sorted((ROOT/'results/v0.28/matmul').rglob('*freeze.json')),
             ROOT/'results/v0.29/matmul/timing-freeze.json',ROOT/'results/v0.29/matmul/full-load/freeze.json',
             ROOT/'results/v0.30/matmul/freeze.json']
    for p in paths:
        if set(triples(ca.read(p))) & set(SHAPES):raise ValueError(f'Device job already registered or measured in {p}')
    run_logs=0
    for folder, dirs, files in os.walk(ROOT/'runs'):
        dirs[:]=[d for d in dirs if d not in ('simulator','ca_workspace','record','profile','log','dump','sources','npusim') and not d.startswith(('OPPROF_','PROF_','npusim_'))]
        if 'run.log' not in files:continue
        run_logs+=1
        with (Path(folder)/'run.log').open(errors='replace') as f:
            for line in f:
                if 'case=matmul' not in line:continue
                mt=re.search(r'case=matmul api=aclnnMatmul M=(\d+) N=(\d+) K=(\d+)',line)
                if mt and tuple(map(int,mt.groups())) in SHAPES:raise ValueError(f'Already run in {folder}')
    host = ca.read(HOST)['rows']
    rows=[]
    for s in SHAPES:
        matches=[r for r in host if tuple(r['shape'])==s]
        assert len(matches)==2 and matches[0]['scenarios']==matches[1]['scenarios']
        preds={}
        for sku in (ca.DT, ca.PR, ca.FUT):
            placement=matmul_placement(s,chip=sku)
            retile=estimate_matmul_retile(s,chip=sku)
            old=estimate_matmul_splitk(s,chip=sku)
            if sku!=ca.FUT:
                assert placement['route']=='split_k' and placement['latency_us']==retile['latency_us']
                assert retile['route']=='retiled_output_split_k'
                assert placement['latency_us']==next(r for r in matches if r['soc']==sku)['placement_us']
            preds[sku]=dict(us=placement['latency_us'],old_us=old['latency_us'],route=placement['route'],
                            geometry=retile,placement=placement)
        rows.append(dict(shape=s,scenarios=matches[0]['scenarios'],predictions=preds,wall_s=3600,raw_mib=4096))
    old=ca.read(PREVIOUS)['acquisition_sha256'];deps={p:ca.digest(ROOT/p) for p in old}
    assert deps==old
    for p in (Path(__file__),HOST,ROOT/'development/validation/matmul28/multi/verify.py',
              *(ROOT/'operator_model'/n for n in ('matmul_multisplit.py','matmul_splitk.py','matmul_placement.py','matmul_resource.py','matmul_waves.py','matmul_v15.py')),
              *(ROOT/'configs/chips'/(c+'.json') for c in (ca.DT,ca.PR,ca.FUT)), ROOT/'results/v0.16/matmul.json'):
        deps[str(p.relative_to(ROOT))]=ca.digest(p)
    with REG.open('x') as f:
        json.dump(dict(utc=ca.now(),rows=rows,source_sha256=deps,
          role='Frozen v0.29 placement and old public resource prices. Prior host-only workspace signals known. First full device timing and actual geometry for each endpoint. 970 is extrapolated.',
          budget='Four serial endpoints. 3600 seconds and 4GiB each, stop on cap. Inherit full permitted CPU mask, nice +12, one serial slot. Preserve any interrupted raw directory and do not rerun automatically.',
          selection='One output Pro indexer at B11 verify Q9, five output Flash wq_a at B11 decode Q1. Contiguous BF16 ND; actual deployed vLLM precision and layout are separate.',
          novelty=dict(device_sources=[str(p.relative_to(ROOT)) for p in paths],raw_logs_checked=run_logs),
          qualification='Require exact SKU/provider, full golden and all-core physical END/retirement. Independently score actual geometry even if timing passes. Missing/failed comparisons stay in denominator of attempted experiments.',
          ),f,indent=2);f.write('\n')


def inspect(job,sku,tag,folder,reg):
    cost=ca.read(folder/'cost.json'); assert cost['started_utc']>reg['utc']
    slug=folder.name
    try:result=ca.qualify(job,sku,folder)
    except (OSError,KeyError,IndexError,AssertionError,ValueError) as e:result=dict(qualified=False,reason=repr(e))
    initial=OUT/(slug+'-initial.json');result.update(case=slug+'-initial',shape=job['shape'],chip=sku,cost=cost)
    ca.save(initial,result)
    if not result['qualified']:
        try:result=recover(job,result,folder,acquisition=initial) or result
        except (OSError,KeyError,IndexError,AssertionError,ValueError) as e:result=dict(qualified=False,reason=repr(e))
    if cost['exit_code'] or cost['stop_reason']:
        result.update(qualified=False,reason='Collection budget/process status failed',ca_us=None)
    if result['qualified']:
        stage,=result['stages']; allowed={v for p,v in reg['source_sha256'].items() if p.endswith('.o')}
        if stage['binary_sha256'] not in allowed:result.update(qualified=False,reason='Provider SHA mismatch',ca_us=None)
        else:
            try:
                assert stage['kernel'].endswith('_4098_mix_aic')
                g=geometry(stage,job['shape']); model=job['predictions'][sku]['geometry'];depth=model['max_segment_k'];slices=model['k_splits'];tail=model['tail_segment_k']
                spans=[[i*depth,i*depth+(tail if i==slices-1 else depth)] for i in range(slices)]
                partition_ok=all([list(s) for s in r['k_spans']]==spans for r in g['output_rectangles'])
                result.update(actual_geometry=g,mechanism_checks=dict(k_spans=partition_ok,groups=g['groups']==model['output_tiles'],
                    productive_aic=g['productive_aic']==model['productive_aic'],active_aic=g['active_aic']==model['active_aic'],
                    active_aiv=g['active_aiv']==model['active_aiv'],mmads=g['actual_mmad']==model['mmads_total']))
            except (OSError,KeyError,AssertionError,ValueError) as e:result.update(mechanism_checks=None,mechanism_reason=repr(e))
    ca.save(OUT/(slug+'.json'),dict(result,chip=sku,shape=job['shape'],cost=cost,registration_sha256=ca.digest(REG),initial_sha256=ca.digest(initial)))


def run():
    reg=ca.read(REG)
    for name,h in reg['source_sha256'].items():
        if ca.digest(ROOT/name)!=h:raise RuntimeError('Changed after freeze: '+name)
    RAW.mkdir(parents=True,exist_ok=True);os.nice(12);os.environ.update(OMP_WAIT_POLICY='PASSIVE',GOMP_SPINCOUNT='0')
    def state(phase,**kw):ca.save(OUT/'state.json',dict(utc=ca.now(),phase=phase,pid=os.getpid(),cpu_count=len(os.sched_getaffinity(0)),**kw))
    with (RAW/'slot.lock').open('a') as fd:
        fcntl.flock(fd,fcntl.LOCK_EX|fcntl.LOCK_NB)
        for job in reg['rows']:
            for sku,tag in CHIPS:
                slug='-'.join(map(str,job['shape']))+'-'+tag; folder=RAW/slug
                if (OUT/(slug+'.json')).exists():continue
                if not (folder/'cost.json').exists():
                    if folder.exists():state('preserved incomplete raw',case=slug);return
                    while shutil.disk_usage(ROOT).free<16*2**30+2*job['raw_mib']*2**20:state('waiting for free disk',case=slug);time.sleep(30)
                    folder.mkdir();state('acquiring',case=slug);ca.measure(job,sku,folder)
                state('qualifying',case=slug);inspect(job,sku,tag,folder,reg)
        state('finished')

if __name__=='__main__':{'freeze':freeze,'run':run}[sys.argv[1]]()
