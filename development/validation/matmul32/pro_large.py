"""Freeze a new Pro projection shape and acquire both SKUs in an existing serial slot."""
import argparse
import fcntl
import gzip
import json
import os
from pathlib import Path
import re
import shutil
import time

from freeze import ROOT, OUT, DT, PR, FUT, digest
from forward import LOCK, ca, recover
from operator_model import estimate

TARGET = OUT / 'pro-large'
REG = TARGET / 'freeze.json'
RAW = ROOT / 'runs/matmul32/pro-large'
PREVIOUS = OUT / 'decode/state.json'
FULL = ROOT / 'results/v0.28/matmul/wide-prefill/freeze.json'
MODEL = 'DeepSeek-V4-Pro'
NODE = 'indexer.weights_proj'
SOURCE = 'vllm_ascend/models/deepseek_v4/indexer.py'
PKG = (ROOT / 'toolchains/cann/9.2.1/cann-9.2.1/opp/built-in/op_impl/ai_core/tbe/kernel'
       '/ascend950/ops_nn/mat_mul_v3/MatMulV3_ND_ND_ND_ND_BF16_BF16_BF16_BF16_high_performance.o')


def relative(p):
    return str(p.relative_to(ROOT))


def shapes(value):
    if isinstance(value, list):
        for item in value:
            yield from shapes(item)
    if isinstance(value, dict):
        if isinstance(value.get('shape'), list):
            yield tuple(value['shape'])
        for item in value.values():
            yield from shapes(item)


def check_novel(shape):
    inventory = []
    for directory in sorted((ROOT/'results').glob('v*/matmul*')):
        for path in sorted(directory.rglob('*freeze.json')):
            if 'raw' in path.parts or path == REG:
                continue
            if tuple(shape) in shapes(ca.read(path)):
                raise ValueError(f'{shape} already reserved by {path}')
            inventory.append(relative(path))
    for path in sorted((ROOT/'results').glob('v*/large/cube/*freeze.json')):
        if tuple(shape) in shapes(ca.read(path)):
            raise ValueError(f'already reserved by {path}')
        inventory.append(relative(path))
    if any(r['shape'] == shape for r in ca.read(OUT/'score.json')['rows']):
        raise ValueError('already scored')
    signature = 'case=matmul api=aclnnMatmul M=%s N=%s K=%s ' % tuple(shape)
    checked = 0
    for root in (ROOT/'runs', ROOT/'results/v0.29/matmul/decode/raw'):
        for folder, dirs, files in os.walk(root):
            dirs[:] = [x for x in dirs if x not in {'sources','profile','simulator','log','log_ca','npusim','dump','ca_workspace','record'}
                       and not x.startswith(('OPPROF_', 'PROF_'))]
            if 'run.log' in files:
                path = Path(folder)/'run.log'
                with path.open(errors='replace') as f:
                    for line in f:
                        if signature in line:
                            raise ValueError(f'already launched by {path}')
                        if 'case=' in line:
                            break
                checked += 1
    return dict(registrations=inventory, existing_host_logs_checked=checked,
                archive_score=relative(OUT/'score.json'), duplicate_acquisitions=[])


def forecasts(shape):
    result = {}
    for chip in (DT, PR, FUT):
        begin = time.perf_counter_ns()
        r = estimate('matmul', shape, dtype='bf16', chip=chip)
        ms = (time.perf_counter_ns()-begin)/1e6
        p = r.projection['target_plan']
        result[chip] = dict(us=r.latency_us, active_aic=r.active_aic, active_aiv=r.active_aiv,
                            waves=p['waves'], mmads=p['mmads_total'], tile=p['tile'], branch=p['branch'],
                            query_ms=ms, extrapolation=chip == FUT)
    return result


def register():
    deps = ca.read(OUT/'freeze.json')['sha256']
    if not all(digest(ROOT/path) == sha for path,sha in deps.items()):
        raise ValueError('changed frozen model')
    locks = ca.read(ROOT/'results/v0.20/workloads/source-lock.json')
    cfg = next(x for x in locks['model_configs'] if x['repo'].endswith('/'+MODEL))
    src = next(x for x in locks['reviewed_source_files'] if x['path'] == SOURCE)
    src_path = ROOT/'runs/workloads19/sources/full'/SOURCE
    if digest(src_path) != src['sha256'] or digest(ROOT/cfg['file']) != cfg['sha256']:
        raise ValueError('workload fingerprint differs')
    config = ca.read(ROOT/cfg['file'])
    dims = [config['index_n_heads'], config['hidden_size']]
    with gzip.open(ca.CATALOG,'rt') as f:
        rows = (json.loads(line) for line in f)
        scenario = next(row for row in rows if all(row['scenario'][key] == val for key,val in
            dict(model=MODEL,seq=32768,batch=8,tp=8,dp=1,chunk=0,step=0).items()))
    nodes = [n for n in scenario['contracts'] if n['op'] == NODE]
    assert nodes and all(n['placement'] == 'replicated' and n['tensors']['weight_logical'] == dims for n in nodes)
    assert config['torch_dtype'] == 'bfloat16'
    shape = [8*4096,*dims]
    full = next(r for r in ca.read(FULL)['rows'] if r['node'] == NODE and r['model'] == MODEL)
    assert full['shape'] == [8*32768,*dims]
    novelty = check_novel(shape)
    files = [Path(__file__), ca.CATALOG, PKG, ca.BIN, ROOT/cfg['file'],
             ROOT/'results/v0.20/workloads/source-lock.json', src_path, FULL,
             ROOT/'development/workloads/prefill_shapes.py', ROOT/'development/validation/matmul28/campaign.py',
             ROOT/'development/validation/matmul28/drain.py', ROOT/'development/validation/matmul28/runner.cpp',
             ROOT/'development/validation/sku17/campaign_collect.py', ROOT/'scripts/cann-run.sh']
    data = dict(utc=ca.now(), model=MODEL, node=NODE, shape=shape, requests=8, per_request_context=32768,
        per_request_query=4096, replica_chunk_token_budget=32768, dp=1, tp=8, weight_logical=dims,
        layers=nodes[0]['layers'], model_config=cfg, indexer_source=src, query_dtype='bf16', placement='replicated',
        weights='Source: ReplicatedLinear(config.hidden_size,self.n_heads,bias=False,quant_config=None). '
            'Synthetic BF16 weights and inputs, dense ND kernel, one isolated projection with every output checked.',
        inputs_bytes=2*shape[0]*dims[1],weights_bytes=2*dims[0]*dims[1],output_bytes=2*shape[0]*dims[0],
        nominal_ops=2*shape[0]*dims[0]*dims[1], predictions=forecasts(shape), installed_matmul_elf=digest(PKG),
        reuse_full=dict(source=relative(FULL),shape=full['shape'],predictions=forecasts(full['shape']),
            state=relative(FULL.with_name('state.json')), budget_per_sku=dict(wall_s=full['wall_s'],raw_mib=full['raw_mib']),
            role='Already acquiring full-sized DT at this new price freeze. Original DT/PR reservation is reused.'),
        novelty=novelty, wall_s=21600, raw_mib=16384, free_disk_required_bytes=48*2**30,
        predecessor=relative(PREVIOUS), slot=relative(LOCK),
        protocol='One official msopprof simulator execution/SKU, --dump=false, compulsory pipeline utilization only, '
                 'all cores. Require one hardware task begin/done pair, SKU, gold for M*N elements, library ELF, '
                 'vendor per-core table, per-core physical END and final engine retirement.',
        policy='DT then PR in retired decode slot. No partial output, wrapper window or scaled short sample as CA latency; '
               'report one attempt/SKU; 970 is an extrapolation. Preserve raw on failure.',
        dependencies={**deps,**{relative(p):digest(p) for p in files}})
    TARGET.mkdir(parents=True,exist_ok=True)
    with REG.open('x') as f:
        json.dump(data,f,ensure_ascii=False,indent=2)
        f.write('\n')


def one_task(raw):
    files = sorted(raw.glob('stars_log*_task.dump'))
    if len(files) != 1:
        raise ValueError('missing or multiple STARS task scopes')
    content = files[0].read_text(errors='replace')
    events = re.findall(r'\[(\d+)\]\s+\[(TASK_BEGIN|TASK_DONE)\].*?stream_id:(\d+), task_id:(\d+)',content)
    if len(events) != 2 or [e[1] for e in events] != ['TASK_BEGIN','TASK_DONE'] or events[0][2:] != events[1][2:] or int(events[0][0]) >= int(events[1][0]):
        raise ValueError('incomplete or multi-task STARS scope')
    return dict(file=relative(files[0]),sha256=digest(files[0]),begin_tick=int(events[0][0]),
                done_tick=int(events[1][0]),stream=int(events[0][2]),task=int(events[0][3]))


def run():
    j = ca.read(REG)
    if not all(digest(ROOT/p) == h for p,h in j['dependencies'].items()):
        raise ValueError('frozen source changed')
    def state(phase, **kw):
        ca.save(TARGET/'state.json',dict(utc=ca.now(),pid=os.getpid(),phase=phase,freeze_sha256=digest(REG),**kw))
    RAW.mkdir(parents=True,exist_ok=True)
    os.nice(12)
    os.environ.update(OMP_WAIT_POLICY='PASSIVE',GOMP_SPINCOUNT='0')
    with (RAW/'worker.lock').open('a') as worker, LOCK.open('a') as slot:
        fcntl.flock(worker,fcntl.LOCK_EX|fcntl.LOCK_NB)
        state('queued behind v0.32 decode pair series',predecessor=relative(PREVIOUS),slot=relative(LOCK))
        while not PREVIOUS.exists() or ca.read(PREVIOUS)['phase'] not in ('finished','incomplete run retained, no retry'):
            time.sleep(30)
        fcntl.flock(slot,fcntl.LOCK_EX)
        for sku,tag in ((DT,'dt'),(PR,'pr')):
            raw=RAW/tag
            dest=TARGET/(tag+'.json')
            if dest.exists():
                continue
            if not (raw/'cost.json').exists():
                if raw.exists():
                    state('interrupted raw preserved',chip=sku,raw=relative(raw))
                    return
                state('waiting for per-endpoint disk budget',chip=sku)
                while shutil.disk_usage(ROOT).free < j['free_disk_required_bytes']:
                    time.sleep(30)
                raw.mkdir()
                state('acquiring',chip=sku,raw=relative(raw))
                ca.measure(j,sku,raw)
            state('verifying',chip=sku,raw=relative(raw))
            cost=ca.read(raw/'cost.json')
            assert j['utc'] < cost['started_utc']
            try:
                if cost['exit_code'] or cost['stop_reason']:
                    raise ValueError('process or acquisition budget')
                task=one_task(raw)
                try:
                    d=ca.qualify(j,sku,raw)
                except (ValueError,KeyError,AssertionError,OSError,IndexError) as exc:
                    initial=TARGET/(tag+'-initial.json')
                    d=dict(qualified=False,ca_us=None,case=tag,chip=sku,shape=j['shape'],reason=repr(exc))
                    ca.save(initial,d)
                    d=recover(j,d,raw,acquisition=initial) or d
                if d['qualified']:
                    stage,=d['stages']
                    if stage['binary_sha256'] != j['installed_matmul_elf']:
                        raise ValueError('installed provider ELF mismatch')
                    p=j['predictions'][sku]
                    d['mechanism_checks']={key:stage[key] == p[key] for key in ('active_aic','active_aiv')}
                    d['mechanism_checks']['mmad_count']=stage['mmad'] == p['mmads']
                d['stars_task']=task
            except (ValueError,KeyError,AssertionError,OSError,IndexError) as exc:
                d=dict(qualified=False,ca_us=None,reason=repr(exc))
            ca.save(dest,dict(d,chip=sku,shape=j['shape'],cost=cost,freeze_sha256=digest(REG)))
        state('finished')


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('action',choices=['freeze','run'])
    {'freeze':register,'run':run}[p.parse_args().action]()
