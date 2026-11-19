"""Freeze and run the seven-shape extension of the FA v0.25 blind test."""
import argparse
import fcntl
import json
import math
import os
import re
import shutil
import subprocess
import time
from collections import defaultdict
from pathlib import Path

from fa_holdout import DT, PR, FUT, OUT, RAW, ROOT, ca, estimate
from fa_exit_proof import verify

DIR = OUT / 'supplement'
REGISTER = DIR / 'register.json'
FIRST = OUT / 'fa-freeze.json'
AUDIT = OUT / 'audit.json'
QUEUE = RAW / 'supplement'
KDA = ROOT / 'results/v0.24/kda-prefetch/supplement/state.json'
PRO = ROOT / 'results/v0.23/pro-prefill/state.json'

# Q, KV, D, B and local H; limits are per job, including the vendor parser.
CASES = [
    ((1, 127, 64, 1, 1), 'one short KV page', 560, 900),
    ((16, 129, 64, 1, 1), 'one full KV page plus one token', 560, 900),
    ((63, 257, 128, 1, 1), 'last partial Q tile, D=128, three KV pages', 620, 1100),
    ((65, 257, 64, 1, 1), 'second Q tile opens', 720, 1500),
    ((129, 383, 64, 1, 2), 'third Q tile per head; two local heads', 900, 2500),
    ((1, 32, 64, 1, 29), '29 jobs, across the 28/32 AIC boundary', 1200, 4800),
    ((9, 32769, 64, 1, 1), 'speculation 8 + 1, 32Ki KV plus a one-token tail', 1800, 5000),
]


def publish(path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix('.tmp')
    ca.write(tmp, obj)
    os.replace(tmp, path)


def novelty():
    """Exclude normalized shapes across versioned measurements, registers, and interrupted run directories."""
    indexed = subprocess.check_output(['git', 'ls-files', '-z', '--', 'results/**/*.json', 'configs/**/*.json'], cwd=ROOT)
    sources = sorted({ROOT / os.fsdecode(name) for name in indexed.split(b'\0') if name})
    shapes = defaultdict(set)
    scanned = {}
    def remember(s, origin):
        if isinstance(s, list) and len(s) in (3, 5) and all(type(i) is int and i > 0 for i in s):
            shapes[tuple(s if len(s) == 5 else s + [1, 1])].add(origin)
    def walk(x, origin, fa=False):
        if isinstance(x, dict):
            name = x.get('op') or x.get('operator') or x.get('api')
            if isinstance(name, str):
                fa = name in ('fa','aclnnFlashAttentionScoreV2','flash_attention_score')
            args = x.get('args')
            if isinstance(args, list) and args[:1] == ['fa'] and all(str(e).isdigit() for e in args[1:]):
                remember(list(map(int, args[1:])), origin)
            if fa:
                remember(x.get('shape'), origin)
            for k, v in x.items():
                if isinstance(v, (dict, list)):
                    walk(v, origin, fa or k == 'fa')
        elif isinstance(x, list):
            for v in x:
                walk(v, origin, fa)
    for src in sources:
        if DIR in src.parents or not src.is_file():
            continue
        blob = src.read_bytes()
        if b'"fa"' not in blob and b'aclnnFlashAttentionScoreV2' not in blob:
            continue
        origin = str(src.relative_to(ROOT))
        walk(json.loads(blob), origin)
        scanned[origin] = ca.sha(src)
    # Run directory names can predate machine-readable receipts. Restrict depth
    # and never walk simulator traces or read their payloads.
    pattern = re.compile(r'(?:^|[-_])fa[-_](?:prefill[-_])?(?:Ascend950(?:DT_9582|PR_957b)[-_])?(\d+)[-_](\d+)[-_](\d+)(?:[-_](\d+)[-_](\d+))?(?:$|[-_])', re.I)
    todo = [(ROOT/'runs', 0), (RAW, 0)]
    while todo:
        folder, depth = todo.pop()
        if not folder.exists():
            continue
        for item in folder.iterdir():
            if not item.is_dir() or item.is_symlink():
                continue
            for match in pattern.finditer(item.name):
                remember([int(n) for n in match.groups() if n is not None], str(item.relative_to(ROOT)))
            if depth < 2 and item.name not in ('profile','simulator','output','dump','raw','supplement') and not item.name.startswith('PROF'):
                todo.append((item, depth + 1))
    for s, _, _, _ in CASES:
        if s in shapes:
            raise ValueError(f'Duplicate FA geometry {s}: {sorted(shapes[s])}')
    return dict(utc=ca.utc(), normalized='Q, KV, D, B, H. Three dimensions imply B=H=1.',
                source_sha256=scanned, known_shapes=[dict(shape=s, sources=sorted(v)) for s,v in sorted(shapes.items())],
                unmatched_supplement_shapes=[s for s, *_ in CASES])


def forecasts(shape):
    values = {}
    for sku in (DT, PR, FUT):
        baseline = estimate('fa',shape,chip=sku,timing_mode='calibrated' if sku == DT else 'resource')
        current = estimate('fa',shape,chip=sku)
        resource = estimate('fa',shape,chip=sku,timing_mode='resource')
        if not math.isclose(current.latency_us, resource.latency_us, abs_tol=1e-12, rel_tol=0):
            raise ValueError('Public entry differs from frozen resource algorithm')
        values[sku] = dict(old=baseline.latency_us, candidate=current.latency_us, active_aic=current.active_aic,
                           active_aiv=current.active_aiv, plan=current.projection['target_plan'],
                           predicted_mmads=current.projection['timing_diagnostics']['mmads_total'],
                           evidence='extrapolation only' if sku == FUT else 'prospective CA target')
    return values


def register():
    if REGISTER.exists():
        raise FileExistsError(REGISTER)
    original = ca.read(FIRST)
    first_jobs = original['jobs']
    for j in first_jobs:
        preds = forecasts(j['shape'])
        for k in (DT, PR, FUT):
            for metric in ('old','candidate'):
                if not math.isclose(preds[k][metric],j['forecasts'][k][metric],rel_tol=0,abs_tol=1e-12):
                    raise ValueError(('original frozen forecast mismatch',j['id'],k,metric))
    exclusions = novelty()
    DIR.mkdir(parents=True,exist_ok=True)
    publish(DIR/'novelty.json',exclusions)
    jobs = []
    for shape, reason, budget, disk in CASES:
        pred = forecasts(shape)
        for sku in (DT, PR):
            ident = 'fa-supp-'+sku+'-'+'-'.join(map(str,shape))
            if (QUEUE/ident).exists() or (DIR/(ident+'.json')).exists():
                raise FileExistsError(ident)
            jobs.append(dict(id=ident, op='fa', chip=sku, shape=shape, reason=reason, args=['fa',*map(str,shape)],
                             runner=first_jobs[0]['runner'], runner_sha256=first_jobs[0]['runner_sha256'],
                             outputs=[dict(op='fa',shape=shape,attrs={},indices=[0])], batch=None,nchecks=1,kernels=1,
                             wall_s=budget,raw_mib=disk,forecasts=pred))
    files = [Path(__file__).resolve(),Path(ca.__file__).resolve(),Path(ca.__file__).parent/'campaign.py',
             Path(__file__).with_name('fa_holdout.py').resolve(),Path(__file__).with_name('fa_exit_proof.py').resolve(),
             ROOT/'scripts/cann-run.sh',ROOT/'smoke/three_ops.cpp',ROOT/first_jobs[0]['runner']]
    sources = [ROOT/'operator_model/backend.py',ROOT/'operator_model/fa_resource.py',ROOT/'results/v0.16/fa.json',
               *[ROOT/'configs/chips'/f'{k}.json' for k in (DT,PR,FUT)]]
    archive = ca.read(ROOT/'results/v0.21/coverage/audit.json')['rows']
    elfs = sorted({r['binary_sha256'] for r in archive if r['op']=='fa' and isinstance(r.get('binary_sha256'),str)})
    reg = dict(utc=ca.utc(), parent_freeze_sha256=ca.sha(FIRST), unchanged_854_audit_sha256=ca.sha(AUDIT),
               novelty_sha256=ca.sha(DIR/'novelty.json'), role='7 extra shapes selected after the first 3; no recalibration',
               normalized=original['normalized'], comparisons='v0.24 auto (DT calibrated, PR/970 resource) vs public v0.25 auto; old matches the first three recorded predictions',
               concurrency=1, wall_budget_s=sum(j['wall_s'] for j in jobs), raw_budget_mib=sum(j['raw_mib'] for j in jobs),
               scheduler=dict(cpu_load_fraction=.72, ram_min_gib=48, disk_min_gib=80, poll_seconds=45,
                              precedence='KDA v0.24 supplement terminal and older Pro not queued; lower priority nice 12'),
               known_released_fa_elf_sha256=elfs,
               gate='one exact SKU BF16 SBH launch, correct physical memory, whole output vs FP64 golden, official launch/roster/durations, per-shard physical END and terminal window, exact released FA ELF',
               scoring='full global device window; compare both frozen estimates on identical eligible rows. Report AIC/AIV/MMAD disagreement separately. CA failure remains N/A.',
               runtime_sha256={str(p.relative_to(ROOT)):ca.sha(p) for p in files},
               model_sha256={str(p.relative_to(ROOT)):ca.sha(p) for p in sources},jobs=jobs)
    with REGISTER.open('x') as f:
        json.dump(reg,f,ensure_ascii=False,indent=2,allow_nan=False)
        f.write('\n')
    return reg


def headroom(policy):
    kda, pro = ca.read(KDA), ca.read(PRO)
    ram = int(re.search(r'^MemAvailable:\s+(\d+) kB',Path('/proc/meminfo').read_text(),re.M)[1])*1024/2**30
    disk = shutil.disk_usage(ROOT).free/2**30
    load = os.getloadavg()[0]
    cores = len(os.sched_getaffinity(0))
    value = dict(kda=kda['state'],pro=pro['state'],pro_case=pro.get('case'),available_ram_gib=round(ram,1),
                 free_disk_gib=round(disk,1),load_1m=load,logical_cpus=cores)
    busy = kda['state'] not in ('complete','CA budget reached','interrupted raw attempt; no duplicate launch') or pro['state']=='queued'
    return not busy and load<cores*policy['cpu_load_fraction'] and ram>=policy['ram_min_gib'] and disk>=policy['disk_min_gib'], value


def summarize(reg):
    cohorts = [('first3',ca.read(FIRST)['jobs'],OUT),('supplement7',reg['jobs'],DIR)]
    rows, pairs, missing = [],[],[]
    for cohort, jobs, dest in cohorts:
        for j in jobs:
            sku,s = j['chip'],j['shape']
            p = dest/(j['id']+'.json')
            if not p.exists():
                missing.append(dict(id=j['id'],status='pending'));continue
            r = ca.read(p)
            if not r['qualified']:
                missing.append(dict(id=j['id'],status='failed',reason=r['reason']));continue
            f=j['forecasts'][sku]
            rows.append(dict(id=j['id'],cohort=cohort,shape=s,sku=sku,ca=r['ca_us'],old=f['old'],
                             candidate=f['candidate'],future_us=j['forecasts'][FUT]['candidate'],cost_s=r['cost']['seconds'],
                             observed_aic=r['active_aic'],observed_aiv=r['active_aiv'],observed_mmads=r['stages'][0]['mmad'],
                             predicted_aic=f['active_aic'],predicted_aiv=f['active_aiv'],predicted_mmads=f.get('predicted_mmads'),
                             receipt=str(p.relative_to(ROOT))))
    for s in sorted({tuple(r['shape']) for r in rows}):
        sides={r['sku']:r for r in rows if tuple(r['shape'])==s}
        if len(sides)==2:
            pairs.append(dict(shape=s,**{m:sides[DT][m]/sides[PR][m] for m in ('ca','old','candidate')}))
    def stats(items):
        if not items:return None
        return {m:dict(wape_pct=100*sum(abs(r[m]-r['ca']) for r in items)/sum(r['ca'] for r in items),
                       max_ape_pct=100*max(abs(r[m]/r['ca']-1) for r in items)) for m in ('old','candidate')}
    return dict(registration_sha256=ca.sha(REGISTER),target=10,rows=rows,missing=missing,paired_dt_over_pr=pairs,
                timing_by_sku={s:dict(shapes=sum(r['sku']==s for r in rows),metrics=stats([r for r in rows if r['sku']==s])) for s in (DT,PR)},
                ratio_metrics=stats(pairs),ratio_direction_correct={m:sum((r[m]-1)*(r['ca']-1)>0 for r in pairs) for m in ('old','candidate')},
                ca_seconds=sum(r['cost_s'] for r in rows),scope='Combined cumulative FA only; old 854-window audit remains unchanged.')


def run():
    reg = ca.read(REGISTER)
    if ca.sha(FIRST)!=reg['parent_freeze_sha256'] or ca.sha(DIR/'novelty.json')!=reg['novelty_sha256']:
        raise ValueError('Registration lineage changed')
    for name, digest in reg['runtime_sha256'].items():
        if ca.sha(ROOT/name)!=digest:raise ValueError(name)
    QUEUE.mkdir(exist_ok=True,parents=True)
    os.environ.update(OMP_WAIT_POLICY='PASSIVE',GOMP_SPINCOUNT='0')
    with (RAW/'lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        def state(name,**kw):
            publish(DIR/'state.json',dict(utc=ca.utc(),state=name,registration_sha256=ca.sha(REGISTER),**kw))
        def score():
            publish(DIR/'score.json',summarize(reg))
        score()
        for j in reg['jobs']:
            dest=DIR/(j['id']+'.json')
            if dest.exists():continue
            d=QUEUE/j['id']
            if not (d/'cost.json').exists():
                if d.exists():
                    state('interrupted raw attempt; preserved without duplicate',id=j['id'])
                    return
                while True:
                    ok, info=headroom(reg['scheduler'])
                    if ok:break
                    state('waiting for older jobs or host capacity',id=j['id'],**info)
                    time.sleep(reg['scheduler']['poll_seconds'])
                state('acquiring',id=j['id'],**info)
                ca.capture(j,j['chip'],d,reg)
            state('qualifying',id=j['id'])
            cost=ca.read(d/'cost.json')
            try:
                r,=ca.qualify(j,j['chip'],d)
                s,=r['stages']
                if s['binary_sha256'] not in reg['known_released_fa_elf_sha256'] or not s['kernel'].startswith('FlashAttentionScore_') or s['mmad']<=0:
                    raise ValueError(('device ELF or FA identity',s['kernel'],s['binary_sha256']))
                q,_,dim,b,h = j['shape']
                if len(r['correctness'])!=1 or not re.search(fr'\belements={q*b*h*dim}\s+bad=0\b',r['correctness'][0]):
                    raise ValueError(('whole output golden',r['correctness']))
                r.update(qualified=True,role='v0.25 seven-shape prospective supplement',registration_sha256=ca.sha(REGISTER))
                provisional=d/'candidate.json'
                ca.write(provisional,r)
                proof=verify(provisional,{(s['kernel'],s['binary_sha256'])})
                proof.update(receipt=str(dest.relative_to(ROOT)), source_sha256=ca.sha(provisional))
                publish(DIR/'physical'/dest.name,proof)
            except (AssertionError,ValueError,KeyError,IndexError,TypeError,OSError) as err:
                r=dict(qualified=False,chip=j['chip'],shape=j['shape'],case_id=j['id'],reason=repr(err),
                       cost=cost,registration_sha256=ca.sha(REGISTER))
            publish(dest,r)
            score()
        state('finished',supplement_receipts=len(reg['jobs']))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action',choices=('novelty','freeze','run','score'))
    a=parser.parse_args().action
    if a=='novelty':
        print(f"{len(novelty()['known_shapes'])} distinct prior shapes; all seven absent.")
    elif a=='freeze':register()
    elif a=='run':run()
    else:publish(DIR/'score.json',summarize(ca.read(REGISTER)))
