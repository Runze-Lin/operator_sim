"""Replay archived full-window receipts through the public API. Never launch CA."""
from collections import defaultdict
from functools import lru_cache
from hashlib import sha256
import json
from pathlib import Path
import re
import sys

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
from operator_model import estimate

OUT = ROOT/'results/v0.21/coverage'
DT, PR, FUTURE = 'Ascend950DT_9582', 'Ascend950PR_957b', 'Ascend970DT'
OP = {'fa': 'aclnnFlashAttentionScoreV2', 'matmul': 'aclnnMatmul', 'quant': 'aclnnDynamicMxQuant'}
OPS = tuple(OP)
repos = {'quant': 'ops-nn', 'matmul': 'ops-nn', 'fa': 'ops-transformer'}
rows = {}
source_index = {}
pending = []


def repo_path(p):
    if not p:
        return None
    return str(p).removeprefix(str(ROOT) + '/')


@lru_cache(None)
def read(p):
    data = (ROOT/p).read_bytes()
    source_index[str(p)] = sha256(data).hexdigest()
    return json.loads(data)


def normalized(d):
    if isinstance(d, (list, tuple)):
        return d[0] if len(d) == 1 else None
    return d


def workload(op, s):
    """Labels only. All admitted complete windows remain in the archive."""
    if op == 'fa':
        return s[1] >= 32767
    if op == 'matmul':
        return max(s) >= 1024 or 2*s[0]*s[1]*s[2] >= 2**26
    return s[0]*s[1] >= 2**20 or (s[0] >= 128 and s[1] >= 5120)


def add(op, shape, chip, m, source, role, attrs=None, contract=None, wall=None):
    if op not in OP:
        return
    if op == 'fa' and len(shape) < 5:
        shape = list(shape) + [1] * (5 - len(shape))
    ca = m.get('ca_us', m.get('reference_us'))
    checks = m.get('correctness', [])
    if not ca or not checks or any(not re.search(r'\bbad=0\b', c) for c in checks):
        return
    digest = normalized(m.get('trace_sha256')) or m.get('db_sha256') or m.get('trace_or_db_sha256')
    if not isinstance(digest, str) or not re.fullmatch('[0-9a-f]{64}', digest):
        return
    chip = chip or DT
    assert chip in (DT,PR), chip
    key = op, chip, digest
    trace = normalized(m.get('trace')) or m.get('db_path')
    if not trace and m.get('report_path'):
        trace = str(Path(m['report_path'])/'simulator/trace.json')
    item = rows.get(key)
    if not item:
        family = 'full instruction SQLite' if (m.get('db_sha256') or 'entry_kernel' in m['kernel']) else 'official full profiler'
        item = rows[key] = dict(op=op,shape=shape,chip=chip,attrs=attrs or {},api=OP[op],
            library=repos[op],kernel=m['kernel'],ca_us=ca,trace_sha256=digest,backend=family,
            correctness=checks,window='all-core last instruction end minus earliest instruction start',
            active_aic=m.get('active_aic'),active_aiv=m.get('active_aiv'),
            large_stress=workload(op,shape),origins=[])
    if item['shape'] != shape or abs(item['ca_us']-ca) > 0.002:
        raise ValueError(('digest identity/window mismatch', key,shape,ca,item['shape'],item['ca_us']))
    if attrs and item['attrs'] != attrs:
        raise ValueError(('attrs differ for same trace',key))
    origin=dict(receipt=repo_path(source),role=role)
    if origin not in item['origins']:
        item['origins'].append(origin)
    if contract:
        item['contract']=contract
    if trace:
        item['trace']=repo_path(trace)
    if wall is not None:
        item['wall_s']=wall
    stages = m.get('stages') or []
    binary=m.get('binary_sha256') or m.get('kernel_binary_sha256') or m.get('runtime_binary_sha256')
    if not binary and len(stages)==1:
        binary=stages[0].get('binary_sha256')
    if binary:
        item['binary_sha256']=normalized(binary)


def capture_history():
    for v in range(1,7):
        p=f'results/v0.{v}/dataset.json'
        for r in read(p)['records']:
            if r['status']=='ok':
                c=r['case']
                add(c['op'],c['shape'],DT,r['measurement'],p,c.get('split','archive'),wall=r.get('wall_seconds'))
    p='results/long-fa/evaluation.json'
    for r in read(p)['new_capture']:
        add('fa',r['shape'],DT,r.get('measurement') or {},p,r['role'],wall=r.get('wall_seconds'))
    for path in sorted((ROOT/'results/v0.8/core').glob('*.json')):
        p=repo_path(path);r=read(p)
        if isinstance(r, dict) and r.get('status')=='ok':
            add(r['op'],r['shape'],DT,{**r['measurement'],'correctness':r['correctness']},p,'v0.8 frozen',wall=r.get('wall_seconds'))
    # This archived aggregate resolves the v0.8 cube SQLite receipts whose hosts
    # logged the all-core span and the DB digest separately.
    p='results/v0.12/evaluation.json'
    for r in read(p)['rows']:
        if r['op'] not in OP:continue
        add(r['op'],r['shape'],DT,r,p,r['role'],r.get('attrs'))
    for f in read('results/v0.21/coverage/local-receipts.json'):
        r=f['observation']
        source_index[f['source']]=f['source_sha256']
        add(r['op'],r['shape'],r['chip'],r,f['source'],f['role'],r['attrs'],wall=r.get('wall_s'))
    for v, files in [(13,['pairs','holdout']),(14,['holdout']),(15,['holdout']),(16,['holdout'])]:
        for f in files:
            p=f'results/v0.{v}/sku/{f}.json'
            for r in read(p):
                if r['op'] not in OP:continue
                for side,chip in [('dt',DT),('pr',PR)]:
                    m=r.get(side) or {}
                    if m.get('correctness'):
                        add(r['op'],r['shape'],chip,m,p,'reused DT archive' if side=='dt' else ('calibration' if f=='pairs' else 'frozen PR holdout'),r.get('attrs'),r.get('contract'),(r.get('pr_cost') or {}).get('seconds') if side=='pr' else None)
    for v in (17,18,19):
        for f in (['records','final-records'] if v==19 else ['records']):
            p=f'results/v0.{v}/sku/{f}.json'
            for r in read(p):
                add(r['op'],r['shape'],r['chip'],r,p,r.get('role','historical'),r.get('attrs'),wall=(r.get('cost') or {}).get('seconds'))
    for side,chip in [('ca',DT),('ca-pr',PR)]:
        p=f'results/v0.20/matmul-prefill/{side}.json';r=read(p)
        if r['qualified']:
            add('matmul',[2048,32,5120],chip,{**r['stage'],'correctness':r['correctness']},p,
                'v0.20 freeze; PR elected after DT shape selection',wall=r['cost'].get('seconds'))
    # Committed extract is sufficient to replay the larger example without local raw runs.
    p='results/v0.20/examples/matmul-tile-pages.json';r=read(p)
    add('matmul',r['shape'],DT,{**r,'correctness':r['golden']},p,'dedicated frozen DT tile diagram',wall=r['ca_wall_seconds'])
    for f in ['fa-pr-32k-pages','fa-dt-128k-pages']:
        p=f'results/v0.20/examples/{f}.json';r=read(p)
        add('fa',r['shape'],r['sku'],{**r,'trace':r['merged_trace'],'trace_sha256':r['merged_sha256']},p,'page-level re-analysis of prior capture')
    # Some DT matched controls exist only in a side dictionary in an older release.
    for v in [15,16]:
        p=f'results/v0.{v}/sku/holdout.json'
        for r in read(p):
            d=r.get('dt') or {}
            if d.get('ca_us') and not d.get('correctness'):
                if not any(x['op']==r['op'] and x['chip']==DT and x['shape']==r['shape'] and abs(x['ca_us']-d['ca_us'])<.0001 for x in rows.values()):
                    pending.append(dict(op=r['op'],shape=r['shape'],source=p,linked_record=d.get('record'),value=d['ca_us'],reason='Matched DT summary omits trace or correctness; not included without its full receipt.'))


def score(rs):
    if not rs:return None
    worst=max(rs,key=lambda r:abs(r['signed_fraction']))
    return dict(windows=len(rs),shapes=len({tuple(r['shape']) for r in rs}),
        wape=sum(abs(r['us']-r['ca_us']) for r in rs)/sum(r['ca_us'] for r in rs),
        max_ape=abs(worst['signed_fraction']),worst_shape=worst['shape'],worst_ca_us=worst['ca_us'],worst_us=worst['us'])


def replay():
    gate=read('results/v0.21/coverage/gate-before-audit.json')['rows']
    for row in rows.values():
        op=row['op'];s=row['shape'];chip=row['chip']
        y=estimate(op,s,row['attrs'],chip=chip)
        assert y.latency_us and y.latency_us>0
        row.update(us=y.latency_us,signed_fraction=y.latency_us/row['ca_us']-1,
            future_us=estimate(op,s,row['attrs'],chip=FUTURE).latency_us)
        byhash=[];byshape=[]
        for g in gate:
            if g['op']!=op or g['chip']!=chip or g['shape']!=s:continue
            digest=normalized(g['trace_sha256'])
            if digest==row['trace_sha256']:
                byhash.append(g['cohort'])
            elif not digest and abs(g['ca_us']-row['ca_us'])<1e-5:
                byshape.append(g['cohort'])
        row['previous_gate']=sorted(set(byhash or byshape))
        row['previous_gate_match']='trace digest' if byhash else 'shape, SKU and same CA within 1e-5 us; no digest in gate' if byshape else None
        row.setdefault('contract',CONTRACTS[op])
        row['current_role']='historical replay'
    results=sorted(rows.values(),key=lambda r:(r['op'],r['chip'],r['shape'],r['trace_sha256']))
    metrics={}
    for op in OPS:
        for chip in (DT,PR):
            rr=[r for r in results if r['op']==op and r['chip']==chip]
            metrics[op,chip]={view:score(group) for view,group in [
                ('all',rr),('large',[r for r in rr if r['large_stress']]),('omitted',[r for r in rr if not r['previous_gate']]),
                ('omitted_large',[r for r in rr if r['large_stress'] and not r['previous_gate']])]}
    return results,{op:{chip:metrics[op,chip] for chip in (DT,PR)} for op in OPS}

CONTRACTS = {r['op']:r['contract'] for r in read('results/v0.14/sku/holdout.json')}


if __name__=='__main__':
    capture_history()
    out,summary=replay()
    (OUT/'audit.json').write_text(json.dumps(dict(contract_version=1,window='all-core instruction makespan (us)',
        score='Public estimate auto, absolute error summed across distinct trace SHA256. 970DT is extrapolation.',
        rows=out,metrics=summary,pending_links=pending,source_sha256=source_index),indent=2)+'\n')
    print(json.dumps(summary,indent=2))
