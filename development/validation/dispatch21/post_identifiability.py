"""Observed latency bounds for BF16 mHC Post under identical visible API inputs."""
from collections import defaultdict
import sqlite3
import json
from pathlib import Path
import statistics
import sys

ROOT=Path(__file__).resolve().parents[3]
sys.path.insert(0,str(ROOT))
from development.validation.sku19.campaign import DT,PR,read,write
sys.path.insert(0,str(ROOT/"development/validation/sku17"))
from campaign_collect import events

OUT=ROOT/'results/v0.21/dispatch/post-identifiability.json'


def good(correct):
    return correct and all('bad=0' in s for s in correct)


def input_state(prior,active):
    return 'first' if prior==0 else 'expand' if prior<active else 'reuse'


def numpy_row(row,file):
    shape=row['shape']; active=row['active_aiv']; prior=row['prior_aiv']
    assert good(row['correctness']) and row['active_aic']==0 and row['attrs']['has_h_res']
    return dict(chip=DT,shape=shape,active=active,backend='npusim compact',
                kernel=row['kernel'],binary=None,us=row['reference_us'],prior_aiv=prior,
                state=input_state(prior,active),scope='standalone Post; prior same-stream Post sequence',
                trace=row.get('db_source',row['source']),sha256=row.get('db_sha256'),
                source=str(file.relative_to(ROOT)),phases_us=row.get('phase_us'),counts=row.get('counts'))


def official_phases(stage):
    core=max(stage['windows'],key=lambda k:stage['windows'][k][1])
    f=ROOT/Path(stage['trace']).parent/core/'trace.json'
    t0=float('inf'); first_dma=float('inf'); vec=float('inf'); rd=wr=0; n=0
    for ev in events(f):
        if ev.get('ph')!='X':
            continue
        t0=min(t0,ev['ts'])
        if ev['dur']<=0:
            continue
        info=json.loads(ev['args']['detail'])
        if info.get('is_wait') or info.get('is_sync') or info.get('is_duration_excluded'):
            continue
        q=info['queue_type']
        if q=='SIMD' and ev['name'].startswith('RV_'):
            vec=min(vec,ev['ts'])
        if q in ('MTE2','MTE3') and ev['name'].startswith('MOV_SRC_TO_DST'):
            b=int(info['processed_bytes'])
            if q=='MTE2':
                rd+=b
                if b>=256:
                    first_dma=min(first_dma,ev['ts'])
            else:
                wr+=b
        n+=1
    origin=t0-stage['windows'][core][0]
    return dict(terminal=core,first_payload_mte2_ge_256B_us=first_dma-origin if first_dma<float('inf') else None,
                first_simd_rv_us=vec-origin if vec<float('inf') else None,
                terminal_dma_read_bytes=rd,terminal_dma_write_bytes=wr,positive_nonwait_records=n)


def collect():
    base=ROOT/'results/v0.12/post/rows.json'
    rows=[numpy_row(r,base) for r in read(base)]
    controls=[ROOT/'results/v0.10/wide/paired-controls.json',
              *(ROOT/'results/v0.11/post').glob('retry-v11_mhc_post_*.json'),
              *(ROOT/'results/v0.12/post').glob('batch-v12_post_*.json')]
    for f in sorted(controls):
        data=read(f)
        for r in data if isinstance(data,list) else data['completed']:
            if r['op']=='mhc_post' and r['role'] in ('paired_control','context_warmup'):
                if 'attrs' not in r:
                    r['attrs']=dict(has_h_res=True,dtype='bf16')  # same mandatory-residual runner
                if 'active_aic' not in r:
                    r['active_aic']=0
                rows.append(numpy_row(r,f))
    registry=ROOT/'results/v0.17/sku/records.json'
    for r in read(registry):
        if r['op'] not in ('mhc','mhc_post'):
            continue
        chain=r['op']=='mhc'
        assert good(r['correctness'])
        if 'stages' not in r:
            if not chain:
                # Previously qualified DT compact records already present in v0.12.
                assert any(x['chip']==DT and x['shape']==r['shape'] and abs(x['us']-r['ca_us'])<1e-6 for x in rows)
                continue
            assert r['source_hash_match'] and r['chip']==DT
            assert r['api'].endswith('aclnnMhcPost') and len(r['trace'])==3
            path=ROOT/r['trace'][-1]
            with sqlite3.connect(path.resolve().as_uri()+'?mode=ro',uri=True) as conn:
                info=dict(conn.execute('SELECT KernelInfoAttr,KernelInfoVal FROM KernelInfo'))
                n,a,b=conn.execute('SELECT COUNT(*),MIN(ExecInstrTickStart),MAX(ExecInstrTickEnd) FROM ExecutedInstructions').fetchone()
                assert info['name']==r['kernel'].split('+')[-1] and int(info['launch_id'])==2
                assert n==int(info['instrs_exec_num']) and b-a==int(info['exec_time_ticks'])
                assert float(info['frequency_ghz'])==1.65 and int(info['ai_core_active'])==r['active_aiv_per_launch'][-1]
            rows.append(dict(chip=DT,shape=r['shape'],active=r['active_aiv_per_launch'][-1],backend='npusim compact',
                             kernel=info['name'],binary=None,us=(b-a)/1650,
                             prior_aiv=None,state='after_pre_sinkhorn',scope='serialized producer chain',
                             prior_stages_aiv=r['active_aiv_per_launch'][:-1],trace=r['trace'][-1],
                             sha256=r['trace_sha256'][-1],source=str(registry.relative_to(ROOT))))
        else:
            ks=r['stages'];k=ks[-1]
            assert k['kernel'].startswith('MhcPost_') and k['active_aic']==0
            assert len(ks)==(3 if chain else 1)
            rows.append(dict(chip=r['chip'],shape=r['shape'],active=k['active_aiv'],backend='msopprof official',
                             kernel=k['kernel'],binary=k['binary_sha256'],us=k['ca_us'],
                             prior_aiv=None if chain else 0,state='after_pre_sinkhorn' if chain else 'first',
                             scope='serialized producer chain' if chain else 'isolated first Post',
                             prior_stages_aiv=[a['active_aiv'] for a in ks[:-1]],trace=k['trace'],
                             sha256=k['trace_sha256'],source=str(registry.relative_to(ROOT)),stage=k))
    seen=set()
    for r in rows:
        p=r['trace']
        assert p not in seen,p
        seen.add(p)
    return rows


def bounds(rows):
    ts=[x['us'] for x in rows]
    a,b=min(ts),max(ts)
    med=statistics.median(ts)
    return dict(min_us=a,max_us=b,minimax_point_us=2*a*b/(a+b),min_possible_max_pct=100*(b-a)/(a+b),
                min_possible_wape_pct_equal_launch_weight=100*sum(abs(x-med) for x in ts)/sum(ts),
                min_wape_point_us=med)


def audit():
    records=collect()
    grouped=defaultdict(list)
    for r in records:
        grouped[r['chip'],tuple(r['shape']),r['backend'],r['kernel'],r['binary'],r['active']].append(r)
    groups=[]
    for (chip,shape,backend,kernel,binary,cores),rows in sorted(grouped.items()):
        if len({r['state'] for r in rows})<2:
            continue
        for r in rows:
            if 'stage' in r:
                r['terminal_phase_evidence']=official_phases(r.pop('stage'))
        groups.append(dict(chip=chip,shape=list(shape),backend=backend,kernel=kernel,binary_sha256=binary,
                           active_aiv=cores,bounds=bounds(rows),samples=rows))
    stats=defaultdict(lambda:defaultdict(int))
    for r in records:
        stats[r['chip']][r['state']]+=1
    write(OUT,dict(scope='BF16 has_h_res=True N=4, complete windows. Grouping requires same SKU, shape, active count, backend and kernel label. Official comparisons match binary SHA256; npusim exposes entry suffix without an object hash.',
          mathematical_bound='For identical visible inputs with observed endpoints a,b>0, max(|p/a-1|,|p/b-1|) minimized at 2ab/(a+b); minimum (b-a)/(b+a). Median minimizes sum absolute errors. Per-group bounds do not imply a bound of the same size on whole-dataset WAPE.',
          state_evidence='first/expand/reuse for Post-only batches use prior dispatch AIV high-water mark. An earlier Pre/Sinkhorn in the same serialized invocation is a different condition: data producers, launch preparation and prior active cores differ. No direct cache residency measurement.',
          pr_repeated_post_evidence='v0.17 standalone PR is exclusively first dispatch, and chain samples follow Pre and Sinkhorn. No consecutive equal-shape Post-only first/expand/reuse sequence on PR in the audited sources.',
          host_contract='smoke/extension_indexer_mhc.cpp:128-131,178,201: same dopost() call and four operands for standalone and chain. Chain feeds producer outputs; contents/addresses can differ.',
          sample_counts=stats,groups=groups,
          eligible_independent_use=False,reason='retrospective identifiability audit; never a fitted shape table'))
    for g in groups:
        print(g['chip'],g['shape'],g['backend'],round(g['bounds']['min_possible_max_pct'],2),[(r['state'],round(r['us'],2)) for r in g['samples']])

if __name__=='__main__':
    audit()
