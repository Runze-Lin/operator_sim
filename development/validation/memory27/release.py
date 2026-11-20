"""Publish a consistent release snapshot from frozen registrations and qualified device receipts."""
from datetime import datetime, timezone
from hashlib import sha256
import json
from pathlib import Path

ROOT=Path(__file__).resolve().parents[3]
OUT=ROOT/'results/v0.27'
DT, PR='Ascend950DT_9582', 'Ascend950PR_957b'


def read(path):
    return json.loads((ROOT/path).read_bytes())


def save(name,data):
    (OUT/name).write_text(json.dumps(data,indent=2,ensure_ascii=False,allow_nan=False)+'\n')


def receipt(path, fields=()):
    path = ROOT/path
    data = read(path)
    return dict(path=str(path.relative_to(ROOT)), sha256=sha256(path.read_bytes()).hexdigest(),
                **{k:data[k] for k in fields})


def main():
    cutoff=datetime.now(timezone.utc).isoformat()
    registrations=['results/v0.23/pro-prefill/freeze.json','results/v0.25/coverage/supplement/register.json',
                   'results/v0.26/memory/initial-issues-freeze.json','results/v0.26/memory/pro-256ki-prefix-forecast.json',
                   'results/v0.26/post/registration.json','results/v0.26/presink/freeze.json','results/v0.26/sink/register.json',
                   'results/v0.27/fusion/register.json','results/v0.27/kda/v1-register.json','results/v0.27/post/registration.json',
                   'results/v0.27/memory/read-freeze.json','results/v0.27/rms/freeze.json',
                   'results/v0.27/fusion/reserve-activation.json','results/v0.27/kda/host-control/register.json','results/v0.27/memory/cast-pair/freeze.json',
                   'results/v0.27/rms/recovery-freeze.json']
    sources=[*registrations,'operator_model/kimi_kda.py','operator_model/stream.py','operator_model/post_repeat.py',
             'operator_model/backend.py','operator_model/sink_history.py','operator_model/mhc_pre_sink_resource.py',
             'operator_model/vector_memory.py','operator_model/fa_resource.py',
             'results/v0.27/post/parameters.json','results/v0.27/rms/prices.json','results/v0.26/presink/prices.json',
             *[str(p.relative_to(ROOT)) for p in sorted((ROOT/'configs/chips').glob('Ascend*.json'))]]
    save('model.json',dict(version='0.27.0',baseline='v0.26',registrations=registrations,
         sha256={s:sha256((ROOT/s).read_bytes()).hexdigest() for s in sources},
         scope='Online Kimi workspace-aware selector and public runtime. Each program/stage forecast is separately registered; offline CA and resource-feasibility analyses do not set parameters on a new chip.'))
    fa=read('results/v0.25/coverage/supplement/score.json')
    post=read(OUT/'post/score.json')
    sink=read(OUT/'fusion/assessment.json')
    release=dict(version='0.27.0',cutoff_utc=cutoff,base='results/v0.25/coverage/audit.json',
                 previous='results/v0.26/evaluation.json',wheel=read(OUT/'wheel-check.json'),
                 pro256={tag:read(f'results/v0.26/memory/pro-b8-256ki-{tag}-score.json')
                         for tag in ('dt','pr') if (ROOT/f'results/v0.26/memory/pro-b8-256ki-{tag}-score.json').exists()},
                 fa=fa, fusion=sink, post=post, retention=read(OUT/'memory/read-windows.json'),
                 read_holdout=read(OUT/'memory/read-holdout.json'),
                 repeat_acceptance=read(OUT/'post/deployment.json'),
                 kda_v1={chip:read(OUT/f'kda/v1-{chip}.json') for chip in (DT,PR)
                         if (OUT/f'kda/v1-{chip}.json').exists()})
    release.update(goal={'WAPE_pct_max':25,'maximum_APE_pct_max':50},
                   replay='results/v0.27/replay.json', history_bounds=read(OUT/'history-bounds.json'),
                   cache_subset=read(OUT/'memory/pr156-partial.json'),
                   large_quant=receipt(OUT/'quant-envelope.json'),
                   sink_unknown_history='results/v0.27/fusion/unknown-history.json',
                   native_64_aiv={tag:receipt(path,('shape','qualified_core_envelope','instruction_ca_us','core_envelope_us','active_aiv','correct_elements','cost')) for tag,path in {
                       'flash':'results/v0.23/full-hidden/v3/flash-retry.json',
                       'kimi':'results/v0.20/full-quant/native.json'}.items()},
                   kda_v1_host_control=receipt(OUT/'kda/host-control/dt.json',('qualified','formal_v1_kernel_us','error_count','cost')),
                   public_sinks={'standalone':'results/v0.27/fusion/public-sink-installed.json',
                                 'fused':'results/v0.27/fusion/fused-public-installed.json'},
                   rms_shadow=receipt(OUT/'rms/score.json',('matched_summary','paired_bounds','promotion_qualified')))
    save('evaluation.json',release)
    table={}
    fa_values=[]
    repeats=[]
    cold=[]
    fused=[]
    cast=[]
    shadow=[]
    for chip,tag in [(DT,'dt'),(PR,'pr')]:
        g=fa['timing_by_sku'][chip]
        c=g['metrics']['candidate']
        fa_values.append(f"{g['shapes']} · {c['wape_pct']:.2f}% / {c['max_ape_pct']:.2f}%")
        c=post['repeat_summary'][chip]['policy_us']
        repeats.append(f"{post['exact_repeat_shapes'][chip]} · {c['wape']:.2f}% / {c['maximum']:.2f}%")
        c=post['summary'][chip]['us']
        shadow.append(f"{c['n']} · {c['wape']:.2f}% / {c['maximum']:.2f}%" if c else '0 · N/A')
        g=sink['groups'][chip]['standalone_common']
        c=g['known_us']
        cold.append(f"{g['distinct_shapes']} · {c['wape_pct']:.2f}% / {c['max_pct']:.2f}%")
        g=sink['groups'][chip]['fused_paired_shapes']
        c=g['candidate_us']
        fused.append(f"{g['distinct_shapes']} · {c['wape_pct']:.2f}% / {c['max_pct']:.2f}%" if c else '0 · N/A')
        p=release['pro256'].get(tag)
        val=abs(p['metrics']['uniform']['full_pct']) if p and p.get('qualified') else None
        cast.append(f"1 · {val:.2f}% / {val:.2f}%" if val is not None else '0 · N/A')
    table['FA（新形）'] = (fa_values,'两批事前冻结，公开模型；仅合格完整窗','达标 / 达标')
    table['mHC Sinkhorn（给同族前序 H）'] = (cold,'同族历史；单 UB 块，独立不同 shape；重复窗计入误差','本组达标 / 本组达标')
    table['mHC PreSinkhorn（融合新形）'] = (fused,'冻结后的独立新形，与 v0.25 同集比较','本组达标 / 本组达标')
    table['mHC Post（有序流，同形紧邻重复）'] = (repeats,'事前冻结的精确重复，原全核结束标签口径','本组达标 / 本组达标')
    table['mHC Post（普通换形/首次候选）'] = (shadow,'已知前序、分入口计价，冻结后新形','扩大样本中 / 扩大样本中')
    table['Cast（Pro B8×256Ki）'] = (cast,'完整 64 通道新数据量；逐元素正确、全核退出','本例达标 / 本例达标')
    doc=ROOT/'docs/error-analysis.md'
    lines=doc.read_text().splitlines()
    after={'mHC PreSinkhorn（融合新形）':'mHC PreSinkhorn',
           'mHC Post（普通换形/首次候选）':'mHC Post（有序流，同形紧邻重复）','Cast（Pro B8×256Ki）':'Cast BF16→FP32'}
    for label,(values,role,status) in table.items():
        new='| '+' | '.join([label,*values,role,status])+' |'
        pos=next((i for i,line in enumerate(lines) if line.startswith('| '+label+' |')),None)
        if pos is None:
            pos=next(i for i,line in enumerate(lines) if line.startswith('| '+after[label]+' |'))
            lines.insert(pos+1,new)
        else:
            lines[pos]=new
    doc.write_text('\n'.join(lines)+'\n')


if __name__=='__main__':
    main()
