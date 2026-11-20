"""Metadata-only replay of decision functions extracted verbatim from the pinned release."""
import argparse
import ast
import builtins
import hashlib
from itertools import accumulate
import json
from pathlib import Path
import statistics
import sys
import time
from types import SimpleNamespace
import zipfile

ROOT=Path(__file__).resolve().parents[3]
OUT=ROOT/'results/v0.27/kda'
LOCAL=ROOT/'results/chunk-recovery/fla2691/raw'
sys.path.insert(0,str(ROOT))
from operator_model.kimi_kda import plan_kimi_kda_prefill


def official_metadata_checker():
    wheel=LOCAL/'fla-a5.whl'
    with zipfile.ZipFile(wheel) as z:
        [member]=[p for p in z.namelist() if p.endswith('/ops/ascendc/_aclnn_ctypes.py')]
        source=z.read(member)
    tree=ast.parse(source)
    names={'_kda_ceil_div','_kda_total_chunks','_chunk_kda_fwd_use_v2'}
    defs=[n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name in names]
    [constant]=[n for n in tree.body if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='_CHUNK_KDA_FWD_V2_MIN_WORK_ITEMS' for t in n.targets)]
    assert len(defs)==3 and ast.literal_eval(constant.value)==4096
    # The actual selector needs only a dtype identity; no Torch operators execute.
    torch=SimpleNamespace(bfloat16=object());imports=dict(vars(builtins));imports['__import__']=lambda name,*args: torch if name=='torch' else builtins.__import__(name,*args)
    ns={'__builtins__':imports};exec(compile(ast.Module(body=[constant,*defs],type_ignores=[]),member,'exec'),ns)
    def check(lens,heads):
        lengths=[n for n in lens if n];cu=tuple(accumulate(lengths,initial=0))
        pairs=tuple(z for i,s in enumerate(lengths) for j in range((s+63)//64) for z in (i,j))
        w=heads*ns['_kda_total_chunks'](1,cu[-1],64,cu,pairs)
        selected=ns['_chunk_kda_fwd_use_v2'](dtype=torch.bfloat16,k_dim=128,v_dim=128,chunk_size=64,cu=cu,work_items=w,force_v2=False)
        return w, 'aclnnChunkKdaFwdV2' if selected else 'aclnnChunkKdaFwd'
    return check,dict(wheel_sha256=hashlib.sha256(wheel.read_bytes()).hexdigest(),member=member,sha256=hashlib.sha256(source).hexdigest(),functions=sorted(names),constant=constant.targets[0].id,stable_cpp_sha256=hashlib.sha256((LOCAL/'stable_chunk_kda_fwd.cpp').read_bytes()).hexdigest())


def main(api_json=None):
    check,source=official_metadata_checker()
    releasefile=OUT/'release-check.json'
    if api_json:
        with open(api_json,'rb') as f: data=f.read()
        source_url='https://api.github.com/repos/flashserve/flash-linear-attention-npu/releases?per_page=10'
        packages=json.loads((ROOT/'results/v0.25/kda-v1/packages.json').read_bytes());known={v['asset']['digest'] for v in packages['versions'].values()}|{packages['reference_kimi_wheel']['digest']}
        releases=[dict(tag=r['tag_name'],published_at=r['published_at'],assets=[dict(name=a['name'],sha256=a['digest'],previously_audited=a['digest'] in known) for a in r['assets'] if '_a5-' in a['name'] and 'aarch64' in a['name']]) for r in json.loads(data)]
        from datetime import datetime, timezone
        releasefile.write_text(json.dumps(dict(checked_utc=datetime.now(timezone.utc).isoformat(),source=source_url,sha256=hashlib.sha256(data).hexdigest(),releases=releases),indent=2)+'\n')
    cases=[('single', [2],96),('two-sequences',[1,1],96),('W4095',[4095*64],96),('W4096',[4096*64],96),('same-tokens',[21824],8),('ragged',[1,0,21823],8),('empty-slots',[0,32768,0,32768],8)]
    cases.extend((f'B8x{n}',[n]*8,8) for n in (512,2048,4096,32768,131072,262144,524288))
    cases.extend((f'TP{tp}',[2048]*8,tp) for tp in (1,2,4,16,32))
    rows=[];latencies=[]
    for tag,lens,tp in cases:
        w,api=check(lens,96//tp);prices={}
        for soc in ('Ascend950DT_9582','Ascend950PR_957b','Ascend970DT'):
            durations=[]
            for _ in range(7):
                t=time.perf_counter_ns();plan=plan_kimi_kda_prefill(lens,attention_tp=tp,chip=soc);durations.append((time.perf_counter_ns()-t)*1e-6)
            assert plan['work_items']==w and plan['aclnn']==api
            prices[soc]=dict(plan=plan,median_query_ms=statistics.median(durations),max_query_ms=max(durations));latencies.extend(durations)
        rows.append(dict(tag=tag,lengths=lens,tp=tp,w=w,official_api=api,targets=prices))
    (OUT/'dispatch.json').write_text(json.dumps(dict(qualification='source and metadata decisions, no production end-to-end CA in these rows',source=source,query_measurement='in-process warm + initial call; 7 calls per row per chip; no farm coordination',max_query_ms=max(latencies),rows=rows),indent=2)+'\n')
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--releases-json');a=p.parse_args();main(a.releases_json)
