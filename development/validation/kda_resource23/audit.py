"""Reprice archived releases, keep diagnostic-only descriptors outside latency error."""
import argparse
from collections import Counter
from hashlib import sha256
import json
from pathlib import Path
from statistics import median
from time import perf_counter_ns
from model import DT,FAMILIES,OUT,ROOT,predict,read

def save(name,obj):
    (OUT/name).write_text(json.dumps(obj,indent=2)+'\n')

def observations():
    rows=[]
    for path in sorted((ROOT/'results/kda-v23').glob('*ascend950*-t*-h*.json')):
        if not path.name.startswith(('ascend950','reused-ascend950','development-ascend950')):
            continue
        rec=read(path)
        if not rec.get('exact_stage_elfs') or not rec.get('full_windows'):
            continue
        case=rec['case']
        pred=predict(case['shape'],case['sku'])
        row=dict(case=case,receipt=str(path.relative_to(ROOT)),qualified=rec['qualified'],
                 role='development replay' if rec['qualified'] else 'mechanism diagnostic only',
                 full_window_us=rec.get('main_stages_us'),predict_us=pred['serialized_device_us'],stages=[])
        for pr,ca in zip(pred['stages'],[s for s in rec['stages'] if s['family']]):
            assert pr['family']==ca['family']
            resource=ca['resource']
            mm=Counter()
            for sig,num in pr['mmads'].items():mm[sig.split(':')[0]]+=num
            observed=resource['queue_instruction_counts']
            rec_v={k:v for k,v in observed.items() if k.startswith('RVEC')}
            changes={k:[v,resource['descriptor_bytes'].get(k,0)] for k,v in pr['dma_bytes'].items()
                     if v!=resource['descriptor_bytes'].get(k,0)}
            mismatched=sorted(set(resource['descriptor_bytes'])-set(pr['dma_bytes']))
            row['stages'].append(dict(family=ca['family'],ca_us=ca['ca_us'],predict_us=pr['device_us'],
                predicted_aic_aiv=[pr['active_aic'],pr['launched_aiv']],observed_aic_aiv=[ca['active_aic'],ca['active_aiv']],
                actual_mmad_counts=resource['mmad_issued_shapes'],modeled_mmad_counts=mm,
                actual_ops=resource['issued_mmad_ops'],modeled_ops=pr['issued_mmad_ops'],
                actual_dma_bytes=resource['descriptor_bytes'],modeled_dma_bytes=pr['dma_bytes'],
                actual_rvec=rec_v,modeled_rvec=pr['estimated_rvec_subqueue_records'],
                equal_mmad=dict(mm)==resource['mmad_issued_shapes'],equal_dma=not changes and not mismatched,
                different_dma=changes,unexpected_paths=mismatched))
        rows.append(row)
    return rows

def benchmark():
    rows=[]
    for chip in (DT,'Ascend950PR_957b','Ascend970DT'):
        for length in (32768,131072,262144):
            shape=[8,length,12,12,128,128]
            calls=[]
            for _ in range(30):
                tic=perf_counter_ns();pred=predict(shape,chip);calls.append((perf_counter_ns()-tic)/1e6)
            rows.append(dict(shape=shape,chip=chip,device_us=pred['serialized_device_us'],stages=pred['stages'],
                             median_query_ms=median(calls),max_query_ms=max(calls)))
    return rows

def provenance():
    tree=read(ROOT/'results/chunk-recovery/fla2691/raw/fla-source-tree.json')['tree']
    parents={'prepare':'kda/chunk_kda_fwd_prepare','finalize':'kda/chunk_kda_fwd_finalize','recur':'gdn/chunk_gdn_fwd/chunk_fwd_h'}
    names={'recur':'chunk_fwd_h','prepare':'chunk_kda_fwd_prepare','finalize':'chunk_kda_fwd_finalize'}
    src_root=ROOT/'results/chunk-recovery/fla2691/raw/fla_npu/opp/vendors/fla_npu_transformer/op_impl/ai_core/tbe/fla_npu_transformer_impl/ascendc'
    rows=[]
    for key,upstream in parents.items():
        name=names[key]
        for suffix in ('_tiling.cpp','_tiling_processor.h'):
            path=f'fla/ops/ascendc/{upstream}/op_host/{name}{suffix}'
            entry=next(t for t in tree if t['path']==path)
            rows.append(dict(authority='pinned official host release commit',file=path,git_blob=entry['sha'],
                         url='https://github.com/flashserve/flash-linear-attention-npu/blob/5cc699069c7f91a35aabd583032c8720a0b626fa/'+path))
        for p in sorted((src_root/name).rglob('*')):
            if not p.is_file() or 'arch22' in p.parts:
                continue
            rows.append(dict(authority='official A5 26.9.1 release wheel',file=str(p.relative_to(ROOT)),sha256=sha256(p.read_bytes()).hexdigest()))
    return rows

def check_freeze():
    reg=read(OUT/'freeze.json')
    for path,expected in reg['sha256'].items():
        assert sha256((ROOT/path).read_bytes()).hexdigest()==expected,path
    for row in reg['predictions']:
        assert predict(row['shape'],row['chip'])==row
    return reg

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--shape',type=int,nargs=6);parser.add_argument('--chip',default=DT)
    args=parser.parse_args()
    if args.shape:
        print(json.dumps(predict(args.shape,args.chip),indent=2))
    else:
        check_freeze()
        save('development-check.json',observations())
        save('prefill-projections.json',benchmark())
        save('source-evidence.json',provenance())
