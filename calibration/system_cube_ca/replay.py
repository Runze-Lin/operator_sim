"""Audit the matrix contracts from real rank-0 capture -> OpFeatures -> full backend."""
import argparse
from collections import Counter
import csv
import gzip
import json
from pathlib import Path
from types import SimpleNamespace

from sim_platform.backend.prediction import convert_node_features
from sim_platform.frontend.inference import _attributes
from sim_platform.graph.ir import NodeRef, OperatorNode, TensorDescriptor

from operator_model.chips import load_chip
from operator_model.system_routes import estimate_full_features
from operator_model.system_routes.cube import NAMES


def descriptor(t):
    physical = tuple(t.get('storage_shape',t['shape']))
    logical = tuple(t['shape']) if 'storage_shape' in t else None
    return TensorDescriptor(str(t['id']),physical,t['dtype'],'meta',tuple(t['stride']),
                            t['storage_offset'],t['layout'],logical_shape=logical)


def replay(root):
    selected = [r['case'] for r in csv.DictReader((root/'summary.csv').open(encoding='utf-8-sig'))
                if r['history_tokens']=='131072']
    chip = load_chip()
    seen, output = {}, {}
    for case in selected:
        stages = {}
        for stage in ('target','draft'):
            with gzip.open(root/case/'native'/f'000_rank0_{stage}.json.gz') as source:
                graph = json.load(source)
            tensors = graph['tensors']
            tallies=Counter()
            for node in graph['nodes']:
                name = node['name']
                if name not in NAMES:
                    continue
                ins = tuple(descriptor(tensors[i]) for i in node['inputs'])
                outs = tuple(descriptor(tensors[i]) for i in node['outputs'])
                attrs = _attributes(SimpleNamespace(**node),ins,outs,rank=graph['rank'])
                ir = OperatorNode(NodeRef(case,str(node['id'])),name,name,ins,outs,attrs,category='cube')
                features = convert_node_features(ir)
                # Module path and partition labels don't change tensor roles or a local product.
                canonical = dict(features.attributes)
                sem = dict(canonical.get('semantic',()))
                for ignored in ('component','partition'):
                    sem.pop(ignored,None)
                canonical['semantic'] = tuple(sorted(sem.items()))
                key = (name,features.inputs,features.outputs,tuple(sorted(canonical.items())))
                if key not in seen:
                    r=estimate_full_features(features,chip=chip)
                    if r.latency_us is None:
                        raise ValueError(f'{case}/{stage}/{name}: {r.reason}')
                    seen[key]=r.contract
                tallies[seen[key]]+=1
            stages[stage]=dict(sorted(tallies.items()))
        output[case]=stages
    return dict(sku=chip.name,configs=len(selected),captures=2*len(selected),cases=output,
                unique_normalized_contracts=dict(sorted(Counter(seen.values()).items())))


if __name__=='__main__':
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('root',type=Path);ap.add_argument('output',type=Path)
    a=ap.parse_args()
    a.output.write_text(json.dumps(replay(a.root),indent=2)+'\n')
