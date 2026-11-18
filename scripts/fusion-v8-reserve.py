"""Pre-freeze substitute validation shapes for the same model and collection budgets."""
from pathlib import Path
import json,sys,hashlib,datetime,concurrent.futures,importlib.util
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from operator_model.fusion_v8 import predict,PRE,FUSED,IDX,FIRST,SECOND,CHAIN
OUT=ROOT/'results/v0.8/fusion'
if __name__=='__main__':
    original=json.loads((OUT/'freeze.json').read_text());model=json.loads((OUT/'model.json').read_text());path=OUT/'reserve.json'
    if not path.exists():
        cases=[]
        for op,shapes in [(IDX,[(1,q,l,8,128) for q,l in [(1,513),(1,576),(2,600),(3,576),(4,528),(1,704),(2,896),(1,1152)]]),
            (FUSED,[(t,4,4096) for t in (10,11,13,14,15,17,18,19)]),
            ('mhc',[(t,4,d) for t,d in ((1,80),(2,96),(1,112),(2,160),(3,96),(4,64),(1,192),(1,224))]),
            (CHAIN,[(m,k,f,len(counts)) for m,k,f,counts in [(2,64,64,[1,1]),(3,128,64,[3]),(4,64,64,[4]),(5,64,64,[1,4]),(6,64,128,[6]),(7,64,64,[7]),(2,128,128,[2]),(5,128,64,[1,4])]])]:
            for i,shape in enumerate(shapes):
                attrs=model['contracts'][op].copy();orig=next(c for c in original['cases'] if c['op']==op);c=dict(orig,shape=shape,attrs=attrs,id=op+'-reserve'+str(i+1))
                if op==CHAIN:
                    attrs['expert_counts']=[[1,1],[3],[4],[1,4],[6],[7],[2],[1,4]][i]
                    c['arguments']=[op,*shape,','.join(map(str,attrs['expert_counts']))]
                elif op==IDX:c['arguments']=[op,*shape[1:4]]
                else:c['arguments']=[op,*shape]
                c['predictions']={p:predict(p,shape,model['contracts'][PRE] if p==PRE else attrs,model)['predicted_us'] for p in c['components']}
                assert not any(list(shape)==x['shape'] for x in original['cases'] if x['op']==op)
                if not any(list(shape)==s['shape'] for s in model['templates'][FIRST if op==CHAIN else PRE if op=='mhc' else op]):cases.append(c)
        path.write_text(json.dumps(dict(original,utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),cases=cases,reason='Budget/failure reserves; replay fixed shape order only until 10 qualified distinct shapes per operator.'),indent=2)+'\n')
    if '--run' in sys.argv:
        spec=importlib.util.spec_from_file_location('collector',ROOT/'scripts/fusion-v8-collect-isolated.py')
        collector=importlib.util.module_from_spec(spec);spec.loader.exec_module(collector);freeze=json.loads(path.read_text())
        for p,d in freeze['sha256'].items():
            assert hashlib.sha256((ROOT/p).read_bytes()).hexdigest()==d
        for c in freeze['cases']:
            rows=json.loads((OUT/'rows.json').read_text())
            if sum(r['op']==c['op'] and r['role']=='validation' for r in rows)>=10:continue
            collector.run(c,freeze)
            import subprocess
            subprocess.run([sys.executable,str(ROOT/'scripts/fusion-v8-report.py')],check=True)
