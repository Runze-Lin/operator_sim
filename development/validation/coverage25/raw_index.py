"""Save otherwise untracked early DB receipts; check DB SHA, host checks and exact stage sum."""
from hashlib import sha256
from pathlib import Path
import json,re
from audit import ROOT, OUT, DT,EXCLUDED


def scan():
    result=[];unusable=[]
    for folder in ('v8cube','v8cube-isolated','v8cv','v9addmm','v9mhc','model-v7'):
        for f in sorted((ROOT/'runs'/folder).glob('*/result.json')):
            cache={};data=f.read_bytes();j=json.loads(data);case=j.get('case') or j
            op=case.get('op') or (case['arguments'][0] if case.get('arguments') else None)
            if not op or op in EXCLUDED:continue
            m=j.get('measurement') or j;ks=m.get('kernels') or [];checks=m.get('correctness') or j.get('correctness') or []
            if j.get('error') or j.get('returncode',0) not in (0,None) or j.get('status','ok')!='ok' or not ks or not checks or any(not re.search(r'\bbad=0\b',c) for c in checks):
                unusable.append(dict(source=str(f.relative_to(ROOT)),op=op,shape=case.get('shape'),reason=str(j.get('error') or j.get('status') or 'no complete DB or goldens')[:300]));continue
            paths=list(f.parent.glob('**/record/*.db'));by={}
            for p in paths:
                if p.stat().st_size in {k.get('db_bytes') for k in ks}:
                    by[sha256(p.read_bytes()).hexdigest()]=p
            ids=[k['db_sha256'] for k in ks]
            if any(k not in by for k in ids) or abs(sum(k['reference_us'] for k in ks)-m['reference_us'])>.002:
                unusable.append(dict(source=str(f.relative_to(ROOT)),op=op,shape=case.get('shape'),reason='source retained but raw DB or additive scope unavailable'));continue
            result.append(dict(op=op,shape=case['shape'],attrs=case.get('attrs',{}),chip=DT,kernel=[k['kernel'] for k in ks],
                        ca_us=m['reference_us'],trace_sha256=ids,raw=[str(by[k].relative_to(ROOT)) for k in ids],
                        source=str(f.relative_to(ROOT)),source_sha256=sha256(data).hexdigest(),correctness=checks))
    return dict(accepted=result,other=unusable)

if __name__=='__main__':
    (OUT/'early-db-receipts.json').write_text(json.dumps(scan(),indent=2)+'\n')
