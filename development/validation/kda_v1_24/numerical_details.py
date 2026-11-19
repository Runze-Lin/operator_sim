"""Read back the exact saved V1 device arrays and compare every coordinate."""
import gzip,hashlib,json
from pathlib import Path
import numpy as np
from investigate import REG, OUT, RAW, read, save


def digest(data):
    return hashlib.sha256(data).hexdigest()


def replay(case):
    ident=case['id'];p=RAW/ident;receipt=OUT/(ident+'.json')
    if not receipt.exists() or read(receipt).get('skipped'):return
    current=read(receipt)
    t,h=case['shape'][1:3];rows={}
    for name,shape in {'attention':(t,h,128),'aqk':(h,t,64),'akk':(h,t,64),'state_vk':(h,128,128)}.items():
        bf=name!='state_vk';file=p/(name+('.bf16' if bf else '.f32'))
        if not file.exists():
            rows[name]=dict(captured=False);continue
        content=file.read_bytes()
        expected_path=RAW/'oracles'/f'{t}-{h}'/(name+'.bin')
        expected=np.fromfile(expected_path,dtype='<f8').reshape(shape)
        actual=(np.frombuffer(content,dtype='<u2').astype('<u4')<<16).view('<f4') if bf else np.frombuffer(content,dtype='<f4')
        actual=actual.astype('f8').reshape(shape);err=np.abs(actual-expected);finite=np.isfinite(actual)
        atol,rtol,hard=(2e-6,2**-6,1) if bf else (2**-16,2**-10,.01)
        bad=~finite|(err>atol+rtol*np.abs(expected))
        dst=OUT/'arrays'/(ident+'-'+file.name+'.gz')
        dst.parent.mkdir(exist_ok=True)
        packed=gzip.compress(content,compresslevel=9,mtime=0);dst.write_bytes(packed)
        locs=np.flatnonzero(~finite)
        if locs.size:idx=int(locs[0])
        else:idx=int(err.argmax())
        loc=tuple(int(i) for i in np.unravel_index(idx,shape))
        x=float(actual[loc]);y=float(expected[loc])
        output=current['output_groups'].get(name)
        if output and (output['bad']!=int(bad.sum()) or output['count']!=int(actual.size)):
            raise ValueError('saved device arrays differ from frozen host group: '+ident+' '+name)
        rows[name]=dict(captured=True,shape=shape,count=int(actual.size),bad=int(bad.sum()),nonfinite=int((~finite).sum()),
            max_finite_abs=float(err[finite].max()) if finite.any() else None,example_index=loc,
            example_actual=x if np.isfinite(x) else str(x),example_expected=y,
            actual_sha256=digest(content),expected_sha256=digest(expected_path.read_bytes()),archived_sha256=digest(packed),
            archived=dst.relative_to(OUT).as_posix(),atol=atol,rtol=rtol,hard_limit=hard)
    save(OUT/(ident+'-arrays.json'),dict(case=ident,role='raw outputs reproduced from frozen release reference',outputs=rows))

if __name__=='__main__':
    for c in read(REG)['cases']:replay(c)
