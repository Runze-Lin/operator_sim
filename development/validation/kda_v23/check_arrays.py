"""Independently check saved BF16 device values against synthetic Prepare goldens."""
import gzip
import hashlib
import json
from pathlib import Path
import numpy as np
from reference import prepare,diff

BASE=Path(__file__).resolve().parents[3]/'results/kda-v23'
ROOT=BASE.parents[1]


def main():
    release=BASE/'aux-actual';release.mkdir(exist_ok=True);records=[]
    for role in ('development','holdout','supplement'):
        for path in sorted(BASE.glob(role+'-ascend950*.json')):
            d=json.loads(path.read_bytes());shape=d['case']['shape'];h,t=shape[3],shape[1];supplied=d['actual_aux_bf16']
            published,math=prepare(t,h)
            groups={}
            for name,pub,mat in zip(('aqk','akk'),published,math):
                dst=release/(d['case']['id']+'-'+name+'.bf16.gz')
                src=ROOT/supplied[name]['path']
                if src.exists():
                    content=src.read_bytes()
                    if not dst.exists():dst.write_bytes(gzip.compress(content,compresslevel=9,mtime=0))
                compressed=dst.read_bytes();content=gzip.decompress(compressed)
                if hashlib.sha256(content).hexdigest()!=supplied[name]['sha256']:raise ValueError('actual BF16 digest '+name)
                actual=(np.frombuffer(content,dtype='<u2').astype('<u4')<<16).view('<f4').astype(np.float64).reshape(h,t,64)
                pub_diff=diff(actual,pub);math_diff=diff(actual,mat)
                group=d['output_groups'][name]
                if group!=[pub_diff['count'],pub_diff['bad']]:raise AssertionError('CPU replay versus device host '+name)
                groups[name]=dict(device_vs_published=pub_diff,device_vs_mathematical=math_diff,
                    published_nonidentical_elements=int((actual!=pub).sum()),zero_ref_nonzero_actual=int(((pub==0)&(actual!=0)).sum()),
                    archive=dst.relative_to(BASE).as_posix(),uncompressed_sha256=supplied[name]['sha256'],compressed_sha256=hashlib.sha256(compressed).hexdigest())
            records.append(dict(case_id=d['case']['id'],shape=shape,role=role,scores=groups))
    path=BASE/'actual-array-checks.json';tmp=path.with_suffix('.tmp')
    tmp.write_text(json.dumps(dict(format='Each gzip file contains H × T × 64 little-endian BF16 values in row-major order, read back from the completed public API. All four reported outputs are checked in each host; the two optional matrices are also retained here.',records=records),indent=2)+'\n');tmp.replace(path)

if __name__=='__main__':main()
