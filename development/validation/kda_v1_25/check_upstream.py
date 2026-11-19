"""Verify current 26.11/main modifications against the exact official Git blobs."""
import difflib
import hashlib
import re
import urllib.request
from collections import Counter
from packages import RAW,OUT,ROOT,read,save,vendor,probe

BASE='fla/ops/ascendc/kda/chunk_kda_fwd/op_kernel/'
PIN='dc1113e2e9f2ed8c2b29ab94b3a385990be0fbeb'
def tree(path):
    x=read(path);assert not x['truncated']
    return x['sha'],{v['path'][len(BASE):]:v['sha'] for v in x['tree'] if v['path'].startswith(BASE) and v['type']=='blob'}
def gitsha(d):return hashlib.sha1(b'blob '+str(len(d)).encode()+b'\0'+d).hexdigest()
def main():
    rhash,r=tree(RAW/'2610-tree.json');mhash,m=tree(RAW/'main-tree.json');bhash,b=tree(RAW/'2611-tree.json')
    if mhash!=PIN:raise ValueError('official source SHA mismatch')
    same=m==b
    wd=vendor('release')/'op_impl/ai_core/tbe/fla_npu_transformer_impl/ascendc/chunk_kda_fwd'
    old=ROOT/'results/chunk-recovery/fla2691/raw/fla_npu/opp/vendors/fla_npu_transformer/op_impl/ai_core/tbe/fla_npu_transformer_impl/ascendc/chunk_kda_fwd'
    changes=[];baseline={}
    for rel in sorted(r):
        ref=wd/rel
        if not ref.is_file():continue
        src=ref.read_bytes()
        if gitsha(src)!=r[rel]:
            if gitsha(src.replace(b'\r\n',b'\n'))!=r[rel]:raise ValueError('package source mismatch: '+rel)
        if (old/rel).exists():
            baseline[rel]={'old_sha256':probe.digest(old/rel),'release_sha256':probe.digest(ref),'all_lines_equal':(old/rel).read_text().splitlines()==ref.read_text().splitlines()}
        if r[rel]==m.get(rel):continue
        dst=RAW/'upstream'/rel;dst.parent.mkdir(exist_ok=True,parents=True)
        data=dst.read_bytes() if dst.exists() else urllib.request.urlopen('https://raw.githubusercontent.com/flashserve/flash-linear-attention-npu/'+PIN+'/'+BASE+rel,timeout=40).read()
        if gitsha(data)!=m[rel]:raise ValueError('new head blob mismatch')
        if not dst.exists():dst.write_bytes(data)
        diff=list(difflib.unified_diff(src.decode().splitlines(),data.decode().splitlines(),n=0))
        lines=[l for l in diff[2:] if l.startswith(('+','-'))]
        changes.append(dict(path=rel,release_blob=r[rel],main_blob=m[rel],changed=lines))
    save(OUT/'upstream-source.json',dict(verified_main=mhash,verified_v2611=bhash,release=rhash,
        official_sources_identical_between_main_and_2611_in_v1_kernel=same,physical_v1_kernel_files_in_tree=len(r),
        release_files_compared_to_kimi_2691=baseline,changes=changes))
    print('files equal 26.9.1/26.10 normalized',sum(x['all_lines_equal'] for x in baseline.values()),'/',len(baseline),'main==26.11',same)
    for c in changes:
        print(c['path'],len(c['changed']),list(dict.fromkeys(c['changed']))[:8])
if __name__=='__main__':main()
