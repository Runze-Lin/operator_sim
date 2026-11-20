"""Resolve the two historical modules named verify and replay their actual qualification."""
import importlib.util
from pathlib import Path
import sys
from campaign import OUT,ROOT,V2_OLD,read,save
import verify


def bind():
    path=ROOT/'development/validation/rms_ordered_verify/verify.py'
    spec=importlib.util.spec_from_file_location('ordered_elf_evidence',path)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    # The release scripts also import a top-level module named verify.
    # Load them against their actual helper, then restore the local verifier.
    sys.modules['verify']=module
    sys.path.insert(0,str(ROOT/'development/validation/kda_v23'))
    try:
        import series
        assert series.probe.text_bytes is module.text_bytes
    finally:
        sys.modules['verify']=verify



def control():
    bind()
    folder=ROOT/'results/kda-v23/raw/holdout-ascend950pr-t3-h1';cost=read(folder/'cost.json')
    job=dict(id='kda-v2',prefix=['ChunkKdaFwdPrepare','ChunkFwdH','ChunkKdaFwdFinalize'],contract='api=aclnnChunkKdaFwdV2 tokens=3 heads=1',golden={'attention':384,'aqk':192,'akk':192,'state_vk':16384})
    cost['stop_reason']=cost['stop']
    r=verify.qualify(job,'Ascend950PR_957b',folder,cost,read(V2_OLD),ROOT)
    assert r['qualified'],r
    save(OUT/'v2-verifier-control.json',r)
    folder=ROOT/'runs/sku17/extension/addmm_128_192_256_Ascend950PR_957b'
    job=dict(id='addmm',prefix=['MatMulV3'],contract='case=addmm api=aclnnAddmm M=128 N=192 K=256',golden={'output':128*192})
    r=verify.qualify(job,'Ascend950PR_957b',folder,read(folder/'cost.json'),read(V2_OLD),ROOT)
    assert r['qualified'],r
    save(OUT/'addmm-verifier-control.json',r)


if __name__=='__main__':control()
