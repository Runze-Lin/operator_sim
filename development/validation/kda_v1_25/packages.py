"""Audit the shipped BF16 fused device objects before committing any simulator cost."""
from pathlib import Path, PurePosixPath
import hashlib,json,stat,struct,sys,zipfile
HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[2]
OUT=ROOT/'results/v0.25/kda-v1'
RAW=OUT/'raw'
OLD=ROOT/'results/chunk-recovery/fla2691/register.json'
sys.path.insert(0,str(HERE.parent/'kda_v1_24'))
import investigate as prior
from series import save,read,when
probe=prior.probe
NAMES={'release':'v26.10.0-release','feature':'v26.10.0-feature'}

def digest_bytes(data):return hashlib.sha256(data).hexdigest()

def text_hash(data):
    if data[:6]!=b'\x7fELF\x02\x01':raise ValueError('bad ELF')
    table=struct.unpack_from('<Q',data,40)[0]
    size,n,names=struct.unpack_from('<HHH',data,58)
    offset,length=struct.unpack_from('<QQ',data,table+size*names+24)
    strings=data[offset:offset+length]
    for i in range(n):
        h=table+size*i;index=struct.unpack_from('<I',data,h)[0]
        if strings[index:].split(b'\0',1)[0]==b'.text':
            a,b=struct.unpack_from('<QQ',data,h+24);return digest_bytes(data[a:a+b])
    raise ValueError('missing .text')

def vendor(kind):return RAW/kind/'fla_npu/opp/vendors/fla_npu_transformer'

def release(kind):
    wheel=RAW/(kind+'.whl');r,=[r for r in read(RAW/'releases.json') if r['tag_name']==NAMES[kind]]
    asset,=[a for a in r['assets'] if '_a5-' in a['name'] and a['name'].endswith('aarch64.whl')]
    if wheel.stat().st_size!=asset['size'] or 'sha256:'+probe.digest(wheel)!=asset['digest']:raise ValueError('invalid '+str(wheel))
    files={};elf={};src={}
    with zipfile.ZipFile(wheel) as z:
        for item in z.infolist():
            if item.is_dir() or '/purelib/fla_npu/' not in item.filename or '/offline/' in item.filename:continue
            path=PurePosixPath(item.filename.split('/purelib/',1)[1])
            if path.is_absolute() or '..' in path.parts or stat.S_ISLNK(item.external_attr>>16):raise ValueError(str(path))
            s=str(path)
            kda='/chunk_kda_fwd/' in s or '/kda_gate_cumsum/' in s
            if kind=='release' or kda or '/op_api/' in s or s.endswith(('_aclnn_ctypes.py','version.info')):
                d=z.read(item);f=RAW/kind/path;f.parent.mkdir(parents=True,exist_ok=True)
                if not f.exists():f.write_bytes(d)
                elif probe.digest(f)!=digest_bytes(d):raise ValueError('extracted bytes differ: '+s)
                if '/kernel/ascend950/chunk_kda_fwd/' in s and s.endswith('.o'):elf[path.name]=text_hash(d)
                if kda and s.endswith(('.hpp','.h','.cpp')):src[s.split('/ascendc/')[-1]]=digest_bytes(d)
                if s.endswith(('libcust_opapi.so','aclnn_chunk_kda_fwd.h','_aclnn_ctypes.py','version.info')):files[s]=digest_bytes(d)
    return dict(release_url=r['html_url'],asset={k:asset[k] for k in ('name','digest','size','browser_download_url','id')},
                published_utc=r['published_at'],commit=r['target_commitish'],wheel_elfs=elf,files=files,sources=src)

def main():
    old=read(OLD);new={k:release(k) for k in NAMES}
    olds=set(old['published_kda_text'].values())
    for d in new.values():
        els=set(d['wheel_elfs'].values());d['number_of_objects']=len(d['wheel_elfs'])
        d['unique_text_count']=len(els);d['old_unique_texts_in_common']=sorted(els&olds)
    a,b=new.values()
    prefix='fla_npu/opp/vendors/fla_npu_transformer/op_impl/ai_core/tbe/fla_npu_transformer_impl/ascendc/'
    old_src=probe.VENDOR/'op_impl/ai_core/tbe/fla_npu_transformer_impl/ascendc'
    src_changed={}
    for k,v in a['sources'].items():
        p=old_src/k
        if p.is_file() and probe.digest(p)!=v:src_changed[k]=dict(prior_sha256=probe.digest(p),release_sha256=v)
    save(OUT/'packages.json',dict(utc=when(),source_api='https://api.github.com/repos/flashserve/flash-linear-attention-npu/releases',
        source_api_sha256=probe.digest(RAW/'releases.json'),reference_kimi_wheel=old['release_asset'],versions=new,
        full_set_of_release_feature_elf_text_equal=set(a['wheel_elfs'].values())==set(b['wheel_elfs'].values()),
        official_v1_or_gate_sources_changed_between_new_packages=[p for p in (set(a['sources'])|set(b['sources'])) if a['sources'].get(p)!=b['sources'].get(p)],
        changed_existing_source_paths_from_2691=src_changed))
    print([(k,d['number_of_objects'],d['unique_text_count'],len(d['old_unique_texts_in_common'])) for k,d in new.items()])
if __name__=='__main__':main()
