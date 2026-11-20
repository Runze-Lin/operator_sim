"""Stage the pinned custom ACLNN with the frontend's prefill flags and output aliasing."""
from pathlib import Path
import subprocess
ROOT=Path(__file__).resolve().parents[4]
OUT=ROOT/'runs/large29/vector-host'
SOURCE=ROOT/'development/workloads/attention_residual_smoke.cpp'
UP=ROOT/'runs/workloads19/sources/full/csrc'
VENDOR=ROOT/'runs/workloads19/sources/fastbuild/vendors/custom_transformer'
HOST=OUT/'attn_res_prefill'
SDK=ROOT/'toolchains/cann/9.2.1/cann-9.2.1'


def build():
    OUT.mkdir(parents=True,exist_ok=True)
    s=SOURCE.read_text()
    for old,new in [
        ('"../../smoke/extension_norm_addmm.cpp"','"'+str(ROOT/'smoke/extension_norm_addmm.cpp')+'"'),
        (', m({T,H},ACL_BF16,init)',''),
        ('1e-5,N,S*H,N,1e-5,true,true,true,false,y.tensor,r.tensor,m.tensor','1e-5,N,S*H,-1,1e-5,false,true,true,true,y.tensor,r.tensor,y.tensor'),
        ('saved=m.read(), ',''),
        ('        if (!(std::abs(fp32(saved[i])-fp32(mix[i])) <= .002f+.009f*std::abs(fp32(mix[i])))) ++bad;',''),
        ('(n==N?raw[t*H+h]:bank[i])','bank[i]'),
        ('groups=output,prefix,materialized,bank','flags=cache,add,mix,nosave,nowrite groups=output,prefix,bank'),
        ('T>64','T>4096')]:
        if s.count(old)!=1:raise ValueError(old)
        s=s.replace(old,new)
    src=OUT/'attn_res_prefill.cpp';src.write_text(s)
    subprocess.run(['bash',str(ROOT/'scripts/cann-run.sh'),'g++','-std=c++17','-O2',str(src),
        '-I'+str(UP/'attention/attn_res_fwd/op_host/op_api'),'-I'+str(UP/'common/include/external'),
        '-I'+str(SDK/'include'),'-L'+str(VENDOR/'op_api/lib'),'-L'+str(SDK/'lib64'),
        '-Wl,-rpath-link,'+str(SDK/'lib64'),'-lcust_opapi','-lopapi','-lnnopbase','-lascendcl','-o',str(HOST)],check=True)

if __name__=='__main__':build()
