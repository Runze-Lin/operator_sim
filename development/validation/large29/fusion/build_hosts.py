"""Relax smoke host size guards; the installed device providers and golden code are unchanged."""
from pathlib import Path
import hashlib
import json
import os
import subprocess

ROOT=Path(__file__).resolve().parents[4];OUT=ROOT/'results/v0.29/large/fusion/build'
SDK=ROOT/'toolchains/cann/9.2.1/cann-9.2.1'
ENV=ROOT/'scripts/cann-run.sh'
HC=ROOT/'build/hcpost23'


def once(s,old,new):
    assert s.count(old)==1,old
    return s.replace(old,new)


def build():
    OUT.mkdir(exist_ok=True);builtin=ROOT/'smoke/extension_indexer_mhc.cpp'
    # 262144 * 4 * 20 Sinkhorn, 4096 * 4 * 5120 chain and indexer [Q9,L32768,G8] all fit signed 32-bit addressing.
    src=once(builtin.read_text(),'n>8192','n>262144')
    src=once(src,'int64_t(a)*b*c>8*1024*1024','int64_t(a)*b*c>128LL*1024*1024')
    path=OUT/'builtin.cpp';path.write_text(src)
    base=['g++','-std=c++17','-O2','-I'+str(SDK/'include'),'-L'+str(SDK/'lib64'),'-Wl,-rpath-link,'+str(SDK/'lib64')]
    subprocess.run(['bash',str(ENV),*base,str(path),'-lopapi','-lnnopbase','-lascendcl','-o',str(OUT/'builtin')],check=True)
    source=ROOT/'development/validation/hcpost23/host.cpp'
    src=once(source.read_text(),'"../../../smoke/extension_indexer_mhc.cpp"','"'+str(builtin)+'"')
    src=once(src,'int64_t(t) * h > 1024 * 1024','int64_t(t) * h > 32LL * 1024 * 1024')
    ext=OUT/'hcpost.cpp';ext.write_text(src)
    subprocess.run(['bash',str(ENV),*base,str(ext),'-I'+str(HC/'host/autogen'),'-L'+str(HC/'vendors/hcpost/op_api/lib'),
                    '-lcust_opapi','-lopapi','-lnnopbase','-lascendcl','-ldl','-o',str(OUT/'hcpost')],check=True)


if __name__=='__main__':build()
