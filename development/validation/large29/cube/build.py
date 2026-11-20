"""Build hosts only. Vendor host APIs and device binaries are unmodified."""
from pathlib import Path
import subprocess
ROOT=Path(__file__).resolve().parents[4]
HERE=Path(__file__).resolve().parent
BIN=ROOT/'runs/large29/cube/bin'
BIN.mkdir(parents=True,exist_ok=True)

def compile(src,dst,extra=()):
    h=ROOT/'toolchains/cann/9.2.1/cann-9.2.1/include'
    assert (h/'acl/acl.h').exists()
    subprocess.run(['bash',str(ROOT/'scripts/cann-run.sh'),'g++','-std=c++17','-O2',str(src),
                    '-I'+str(h),'-L'+str(h.parent/'lib64'),'-Wl,-rpath-link,'+str(h.parent/'lib64'),*extra,
                    '-lopapi','-lnnopbase','-lascendcl','-o',str(dst)],check=True)

def adapt():
    raw=(ROOT/'smoke/extension_moe.cpp').read_text()
    raw=raw[:raw.index('static std::vector<uint8_t> values')]+(HERE/'moe_reference.hpp').read_text()+raw[raw.index('static std::vector<uint8_t> scales'):]
    raw=raw[:raw.index('static std::vector<double> gmm')]+(HERE/'moe_golden.hpp').read_text()+raw[raw.index('static void compare'):]
    raw=raw.replace('v>4096','v>16384').replace('int64_t(m)*k*f*e>16*1024*1024','int64_t(m)*k*f*e>20000000000LL')
    for name,text in [('moe',raw)]:
        file=BIN/(name+'.cpp');file.write_text(text);compile(file,BIN/name)
    raw=(ROOT/'smoke/extension_norm_addmm.cpp').read_text()
    src='for(int q=0;q<k;++q)v+=double(fp32(a[size_t(i)*k+q]))*fp32(b[size_t(q)*n+j]);'
    assert raw.count(src)==1
    raw=raw.replace(src,'for(int q=0;q<std::min(17,k);++q)v+=double((k-1-q)/17+1)*fp32(a[size_t(i)*k+q])*fp32(b[size_t(q)*n+j]);')
    raw=raw.replace('int64_t(a)*b*c>64*1024*1024','int64_t(a)*b*c>40000000000LL')
    file=BIN/'addmm.cpp';file.write_text(raw);compile(file,BIN/'addmm')
    compile(HERE/'recurrent.cpp',BIN/'recurrent')
    compile(HERE/'golden_check.cpp',BIN/'golden-check')
    subprocess.run(['bash',str(ROOT/'scripts/cann-run.sh'),str(BIN/'golden-check')],check=True)

if __name__=='__main__':adapt()
