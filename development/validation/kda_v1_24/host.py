"""Generate a host-only oracle harness; all device launches retain the FLA release ELF."""
import subprocess
from pathlib import Path
import sys

HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[2]
sys.path.insert(0,str(HERE.parent/'chunk_recovery/fla2691/v2'))
from probe_v2 import replace_once,probe
OUT=ROOT/'results/v0.24/kda-v1'
RAW=OUT/'raw';BIN=RAW/'kimi-v1-default'

def build():
    src=probe.SOURCE.read_text().replace('"../../smoke/extension_kda.cpp"','"'+str(ROOT/'smoke/extension_kda.cpp')+'"')
    src='#include <dlfcn.h>\n'+src
    inc=HERE.parent/'kda_v23/precise_host.hpp'
    src=replace_once(src,'void prefill(aclrtStream stream, int tokens, int heads) {',f'#include "{inc}"\nvoid prefill(aclrtStream stream, int tokens, int heads, bool dense) {{')
    src=replace_once(src,'lengths.array,chunks.array,"BSND"','dense ? nullptr : lengths.array, dense ? nullptr : chunks.array,"BSND"')
    a=src.index('    compare(decode(O.read())');b=src.index('\n}',a)
    src=src[:a]+r'''    const std::string root="FIXTURE/"+std::to_string(tokens)+"-"+std::to_string(heads)+"/";
    bool ok=true;
    auto check=[&](const char* name,const std::vector<double>& got,const std::vector<double>& math,bool bf) {
        auto expected=readPreparedGolden(root+name+".bin",got.size());
        try {compare(got,expected,bf,name);} catch (const std::runtime_error&) {ok=false;}
        try {compare(got,math,bf,(std::string("math_")+name).c_str());} catch (const std::runtime_error&) {}
    };
    auto attention=O.read(), qk=QK.read(), kk=KK.read();auto f=F.read();
    writeActual("attention",attention);writeActual("aqk",qk);writeActual("akk",kk);
    {std::ofstream disk("state_vk.f32",std::ios::binary);disk.write(reinterpret_cast<const char*>(f.data()),f.size()*sizeof(float));}
    check("attention",decode(attention),out,true);
    check("aqk",decode(qk),aqk,true);
    check("akk",decode(kk),akk,true);
    check("state_vk",{f.begin(),f.end()},final,false);
    if (I.read()!=state) throw std::runtime_error("modified initial VK");
    std::puts("input_vk_unchanged=1");
    if(!ok) throw std::runtime_error("fused V1 reference comparison failed");'''.replace('FIXTURE',str(RAW/'oracles'))+src[b:]
    src=replace_once(src,'if (argc != 3) return 2;',r'''if (argc != 4) return 2;
    std::setvbuf(stdout,nullptr,_IOLBF,0);
    for(const char* name: {"aclnnChunkKdaFwd","aclnnChunkKdaFwdGetWorkspaceSize"}) {
        Dl_info info{};const char* required=std::getenv("FLA_NPU_OP_API_LIB");
        if (!required || !dladdr(dlsym(RTLD_DEFAULT,name),&info) || std::string(required)!=info.dli_fname) return 7;
        std::printf("bound_v1_api=%s library=%s\n",name,info.dli_fname);
    }''')
    src=replace_once(src,'prefill(stream,tokens,heads);',r'''bool dense=std::string(argv[3])=="dense";
        if(!dense && std::string(argv[3])!="varlen") return 2;
        std::printf("metadata=%s\n",dense?"dense":"varlen");
        prefill(stream,tokens,heads,dense);''')
    cpp=BIN.with_suffix('.cpp');cpp.write_text(src);cann=probe.CANN;api=probe.VENDOR/'op_api'
    subprocess.run(['bash',str(ROOT/'scripts/cann-run.sh'),'g++','-std=c++17','-O2','-fPIE','-pie',str(cpp),
        '-I'+str(cann/'include'),'-I'+str(api/'include'),'-L'+str(api/'lib'),'-Wl,-rpath,'+str(api/'lib'),'-L'+str(cann/'lib64'),
        '-Wl,-rpath-link,'+str(cann/'lib64'),str(ROOT/'toolchains/envs/cann-9.2.1/lib/libstdc++.so.6'),'-lcust_opapi','-lopapi','-lnnopbase','-lascendcl','-ldl','-o',str(BIN)],check=True)
