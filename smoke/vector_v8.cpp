// Build once per operator family to reuse the exact single-call CPU references.
#define main archived_single_case_main
#if defined(V8_NORM)
#include "extension_norm_addmm.cpp"
#elif defined(V8_MHC)
#include "extension_indexer_mhc.cpp"
#else
#include "extension_kda.cpp"
#endif
#undef main
#include <fstream>
#include <sstream>

#if !defined(V8_NORM) && !defined(V8_MHC)
static void recurrent(aclrtStream stream,int s,int v) {
    const int k=128;
    Inputs in(s,k,v);
    Tensor<uint16_t> q({1,s,1,k},ACL_BF16,in.q),key({1,s,1,k},ACL_BF16,in.key),value({1,s,1,v},ACL_BF16,in.value);
    Tensor<float> gate({1,s,1,k},ACL_FLOAT,in.gate),beta({1,s,1},ACL_FLOAT,in.beta),initial({1,1,k,v},ACL_FLOAT,in.state);
    Tensor<uint16_t> out({1,s,1,v},ACL_BF16,std::vector<uint16_t>(s*v,0x7fc0));
    Tensor<float> final({1,1,k,v},ACL_FLOAT,std::vector<float>(k*v,NAN));
    uint64_t bytes=0;aclOpExecutor* executor=nullptr;
    ACL_CHECK(aclnnRecurrentKdaGetWorkspaceSize(q.tensor,key.tensor,value.tensor,gate.tensor,beta.tensor,initial.tensor,
        nullptr,nullptr,nullptr,nullptr,nullptr,"BSND",1./std::sqrt(double(k)),true,false,false,false,false,false,false,-5.,false,
        out.tensor,final.tensor,&bytes,&executor));
    Buffer workspace(bytes);
    ACL_CHECK(aclnnRecurrentKda(workspace.ptr,bytes,executor,stream));
    ACL_CHECK(aclrtSynchronizeStream(stream));
    auto gold=recurrent_gold(in);
    compare(decode(out.read()),gold.first,true,"attn_out");
    auto f=final.read(),st=initial.read();
    compare(std::vector<double>(f.begin(),f.end()),gold.second,false,"final_state");
    compare(std::vector<double>(st.begin(),st.end()),std::vector<double>(in.state.begin(),in.state.end()),false,"initial_state_unchanged");
}
#endif
int main(int argc,char** argv) {
    try {
        if(argc!=2)throw std::runtime_error("Expected batch list");
        std::ifstream input(argv[1]);if(!input)throw std::runtime_error("Cannot open list");
        ACL_CHECK(aclInit(nullptr));ACL_CHECK(aclrtSetDevice(0));
        aclrtStream stream=nullptr;ACL_CHECK(aclrtCreateStream(&stream));
        std::printf("runtime_soc=%s\n",aclrtGetSocName());
        std::string op;int id,a,b,c;
        while(input>>op>>id>>a>>b>>c) {
            std::printf("BATCH_BEGIN %d\n",id);std::fflush(stdout);
#if defined(V8_NORM)
            rmsnorm(stream,a,b);
#elif defined(V8_MHC)
            if(op=="mhc_post")post(stream,a,b,c);
            else if(op=="mhc_sinkhorn")sink(stream,a,b,c);
            else throw std::runtime_error("Bad mHC mode");
#else
            recurrent(stream,a,b);
#endif
            std::printf("BATCH_END %d\n",id);std::fflush(stdout);
        }
        ACL_CHECK(aclrtDestroyStream(stream));ACL_CHECK(aclrtResetDevice(0));ACL_CHECK(aclFinalize());std::puts("SMOKE_PASS");return 0;
    }catch(const std::exception& e){std::fprintf(stderr,"SMOKE_FAIL %s\n",e.what());return 1;}
}
