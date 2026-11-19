// Single numerical probe of the installed generic MIX template, FP16 Q/K/V.
#define main legacy_smoke_main
#include "../../../smoke/extension_kda.cpp"
#undef main

static std::vector<__fp16> half_input(const std::vector<uint16_t>& input) {
    std::vector<__fp16> result(input.size());
    std::transform(input.begin(), input.end(), result.begin(), [](uint16_t x) { return __fp16(fp32(x)); });
    return result;
}

static void generic_mix(aclrtStream stream) {
    constexpr int s=3, d=16, c=128;
    Inputs input(s,d,d);
    std::fill(input.state.begin(),input.state.end(),0.);
    const double scale=1./std::sqrt(double(d));
    auto oracle = recurrent_gold(input);
    std::vector<double> aqk(s*c),akk(s*c),prefix(s*d),strict_lower(s*s);
    for (int i=0;i<s;i++) {
        for (int k=0;k<d;k++) prefix[i*d+k]=input.gate[i*d+k]+(i?prefix[(i-1)*d+k]:0.);
        for (int j=0;j<=i;j++) {
            double qdot=0,kdot=0;
            for (int k=0;k<d;k++) {
                double key=fp32(input.key[j*d+k])*std::exp(prefix[i*d+k]-prefix[j*d+k]);
                qdot+=fp32(input.q[i*d+k])*key;
                kdot+=fp32(input.key[i*d+k])*key;
            }
            aqk[i*c+j]=scale*qdot;
            if(j<i) strict_lower[i*s+j]=input.beta[i]*kdot;
        }
        for(int j=0;j<=i;j++) {
            double x=i==j ? 1.:0.;
            for (int z=j;z<i;z++) x-=strict_lower[i*s+z]*akk[z*c+j];
            akk[i*c+j]=x;
        }
    }
    Tensor<__fp16> Q({1,1,s,d},ACL_FLOAT16,half_input(input.q)), K({1,1,s,d},ACL_FLOAT16,half_input(input.key));
    Tensor<__fp16> V({1,1,s,d},ACL_FLOAT16,half_input(input.value));
    Tensor<float> G({1,1,s,d},ACL_FLOAT,input.gate),B({1,1,s},ACL_FLOAT,input.beta);
    Tensor<__fp16> O({1,s,1,d},ACL_FLOAT16,std::vector<__fp16>(s*d,NAN));
    Tensor<__fp16> QK({1,1,s,c},ACL_FLOAT16,std::vector<__fp16>(s*c,NAN)),KK({1,1,s,c},ACL_FLOAT16,std::vector<__fp16>(s*c,NAN));
    uint64_t bytes=0;
    aclOpExecutor* executor=nullptr;
    ACL_CHECK(aclnnChunkKdaFwdGetWorkspaceSize(Q.tensor,K.tensor,V.tensor,G.tensor,B.tensor,
        nullptr,nullptr,nullptr,nullptr,nullptr,"BNSD",scale,c,false,-5.,false,false,
        O.tensor,nullptr,nullptr,QK.tensor,KK.tensor,nullptr,nullptr,nullptr,nullptr,nullptr,nullptr,&bytes,&executor));
    Buffer ws(bytes);
    std::printf("api=aclnnChunkKdaFwd workspace_bytes=%llu\n",(unsigned long long)bytes);
    std::fflush(stdout);
    ACL_CHECK(aclnnChunkKdaFwd(ws.ptr,bytes,executor,stream));
    ACL_CHECK(aclrtSynchronizeStream(stream));
    auto compare_output=[](const auto& actual,const auto& gold,const char* label) {
        if(actual.size()!=gold.size()) throw std::runtime_error("output shape mismatch");
        double worst=0;
        size_t bad=0;
        for(size_t i=0;i<gold.size();i++) {
            double value=double(float(actual[i])), err=std::abs(value-gold[i]);
            if(!std::isfinite(value) || err>2e-6+0.001*std::abs(gold[i])) ++bad;
            worst=std::max(worst,err);
        }
        std::printf("check=%s count=%zu bad=%zu max_abs=%g atol=2e-6 rtol=0.001\n",label,gold.size(),bad,worst);
        if(bad) throw std::runtime_error("numerical check failed");
    };
    compare_output(O.read(),oracle.first,"output");
    compare_output(QK.read(),aqk,"aqk");
    compare_output(KK.read(),akk,"akk");
}

int main() {
    try {
        ACL_CHECK(aclInit(nullptr));ACL_CHECK(aclrtSetDevice(0));aclrtStream stream;
        ACL_CHECK(aclrtCreateStream(&stream));std::printf("runtime_soc=%s\n",aclrtGetSocName());
        generic_mix(stream);
        ACL_CHECK(aclrtDestroyStream(stream));ACL_CHECK(aclrtResetDevice(0));ACL_CHECK(aclFinalize());
        std::puts("SMOKE_PASS");return 0;
    } catch (const std::exception& e) { std::fprintf(stderr,"SMOKE_FAIL %s\n",e.what());return 1; }
}
