// Test the pinned custom ACLNN entry on one unpadded Kimi K3 hidden dimension.
#define main archived_norm_host
#include "../../smoke/extension_norm_addmm.cpp"
#undef main
#include "aclnn_attn_res_fwd.h"

static void run_residual(aclrtStream stream, int T, int N) {
    const int H=7168, S=8;
    auto prefix = data(T*H,1), addend = data(T*H,4), bank = data(T*S*H,7);
    std::vector<uint16_t> projection(H,bf16(0.002f)), norm(H,bf16(1.f)), outWeight(H,bf16(.75f));
    std::vector<uint16_t> init(T*H,bf16(NAN)), raw(T*H), mix(T*H), golden(T*H);
    for(int t=0;t<T;++t) {
        std::vector<double> logits(N+1), probs(N+1);
        for(int h=0;h<H;++h)
            raw[t*H+h]=bf16(fp32(prefix[t*H+h])+fp32(addend[t*H+h]));
        auto row_at=[&](int n, int h) {
            return fp32(n==N ? raw[t*H+h] : bank[(t*S+n)*H+h]);
        };
        for(int n=0;n<=N;++n) {
            double ss=0, dot=0;
            for(int h=0;h<H;++h) {
                double x=row_at(n,h);
                ss+=x*x;
                dot+=x*fp32(projection[h])*fp32(norm[h]);
            }
            logits[n]=dot/std::sqrt(ss/H+1e-5);
        }
        double max_logit=*std::max_element(logits.begin(),logits.end()), denom=0, ss=0;
        for(int n=0;n<=N;++n) { probs[n]=std::exp(logits[n]-max_logit); denom+=probs[n]; }
        for(int h=0;h<H;++h) {
            double y=0;
            for(int n=0;n<=N;++n) y+=row_at(n,h)*probs[n]/denom;
            mix[t*H+h]=bf16(y);
            double z=fp32(mix[t*H+h]); ss+=z*z;
        }
        for(int h=0;h<H;++h) golden[t*H+h]=bf16(.75f*fp32(mix[t*H+h])/std::sqrt(ss/H+1e-5));
    }
    Tensor<uint16_t> p({T,H},ACL_BF16,prefix), b({T,S,H},ACL_BF16,bank);
    Tensor<uint16_t> proj({1,H},ACL_BF16,projection), g({H},ACL_BF16,norm);
    Tensor<uint16_t> add({T,H},ACL_BF16,addend), og({H},ACL_BF16,outWeight);
    Tensor<uint16_t> y({T,H},ACL_BF16,init), r({T,H},ACL_BF16,init), m({T,H},ACL_BF16,init);
    uint64_t bytes=0; aclOpExecutor* exe=nullptr;
    ACL_CHECK(aclnnAttnResFwdGetWorkspaceSize(p.tensor,b.tensor,proj.tensor,g.tensor,add.tensor,og.tensor,
        1e-5,N,S*H,N,1e-5,true,true,true,false,y.tensor,r.tensor,m.tensor,&bytes,&exe));
    Buffer workspace(bytes);
    ACL_CHECK(aclnnAttnResFwd(workspace.ptr,bytes,exe,stream));
    ACL_CHECK(aclrtSynchronizeStream(stream));
    const auto got=y.read(), saved=m.read(), raw_result=r.read(), bank_result=b.read();
    size_t bad=0;
    for (int i=0;i<T*H;++i) {
        if (!(std::abs(fp32(got[i])-fp32(golden[i])) <= .01f+.012f*std::abs(fp32(golden[i])))) ++bad;
        if (!(std::abs(fp32(saved[i])-fp32(mix[i])) <= .002f+.009f*std::abs(fp32(mix[i])))) ++bad;
        if (raw_result[i]!=raw[i]) ++bad;
    }
    for(int t=0;t<T;++t) for(int n=0;n<S;++n) for(int h=0;h<H;++h) {
        auto i=(t*S+n)*H+h;
        if(bank_result[i]!=(n==N?raw[t*H+h]:bank[i])) ++bad;
    }
    std::printf("correctness shape=%dx%d S=%d N=%d groups=output,prefix,materialized,bank bad=%zu\n",T,H,S,N,bad);
    if(bad) throw std::runtime_error("CA failed golden");
}
int main(int argc,char** argv) {
    setvbuf(stdout,nullptr,_IOLBF,0);
    try {
        if(argc!=3) throw std::runtime_error("expected T N");
        const int T=std::stoi(argv[1]), N=std::stoi(argv[2]);
        if(T<1 || T>64 || N<1 || N>7) throw std::runtime_error("shape outside smoke scope");
        ACL_CHECK(aclInit(nullptr)); ACL_CHECK(aclrtSetDevice(0));
        std::printf("runtime_soc=%s api=aclnnAttnResFwd\n",aclrtGetSocName());
        aclrtStream s=nullptr; ACL_CHECK(aclrtCreateStream(&s));
        run_residual(s,T,N);
        ACL_CHECK(aclrtDestroyStream(s)); ACL_CHECK(aclrtResetDevice(0)); ACL_CHECK(aclFinalize());
        std::puts("SMOKE_PASS");
        return 0;
    } catch(const std::exception& e) { std::fprintf(stderr,"SMOKE_FAIL: %s\n",e.what()); return 1; }
}
