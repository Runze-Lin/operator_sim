// Explicit dense B,S,H,K/V. Reuse the numerical oracle, Tensor and ABI from the small official runner.
#define main small_kda_main
#include "../../../../smoke/extension_kda.cpp"
#undef main
int main(int argc,char** argv) {
    if(argc!=4) return 2;
    const int b=std::stoi(argv[1]),s=std::stoi(argv[2]),h=std::stoi(argv[3]),d=128;
    if(b<1 || b>256 || s<1 || s>8 || h<1 || h>96) return 2;
    try {
        std::setvbuf(stdout,nullptr,_IOLBF,0);
        ACL_CHECK(aclInit(nullptr));ACL_CHECK(aclrtSetDevice(0));
        aclrtStream stream;ACL_CHECK(aclrtCreateStream(&stream));
        std::printf("runtime_soc=%s\ncase=kda_recurrent api=aclnnRecurrentKda B=%d S=%d Hq=%d Hv=%d K=128 V=128 dtype=BF16\n",aclrtGetSocName(),b,s,h,h);
        {
        const size_t tokens=size_t(b)*s*h;
        std::vector<uint16_t> q(tokens*d),key(tokens*d),v(tokens*d);
        std::vector<float> gate(tokens*d),beta(tokens),state(size_t(b)*h*d*d);
        std::vector<double> outgold(tokens*d),stategold(state.size());
        for(int bi=0;bi<b;++bi)for(int hi=0;hi<h;++hi) {
            Inputs chain(s,d,d);int bh=bi*h+hi;
            // Differentiate heads and request slots in every list of final-state outputs.
            for(auto &x:chain.state)x+=(bh%37-18)/65536.f;
            for(auto &x:chain.beta)x*=1.f+float(bh%11)/128.f;
            const auto gold=recurrent_gold(chain);
            std::copy(chain.state.begin(),chain.state.end(),state.begin()+size_t(bh)*d*d);
            std::copy(gold.second.begin(),gold.second.end(),stategold.begin()+size_t(bh)*d*d);
            for(int t=0;t<s;++t) {
                size_t ti=(size_t(bi)*s+t)*h+hi;
                std::copy_n(chain.q.begin()+t*d,d,q.begin()+ti*d);
                std::copy_n(chain.key.begin()+t*d,d,key.begin()+ti*d);
                std::copy_n(chain.value.begin()+t*d,d,v.begin()+ti*d);
                std::copy_n(chain.gate.begin()+t*d,d,gate.begin()+ti*d);
                std::copy_n(gold.first.begin()+t*d,d,outgold.begin()+ti*d);
                beta[ti]=chain.beta[t];
            }
        }
        Tensor<uint16_t> tq({b,s,h,d},ACL_BF16,q),tk({b,s,h,d},ACL_BF16,key),tv({b,s,h,d},ACL_BF16,v);
        Tensor<float> tg({b,s,h,d},ACL_FLOAT,gate),tb({b,s,h},ACL_FLOAT,beta),ti({b,h,d,d},ACL_FLOAT,state);
        Tensor<uint16_t> to({b,s,h,d},ACL_BF16,std::vector<uint16_t>(tokens*d,0x7fc0));
        Tensor<float> tf({b,h,d,d},ACL_FLOAT,std::vector<float>(state.size(),NAN));
        uint64_t bytes=0;aclOpExecutor* exe=nullptr;
        ACL_CHECK(aclnnRecurrentKdaGetWorkspaceSize(tq.tensor,tk.tensor,tv.tensor,tg.tensor,tb.tensor,ti.tensor,
            nullptr,nullptr,nullptr,nullptr,nullptr,"BSND",1./std::sqrt(128.),true,false,false,false,false,false,false,-5.,false,
            to.tensor,tf.tensor,&bytes,&exe));
        Buffer scratch(bytes);ACL_CHECK(aclnnRecurrentKda(scratch.ptr,bytes,exe,stream));ACL_CHECK(aclrtSynchronizeStream(stream));
        compare(decode(to.read()),outgold,true,"attention");auto actual=tf.read();
        compare(std::vector<double>(actual.begin(),actual.end()),stategold,false,"state");
        }
        ACL_CHECK(aclrtDestroyStream(stream));ACL_CHECK(aclrtResetDevice(0));ACL_CHECK(aclFinalize());
        std::puts("SMOKE_PASS");
        return 0;
    } catch(const std::exception& e){std::fprintf(stderr,"SMOKE_FAIL %s\n",e.what());return 1;}
}
