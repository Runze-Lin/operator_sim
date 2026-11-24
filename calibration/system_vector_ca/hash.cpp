// Native hash-routing ABI for DeepSeek decode, with nonconstant logits and token IDs.
#include "../acl_tensor.hpp"
#include <aclnnop/aclnn_moe_gating_top_k_v2.h>
#include <cmath>
#include <cstdio>
#include <cstring>
template<class T> static std::vector<uint8_t> bytes(const std::vector<T>& a) {
    std::vector<uint8_t> result(a.size()*sizeof(T));std::memcpy(result.data(),a.data(),result.size());return result;
}
static void run(int r,int e,int k,aclrtStream stream) {
    const int vocab=129280;
    std::vector<float> logit(r*e);
    std::vector<int32_t> tokens(r), table(vocab*k);
    for(int row=0;row<r;++row) {tokens[row]=1+row*17;for(int c=0;c<e;++c) logit[row*e+c]=(c%13-6)*.25f;}
    for(int i=0;i<vocab;++i) for(int j=0;j<k;++j) table[i*k+j]=(i*11+j*7)%e;
    Tensor x({r,e},ACL_FLOAT,bytes(logit)),t({r},ACL_INT32,bytes(tokens)),
           m({vocab,k},ACL_INT32,bytes(table)),y({r,k},ACL_FLOAT,bytes(std::vector<float>(r*k,NAN))),
           ids({r,k},ACL_INT32,bytes(std::vector<int32_t>(r*k,-1))),
           aux({r,e},ACL_FLOAT,bytes(std::vector<float>(r*e,NAN)));
    aclOpExecutor *ex=nullptr;uint64_t size=0;
    CHECK(aclnnMoeGatingTopKV2GetWorkspaceSize(x.desc,nullptr,t.desc,m.desc,k,1,1,1,0,2,false,1.,1e-20,
                                                y.desc,ids.desc,aux.desc,&size,&ex));
    Buffer workspace(size);CHECK(aclnnMoeGatingTopKV2(workspace.ptr,size,ex,stream));CHECK(aclrtSynchronizeStream(stream));
    std::vector<float> got(r*k);std::vector<int32_t> which(r*k);
    CHECK(aclrtMemcpy(got.data(),got.size()*4,y.buffer.ptr,got.size()*4,ACL_MEMCPY_DEVICE_TO_HOST));
    CHECK(aclrtMemcpy(which.data(),which.size()*4,ids.buffer.ptr,which.size()*4,ACL_MEMCPY_DEVICE_TO_HOST));
    size_t bad=0;
    for(int i=0;i<r;++i) {
        std::vector<double> s(k);double sum=0;
        for(int j=0;j<k;++j){int idx=table[tokens[i]*k+j];bad+=which[i*k+j]!=idx;
            sum+=s[j]=std::sqrt(std::log1p(std::exp(double(logit[i*e+idx]))));}
        for(int j=0;j<k;++j)bad+=!(std::abs(got[i*k+j]-s[j]/sum)<1e-5);
    }
    printf("correctness Y count=%d bad=%zu\n",2*r*k,bad);if(bad) throw std::runtime_error("hash golden differs");
}
int main(int argc,char** argv) {
    std::setvbuf(stdout,nullptr,_IONBF,0);if(argc!=5)return 2;
    try{CHECK(aclInit(nullptr));CHECK(aclrtSetDevice(0));aclrtStream s;CHECK(aclrtCreateStream(&s));
        std::printf("runtime_soc=%s\n",aclrtGetSocName());run(std::stoi(argv[2]),std::stoi(argv[3]),std::stoi(argv[4]),s);
        CHECK(aclrtDestroyStream(s));CHECK(aclrtResetDevice(0));CHECK(aclFinalize());std::puts("SMOKE_PASS");std::_Exit(0);
    } catch(const std::exception& e){std::fprintf(stderr,"SMOKE_FAIL %s\n",e.what());return 1;}
}
