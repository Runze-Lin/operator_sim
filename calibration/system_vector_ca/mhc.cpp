// Validate the library PreSinkhorn inference outputs on a captured wide hidden dimension.
#include "../acl_tensor.hpp"
#include <aclnnop/aclnn_mhc_pre_sinkhorn.h>
#include <cmath>
#include <cstdio>
#include <cstring>

template<class T> static std::vector<uint8_t> bytes(const std::vector<T>& a) {
    std::vector<uint8_t> result(a.size()*sizeof(T));
    std::memcpy(result.data(),a.data(),result.size());
    return result;
}
static uint16_t bf(float x) {
    uint32_t u; std::memcpy(&u,&x,4); return (u+32767+((u>>16)&1))>>16;
}
static void probe(int t,int n,int d,aclrtStream stream) {
    const int c=2*n+n*n, k=n*d;
    std::vector<uint16_t> input(t*k),gold(t*d);
    for(int i=0;i<t;++i) for(int j=0;j<d;++j) {
        float sum=0;
        for(int h=0;h<n;++h) {
            float x=float((i+j+h)%11-5)/4;
            input[(i*n+h)*d+j]=bf(x); sum+=x;
        }
        gold[i*d+j]=bf((.5f+1e-6f)*sum);
    }
    auto floats=[](size_t count,float fill=0.f){return bytes(std::vector<float>(count,fill));};
    Tensor x({t,n,d},ACL_BF16,bytes(input)),phi({c,k},ACL_FLOAT,floats(c*k)),
           alpha({3},ACL_FLOAT,floats(3,1.f)),bias({c},ACL_FLOAT,floats(c)),
           hin({t,d},ACL_BF16,bytes(std::vector<uint16_t>(t*d,0x7fc0))),
           post({t,n},ACL_FLOAT,floats(t*n,NAN)),res({t,n*n},ACL_FLOAT,floats(t*n*n,NAN)),
           pre({t,n},ACL_FLOAT,floats(t*n)),mix({t,c},ACL_FLOAT,floats(t*c)),
           inv({t,1},ACL_FLOAT,floats(t)),sums({40,t,n},ACL_FLOAT,floats(40*t*n)),
           norm({40,t,n,n},ACL_FLOAT,floats(40*t*n*n));
    uint64_t size=0;aclOpExecutor *ex=nullptr;
    CHECK(aclnnMhcPreSinkhornGetWorkspaceSize(x.desc,phi.desc,alpha.desc,bias.desc,n,20,1e-6,1e-6,
          false,hin.desc,post.desc,res.desc,pre.desc,mix.desc,inv.desc,sums.desc,norm.desc,&size,&ex));
    Buffer workspace(size);
    CHECK(aclnnMhcPreSinkhorn(workspace.ptr,size,ex,stream));CHECK(aclrtSynchronizeStream(stream));
    std::vector<uint16_t> actual(t*d);CHECK(aclrtMemcpy(actual.data(),2*actual.size(),hin.buffer.ptr,
          2*actual.size(),ACL_MEMCPY_DEVICE_TO_HOST));
    size_t bad=0;
    for(size_t i=0;i<gold.size();++i) bad+=actual[i]!=gold[i] && !(gold[i]==0 && actual[i]==0x8000);
    auto check_fp=[&](Tensor& tensor,size_t count,float want){
        std::vector<float> out(count);CHECK(aclrtMemcpy(out.data(),4*count,tensor.buffer.ptr,4*count,ACL_MEMCPY_DEVICE_TO_HOST));
        for(float v:out) bad+=!(std::abs(v-want)<1e-4f);
    };
    check_fp(post,t*n,1.f);check_fp(res,t*n*n,1.f/n);
    std::printf("correctness Y count=%d bad=%zu\n",t*(d+n+n*n),bad);
    if(bad) throw std::runtime_error("PreSinkhorn inference values differ from golden");
}
int main(int argc,char** argv) {
    std::setvbuf(stdout,nullptr,_IONBF,0);
    if(argc!=5) return 2;
    try {
        CHECK(aclInit(nullptr));CHECK(aclrtSetDevice(0));aclrtStream stream;CHECK(aclrtCreateStream(&stream));
        std::printf("runtime_soc=%s\n",aclrtGetSocName());probe(std::stoi(argv[2]),std::stoi(argv[3]),std::stoi(argv[4]),stream);
        CHECK(aclrtDestroyStream(stream));CHECK(aclrtResetDevice(0));CHECK(aclFinalize());std::puts("SMOKE_PASS");std::_Exit(0);
    } catch(const std::exception& e){std::fprintf(stderr,"SMOKE_FAIL %s\n",e.what());return 1;}
}
