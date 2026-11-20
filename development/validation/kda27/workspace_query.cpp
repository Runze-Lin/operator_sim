// Shape and host workspace-query experiment. Contains no kernel execution call.
#include <acl/acl.h>
#include <aclnnop/aclnn_chunk_kda_fwd_v2.h>
#include <cstdio>
#include <cmath>
#include <cstdlib>
#include <dlfcn.h>
#include <stdexcept>
#include <string>
#include <vector>

struct Tensors {
    std::vector<aclTensor*> storage;
    aclTensor* tensor(std::vector<int64_t> shape,aclDataType type) {
        std::vector<int64_t> stride(shape.size(),1);
        for (size_t i=shape.size()-1;i>0;--i) stride[i-1]=shape[i]*stride[i];
        auto* t=aclCreateTensor(shape.data(),shape.size(),type,stride.data(),0,ACL_FORMAT_ND,shape.data(),shape.size(),nullptr);
        if (!t) throw std::runtime_error("null tensor descriptor");
        storage.push_back(t);return t;
    }
    ~Tensors(){for(auto* t:storage) aclDestroyTensor(t);}
};

int main(int argc,char** argv) {
    if(argc!=4) return 2;
    std::setvbuf(stdout,nullptr,_IOLBF,0);
    const int64_t seq=std::stoll(argv[1]),batch=std::stoll(argv[2]),h=std::stoll(argv[3]);
    if(seq<=0 || batch<=0 || h<=0) return 2;
    Dl_info provider{};
    const char *sym="aclnnChunkKdaFwdV2GetWorkspaceSize",*expected=std::getenv("FLA_NPU_OP_API_LIB");
    if(!expected||!dladdr(dlsym(RTLD_DEFAULT,sym),&provider)||std::string(expected)!=provider.dli_fname) return 7;
    printf("symbol=%s provider=%s\n",sym,provider.dli_fname);
    // No aclInit or aclrtSetDevice. The phase-1 library may discover its own
    // platform. Only report what its underlying HAL actually returns.
    const char* soc=aclrtGetSocName(); printf("soc=%s\n",soc?soc:"unresolved");
    Tensors x;int64_t s=seq*batch;std::vector<int64_t> cu{0},idx;
    for(int64_t b=0;b<batch;b++){cu.push_back((b+1)*seq);for(int64_t i=0;i<(seq+63)/64;i++){idx.push_back(b);idx.push_back(i);}}
    auto offsets=aclCreateIntArray(cu.data(),cu.size()),indices=aclCreateIntArray(idx.data(),idx.size());
    std::vector<int64_t> bsnd{1,s,h,128},vk{batch,h,128,128},matrix{1,h,s,64};
    auto q=x.tensor(bsnd,ACL_BF16),k=x.tensor(bsnd,ACL_BF16),v=x.tensor(bsnd,ACL_BF16),g=x.tensor(bsnd,ACL_BF16);
    auto beta=x.tensor({1,s,h},ACL_FLOAT),a=x.tensor({h},ACL_FLOAT),bias=x.tensor({h*128},ACL_FLOAT);
    auto init=x.tensor(vk,ACL_FLOAT),out=x.tensor(bsnd,ACL_BF16),f=x.tensor(vk,ACL_FLOAT),aq=x.tensor(matrix,ACL_BF16),ak=x.tensor(matrix,ACL_BF16);
    uint64_t bytes=0;aclOpExecutor* exec=nullptr;
    int code=aclnnChunkKdaFwdV2GetWorkspaceSize(q,k,v,g,beta,a,bias,init,offsets,indices,"BSND",1./std::sqrt(128.),64,true,-5.,true,true,1e-6,false,false,false,true,out,f,nullptr,aq,ak,nullptr,nullptr,nullptr,nullptr,nullptr,nullptr,nullptr,nullptr,nullptr,nullptr,nullptr,&bytes,&exec);
    printf("query_status=%d executor=%d workspace=%llu total_tokens=%lld heads=%lld sequences=%lld\n",code,exec!=nullptr,(unsigned long long)bytes,(long long)s,(long long)h,(long long)batch);
    aclDestroyIntArray(indices);aclDestroyIntArray(offsets);
    // Executor owns transient host descriptors until process exit. No workspace
    // or tensor memory is allocated. Neither launch nor synchronization runs.
    return (code==0 && exec!=nullptr)?0:4;
}
