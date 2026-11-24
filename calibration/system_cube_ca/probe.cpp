#include "../acl_tensor.hpp"
#include <aclnnop/aclnn_mm.h>
#include <aclnnop/aclnn_addmm.h>
#include <algorithm>
#include <cstdio>
#include <cstring>

// Identity-coded rows select a different K and column for every output. All
// numbers are dyadic and exactly representable in both BF16 and FP32.
static std::vector<uint8_t> encode(const std::vector<float>& x, bool bf16) {
    std::vector<uint8_t> result(x.size() * (bf16 ? 2 : 4));
    if (!bf16) std::memcpy(result.data(),x.data(),result.size());
    else for(size_t i=0; i<x.size(); ++i) {
        uint32_t bits; std::memcpy(&bits,&x[i],4); uint16_t hi=bits>>16;
        std::memcpy(&result[2*i],&hi,2);
    }
    return result;
}

static void measure(int m,int n,int k,bool bf16,bool bias,aclrtStream stream,bool fp32_output=false) {
    std::vector<float> a(m*k,0), w(n*k), c(n), y(m*n,0);
    for(int i=0;i<m;++i) a[i*k+(7*i+11)%k] = 1;
    for(int j=0;j<n;++j) {
        c[j] = .25f;
        for(int l=0;l<k;++l) w[j*k+l] = ((3*j+l)%17-8)*.125f;
    }
    auto dt = bf16 ? ACL_BF16 : ACL_FLOAT;
    Tensor x({m,k},dt,encode(a,bf16)), b({n,k},dt,encode(w,bf16));
    Tensor offset({n},dt,encode(c,bf16)), out({m,n},fp32_output?ACL_FLOAT:dt,encode(y,bf16 && !fp32_output));
    int64_t dims[]={k,n}, strides[]={1,k}, storage[]={n,k};
    auto wt=aclCreateTensor(dims,2,dt,strides,0,ACL_FORMAT_ND,storage,2,b.buffer.ptr);
    if (!wt) throw std::runtime_error("invalid transposed weight");
    uint64_t ws=0; aclOpExecutor* e=nullptr; float one=1;
    auto scale=aclCreateScalar(&one,ACL_FLOAT);
    std::printf("case=%s dtype=%s M=%d N=%d K=%d\n",bias?"Addmm":"Mm",fp32_output?"BF16_FP32":bf16?"BF16":"FP32",m,n,k);
    if(bias) CHECK(aclnnAddmmGetWorkspaceSize(offset.desc,x.desc,wt,scale,scale,out.desc,0,&ws,&e));
    else CHECK(aclnnMmGetWorkspaceSize(x.desc,wt,out.desc,0,&ws,&e));
    Buffer scratch(ws);
    if(bias) CHECK(aclnnAddmm(scratch.ptr,ws,e,stream)); else CHECK(aclnnMm(scratch.ptr,ws,e,stream));
    CHECK(aclrtSynchronizeStream(stream));
    std::vector<uint8_t> result((bf16 && !fp32_output?2:4)*m*n);
    CHECK(aclrtMemcpy(result.data(),result.size(),out.buffer.ptr,result.size(),ACL_MEMCPY_DEVICE_TO_HOST));
    for(int i=0;i<m;++i) for(int j=0;j<n;++j) y[i*n+j]=w[j*k+(7*i+11)%k]+(bias?c[j]:0);
    if(result!=encode(y,bf16 && !fp32_output)) throw std::runtime_error("incorrect output");
    std::printf("checked=%d bad=0\n",m*n);
    aclDestroyScalar(scale);aclDestroyTensor(wt);
}

int main(int argc,char** argv) {
    std::setvbuf(stdout,nullptr,_IONBF,0);
    try {
        CHECK(aclInit(nullptr)); CHECK(aclrtSetDevice(0));aclrtStream s;CHECK(aclrtCreateStream(&s));
        std::printf("runtime_soc=%s\n",aclrtGetSocName());
        if (argc == 2 && std::strcmp(argv[1],"router-holdout") == 0) {
            measure(144,896,7168,true,false,s,true); // next distinct Kimi capture B16, M=16x9
        } else if (argc == 2 && std::strcmp(argv[1],"router") == 0) {
            measure(9,896,7168,true,false,s,true);  // Kimi BF16 projection retains FP32
        } else if (argc == 2 && std::strcmp(argv[1],"holdout") == 0) {
            measure(9,4,28672,false,false,s);    // Pro mHC
            measure(9,384,7168,false,false,s);   // Pro router
            measure(128,1,5376,false,true,s);    // Flash draft B16 acceptance
            measure(144,128,512,true,false,s);   // Flash B16 projection
        } else {
            measure(9,4,20480,false,false,s);     // Flash FP32 mHC projection
            measure(9,384,5120,false,false,s);    // Flash FP32 router
            measure(8,1,7680,false,true,s);       // Kimi FP32 output head
            measure(9,128,512,true,false,s);      // Flash transposed BF16 projection
        }
        CHECK(aclrtDestroyStream(s));CHECK(aclrtResetDevice(0));CHECK(aclFinalize());
        std::puts("SMOKE_PASS");std::_Exit(EXIT_SUCCESS);
    } catch(const std::exception& e){std::fprintf(stderr,"%s\n",e.what());return 1;}
}
