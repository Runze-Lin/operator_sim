// Small, independently verifiable dispatches for the decode graph's tensor primitives.
#include "../acl_tensor.hpp"
#include <aclnnop/aclnn_pow.h>
#include <aclnnop/aclnn_mean.h>
#include <aclnnop/aclnn_reduce_sum.h>
#include <aclnnop/aclnn_rsqrt.h>
#include <aclnnop/aclnn_neg.h>
#include <aclnnop/aclnn_lt_scalar.h>
#include <aclnnop/aclnn_ge_scalar.h>
#include <aclnnop/aclnn_bitwise_or_tensor.h>
#include <aclnnop/aclnn_sub.h>
#include <aclnnop/aclnn_masked_fill_scalar.h>
#include <aclnnop/aclnn_embedding.h>
#include <aclnnop/aclnn_gather_v2.h>
#include <aclnnop/aclnn_cat.h>
#include <aclnnop/aclnn_constant_pad_nd.h>
#include <aclnnop/aclnn_cast.h>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>

template<class T> static std::vector<uint8_t> bytes(const std::vector<T>& v) {
    std::vector<uint8_t> out(v.size()*sizeof(T));
    std::memcpy(out.data(),v.data(),out.size()); return out;
}
static uint16_t bf(float v) {
    uint32_t u; std::memcpy(&u,&v,4); return u>>16;
}
template<class T, class F> static void verify(const char* api, Tensor& result, size_t n, F golden) {
    std::vector<T> out(n);
    CHECK(aclrtMemcpy(out.data(),n*sizeof(T),result.buffer.ptr,n*sizeof(T),ACL_MEMCPY_DEVICE_TO_HOST));
    size_t bad=0;
    for (size_t i=0;i<n;++i) {
        const double error = double(out[i]) - double(golden(i));
        if (!std::isfinite(error) || std::abs(error) > 1e-5) ++bad;
    }
    std::printf("verified %s n=%zu bad=%zu\n",api,n,bad);
    if(bad) throw std::runtime_error("primitive golden mismatch");
}
#define RUN(op, ...) do { \
    std::printf("launch=%s\n", #op); uint64_t ws=0; aclOpExecutor* ex=nullptr; \
    CHECK(op##GetWorkspaceSize(__VA_ARGS__, &ws,&ex)); Buffer scratch(ws); \
    CHECK(op(scratch.ptr,ws,ex,stream)); CHECK(aclrtSynchronizeStream(stream)); \
} while(false)

static void scalar_ops(int r,aclrtStream stream) {
    std::vector<int64_t> data(r); for(int i=0;i<r;++i) data[i]=i-4;
    Tensor x({r},ACL_INT64,bytes(data)),sub({r},ACL_INT64,bytes(data));
    Tensor left({r},ACL_BOOL,std::vector<uint8_t>(r)),right({r},ACL_BOOL,std::vector<uint8_t>(r)),both({r},ACL_BOOL,std::vector<uint8_t>(r));
    int64_t zero=0,one=1,four=4;
    auto* z=aclCreateScalar(&zero,ACL_INT64);auto* unity=aclCreateScalar(&one,ACL_INT64);auto* hi=aclCreateScalar(&four,ACL_INT64);
    RUN(aclnnLtScalar,x.desc,z,left.desc); verify<uint8_t>("aclnnLtScalar",left,r,[&](size_t i){return data[i]<0;});
    RUN(aclnnGeScalar,x.desc,hi,right.desc); verify<uint8_t>("aclnnGeScalar",right,r,[&](size_t i){return data[i]>=4;});
    RUN(aclnnBitwiseOrTensor,left.desc,right.desc,both.desc); verify<uint8_t>("aclnnBitwiseOrTensor",both,r,[&](size_t i){return data[i]<0 || data[i]>=4;});
    RUN(aclnnSubs,x.desc,hi,unity,sub.desc); verify<int64_t>("aclnnSubs",sub,r,[&](size_t i){return data[i]-4;});
    RUN(aclnnInplaceMaskedFillScalar,sub.desc,both.desc,z); verify<int64_t>("aclnnInplaceMaskedFillScalar",sub,r,[&](size_t i){return data[i]<0 || data[i]>=4 ? 0 : data[i]-4;});
    aclDestroyScalar(z);aclDestroyScalar(unity);aclDestroyScalar(hi);
}
static void math_ops(int r,aclrtStream stream) {
    const int d=64,n=r*d;
    std::vector<float> v(n); std::vector<uint16_t> b(n); for(int i=0;i<n;++i){v[i]=1+(i%4);b[i]=bf(v[i]);}
    Tensor x({r,d},ACL_FLOAT,bytes(v)),y({r,d},ACL_FLOAT,bytes(v)), bx({r,1,1,d},ACL_BF16,bytes(b)), by({r,1,1,d},ACL_BF16,bytes(b));
    RUN(aclnnNeg,bx.desc,by.desc);verify<uint16_t>("aclnnNeg",by,n,[&](size_t i){return bf(-v[i]);});
    float two=2;auto* exponent=aclCreateScalar(&two,ACL_FLOAT);
    RUN(aclnnPowTensorScalar,x.desc,exponent,y.desc);verify<float>("aclnnPowTensorScalar",y,n,[&](size_t i){return v[i]*v[i];});
    Tensor red({r},ACL_FLOAT,std::vector<uint8_t>(r*4)); int64_t axis=1; auto* dims=aclCreateIntArray(&axis,1);
    RUN(aclnnReduceSum,x.desc,dims,false,ACL_FLOAT,red.desc);verify<float>("aclnnReduceSum",red,r,[](size_t){return 160;});
    RUN(aclnnMean,x.desc,dims,false,ACL_FLOAT,red.desc);verify<float>("aclnnMean",red,r,[](size_t){return 2.5f;});
    RUN(aclnnRsqrt,x.desc,y.desc);verify<float>("aclnnRsqrt",y,n,[&](size_t i){return 1/std::sqrt(v[i]);});
    aclDestroyIntArray(dims);aclDestroyScalar(exponent);
}
static void memory_ops(int r,aclrtStream stream) {
    const int d=64,rows=128,n=r*d;
    std::vector<uint16_t> raw(rows*d); for(int i=0;i<rows*d;++i) raw[i]=bf(1+(i%61));
    std::vector<int64_t> ids(r);for(int i=0;i<r;++i) ids[i]=(i*7)%rows;
    Tensor table({rows,d},ACL_BF16,bytes(raw)),indices({r},ACL_INT64,bytes(ids));
    Tensor value({r,d},ACL_BF16,std::vector<uint8_t>(2*n)),copy({r,d},ACL_BF16,std::vector<uint8_t>(2*n));
    auto val=[&](size_t i){return raw[ids[i/d]*d+i%d];};
    RUN(aclnnEmbedding,table.desc,indices.desc,value.desc);verify<uint16_t>("aclnnEmbedding",value,n,val);
    RUN(aclnnGatherV2,table.desc,0,indices.desc,copy.desc);verify<uint16_t>("aclnnGatherV2",copy,n,val);
    Tensor cast({r,d},ACL_FLOAT,std::vector<uint8_t>(4*n));
    RUN(aclnnCast,value.desc,ACL_FLOAT,cast.desc);verify<float>("aclnnCast",cast,n,[&](size_t i){return 1+((ids[i/d]*d+i%d)%61);});
    Tensor joined({r,2*d},ACL_BF16,std::vector<uint8_t>(4*n));
    aclTensor* ts[]={value.desc,copy.desc};auto* list=aclCreateTensorList(ts,2);
    RUN(aclnnCat,list,1,joined.desc);verify<uint16_t>("aclnnCat",joined,2*n,[&](size_t i){return val(i/(2*d)*d+i%d);});
    // The tensor list owns its descriptors; these two are still used below.
    int64_t padding[]={0,0,0,7};auto* pads=aclCreateIntArray(padding,4);
    float zero=0;auto* z=aclCreateScalar(&zero,ACL_FLOAT);
    Tensor padded({r+7,d},ACL_BF16,std::vector<uint8_t>(2*(r+7)*d));
    RUN(aclnnConstantPadNd,value.desc,pads,z,padded.desc);verify<uint16_t>("aclnnConstantPadNd",padded,(r+7)*d,[&](size_t i){return i<size_t(n)?val(i):0;});
    aclDestroyIntArray(pads);aclDestroyScalar(z);
    aclDestroyTensorList(list);value.desc=nullptr;copy.desc=nullptr;
}
int main(int argc,char** argv) {
    std::setvbuf(stdout,nullptr,_IONBF,0);
    try {
        if(argc!=3) return 2;
        std::string group=argv[1];int rows=std::stoi(argv[2]);if(rows<=0 || rows>4096) return 2;
        CHECK(aclInit(nullptr));CHECK(aclrtSetDevice(0));aclrtStream stream;CHECK(aclrtCreateStream(&stream));
        std::printf("runtime_soc=%s group=%s rows=%d\n",aclrtGetSocName(),group.c_str(),rows);
        if(group=="mask") scalar_ops(rows,stream);
        else if(group=="math") math_ops(rows,stream);
        else if(group=="memory") memory_ops(rows,stream);
        else return 2;
        CHECK(aclrtDestroyStream(stream));CHECK(aclrtResetDevice(0));CHECK(aclFinalize());
        std::puts("SMOKE_PASS");std::_Exit(EXIT_SUCCESS);
    } catch(const std::exception& e){std::fprintf(stderr,"SMOKE_FAIL %s\n",e.what());return 1;}
}
