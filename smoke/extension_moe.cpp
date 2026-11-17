#include <acl/acl.h>
#include <aclnnop/aclnn_grouped_matmul_swiglu_quant_v2.h>
#include <aclnnop/aclnn_grouped_matmul_v5.h>
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <stdexcept>
#include <string>
#include <sstream>
#include <vector>
static void check(int code, const char* operation) {
    if (code) throw std::runtime_error(std::string(operation) + ": " + std::to_string(code));
}
#define ACL_CHECK(expr) check((expr), #expr)

static uint16_t bf16(float x) {
    uint32_t bits;
    std::memcpy(&bits, &x, sizeof(bits));
    bits += 0x7fff + ((bits >> 16) & 1);
    return static_cast<uint16_t>(bits >> 16);
}
static float fp32(uint16_t x) {
    uint32_t bits = uint32_t(x) << 16;
    float result;
    std::memcpy(&result, &bits, sizeof(result));
    return result;
}

struct Buffer {
    void* ptr = nullptr;
    explicit Buffer(size_t bytes) {
        if (bytes) ACL_CHECK(aclrtMalloc(&ptr, bytes, ACL_MEM_MALLOC_HUGE_FIRST));
    }
    ~Buffer() { if (ptr) aclrtFree(ptr); }
    Buffer(const Buffer&) = delete;
    Buffer& operator=(const Buffer&) = delete;
};

template <class T> struct Tensor {
    Buffer memory;
    aclTensor* tensor = nullptr;
    size_t count;
    Tensor(std::vector<int64_t> shape, aclDataType dtype, const std::vector<T>& input)
        : memory(input.size() * sizeof(T)), count(input.size()) {
        int64_t elements = 1;
        for (auto d : shape) elements *= d;
        if (elements != int64_t(count)) throw std::runtime_error("Shape/data mismatch");
        std::vector<int64_t> strides(shape.size(), 1);
        for (int i = int(shape.size()) - 2; i >= 0; --i) strides[i] = strides[i+1] * shape[i+1];
        ACL_CHECK(aclrtMemcpy(memory.ptr, count*sizeof(T), input.data(), count*sizeof(T), ACL_MEMCPY_HOST_TO_DEVICE));
        tensor = aclCreateTensor(shape.data(), shape.size(), dtype, strides.data(), 0,
                                 ACL_FORMAT_ND, shape.data(), shape.size(), memory.ptr);
        if (!tensor) throw std::runtime_error("aclCreateTensor failed");
    }
    ~Tensor() { if (tensor) aclDestroyTensor(tensor); }
    std::vector<T> read() const {
        std::vector<T> out(count);
        ACL_CHECK(aclrtMemcpy(out.data(), count*sizeof(T), memory.ptr, count*sizeof(T), ACL_MEMCPY_DEVICE_TO_HOST));
        return out;
    }
};


struct TensorList {
    aclTensorList* value;
    explicit TensorList(aclTensor*& t) {
        const aclTensor* ts[]={t}; value=aclCreateTensorList(ts,1);
        if(!value) throw std::runtime_error("aclCreateTensorList failed");
        // aclDestroyTensorList destroys each member metadata object too. Transfer
        // ownership only after successful construction; Tensor still owns Buffer.
        // Leaving the original pointer would return metadata twice to BlockCache.
        t=nullptr;
    }
    ~TensorList(){aclDestroyTensorList(value);}
    TensorList(const TensorList&)=delete;
    TensorList& operator=(const TensorList&)=delete;
};
static double fp8(uint8_t b) {
    int e=(b>>3)&15, m=b&7;
    double v=e?std::ldexp(1.+m/8.,e-7):std::ldexp(double(m),-9);
    if(e==15 && m==7) return NAN;
    return b&128?-v:v;
}
static uint8_t encode8(double x) {
    double v=std::min(std::abs(x),448.);int best=0;double err=1e300;
    for(int i=0;i<127;++i) {
        double d=std::abs(fp8(i)-v);
        if(d<err || (d==err && !(i&1))){best=i;err=d;}
    }
    return uint8_t(best|(std::signbit(x)?128:0));
}
static std::vector<uint8_t> values(size_t n,int seed) {
    std::vector<uint8_t> result(n);
    // Nonzero, signed, nonuniform values. Powers of two scales keep CPU dequant exact.
    for(size_t i=0;i<n;++i) {
        int v=int((i*13+i/7+seed*3)%15)-7;
        result[i]=encode8((v?v:1)/8.);
    }
    return result;
}
static std::vector<uint8_t> scales(size_t n,int seed) {
    std::vector<uint8_t> result(n);
    for(size_t i=0;i<n;++i)result[i]=uint8_t(125+((i+seed)%3));
    return result;
}
static double s8(uint8_t v){return std::ldexp(1.,int(v)-127);}
static std::vector<double> gmm(const std::vector<uint8_t>& x,const std::vector<uint8_t>& w,
 const std::vector<uint8_t>& xs,const std::vector<uint8_t>& ws,
 int m,int k,int n,const std::vector<int64_t>& counts) {
    std::vector<double> y(size_t(m)*n,NAN);int row=0;int kg=(k+63)/64;
    for(size_t e=0;e<counts.size();++e)for(int rr=0;rr<counts[e];++rr,++row)
        for(int j=0;j<n;++j) {
            double sum=0;
            for(int q=0;q<k;++q)sum+=fp8(x[size_t(row)*k+q])*fp8(w[(e*k+q)*n+j])*
                s8(xs[(size_t(row)*kg+q/64)*2+(q/32)%2])*
                s8(ws[((e*kg+q/64)*n+j)*2+(q/32)%2]);
            y[size_t(row)*n+j]=sum;
        }
    return y;
}
static void compare(const char* name,const std::vector<double>& actual,
                    const std::vector<double>& expected,double atol,double rtol,double hard_limit) {
    size_t bad=0;double worst=0;
    for(size_t i=0;i<actual.size();++i) {
        double err=std::abs(actual[i]-expected.at(i));worst=std::max(worst,err);
        if(!std::isfinite(actual[i]) || !std::isfinite(expected[i]) || err>atol+rtol*std::abs(expected[i])) {
            if(bad<3)std::printf("mismatch name=%s index=%zu actual=%g expected=%g\n",name,i,actual[i],expected[i]);
            ++bad;
        }
    }
    std::printf("correctness name=%s count=%zu bad=%zu max_abs_error=%g atol=%g rtol=%g hard_limit=%g\n",
        name,actual.size(),bad,worst,atol,rtol,hard_limit);
    if(bad || worst>hard_limit)throw std::runtime_error(std::string(name)+" correctness failed");
}
static void run(aclrtStream stream,const std::string& op,int m,int k,int f,const std::vector<int64_t>& counts) {
    int e=int(counts.size()),valid=0;for(auto c:counts)valid+=int(c);
    int kg=(k+63)/64,fg=(f+63)/64;
    auto xd=values(size_t(m)*k,1),w1d=values(size_t(e)*k*2*f,2);
    auto xsd=scales(size_t(m)*kg*2,0),w1sd=scales(size_t(e)*kg*2*f*2,1);
    auto w2d=values(size_t(e)*f*k,3),w2sd=scales(size_t(e)*fg*k*2,2);
    auto yd=values(size_t(m)*f,4),ysd=scales(size_t(m)*fg*2,2);
    if(op!="gmm2") {std::fill(yd.begin(),yd.end(),0x7f);std::fill(ysd.begin(),ysd.end(),0xff);}
    Tensor<uint8_t> x({m,k},ACL_FLOAT8_E4M3FN,xd),w1({e,k,2*f},ACL_FLOAT8_E4M3FN,w1d);
    Tensor<uint8_t> xs({m,kg,2},ACL_FLOAT8_E8M0,xsd),w1s({e,kg,2*f,2},ACL_FLOAT8_E8M0,w1sd);
    Tensor<uint8_t> y({m,f},ACL_FLOAT8_E4M3FN,yd),ys({m,fg,2},ACL_FLOAT8_E8M0,ysd);
    Tensor<int64_t> groups({e},ACL_INT64,counts);
    std::vector<uint8_t> goldenY=yd,goldenS=ysd;
    if(op!="gmm2") {
        TensorList wl(w1.tensor),sl(w1s.tensor);uint64_t bytes=0;aclOpExecutor* exec=nullptr;
        ACL_CHECK(aclnnGroupedMatmulSwigluQuantV2GetWorkspaceSize(x.tensor,wl.value,sl.value,nullptr,
            nullptr,xs.tensor,nullptr,groups.tensor,2,0,2,1,nullptr,y.tensor,ys.tensor,&bytes,&exec));
        Buffer workspace(bytes);
        std::printf("stage=grouped_matmul_swiglu_quant_v2 api=aclnnGroupedMatmulSwigluQuantV2 workspace=%llu\n",(unsigned long long)bytes);std::fflush(stdout);
        ACL_CHECK(aclnnGroupedMatmulSwigluQuantV2(workspace.ptr,bytes,exec,stream));
        ACL_CHECK(aclrtSynchronizeStream(stream));
        std::puts("stage_complete=grouped_matmul_swiglu_quant_v2");
        auto c=gmm(xd,w1d,xsd,w1sd,m,k,2*f,counts);std::vector<double> sw(size_t(valid)*f);
        for(int r=0;r<valid;++r)for(int j=0;j<f;++j) {
            double a=c[size_t(r)*2*f+j],gate=c[size_t(r)*2*f+f+j];
            // Installed CGMCT epilogue rounds SwiGLU to BF16 before MX exponent reduction.
            sw[size_t(r)*f+j]=fp32(bf16(float(a/(1.+std::exp(-a))*gate)));
        }
        for(int r=0;r<valid;++r)for(int b=0;b<f/32;++b) {
            double mx=0;for(int j=0;j<32;++j)mx=std::max(mx,std::abs(sw[size_t(r)*f+b*32+j]));
            int exponent=mx?std::max(-127,int(std::floor(std::log2(mx)))-8):-127;
            goldenS[size_t(r)*fg*2+b]=uint8_t(exponent+127);
            for(int j=0;j<32;++j)goldenY[size_t(r)*f+b*32+j]=encode8(std::ldexp(sw[size_t(r)*f+b*32+j],-exponent));
        }
        auto ay=y.read(),as=ys.read();std::vector<double> actual,expected;
        for(int r=0;r<valid;++r)for(int j=0;j<f;++j){actual.push_back(fp8(ay[size_t(r)*f+j]));expected.push_back(fp8(goldenY[size_t(r)*f+j]));}
        compare("gmm1_output",actual,expected,1./16.,1./4.,4.);
        actual.clear();expected.clear();for(int r=0;r<valid;++r)for(int j=0;j<fg*2;++j){actual.push_back(as[size_t(r)*fg*2+j]);expected.push_back(goldenS[size_t(r)*fg*2+j]);}
        compare("gmm1_mx_scale",actual,expected,0,0,0);
    }
    if(op!="grouped_matmul_swiglu_quant_v2") {
        Tensor<uint8_t> w2({e,f,k},ACL_FLOAT8_E4M3FN,w2d),w2s({e,fg,k,2},ACL_FLOAT8_E8M0,w2sd);
        Tensor<uint16_t> out({m,k},ACL_BF16,std::vector<uint16_t>(size_t(m)*k,0x7fc0));
        TensorList xl(y.tensor),wl(w2.tensor),sl(w2s.tensor),ps(ys.tensor),ol(out.tensor);
        uint64_t bytes=0;aclOpExecutor* exec=nullptr;
        ACL_CHECK(aclnnGroupedMatmulV5GetWorkspaceSize(xl.value,wl.value,nullptr,sl.value,nullptr,nullptr,
            nullptr,ps.value,groups.tensor,nullptr,nullptr,nullptr,3,0,1,0,nullptr,ol.value,nullptr,nullptr,&bytes,&exec));
        Buffer workspace(bytes);
        std::printf("stage=gmm2 api=aclnnGroupedMatmulV5 workspace=%llu\n",(unsigned long long)bytes);std::fflush(stdout);
        ACL_CHECK(aclnnGroupedMatmulV5(workspace.ptr,bytes,exec,stream));ACL_CHECK(aclrtSynchronizeStream(stream));
        std::puts("stage_complete=gmm2");
        auto gold=gmm(goldenY,w2d,goldenS,w2sd,m,f,k,counts);gold.resize(size_t(valid)*k);
        auto actualRaw=out.read();std::vector<double> actual;for(int i=0;i<valid*k;++i)actual.push_back(fp32(actualRaw[i]));
        compare("gmm2_output",actual,gold,1./64.,1./64.,1.);
    }
}
static int metadata_selftest() {
    // CPU-only: exercise the real libnnopbase ownership/cache destruction path,
    // without ACL initialization, a device allocation or CA simulation.
    int64_t shape[]={1,64},stride[]={64,1};uint8_t backing[64]={};
    for(int i=0;i<1000;++i) {
        aclTensor* metadata=aclCreateTensor(shape,2,ACL_FLOAT8_E4M3FN,stride,0,ACL_FORMAT_ND,shape,2,backing);
        if(!metadata)throw std::runtime_error("selftest metadata create failed");
        {TensorList owner(metadata);if(metadata)throw std::runtime_error("metadata ownership was not transferred");}
        if(metadata)aclDestroyTensor(metadata);
    }
    std::puts("METADATA_SELFTEST_PASS transferred=1000 device_allocations=0");
    return 0;
}
int main(int argc,char** argv) {
    if(argc==2 && std::string(argv[1])=="--metadata-selftest")return metadata_selftest();
    aclrtStream stream=nullptr;bool initialized=false,device=false;int result=0;
    try {
        if(argc!=7)throw std::runtime_error("Usage: extension_moe moe_ffn|grouped_matmul_swiglu_quant_v2|gmm2 M K F E counts_csv");
        std::string op=argv[1];if(op!="moe_ffn"&&op!="grouped_matmul_swiglu_quant_v2"&&op!="gmm2")throw std::runtime_error("Unknown op");
        auto dim=[](const char* s){size_t p;long long v=std::stoll(s,&p);if(p!=std::strlen(s)||v<1||v>4096)throw std::runtime_error("Bad dimension");return int(v);};
        int m=dim(argv[2]),k=dim(argv[3]),f=dim(argv[4]),e=dim(argv[5]);
        std::vector<int64_t> counts;std::stringstream ss(argv[6]);std::string token;int64_t total=0;
        while(std::getline(ss,token,',')){size_t p;auto v=std::stoll(token,&p);if(p!=token.size()||v<0)throw std::runtime_error("Bad counts");counts.push_back(v);total+=v;}
        if(int(counts.size())!=e||total<1||total>m||k%64||f%64||int64_t(m)*k*f*e>16*1024*1024)throw std::runtime_error("Unsupported smoke shape/budget");
        ACL_CHECK(aclInit(nullptr));initialized=true;ACL_CHECK(aclrtSetDevice(0));device=true;
        std::printf("runtime_soc=%s\ncase=%s M=%d K=%d F=%d E=%d expert_counts=%s valid_rows=%lld groupListType=1 format=ND input=E4M3 output_gmm1=E4M3 output_gmm2=BF16\n",aclrtGetSocName(),op.c_str(),m,k,f,e,argv[6],(long long)total);
        ACL_CHECK(aclrtCreateStream(&stream));run(stream,op,m,k,f,counts);
        std::puts("cleanup=tensor_metadata_and_buffers_complete");std::puts("SMOKE_PASS");
    } catch(const std::exception& e){std::fprintf(stderr,"SMOKE_FAIL %s\n",e.what());result=1;}
    if(stream){std::puts("cleanup=stream_begin");aclrtDestroyStream(stream);std::puts("cleanup=stream_complete");}
    if(device){std::puts("cleanup=device_reset_begin");aclrtResetDevice(0);std::puts("cleanup=device_reset_complete");}
    if(initialized){std::puts("cleanup=acl_finalize_begin");aclFinalize();std::puts("cleanup=acl_finalize_complete");}
    std::puts("cleanup=main_return");std::fflush(stdout);
    return result;
}
