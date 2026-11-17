#include <acl/acl.h>
#include <aclnnop/aclnn_recurrent_kda.h>
#include <aclnnop/aclnn_chunk_kda_fwd.h>
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <stdexcept>
#include <string>
#include <vector>

// Standalone callers of the installed CANN operators; no replacement kernels.
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

static void compare(const std::vector<double>& actual, const std::vector<double>& expected,
                    bool bfloat, const char* name) {
    const double atol=bfloat?2e-6:std::ldexp(1.,-16);
    const double rtol=bfloat?std::ldexp(1.,-6):std::ldexp(1.,-10);
    const double limit=bfloat?1.:.01;
    size_t bad=0; double worst=0;
    for(size_t i=0;i<actual.size();++i) {
        double err=std::abs(actual[i]-expected.at(i));
        worst=std::max(worst,err);
        if(!std::isfinite(actual[i]) || err>atol+rtol*std::abs(expected[i])) ++bad;
    }
    std::printf("correctness name=%s count=%zu bad=%zu max_abs_error=%g atol=%g rtol=%g hard_limit=%g\n",
                name,actual.size(),bad,worst,atol,rtol,limit);
    // A small smoke checks every element, stricter than the skill's 99% matched ratio.
    if(bad || worst>limit) throw std::runtime_error("Correctness failed");
}
static std::vector<double> decode(const std::vector<uint16_t>& x) {
    std::vector<double> result; result.reserve(x.size());
    for(auto v:x) result.push_back(fp32(v));
    return result;
}
// High precision CPU reference operates on exactly the rounded BF16 inputs.
// Small but nonzero state/value keeps FP32-output accumulated BF16 error bounded;
// BF16 output absolute tolerance is 2e-6, much tighter than the generic 2^-6.
struct Inputs {
    int s,k,v;
    std::vector<uint16_t> q,key,value;
    std::vector<float> gate,beta,state;
    Inputs(int steps,int kd,int vd):s(steps),k(kd),v(vd),q(s*k),key(s*k),value(s*v),gate(s*k),beta(s),state(k*v) {
        for(int i=0;i<s*k;++i) {
            q[i]=bf16(float((i*7+3)%29-14)/64.f);
            key[i]=bf16(float((i*11+5)%31-15)/128.f);
            gate[i]=-float(1+(i*3)%7)/256.f;
        }
        for(int i=0;i<s*v;++i)value[i]=bf16(float((i*5+7)%23-11)/2048.f);
        for(int i=0;i<s;++i)beta[i]=float(3+i%5)/16.f;
        for(int i=0;i<k*v;++i)state[i]=float((i*13+1)%19-9)/8192.f;
    }
};
static std::pair<std::vector<double>,std::vector<double>> recurrent_gold(const Inputs& x) {
    std::vector<double> h(x.state.begin(),x.state.end()),o(x.s*x.v);
    for(int t=0;t<x.s;++t) {
        for(int d=0;d<x.k;++d)for(int e=0;e<x.v;++e)h[d*x.v+e]*=std::exp(double(x.gate[t*x.k+d]));
        for(int e=0;e<x.v;++e) {
            double dot=0;
            for(int d=0;d<x.k;++d)dot+=h[d*x.v+e]*fp32(x.key[t*x.k+d]);
            double delta=x.beta[t]*(fp32(x.value[t*x.v+e])-dot);
            for(int d=0;d<x.k;++d)h[d*x.v+e]+=fp32(x.key[t*x.k+d])*delta;
            for(int d=0;d<x.k;++d)o[t*x.v+e]+=h[d*x.v+e]*fp32(x.q[t*x.k+d])/std::sqrt(double(x.k));
        }
    }
    return {o,h};
}
static void run(aclrtStream stream,bool chunk,int s,int k) {
    const int v=k,c=64;
    Inputs in(s,k,v);
    Tensor<uint16_t> q({1,s,1,k},ACL_BF16,in.q),key({1,s,1,k},ACL_BF16,in.key),value({1,s,1,v},ACL_BF16,in.value);
    Tensor<float> gate({1,s,1,k},ACL_FLOAT,in.gate),beta({1,s,1},ACL_FLOAT,in.beta),initial({1,1,k,v},ACL_FLOAT,in.state);
    Tensor<uint16_t> out({1,s,1,v},ACL_BF16,std::vector<uint16_t>(s*v,0x7fc0));
    Tensor<float> final({1,1,k,v},ACL_FLOAT,std::vector<float>(k*v,NAN));
    uint64_t bytes=0;aclOpExecutor* executor=nullptr;double scale=1./std::sqrt(double(k));
    auto golden=recurrent_gold(in);
    if(!chunk) {
        ACL_CHECK(aclnnRecurrentKdaGetWorkspaceSize(q.tensor,key.tensor,value.tensor,gate.tensor,beta.tensor,initial.tensor,
            nullptr,nullptr,nullptr,nullptr,nullptr,"BSND",scale,true,false,false,false,false,false,false,-5.,false,
            out.tensor,final.tensor,&bytes,&executor));
        Buffer workspace(bytes);
        std::printf("case=kda_recurrent api=aclnnRecurrentKda B=1 S=%d Hq=1 Hv=1 K=%d V=%d dtype=BF16 workspace=%llu\n",s,k,v,(unsigned long long)bytes);
        ACL_CHECK(aclnnRecurrentKda(workspace.ptr,bytes,executor,stream));
        ACL_CHECK(aclrtSynchronizeStream(stream));
    } else {
        Tensor<uint16_t> aqk({1,1,s,c},ACL_BF16,std::vector<uint16_t>(s*c,0x7fc0));
        Tensor<uint16_t> akk({1,1,s,c},ACL_BF16,std::vector<uint16_t>(s*c,0x7fc0));
        Tensor<float> gk({1,1,s,k},ACL_FLOAT,std::vector<float>(s*k,NAN));
        ACL_CHECK(aclnnChunkKdaFwdGetWorkspaceSize(q.tensor,key.tensor,value.tensor,gate.tensor,beta.tensor,
            nullptr,nullptr,initial.tensor,nullptr,nullptr,"BSND",scale,c,false,-5.,false,false,
            out.tensor,final.tensor,gk.tensor,aqk.tensor,akk.tensor,nullptr,nullptr,nullptr,nullptr,nullptr,nullptr,&bytes,&executor));
        Buffer workspace(bytes);
        std::printf("case=kda_chunk api=aclnnChunkKdaFwd B=1 S=%d Hq=1 Hv=1 K=%d V=%d chunk=64 dtype=BF16 workspace=%llu\n",s,k,v,(unsigned long long)bytes);
        ACL_CHECK(aclnnChunkKdaFwd(workspace.ptr,bytes,executor,stream));
        ACL_CHECK(aclrtSynchronizeStream(stream));
        std::vector<double> gg(s*k),ga(s*c),gi(s*c);
        for(int start=0;start<s;start+=c) {
            int n=std::min(c,s-start);
            for(int i=0;i<n;++i)for(int d=0;d<k;++d)gg[(start+i)*k+d]=in.gate[(start+i)*k+d]/std::log(2.)+(i?gg[(start+i-1)*k+d]:0.);
            std::vector<double> lower(n*n);
            for(int i=0;i<n;++i)for(int j=0;j<=i;++j) {
                double qk=0,kk=0;
                for(int d=0;d<k;++d) {
                    double decay=std::exp2(gg[(start+i)*k+d]-gg[(start+j)*k+d]);
                    qk+=fp32(in.q[(start+i)*k+d])*fp32(in.key[(start+j)*k+d])*decay;
                    kk+=fp32(in.key[(start+i)*k+d])*fp32(in.key[(start+j)*k+d])*decay;
                }
                ga[(start+i)*c+j]=scale*qk;
                if(j<i)lower[i*n+j]=in.beta[start+i]*kk;
            }
            for(int i=0;i<n;++i)for(int j=0;j<=i;++j) {
                double x=(i==j)?1.:0.;
                for(int p=j;p<i;++p)x-=lower[i*n+p]*gi[(start+p)*c+j];
                gi[(start+i)*c+j]=x;
            }
        }
        compare(decode(aqk.read()),ga,true,"aqk");
        compare(decode(akk.read()),gi,true,"akk");
        auto actual=gk.read();compare(std::vector<double>(actual.begin(),actual.end()),gg,false,"gk");
    }
    compare(decode(out.read()),golden.first,true,"attn_out");
    auto actual=final.read();compare(std::vector<double>(actual.begin(),actual.end()),golden.second,false,"final_state");
    auto unchanged=initial.read();compare(std::vector<double>(unchanged.begin(),unchanged.end()),std::vector<double>(in.state.begin(),in.state.end()),false,"initial_state_unchanged");
}
int main(int argc,char** argv) {
    aclrtStream stream=nullptr;bool initialized=false,device=false;int result=0;
    try {
        if(argc<3||argc>4)throw std::runtime_error("Usage: extension_kda recurrent S | chunk S [K]");
        std::string op=argv[1];bool chunk=op=="chunk";int s=std::stoi(argv[2]);
        if(op!="recurrent"&&!chunk)throw std::runtime_error("Unknown operator");
        if(!chunk&&argc!=3)throw std::runtime_error("Recurrent K is fixed at128");
        int k=chunk?(argc==4?std::stoi(argv[3]):16):128;
        if(k!=16&&k!=128)throw std::runtime_error("Chunk smoke supports K=V=16 or128");
        if(s<1||s>(chunk?128:8))throw std::runtime_error("Unsupported smoke sequence length; recurrent S<=8, chunk S<=128");
        ACL_CHECK(aclInit(nullptr));initialized=true;ACL_CHECK(aclrtSetDevice(0));device=true;
        std::printf("runtime_soc=%s\n",aclrtGetSocName());ACL_CHECK(aclrtCreateStream(&stream));
        run(stream,chunk,s,k);std::puts("SMOKE_PASS");
    } catch(const std::exception& e){std::fprintf(stderr,"SMOKE_FAIL %s\n",e.what());result=1;}
    if(stream)aclrtDestroyStream(stream);
    if(device)aclrtResetDevice(0);
    if(initialized)aclFinalize();
    return result;
}
