#include <acl/acl.h>
#include <aclnnop/aclnn_dynamic_mx_quant.h>
#include <aclnnop/aclnn_matmul.h>
#include <aclnnop/aclnn_flash_attention_score.h>
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

static void compare(const std::vector<uint16_t>& actual, const std::vector<double>& expected,
                    double atol, double rtol) {
    size_t bad = 0;
    double max_error = 0;
    for (size_t i = 0; i < actual.size(); ++i) {
        double value = fp32(actual[i]);
        double error = std::abs(value - expected.at(i));
        max_error = std::max(max_error, error);
        if (!std::isfinite(value) || error > atol + rtol * std::abs(expected[i])) ++bad;
    }
    std::printf("correctness elements=%zu bad=%zu max_abs_error=%.9g atol=%g rtol=%g\n",
                actual.size(), bad, max_error, atol, rtol);
    if (bad) throw std::runtime_error("Numerical comparison failed");
}

static void quant(aclrtStream stream, int rows, int cols) {
    const float pattern[] = {0, 8, 64, 512};
    const uint8_t expected[] = {0, 72, 96, 120}; // E4M3FN: 0,4,32,256; scale=2.
    std::vector<uint16_t> data(rows*cols);
    for (size_t i = 0; i < data.size(); ++i) data[i] = bf16(pattern[i%4]);
    Tensor<uint16_t> x({rows,cols}, ACL_BF16, data);
    Tensor<uint8_t> y({rows,cols}, ACL_FLOAT8_E4M3FN, std::vector<uint8_t>(rows*cols, 0xff));
    Tensor<uint8_t> scales({rows,cols/64,2}, ACL_FLOAT8_E8M0, std::vector<uint8_t>(rows*cols/32, 0xff));
    uint64_t bytes = 0;
    aclOpExecutor* executor = nullptr;
    char rounding[] = "rint";
    ACL_CHECK(aclnnDynamicMxQuantGetWorkspaceSize(x.tensor, -1, rounding, 36, 32, 0,
                                                 y.tensor, scales.tensor, &bytes, &executor));
    Buffer workspace(bytes);
    std::printf("case=dynamic_mx_quant rows=%d cols=%d input=BF16 output=E4M3FN block=32 workspace=%llu\n",
                rows, cols, static_cast<unsigned long long>(bytes));
    ACL_CHECK(aclnnDynamicMxQuant(workspace.ptr, bytes, executor, stream));
    ACL_CHECK(aclrtSynchronizeStream(stream));
    auto result = y.read(); auto scale_result = scales.read();
    size_t bad = 0;
    for (size_t i = 0; i < result.size(); ++i) bad += result[i] != expected[i%4];
    for (auto s : scale_result) bad += s != 128;
    std::printf("correctness quant_elements=%zu scale_elements=%zu bad=%zu comparison=exact_bytes\n",
                result.size(), scale_result.size(), bad);
    if (bad) throw std::runtime_error("Quantization golden mismatch");
}

static void matmul(aclrtStream stream, int m, int n, int k) {
    std::vector<uint16_t> a(m*k), b(k*n);
    for (int i=0;i<m*k;++i) a[i]=bf16(float((i*7+3)%17-8)/8);
    for (int i=0;i<k*n;++i) b[i]=bf16(float((i*11+1)%19-9)/8);
    std::vector<double> golden(m*n,0);
    for (int i=0;i<m;++i) for (int j=0;j<n;++j) for (int t=0;t<k;++t)
        golden[i*n+j]+=double(fp32(a[i*k+t]))*fp32(b[t*n+j]);
    Tensor<uint16_t> x({m,k},ACL_BF16,a), w({k,n},ACL_BF16,b);
    Tensor<uint16_t> y({m,n},ACL_BF16,std::vector<uint16_t>(m*n,0x7fc0));
    uint64_t bytes=0; aclOpExecutor* executor=nullptr;
    ACL_CHECK(aclnnMatmulGetWorkspaceSize(x.tensor,w.tensor,y.tensor,0,&bytes,&executor));
    Buffer workspace(bytes);
    std::printf("case=matmul M=%d N=%d K=%d dtype=BF16 workspace=%llu\n",m,n,k,static_cast<unsigned long long>(bytes));
    ACL_CHECK(aclnnMatmul(workspace.ptr,bytes,executor,stream));
    ACL_CHECK(aclrtSynchronizeStream(stream));
    compare(y.read(),golden,0.01,0.005);
}

static void fa(aclrtStream stream, int sq, int sk, int d, int batch, int heads) {
    const int hidden=heads*d;
    auto index=[=](int s,int b,int h,int t) {return ((s*batch+b)*heads+h)*d+t;};
    std::vector<uint16_t> q(sq*batch*hidden), k(sk*batch*hidden), v(sk*batch*hidden);
    for(size_t i=0;i<q.size();++i) q[i]=bf16(float((int(i)*7+3)%17-8)/16);
    for(size_t i=0;i<k.size();++i) {
        k[i]=bf16(float((int(i)*11+1)%19-9)/16);
        v[i]=bf16(float((int(i)*13+5)%23-11)/8);
    }
    std::vector<double> golden(q.size(),0), scores(sk);
    double scale=1/std::sqrt(double(d));
    for(int b=0;b<batch;++b) for(int h=0;h<heads;++h) for(int i=0;i<sq;++i) {
        for(int j=0;j<sk;++j) {
            double dot=0;
            for(int t=0;t<d;++t) dot+=double(fp32(q[index(i,b,h,t)]))*fp32(k[index(j,b,h,t)]);
            scores[j]=dot*scale;
        }
        double mx=*std::max_element(scores.begin(),scores.end()),sum=0;
        for(auto& s:scores) {s=std::exp(s-mx);sum+=s;}
        for(int t=0;t<d;++t) for(int j=0;j<sk;++j) golden[index(i,b,h,t)]+=scores[j]/sum*fp32(v[index(j,b,h,t)]);
    }
    Tensor<uint16_t> qt({sq,batch,hidden},ACL_BF16,q),kt({sk,batch,hidden},ACL_BF16,k),vt({sk,batch,hidden},ACL_BF16,v);
    Tensor<uint16_t> out({sq,batch,hidden},ACL_BF16,std::vector<uint16_t>(q.size(),0x7fc0));
    Tensor<float> mx({batch,heads,sq,8},ACL_FLOAT,std::vector<float>(batch*heads*sq*8,0));
    Tensor<float> sum({batch,heads,sq,8},ACL_FLOAT,std::vector<float>(batch*heads*sq*8,0));
    uint64_t bytes=0;aclOpExecutor* executor=nullptr;char layout[]="SBH";
    ACL_CHECK(aclnnFlashAttentionScoreV2GetWorkspaceSize(qt.tensor,kt.tensor,vt.tensor,
        nullptr,nullptr,nullptr,nullptr,nullptr,nullptr,nullptr,
        scale,1.0,2147483647,2147483647,heads,layout,0,0,1,
        mx.tensor,sum.tensor,nullptr,out.tensor,&bytes,&executor));
    Buffer workspace(bytes);
    std::printf("case=fa B=%d H=%d Q=%d KV=%d D=%d dtype=BF16 layout=SBH causal=false workspace=%llu\n",
                batch,heads,sq,sk,d,static_cast<unsigned long long>(bytes));
    ACL_CHECK(aclnnFlashAttentionScoreV2(workspace.ptr,bytes,executor,stream));
    ACL_CHECK(aclrtSynchronizeStream(stream));
    compare(out.read(),golden,0.002,0.01);
}

int main(int argc,char** argv) {
    std::setvbuf(stdout,nullptr,_IOLBF,0);
    if(argc<2 || (std::string(argv[1])!="quant" && std::string(argv[1])!="matmul" && std::string(argv[1])!="fa")) {
        std::fprintf(stderr,"Usage: %s quant [rows cols] | matmul [M N K] | fa [Q KV D [B H]]\n",argv[0]);return 2;
    }
    aclrtStream stream=nullptr;bool initialized=false,device=false;
    int result=0;
    try {
        std::string name=argv[1];
        std::vector<int> dims = name=="quant" ? std::vector<int>{4,256} :
                                name=="matmul" ? std::vector<int>{32,64,64} : std::vector<int>{16,64,64};
        if(name=="fa" && argc==7) {dims.push_back(1);dims.push_back(1);}
        if (argc!=2) {
            if (argc!=int(dims.size())+2) throw std::runtime_error("Wrong number of dimensions");
            for (size_t i=0;i<dims.size();++i) {
                std::string arg=argv[i+2];size_t used=0;
                dims[i]=std::stoi(arg,&used);
                if(used!=arg.size() || dims[i]<1 || dims[i]>1048576) throw std::runtime_error("Dimension outside [1,1048576]");
            }
        }
        if (name=="quant" && dims[1]%64) throw std::runtime_error("This quant golden requires cols divisible by 64");
        const int64_t work_limit = (name=="fa" ? 512LL : 64LL)*1024*1024;
        int64_t work=1;
        for(auto dimension:dims) {
            if(work>work_limit/dimension) throw std::runtime_error("Bounded-case work budget exceeded");
            work*=dimension;
        }
        if (name=="fa") {
            int64_t bh = dims.size()==5 ? int64_t(dims[3])*dims[4] : 1;
            if (int64_t(std::max(dims[0],dims[1]))*dims[2]*bh > 64LL*1024*1024)
                throw std::runtime_error("FA input allocation exceeds 64M elements per tensor");
        }
        ACL_CHECK(aclInit(nullptr));initialized=true;
        ACL_CHECK(aclrtSetDevice(0));device=true;
        const char* soc=aclrtGetSocName();
        std::printf("runtime_soc=%s\n",soc?soc:"unknown");
        ACL_CHECK(aclrtCreateStream(&stream));
        if(name=="quant") quant(stream,dims[0],dims[1]);
        else if(name=="matmul") matmul(stream,dims[0],dims[1],dims[2]);
        else fa(stream,dims[0],dims[1],dims[2],dims.size()==5?dims[3]:1,dims.size()==5?dims[4]:1);
        std::puts("SMOKE_PASS");
    } catch(const std::exception& e) {std::fprintf(stderr,"SMOKE_FAIL: %s\n",e.what());result=1;}
    if(stream) aclrtDestroyStream(stream);
    if(device) aclrtResetDevice(0);
    if(initialized) aclFinalize();
    return result;
}
