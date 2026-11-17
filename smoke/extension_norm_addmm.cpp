#include <acl/acl.h>
#include <aclnnop/aclnn_rms_norm.h>
#include <aclnnop/aclnn_addmm.h>
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
                    bool bfloat) {
    const double atol=bfloat?std::ldexp(1.,-6):std::ldexp(1.,-16);
    const double rtol=bfloat?std::ldexp(1.,-6):std::ldexp(1.,-10);
    const double limit=bfloat?1.:.01;
    size_t bad=0; double worst=0;
    for(size_t i=0;i<actual.size();++i) {
        double err=std::abs(actual[i]-expected.at(i));
        worst=std::max(worst,err);
        if(!std::isfinite(actual[i]) || err>atol+rtol*std::abs(expected[i])) ++bad;
    }
    std::printf("correctness count=%zu bad=%zu max_abs_error=%g atol=%g rtol=%g hard_limit=%g\n",
                actual.size(),bad,worst,atol,rtol,limit);
    // A small smoke checks every element, stricter than the skill's 99% matched ratio.
    if(bad || worst>limit) throw std::runtime_error("Correctness failed");
}
static std::vector<double> decode(const std::vector<uint16_t>& x) {
    std::vector<double> result; result.reserve(x.size());
    for(auto v:x) result.push_back(fp32(v));
    return result;
}
static std::vector<uint16_t> data(size_t count,int offset) {
    std::vector<uint16_t> x(count);
    for(size_t i=0;i<count;++i) x[i]=bf16(float(int((i+offset)%17)-8)/16.f);
    return x;
}
static void rmsnorm(aclrtStream stream,int r,int c) {
    auto xdata=data(size_t(r)*c,0); auto gdata=data(c,3);
    Tensor<uint16_t> x({r,c},ACL_BF16,xdata),g({c},ACL_BF16,gdata);
    Tensor<uint16_t> y({r,c},ACL_BF16,std::vector<uint16_t>(size_t(r)*c,0x7fc0));
    Tensor<float> rstd({r,1},ACL_FLOAT,std::vector<float>(r,NAN));
    uint64_t bytes=0; aclOpExecutor* executor=nullptr; constexpr double epsilon=1e-6;
    ACL_CHECK(aclnnRmsNormGetWorkspaceSize(x.tensor,g.tensor,epsilon,y.tensor,rstd.tensor,&bytes,&executor));
    Buffer workspace(bytes);
    std::printf("case=rmsnorm api=aclnnRmsNorm R=%d C=%d dtype=BF16 epsilon=%g workspace=%llu\n",r,c,epsilon,(unsigned long long)bytes);
    ACL_CHECK(aclnnRmsNorm(workspace.ptr,bytes,executor,stream));
    ACL_CHECK(aclrtSynchronizeStream(stream));
    std::vector<double> gy(size_t(r)*c),gr(r);
    for(int row=0;row<r;++row) {
        double sum=0;
        for(int col=0;col<c;++col) {double v=fp32(xdata[size_t(row)*c+col]);sum+=v*v;}
        gr[row]=1./std::sqrt(sum/c+epsilon);
        for(int col=0;col<c;++col)gy[size_t(row)*c+col]=fp32(xdata[size_t(row)*c+col])*gr[row]*fp32(gdata[col]);
    }
    compare(decode(y.read()),gy,true);
    auto actual=rstd.read(); compare(std::vector<double>(actual.begin(),actual.end()),gr,false);
}
struct Scalar {
    aclScalar* value=nullptr;
    explicit Scalar(float v):value(aclCreateScalar(&v,ACL_FLOAT)) {
        if(!value)throw std::runtime_error("aclCreateScalar failed");
    }
    ~Scalar(){if(value)aclDestroyScalar(value);}
};
static void addmm(aclrtStream stream,int m,int n,int k,bool matrix) {
    auto a=data(size_t(m)*k,0),b=data(size_t(k)*n,2),s=data(matrix?size_t(m)*n:size_t(n),5);
    Tensor<uint16_t> at({m,k},ACL_BF16,a),bt({k,n},ACL_BF16,b);
    Tensor<uint16_t> st(matrix?std::vector<int64_t>{m,n}:std::vector<int64_t>{n},ACL_BF16,s);
    Tensor<uint16_t> out({m,n},ACL_BF16,std::vector<uint16_t>(size_t(m)*n,0x7fc0));
    Scalar one(1.f);uint64_t bytes=0;aclOpExecutor* executor=nullptr;
    ACL_CHECK(aclnnAddmmGetWorkspaceSize(st.tensor,at.tensor,bt.tensor,one.value,one.value,out.tensor,0,&bytes,&executor));
    Buffer workspace(bytes);
    std::printf("case=addmm api=aclnnAddmm M=%d N=%d K=%d self=%s dtype=BF16 alpha=1 beta=1 cubeMathType=0 workspace=%llu\n",m,n,k,matrix?"matrix":"bias",(unsigned long long)bytes);
    ACL_CHECK(aclnnAddmm(workspace.ptr,bytes,executor,stream));ACL_CHECK(aclrtSynchronizeStream(stream));
    std::vector<double> expected(size_t(m)*n);
    for(int i=0;i<m;++i)for(int j=0;j<n;++j) {
        double v=fp32(s[matrix?size_t(i)*n+j:size_t(j)]);
        for(int q=0;q<k;++q)v+=double(fp32(a[size_t(i)*k+q]))*fp32(b[size_t(q)*n+j]);
        expected[size_t(i)*n+j]=v;
    }
    compare(decode(out.read()),expected,true);
}
static int dim(const char* arg) {
    size_t pos=0;long long value=std::stoll(arg,&pos);
    if(pos!=std::strlen(arg)||value<1||value>1048576)throw std::runtime_error("Invalid dimension");
    return int(value);
}
int main(int argc,char** argv) {
    aclrtStream stream=nullptr;bool initialized=false,device=false;int result=0;
    try {
        if(argc<4)throw std::runtime_error("Usage: extension_norm_addmm rmsnorm R C | addmm M N K bias|matrix");
        std::string op=argv[1];int a=dim(argv[2]),b=dim(argv[3]),c=1;
        if(op=="rmsnorm"&&argc!=4)throw std::runtime_error("RMSNorm needs R C");
        bool matrix=false;
        if(op=="addmm") {
            if(argc!=6)throw std::runtime_error("Addmm needs M N K bias|matrix");
            c=dim(argv[4]);std::string self=argv[5];matrix=self=="matrix";
            if(!matrix&&self!="bias")throw std::runtime_error("Expected bias|matrix");
        } else if(op!="rmsnorm")throw std::runtime_error("Unknown operator");
        if(int64_t(a)*b*c>64*1024*1024)throw std::runtime_error("Smoke work budget exceeded");
        ACL_CHECK(aclInit(nullptr));initialized=true;
        ACL_CHECK(aclrtSetDevice(0));device=true;
        std::printf("runtime_soc=%s\n",aclrtGetSocName());
        ACL_CHECK(aclrtCreateStream(&stream));
        if(op=="rmsnorm")rmsnorm(stream,a,b);else addmm(stream,a,b,c,matrix);
        std::puts("SMOKE_PASS");
    } catch(const std::exception& e) {std::fprintf(stderr,"SMOKE_FAIL %s\n",e.what());result=1;}
    if(stream)aclrtDestroyStream(stream);
    if(device)aclrtResetDevice(0);
    if(initialized)aclFinalize();
    return result;
}
