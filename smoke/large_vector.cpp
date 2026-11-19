// Memory-bounded host for a single unchunked prefill dispatch of the installed ops.
#include <acl/acl.h>
#include <aclnnop/aclnn_dynamic_mx_quant.h>
#include <aclnnop/aclnn_rms_norm.h>
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <stdexcept>
#include <string>
#include <vector>

static void check(int code, const char* what) {
    if (code) throw std::runtime_error(std::string(what) + " failed: " + std::to_string(code));
}
#define CHECK(call) check((call), #call)

struct Buffer {
    void* ptr = nullptr;
    explicit Buffer(size_t bytes) {
        if (bytes) CHECK(aclrtMalloc(&ptr, bytes, ACL_MEM_MALLOC_HUGE_FIRST));
    }
    ~Buffer() { if (ptr) aclrtFree(ptr); }
    Buffer(const Buffer&) = delete;
    Buffer& operator=(const Buffer&) = delete;
};

template<class T> struct Tensor {
    size_t count;
    Buffer buffer;
    aclTensor* handle;
    Tensor(std::vector<int64_t> dims, aclDataType dtype, size_t size)
        : count(size), buffer(size * sizeof(T)) {
        std::vector<int64_t> stride(dims.size(), 1);
        for (size_t i=dims.size()-1; i>0; --i) stride[i-1] = stride[i]*dims[i];
        if (static_cast<size_t>(dims[0]*stride[0])!=size) throw std::runtime_error("size mismatch");
        handle=aclCreateTensor(dims.data(), dims.size(), dtype, stride.data(), 0,
                               ACL_FORMAT_ND, dims.data(), dims.size(), buffer.ptr);
        if (!handle) throw std::runtime_error("aclCreateTensor");
    }
    ~Tensor() { aclDestroyTensor(handle); }
    template<class F> void fill(F element) {
        std::vector<T> chunk(std::min<size_t>(1U<<19,count));
        for (size_t off=0;off<count;off+=chunk.size()) {
            size_t n=std::min(chunk.size(), count-off);
            for(size_t i=0;i<n;++i) chunk[i]=element(off+i);
            CHECK(aclrtMemcpy(static_cast<T*>(buffer.ptr)+off,n*sizeof(T),chunk.data(),n*sizeof(T),ACL_MEMCPY_HOST_TO_DEVICE));
        }
    }
    template<class F> void verify(F good) {
        size_t bad=0;
        std::vector<T> chunk(std::min<size_t>(1U<<19,count));
        for (size_t off=0;off<count;off+=chunk.size()) {
            size_t n=std::min(chunk.size(), count-off);
            CHECK(aclrtMemcpy(chunk.data(),n*sizeof(T),static_cast<T*>(buffer.ptr)+off,n*sizeof(T),ACL_MEMCPY_DEVICE_TO_HOST));
            for (size_t i=0;i<n;++i) bad+=!good(off+i,chunk[i]);
        }
        std::printf("correctness count=%zu bad=%zu\n",count,bad);
        if (bad) throw std::runtime_error("golden mismatch");
    }
};

static void measure(std::string op, int64_t rows, int64_t cols, aclrtStream stream) {
    size_t n=static_cast<size_t>(rows)*cols;
    Tensor<uint16_t> x({rows,cols},ACL_BF16,n);
    uint64_t bytes=0;
    aclOpExecutor* exe=nullptr;
    if (op=="quant") {
        const uint16_t xbits[]={0,0x4100,0x4280,0x4400}; // BF16 0,8,64,512
        const uint8_t expect[]={0,72,96,120};
        x.fill([&](size_t i){return xbits[i%4];});
        Tensor<uint8_t> y({rows,cols},ACL_FLOAT8_E4M3FN,n);
        Tensor<uint8_t> scale({rows,cols/64,2},ACL_FLOAT8_E8M0,n/32);
        y.fill([](size_t){return 255;}); scale.fill([](size_t){return 255;});
        char mode[]="rint";
        CHECK(aclnnDynamicMxQuantGetWorkspaceSize(x.handle,-1,mode,36,32,0,y.handle,scale.handle,&bytes,&exe));
        Buffer workspace(bytes);
        std::printf("case=dynamic_mx_quant rows=%ld cols=%ld input=BF16 output=E4M3FN block=32\n",rows,cols);
        CHECK(aclnnDynamicMxQuant(workspace.ptr,bytes,exe,stream));
        CHECK(aclrtSynchronizeStream(stream));
        y.verify([&](size_t i,uint8_t v){return v==expect[i%4];});
        scale.verify([](size_t,uint8_t v){return v==128;});
    } else {
        auto sign=[](size_t i){return static_cast<uint16_t>((i%2)?0xbf80:0x3f80);};
        x.fill(sign);
        Tensor<uint16_t> g({cols},ACL_BF16,cols),y({rows,cols},ACL_BF16,n);
        Tensor<float> stats({rows,1},ACL_FLOAT,rows);
        g.fill([](size_t){return 0x3f80;}); y.fill([](size_t){return 0x7fc0;});
        stats.fill([](size_t){return NAN;});
        CHECK(aclnnRmsNormGetWorkspaceSize(x.handle,g.handle,1e-6,y.handle,stats.handle,&bytes,&exe));
        Buffer workspace(bytes);
        std::printf("case=rmsnorm api=aclnnRmsNorm R=%ld C=%ld dtype=BF16 epsilon=1e-6\n",rows,cols);
        CHECK(aclnnRmsNorm(workspace.ptr,bytes,exe,stream));
        CHECK(aclrtSynchronizeStream(stream));
        y.verify([&](size_t i,uint16_t v){return v==sign(i);});
        stats.verify([](size_t,float v){return std::abs(v-1./std::sqrt(1.+1e-6))<1e-4;});
    }
}

int main(int argc, char** argv) {
    std::setvbuf(stdout,nullptr,_IOLBF,0);
    try {
        if (argc!=4) throw std::runtime_error("expected rmsnorm|quant rows cols");
        const std::string op=argv[1];
        const long r=std::stol(argv[2]),c=std::stol(argv[3]);
        if ((op!="quant" && op!="rmsnorm") || r<=0 || r>262144 || c<=0 || c>7168 || c%64)
            throw std::runtime_error("invalid prefill contract");
        CHECK(aclInit(nullptr)); CHECK(aclrtSetDevice(0));
        std::printf("runtime_soc=%s\n",aclrtGetSocName());
        aclrtStream stream=nullptr; CHECK(aclrtCreateStream(&stream));
        measure(op,r,c,stream);
        CHECK(aclrtDestroyStream(stream)); CHECK(aclrtResetDevice(0)); CHECK(aclFinalize());
        std::puts("SMOKE_PASS"); return 0;
    } catch(const std::exception& e) { std::fprintf(stderr,"SMOKE_FAIL: %s\n",e.what());return 1; }
}
