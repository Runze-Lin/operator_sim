// Offline CANN 9.2.1 CA probe for aclnnQuantMatmulV5. No bias, transposed ND W.
#include <acl/acl.h>
#include <aclnnop/aclnn_quant_matmul_v5.h>
#include <cstdint>
#include <cstring>
#include <cstdio>
#include <stdexcept>
#include <string>
#include <vector>

static void check(int code, const char *expr) {
    if (code) {
        const char *detail = aclGetRecentErrMsg();
        throw std::runtime_error(std::string(expr) + " code=" + std::to_string(code)
                                 + (detail ? std::string(" ") + detail : ""));
    }
}
#define CHECK(expr) check(expr,#expr)

struct Buffer {
    void *ptr = nullptr;
    explicit Buffer(size_t bytes) {
        if (bytes) CHECK(aclrtMalloc(&ptr, bytes, ACL_MEM_MALLOC_HUGE_FIRST));
    }
    ~Buffer() { if (ptr) aclrtFree(ptr); }
    Buffer(const Buffer&) = delete;
};

struct Tensor {
    Buffer buffer;
    aclTensor *desc;
    Tensor(const std::vector<int64_t>& shape, aclDataType type, const std::vector<uint8_t>& bytes)
        : buffer(bytes.size()) {
        std::vector<int64_t> stride(shape.size(), 1);
        for (size_t i=shape.size()-1; i>0; --i) stride[i-1]=stride[i]*shape[i];
        CHECK(aclrtMemcpy(buffer.ptr,bytes.size(),bytes.data(),bytes.size(),ACL_MEMCPY_HOST_TO_DEVICE));
        desc=aclCreateTensor(shape.data(),shape.size(),type,stride.data(),0,ACL_FORMAT_ND,
                             shape.data(),shape.size(),buffer.ptr);
        if (!desc) throw std::runtime_error("aclCreateTensor failed");
    }
    ~Tensor() { aclDestroyTensor(desc); }
};

static uint16_t bf16(float value) {
    uint32_t bits;
    std::memcpy(&bits,&value,sizeof(bits));
    bits+=0x7fff+((bits>>16)&1);
    return bits>>16;
}

static void run(bool mixed,int m,int n,int k,aclrtStream stream) {
    if (m<=0 || n<=0 || k<=0 || k%64) throw std::runtime_error("Use positive dimensions and K aligned to 64");
    // Sign patterns and scales vary across rows, columns and K groups. The
    // magnitude of each product is 1 before application of both E8M0 scales.
    auto a_sign=[](int i,int t){return ((i%7+t%5)%4)==0 ? -1 : 1;};
    auto b_sign=[](int j,int t){return ((j%5+t%7)%3)==0 ? -1 : 1;};
    auto a_exp=[](int i,int g){return (i+g)%2;};
    auto b_exp=[](int j,int g){return (j+g/2)%2;};
    std::vector<uint8_t> a(size_t(m)*k/(mixed?1:2)), b(size_t(n)*k/2);
    std::vector<uint8_t> as(size_t(m)*k/32),bs(size_t(n)*k/32);
    for(int i=0;i<m;++i) {
        for(int t=0;t<k;++t) {
            size_t pos=size_t(i)*k+t;
            bool neg=a_sign(i,t)<0;
            if(mixed) a[pos]=neg?0xb8:0x38; // E4M3FN -1/+1
            else a[pos/2]|=(neg?0xa:0x2)<<((pos%2)*4); // E2M1 -1/+1
        }
        for(int g=0;g<k/32;++g) as[size_t(i)*k/32+g]=127+a_exp(i,g);
    }
    for(int j=0;j<n;++j) {
        for(int t=0;t<k;++t) {
            size_t pos=size_t(j)*k+t;
            b[pos/2]|=(b_sign(j,t)<0?0xa:0x2)<<((pos%2)*4);
        }
        for(int g=0;g<k/32;++g) bs[size_t(j)*k/32+g]=127+b_exp(j,g);
    }
    Tensor x({m,k},mixed?ACL_FLOAT8_E4M3FN:ACL_FLOAT4_E2M1,a);
    Tensor w({n,k},ACL_FLOAT4_E2M1,b);
    Tensor sa({m,k/64,2},ACL_FLOAT8_E8M0,as),sb({n,k/64,2},ACL_FLOAT8_E8M0,bs);
    Tensor y({m,n},ACL_BF16,std::vector<uint8_t>(size_t(m)*n*2,0xff));
    uint64_t ws=0; aclOpExecutor *executor=nullptr;
    int64_t group=(1LL<<32)+(1LL<<16)+32;
    CHECK(aclnnQuantMatmulV5GetWorkspaceSize(x.desc,w.desc,sa.desc,sb.desc,nullptr,nullptr,nullptr,nullptr,
                                              nullptr,false,true,group,y.desc,&ws,&executor));
    std::printf("case=matmul api=aclnnQuantMatmulV5 route=%s M=%d N=%d K=%d layout=ND_NDT scales=E8M0 group=32 out=BF16 workspace=%llu\n",
                mixed?"a8w4":"a4w4",m,n,k,static_cast<unsigned long long>(ws));
    Buffer workspace(ws);
    CHECK(aclnnQuantMatmulV5(workspace.ptr,ws,executor,stream));
    CHECK(aclrtSynchronizeStream(stream));
    std::vector<uint16_t> out(size_t(m)*n);
    CHECK(aclrtMemcpy(out.data(),out.size()*2,y.buffer.ptr,out.size()*2,ACL_MEMCPY_DEVICE_TO_HOST));
    // The inputs repeat every lcm(7,2) rows and lcm(5,2) columns;
    // first compute the exact signed integer dot for each distinct pair.
    uint16_t gold[14][10];
    for(int i=0;i<14;++i) for(int j=0;j<10;++j) {
        int value=0;
        for(int t=0;t<k;++t)
            value+=a_sign(i,t)*b_sign(j,t)*(1<<(a_exp(i,t/32)+b_exp(j,t/32)));
        gold[i][j]=bf16(static_cast<float>(value));
    }
    size_t bad=0;
    for(int i=0;i<m;++i) for(int j=0;j<n;++j) {
        uint16_t observed=out[size_t(i)*n+j], expected=gold[i%14][j%10];
        if(observed!=expected) {
            if(!bad) std::fprintf(stderr,"first_mismatch row=%d col=%d got=%04x expected=%04x\n",i,j,observed,expected);
            ++bad;
        }
    }
    std::printf("correctness count=%zu bad=%zu\n",out.size(),bad);
    if(bad) throw std::runtime_error("Numerical validation failed");
}

int main(int argc,char** argv) {
    std::setvbuf(stdout,nullptr,_IONBF,0);
    if(argc!=5 || (std::string(argv[1])!="a4w4" && std::string(argv[1])!="a8w4")) return 2;
    try {
        CHECK(aclInit(nullptr)); CHECK(aclrtSetDevice(0));
        std::printf("runtime_soc=%s\n",aclrtGetSocName());
        aclrtStream stream=nullptr; CHECK(aclrtCreateStream(&stream));
        run(std::string(argv[1])=="a8w4",std::stoi(argv[2]),std::stoi(argv[3]),std::stoi(argv[4]),stream);
        CHECK(aclrtDestroyStream(stream)); CHECK(aclrtResetDevice(0)); CHECK(aclFinalize());
        std::puts("SMOKE_PASS"); return 0;
    } catch(const std::exception& e) { std::fprintf(stderr,"SMOKE_FAIL %s\n",e.what()); return 1; }
}
