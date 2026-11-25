// BF16 BMM. A is a view of packed [M,heads,192]; B has independent per-head values.
#include "../acl_tensor.hpp"
#include <aclnnop/aclnn_batch_matmul.h>
#include <cstdio>
#include <cstring>

static void put(std::vector<uint8_t>& bytes, size_t i, float value) {
    uint32_t bits;
    std::memcpy(&bits, &value, sizeof(bits));
    uint16_t word = bits >> 16;
    std::memcpy(bytes.data()+2*i, &word, 2);
}

static void probe(aclrtStream stream, int m) {
    const int g=12,k=128,n=512,pitch=192;
    std::vector<uint8_t> a(2*m*g*pitch), b(2*g*k*n), expected(2*g*m*n);
    for (int h=0;h<g;h++) {
        for (int r=0;r<k;r++) for(int j=0;j<n;j++)
            put(b,(h*k+r)*n+j,((h+r+j)%13-6)*0.125f);
        for (int i=0;i<m;i++) {
            int ix=(h+3*i)%k;
            put(a,(i*g+h)*pitch+ix,1);
            for (int j=0;j<n;j++) put(expected,(h*m+i)*n+j,((h+ix+j)%13-6)*0.125f);
        }
    }
    Tensor backing({m,g,pitch},ACL_BF16,a), w({g,k,n},ACL_BF16,b);
    Tensor y({g,m,n},ACL_BF16,std::vector<uint8_t>(expected.size()));
    int64_t dims[]={g,m,k}, strides[]={pitch,g*pitch,1}, storage[]={m,g,pitch};
    auto x=aclCreateTensor(dims,3,ACL_BF16,strides,0,ACL_FORMAT_ND,storage,3,backing.buffer.ptr);
    if (!x) throw std::runtime_error("Invalid activation view");
    aclOpExecutor* executor=nullptr;
    uint64_t bytes=0;
    CHECK(aclnnBatchMatMulGetWorkspaceSize(x,w.desc,y.desc,0,&bytes,&executor));
    Buffer ws(bytes);
    CHECK(aclnnBatchMatMul(ws.ptr,bytes,executor,stream));
    CHECK(aclrtSynchronizeStream(stream));
    std::vector<uint8_t> result(expected.size());
    CHECK(aclrtMemcpy(result.data(),result.size(),y.buffer.ptr,result.size(),ACL_MEMCPY_DEVICE_TO_HOST));
    if(result!=expected) throw std::runtime_error("BMM golden mismatch");
    std::printf("GOLDEN_PASS G=%d M=%d K=%d N=%d checked=%zu\n",g,m,k,n,result.size()/2);
    aclDestroyTensor(x);
}

int main(int argc,char** argv){
    try {
        std::setvbuf(stdout,nullptr,_IONBF,0);
        CHECK(aclInit(nullptr)); CHECK(aclrtSetDevice(0)); aclrtStream s; CHECK(aclrtCreateStream(&s));
        probe(s,argc>1?std::stoi(argv[1]):1);
        CHECK(aclrtDestroyStream(s)); CHECK(aclrtResetDevice(0)); CHECK(aclFinalize());
    } catch (const std::exception& e) {std::fprintf(stderr,"%s\n",e.what()); return 1;}
}
