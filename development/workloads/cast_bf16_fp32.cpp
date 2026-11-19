// Reuse CANN tensor lifecycle and finite BF16 test values from the norm host.
#define main archived_norm_host
#include "../../smoke/extension_norm_addmm.cpp"
#undef main
#include <aclnnop/aclnn_cast.h>

static void run_cast(aclrtStream stream, int rows, int width) {
    const size_t n = size_t(rows) * width;
    auto values = data(n, 5);
    Tensor<uint16_t> x({rows,width},ACL_BF16,values);
    Tensor<float> y({rows,width},ACL_FLOAT,std::vector<float>(n,NAN));
    uint64_t bytes=0;
    aclOpExecutor* exe=nullptr;
    ACL_CHECK(aclnnCastGetWorkspaceSize(x.tensor,ACL_FLOAT,y.tensor,&bytes,&exe));
    Buffer ws(bytes);
    ACL_CHECK(aclnnCast(ws.ptr,bytes,exe,stream));
    ACL_CHECK(aclrtSynchronizeStream(stream));
    size_t bad=0;
    auto result=y.read();
    for (size_t i=0;i<n;++i) {
        uint32_t bits;
        std::memcpy(&bits,&result[i],sizeof(bits));
        if (bits != (uint32_t(values[i]) << 16)) ++bad;
    }
    std::printf("correctness count=%zu bad=%zu exact=bf16_widen_to_fp32\n",n,bad);
    if (bad) throw std::runtime_error("Cast failed exact bit comparison");
}
int main(int argc,char** argv) {
    setvbuf(stdout,nullptr,_IOLBF,0);
    try {
        if (argc!=3) throw std::runtime_error("Expected rows width");
        int rows=std::stoi(argv[1]),width=std::stoi(argv[2]);
        if (rows<=0 || width<=0 || size_t(rows)*width>1048576) throw std::runtime_error("Invalid test dimensions");
        ACL_CHECK(aclInit(nullptr)); ACL_CHECK(aclrtSetDevice(0));
        std::printf("runtime_soc=%s\ncase=cast_bf16_fp32 api=aclnnCast R=%d D=%d\n",aclrtGetSocName(),rows,width);
        aclrtStream stream=nullptr; ACL_CHECK(aclrtCreateStream(&stream));
        run_cast(stream,rows,width);
        ACL_CHECK(aclrtDestroyStream(stream)); ACL_CHECK(aclrtResetDevice(0)); ACL_CHECK(aclFinalize());
        std::puts("SMOKE_PASS"); return 0;
    } catch(const std::exception& e) {std::fprintf(stderr,"SMOKE_FAIL: %s\n",e.what());return 1;}
}
