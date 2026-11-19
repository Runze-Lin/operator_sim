// The v0.19 golden and kernels, with streaming device/host staging for large prefills.
#define main vector_smoke_main
#include "../../../smoke/large_vector.cpp"
#undef main
#include <aclnnop/aclnn_cast.h>
#include <cstring>

int main(int argc, char** argv) {
    std::setvbuf(stdout, nullptr, _IOLBF, 0);
    try {
        if (argc != 3) throw std::runtime_error("expected token_rows channels");
        int64_t rows = std::stoll(argv[1]), cols = std::stoll(argv[2]);
        if (rows<=0 || rows>(1<<21) || cols<=0 || cols>7168) throw std::runtime_error("invalid dimensions");
        size_t n = size_t(rows)*cols;
        // The largest committed scenario has <2^31 elements.
        if (n>(size_t(1)<<31)) throw std::runtime_error("host capacity exceeded");
        CHECK(aclInit(nullptr)); CHECK(aclrtSetDevice(0));
        std::printf("runtime_soc=%s\ncase=cast_bf16_fp32 api=aclnnCast R=%ld D=%ld\n",aclrtGetSocName(),rows,cols);
        aclrtStream stream=nullptr; CHECK(aclrtCreateStream(&stream));
        {
            Tensor<uint16_t> x({rows,cols}, ACL_BF16, n);
            Tensor<float> y({rows,cols}, ACL_FLOAT, n);
            // Safe finite BF16 normals with nontrivial mantissa and both signs.
            auto val=[](size_t i){return uint16_t(0x3d00u+((i*17)%1024)+((i&1)<<15));};
            x.fill(val); y.fill([](size_t){return NAN;});
            uint64_t bytes=0; aclOpExecutor* exe=nullptr;
            CHECK(aclnnCastGetWorkspaceSize(x.handle,ACL_FLOAT,y.handle,&bytes,&exe));
            Buffer ws(bytes);
            CHECK(aclnnCast(ws.ptr,bytes,exe,stream)); CHECK(aclrtSynchronizeStream(stream));
            y.verify([&](size_t i,float output) {
                uint32_t bits; std::memcpy(&bits,&output,sizeof(bits));
                return bits==(uint32_t(val(i))<<16);
            });
        }
        CHECK(aclrtDestroyStream(stream)); CHECK(aclrtResetDevice(0)); CHECK(aclFinalize());
        std::puts("SMOKE_PASS"); return 0;
    } catch (const std::exception& e) {
        std::fprintf(stderr,"SMOKE_FAIL: %s\n",e.what()); return 1;
    }
}
