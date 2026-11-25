#include "../acl_tensor.hpp"
#include <aclnnop/aclnn_add_rms_norm.h>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>

static std::vector<uint8_t> bf16_bytes(size_t n, uint16_t bits) {
    std::vector<uint8_t> bytes(n * 2);
    for (size_t i = 0; i < n; ++i) std::memcpy(bytes.data() + 2*i, &bits, 2);
    return bytes;
}

static void run(int rows, int width, aclrtStream stream) {
    if (rows <= 0 || width <= 0) throw std::invalid_argument("positive shape required");
    const size_t n = size_t(rows)*width;
    auto ones = bf16_bytes(n, 0x3f80);
    // Vary the sign across elements. With equal operands the sum is +/-2,
    // its RMS is 2, and a gamma of 1 leaves normalized outputs +/-1.
    for (size_t i = 1; i < n; i += 2) ones[2*i+1] |= 0x80;
    auto out = bf16_bytes(n, 0);
    std::vector<uint8_t> rs(4*rows);
    Tensor x({rows,width}, ACL_BF16, ones), residual({rows,width}, ACL_BF16, ones);
    Tensor gamma({width}, ACL_BF16, bf16_bytes(width, 0x3f80));
    Tensor y({rows,width}, ACL_BF16, out), sum({rows,width}, ACL_BF16, out);
    Tensor rstd({rows,1}, ACL_FLOAT, rs);
    uint64_t bytes = 0;
    aclOpExecutor* executor = nullptr;
    CHECK(aclnnAddRmsNormGetWorkspaceSize(x.desc,residual.desc,gamma.desc,1e-6,
                                         y.desc,rstd.desc,sum.desc,&bytes,&executor));
    Buffer workspace(bytes);
    CHECK(aclnnAddRmsNorm(workspace.ptr,bytes,executor,stream));
    CHECK(aclrtSynchronizeStream(stream));
    CHECK(aclrtMemcpy(out.data(),out.size(),y.buffer.ptr,out.size(),ACL_MEMCPY_DEVICE_TO_HOST));
    if (out != ones) throw std::runtime_error("normalized BF16 mismatch");
    CHECK(aclrtMemcpy(out.data(),out.size(),sum.buffer.ptr,out.size(),ACL_MEMCPY_DEVICE_TO_HOST));
    for (size_t i = 0; i < n; ++i) {
        uint16_t got;
        std::memcpy(&got,out.data()+2*i,2);
        if (got != (i%2 ? 0xc000 : 0x4000)) throw std::runtime_error("residual mismatch");
    }
    CHECK(aclrtMemcpy(rs.data(),rs.size(),rstd.buffer.ptr,rs.size(),ACL_MEMCPY_DEVICE_TO_HOST));
    for (int i = 0; i < rows; ++i) {
        float got;
        std::memcpy(&got,rs.data()+4*i,4);
        if (!(std::abs(got-0.5f)<1e-5f)) throw std::runtime_error("rstd mismatch");
    }
    std::printf("case=add_rms_norm api=aclnnAddRmsNorm rows=%d width=%d\ncorrectness Y count=%zu bad=0\n",rows,width,2*n+rows);
}

int main(int argc, char** argv) {
    if (argc != 4 || std::string(argv[1]) != "add_rms_norm") return 2;
    try {
        CHECK(aclInit(nullptr));
        CHECK(aclrtSetDevice(0));
        std::printf("runtime_soc=%s\n",aclrtGetSocName());
        aclrtStream stream;
        CHECK(aclrtCreateStream(&stream));
        run(std::stoi(argv[2]),std::stoi(argv[3]),stream);
        CHECK(aclrtDestroyStream(stream));
        CHECK(aclrtResetDevice(0));
        CHECK(aclFinalize());
        std::puts("SMOKE_PASS");
        std::fflush(stdout);
        std::_Exit(EXIT_SUCCESS);
    } catch (const std::exception& e) {
        std::fprintf(stderr,"SMOKE_FAIL %s\n",e.what());
        return 1;
    }
}
