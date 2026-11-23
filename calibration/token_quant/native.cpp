// BF16 rows to E4M3FN plus one FP32 scale per row; compare all outputs on host.
#include "../acl_tensor.hpp"
#include <aclnnop/aclnn_dynamic_quant_v2.h>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>

static void run(int r, int d, aclrtStream stream) {
    if (r < 1 || d < 8) throw std::runtime_error("Expected positive rows and width >=8");
    const size_t count = size_t(r)*d;
    const float values[] = {448, -448, 0, 0.5f, -1, 1.5f, 32, -64};
    const uint8_t encoding[] = {126,254,0,48,184,60,96,232};
    std::vector<uint8_t> input(2*count), gold(count), output(count);
    std::vector<float> golden_scales(r), scales(r);
    for (int i=0; i<r; ++i) {
        float scale = std::ldexp(1.0f, i%5-2);
        golden_scales[i] = scale;
        for (int j=0; j<d; ++j) {
            int pattern = (i+j)%8;
            float v = values[pattern]*scale;
            uint32_t bits;
            std::memcpy(&bits, &v, 4);
            uint16_t bf = bits >> 16;
            size_t n = size_t(i)*d+j;
            std::memcpy(input.data()+2*n, &bf, 2);
            gold[n] = encoding[pattern];
        }
    }
    Tensor x({r,d},ACL_BF16,input);
    Tensor y({r,d},ACL_FLOAT8_E4M3FN,std::vector<uint8_t>(count,0xcd));
    Tensor s({r},ACL_FLOAT,std::vector<uint8_t>(4*r,0xcd));
    uint64_t bytes = 0;
    aclOpExecutor* executor = nullptr;
    CHECK(aclnnDynamicQuantV2GetWorkspaceSize(x.desc,nullptr,nullptr,ACL_FLOAT8_E4M3FN,
                                              y.desc,s.desc,nullptr,&bytes,&executor));
    Buffer scratch(bytes);
    std::printf("case=token_quant rows=%d cols=%d api=aclnnDynamicQuantV2\n",r,d);
    CHECK(aclnnDynamicQuantV2(scratch.ptr,bytes,executor,stream));
    CHECK(aclrtSynchronizeStream(stream));
    CHECK(aclrtMemcpy(output.data(),count,y.buffer.ptr,count,ACL_MEMCPY_DEVICE_TO_HOST));
    CHECK(aclrtMemcpy(scales.data(),4*r,s.buffer.ptr,4*r,ACL_MEMCPY_DEVICE_TO_HOST));
    size_t bad=0, sbad=0;
    for (size_t i=0; i<count; ++i) bad += output[i] != gold[i];
    for (int i=0; i<r; ++i) sbad += scales[i] != golden_scales[i];
    std::printf("correctness Y count=%zu bad=%zu\ncorrectness scale count=%d bad=%zu\n",count,bad,r,sbad);
    if (bad || sbad) throw std::runtime_error("Packed output or FP32 row scale mismatched");
}

int main(int argc, char** argv) {
    std::setvbuf(stdout,nullptr,_IONBF,0);
    if (argc!=4 || std::string(argv[1])!="token_quant") return 2;
    try {
        CHECK(aclInit(nullptr)); CHECK(aclrtSetDevice(0));
        std::printf("runtime_soc=%s\n",aclrtGetSocName());
        aclrtStream stream=nullptr; CHECK(aclrtCreateStream(&stream));
        run(std::stoi(argv[2]),std::stoi(argv[3]),stream);
        CHECK(aclrtDestroyStream(stream)); CHECK(aclrtResetDevice(0)); CHECK(aclFinalize());
        // The CA libraries can run further static destructors after aclFinalize.
        std::puts("SMOKE_PASS"); std::_Exit(EXIT_SUCCESS);
    } catch (const std::exception& error) { std::fprintf(stderr,"SMOKE_FAIL %s\n",error.what()); return 1; }
}
