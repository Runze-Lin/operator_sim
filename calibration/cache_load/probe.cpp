// Run exactly one contiguous tail-axis E2M1/E8M0 -> BF16 AntiMxQuant.
#include "../acl_tensor.hpp"
#include <aclnnop/aclnn_anti_mx_quant.h>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <numeric>

static void run(const std::vector<int64_t> &shape, aclrtStream stream) {
    int64_t width = shape.back();
    if (width <= 0 || width % 2 || shape.size() > 7) throw std::runtime_error("Expected an even storage width and rank <=7");
    size_t rows = 1;
    for (size_t j = 0; j+1 < shape.size(); ++j) {
        if (shape[j] < 1) throw std::runtime_error("Expected positive extents");
        rows *= shape[j];
    }
    const size_t values = rows*width, scales_per_row=2*((width+63)/64);
    std::vector<uint8_t> packed(values/2), scales(rows*scales_per_row, 239);
    std::vector<uint16_t> expected(values), observed(values);
    const float lut[] = {0,.5,1,1.5,2,3,4,6};
    for (size_t i=0; i<values; ++i) {
        const size_t row=i/width, col=i%width, group=col/32;
        const uint8_t code=(col+row+group)%16;
        const int exponent=int((row+group)%9)-4;
        scales[row*scales_per_row+group]=127+exponent;
        packed[i/2] |= code << (4*(i%2));
        const float value=std::ldexp((code&8)?-lut[code&7]:lut[code&7],exponent);
        uint32_t bits; std::memcpy(&bits,&value,sizeof(bits)); expected[i]=bits>>16;
    }
    std::vector<int64_t> scale_shape=shape;
    scale_shape.back()=scales_per_row/2; scale_shape.push_back(2);
    Tensor x(shape, ACL_FLOAT4_E2M1, packed), s(scale_shape, ACL_FLOAT8_E8M0, scales);
    Tensor y(shape,ACL_BF16,std::vector<uint8_t>(values*2,0xcc));
    uint64_t bytes=0; aclOpExecutor *exe=nullptr;
    CHECK(aclnnAntiMxQuantGetWorkspaceSize(x.desc,s.desc,-1,ACL_BF16,y.desc,&bytes,&exe));
    Buffer workspace(bytes);
    std::printf("case=cache_load api=aclnnAntiMxQuant axis=-1 group=32 dst=bf16 shape=");
    for(size_t j=0;j<shape.size();++j) std::printf("%s%lld",j?"x":"",(long long)shape[j]);
    std::puts("");
    CHECK(aclnnAntiMxQuant(workspace.ptr,bytes,exe,stream));
    CHECK(aclrtSynchronizeStream(stream));
    CHECK(aclrtMemcpy(observed.data(),values*2,y.buffer.ptr,values*2,ACL_MEMCPY_DEVICE_TO_HOST));
    size_t bad=0;
    for (size_t i=0;i<values;++i) if (expected[i]!=observed[i]) {
        if (bad<3) std::printf("mismatch i=%zu expected=%04x got=%04x\n",i,expected[i],observed[i]);
        ++bad;
    }
    std::printf("correctness Y count=%zu bad=%zu\n",values,bad);
    if (bad) throw std::runtime_error("Packed-cache output differs from CPU golden");
}
int main(int argc,char **argv) {
    std::setvbuf(stdout,nullptr,_IONBF,0);
    if(argc<4 || std::string(argv[1])!="cache_load") return 2;
    try {
        CHECK(aclInit(nullptr)); CHECK(aclrtSetDevice(0)); std::printf("runtime_soc=%s\n",aclrtGetSocName());
        aclrtStream stream=nullptr; CHECK(aclrtCreateStream(&stream));
        std::vector<int64_t> shape; for(int i=2;i<argc;++i) shape.push_back(std::stoll(argv[i]));
        run(shape,stream);
        CHECK(aclrtDestroyStream(stream)); CHECK(aclrtResetDevice(0)); CHECK(aclFinalize());
        std::puts("SMOKE_PASS"); return 0;
    } catch(const std::exception &e) { std::fprintf(stderr,"SMOKE_FAIL %s\n",e.what()); return 1; }
}
