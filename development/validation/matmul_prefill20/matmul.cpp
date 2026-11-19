// BF16 all-output golden for the fixed rank-local projection geometry.
#define main large_vector_smoke_main
#include "../../../smoke/large_vector.cpp"
#undef main
#include <aclnnop/aclnn_matmul.h>

int main() {
    std::setvbuf(stdout, nullptr, _IOLBF, 0);
    constexpr int64_t m=2048, n=32, k=5120;
    constexpr uint16_t plus_one=0x3f80, minus_one=0xbf80;
    constexpr uint16_t plus_1280=0x44a0, minus_1280=0xc4a0;
    try {
        CHECK(aclInit(nullptr));
        CHECK(aclrtSetDevice(0));
        std::printf("runtime_soc=%s\n",aclrtGetSocName());
        aclrtStream stream=nullptr;
        CHECK(aclrtCreateStream(&stream));
        {
            Tensor<uint16_t> x({m,k},ACL_BF16,m*k), w({k,n},ACL_BF16,k*n), y({m,n},ACL_BF16,m*n);
            x.fill([&](size_t i){return ((i/k)%2==1) != (i%4==3) ? minus_one : plus_one;});
            w.fill([&](size_t i){return (i%2==1) != ((i/n)%8>=5) ? minus_one : plus_one;});
            y.fill([](size_t){return uint16_t(0x7fc0);});
            uint64_t bytes=0;
            aclOpExecutor* executor=nullptr;
            CHECK(aclnnMatmulGetWorkspaceSize(x.handle,w.handle,y.handle,0,&bytes,&executor));
            Buffer workspace(bytes);
            std::printf("case=matmul_prefill api=aclnnMatmul M=2048 N=32 K=5120 dtype=BF16 layout=ND_ND workspace=%llu\n",static_cast<unsigned long long>(bytes));
            CHECK(aclnnMatmul(workspace.ptr,bytes,executor,stream));
            CHECK(aclrtSynchronizeStream(stream));
            y.verify([&](size_t i,uint16_t v){return v==(((i/n)%2==1)!=(i%2==1)?minus_1280:plus_1280);});
        }
        CHECK(aclrtDestroyStream(stream));
        CHECK(aclrtResetDevice(0));
        CHECK(aclFinalize());
        std::puts("SMOKE_PASS");
        return 0;
    } catch(const std::exception& e) {
        std::fprintf(stderr,"SMOKE_FAIL %s\n",e.what());
        return 1;
    }
}
