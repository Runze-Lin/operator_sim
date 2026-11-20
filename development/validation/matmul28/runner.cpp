// One official, contiguous BF16 ND Matmul dispatch. Exact, bounded host verification.
#define main archived_vector_smoke_main
#include "../../../smoke/large_vector.cpp"
#undef main
#include <aclnnop/aclnn_matmul.h>
#include <cstring>

static int64_t extent(const char* value) {
    std::string text(value);
    size_t used = 0;
    const auto n = std::stoll(text, &used);
    if (used != text.size() || n <= 0 || n > (1 << 24)) throw std::runtime_error("bad dimension");
    return n;
}

static int left_value(size_t row, size_t inner) {
    const bool negative = (inner + 3*(row%4))%8 >= 5;
    return negative != (row%8 >= 4) ? -1 : 1;
}

static int right_value(size_t inner, size_t column) {
    const bool negative = (inner + column%4)%8 >= 6;
    return negative != (column%8 >= 4) ? -1 : 1;
}

static uint16_t bf16(int value) {
    float converted = static_cast<float>(value);
    uint32_t bits;
    std::memcpy(&bits, &converted, sizeof(bits));
    return static_cast<uint16_t>((bits + 0x7fff + ((bits >> 16) & 1)) >> 16);
}

int main(int argc, char** argv) {
    std::setvbuf(stdout, nullptr, _IOLBF, 0);
    if (argc != 4) return 2;
    try {
        const size_t m=extent(argv[1]), n=extent(argv[2]), k=extent(argv[3]);
        uint16_t golden[8][8];
        for (size_t i=0; i<8; ++i) {
            for (size_t j=0; j<8; ++j) {
                int dot=0;
                for (size_t t=0; t<8; ++t) {
                    const int repetitions = static_cast<int>(k/8 + (t < k%8));
                    dot += repetitions * left_value(i,t) * right_value(t,j);
                }
                golden[i][j]=bf16(dot);
            }
        }
        CHECK(aclInit(nullptr));
        CHECK(aclrtSetDevice(0));
        aclrtStream stream=nullptr;
        CHECK(aclrtCreateStream(&stream));
        std::printf("runtime_soc=%s\n",aclrtGetSocName());
        {
            Tensor<uint16_t> x({int64_t(m),int64_t(k)},ACL_BF16,m*k);
            Tensor<uint16_t> w({int64_t(k),int64_t(n)},ACL_BF16,k*n);
            Tensor<uint16_t> y({int64_t(m),int64_t(n)},ACL_BF16,m*n);
            x.fill([&](size_t i){return left_value(i/k,i%k)<0 ? uint16_t(0xbf80) : uint16_t(0x3f80);});
            w.fill([&](size_t i){return right_value(i/n,i%n)<0 ? uint16_t(0xbf80) : uint16_t(0x3f80);});
            y.fill([](size_t){return uint16_t(0x7fc0);});
            uint64_t bytes=0;
            aclOpExecutor* executor=nullptr;
            CHECK(aclnnMatmulGetWorkspaceSize(x.handle,w.handle,y.handle,0,&bytes,&executor));
            Buffer workspace(bytes);
            std::printf("case=matmul api=aclnnMatmul M=%zu N=%zu K=%zu dtype=BF16 layout=ND_ND workspace=%llu\n",m,n,k,(unsigned long long)bytes);
            CHECK(aclnnMatmul(workspace.ptr,bytes,executor,stream));
            CHECK(aclrtSynchronizeStream(stream));
            y.verify([&](size_t i,uint16_t v){return v==golden[(i/n)%8][(i%n)%8];});
        }
        CHECK(aclrtDestroyStream(stream));
        CHECK(aclrtResetDevice(0));
        CHECK(aclFinalize());
        std::puts("SMOKE_PASS");
    } catch(const std::exception& error) {
        std::fprintf(stderr,"SMOKE_FAIL %s\n",error.what());
        return 1;
    }
    return 0;
}
