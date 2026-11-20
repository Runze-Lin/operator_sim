// Exercise only the installed host tiler. No tensor payload is read or kernel submitted.
#define main archived_vector_smoke_main
#include "../../../smoke/large_vector.cpp"
#undef main
#include <aclnnop/aclnn_matmul.h>
#include <fstream>

int main(int argc, char **argv) {
    if (argc != 2) return 2;
    std::setvbuf(stdout, nullptr, _IOLBF, 0);
    try {
        std::ifstream cases(argv[1]);
        if (!cases) throw std::runtime_error("cannot open shape list");
        CHECK(aclInit(nullptr));
        CHECK(aclrtSetDevice(0));
        std::printf("host_query_soc=%s\n", aclrtGetSocName());
        int64_t m, n, k;
        while (cases >> m >> n >> k) {
            if (m <= 0 || n <= 0 || k <= 0 || std::max({m,n,k}) > 16384)
                throw std::runtime_error("shape outside registered host query range");
            Tensor<uint16_t> x({m,k}, ACL_BF16, m*k);
            Tensor<uint16_t> w({k,n}, ACL_BF16, k*n);
            Tensor<uint16_t> y({m,n}, ACL_BF16, m*n);
            uint64_t bytes = 0;
            aclOpExecutor *executor = nullptr;
            CHECK(aclnnMatmulGetWorkspaceSize(x.handle, w.handle, y.handle, 0, &bytes, &executor));
            if (!executor) throw std::runtime_error("missing phase-one executor");
            std::printf("HOST_WS %lld %lld %lld %llu\n", (long long)m,(long long)n,(long long)k,(unsigned long long)bytes);
            CHECK(aclDestroyAclOpExecutor(executor));
        }
        if (!cases.eof()) throw std::runtime_error("invalid shape list");
        CHECK(aclrtResetDevice(0));
        CHECK(aclFinalize());
        std::puts("HOST_QUERY_DONE");
    } catch (const std::exception& error) {
        std::fprintf(stderr, "HOST_QUERY_FAILED %s\n", error.what());
        return 1;
    }
}
