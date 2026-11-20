// Ask the installed CANN host tiler about exact rank-local BF16 tensors.
#define main archived_large_vector_main
#include "../../../smoke/large_vector.cpp"
#undef main
#include <aclnnop/aclnn_matmul.h>
#include <fstream>

int main(int argc, char** argv) {
    if (argc != 2) return 2;
    std::setvbuf(stdout, nullptr, _IOLBF, 0);
    try {
        std::ifstream cases(argv[1]);
        if (!cases) throw std::runtime_error("unreadable shape list");
        CHECK(aclInit(nullptr));
        CHECK(aclrtSetDevice(0));
        std::printf("host_query_soc=%s\n", aclrtGetSocName());
        int64_t m, n, k;
        while (cases >> m >> n >> k) {
            if (m <= 0 || n <= 0 || k <= 0 || m > 2097152 || n > 3072 || k > 7168)
                throw std::runtime_error("geometry outside registered campaign");
            Tensor<uint16_t> x({m,k}, ACL_BF16, m*k);
            Tensor<uint16_t> weight({k,n}, ACL_BF16, k*n);
            Tensor<uint16_t> y({m,n}, ACL_BF16, m*n);
            aclOpExecutor* executor = nullptr;
            uint64_t bytes = 0;
            CHECK(aclnnMatmulGetWorkspaceSize(x.handle, weight.handle, y.handle, 0, &bytes, &executor));
            if (!executor) throw std::runtime_error("missing Matmul executor");
            std::printf("HOST_WS %lld %lld %lld %llu\n", (long long)m, (long long)n,
                        (long long)k, (unsigned long long)bytes);
            CHECK(aclDestroyAclOpExecutor(executor));
        }
        if (!cases.eof()) throw std::runtime_error("unreadable geometry");
        CHECK(aclrtResetDevice(0));
        CHECK(aclFinalize());
        std::puts("HOST_QUERY_DONE");
    } catch (const std::exception& e) {
        std::fprintf(stderr, "HOST_QUERY_FAILED %s\n",e.what());
        return 1;
    }
}
