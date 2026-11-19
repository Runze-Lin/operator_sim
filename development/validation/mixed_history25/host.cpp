// Real aclnnRmsNorm and aclnnAddmm calls share one stream and the existing independent CPU goldens.
#define main standalone_program
#include "../../../smoke/extension_norm_addmm.cpp"
#undef main
#include <fstream>

int main(int argc, char** argv) {
    aclrtStream stream = nullptr;
    try {
        if (argc != 2) throw std::runtime_error("Expected registered input file");
        std::ifstream in(argv[1]);
        if (!in) throw std::runtime_error("Missing input file");
        ACL_CHECK(aclInit(nullptr));
        ACL_CHECK(aclrtSetDevice(0));
        ACL_CHECK(aclrtCreateStream(&stream));
        std::printf("runtime_soc=%s\n", aclrtGetSocName());
        std::string op;
        int a, b, c, i = 0;
        while (in >> op >> a >> b >> c) {
            if (a < 1 || b < 1 || c < 0 || int64_t(a)*b*std::max(c,1)>64*1024*1024)
                throw std::runtime_error("Test work budget exceeded");
            std::printf("BATCH_BEGIN %d\n", i);
            if (op == "rmsnorm" && c == 0) rmsnorm(stream, a, b);
            else if (op == "addmm" && c > 0) addmm(stream, a, b, c, false);
            else throw std::runtime_error("Unexpected input signature");
            std::printf("BATCH_END %d\n", i++);
            std::fflush(stdout);
        }
        if (!in.eof() || !i) throw std::runtime_error("Incomplete input");
        ACL_CHECK(aclrtDestroyStream(stream));
        ACL_CHECK(aclrtResetDevice(0));
        ACL_CHECK(aclFinalize());
        std::puts("SMOKE_PASS");
        return 0;
    } catch (const std::exception& e) {
        std::fprintf(stderr, "SMOKE_FAIL %s\n", e.what());
        return 1;
    }
}
