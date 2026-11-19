// Exercise the exact batched binding used by both pinned DeepSeek model adapters.
#define main archived_mhc_test_main
#include "../../../smoke/extension_indexer_mhc.cpp"
#undef main
#include "aclnn_hc_post.h"
#include <dlfcn.h>

static void test(aclrtStream stream, int tokens, int hidden) {
    constexpr int hc = 4;
    auto x = data(tokens * hidden, 7);
    auto residual = data(tokens * hc * hidden, 4);
    std::vector<float> post(tokens * hc), comb(tokens * hc * hc);
    for (size_t i = 0; i < post.size(); ++i)
        post[i] = (static_cast<int>(i % 7) - 3) / 8.f;
    for (size_t i = 0; i < comb.size(); ++i)
        comb[i] = (static_cast<int>(i % 11) - 5) / 16.f;
    Tensor<uint16_t> input({1,tokens,hidden},ACL_BF16,x);
    Tensor<uint16_t> prev({1,tokens,hc,hidden},ACL_BF16,residual);
    Tensor<float> gates({1,tokens,hc},ACL_FLOAT,post);
    Tensor<float> mixes({1,tokens,hc,hc},ACL_FLOAT,comb);
    Tensor<uint16_t> output({1,tokens,hc,hidden},ACL_BF16,std::vector<uint16_t>(residual.size(),0x7fc0));
    uint64_t bytes = 0;
    aclOpExecutor* ex = nullptr;
    ACL_CHECK(aclnnHcPostGetWorkspaceSize(input.tensor,prev.tensor,gates.tensor,mixes.tensor,output.tensor,&bytes,&ex));
    Buffer ws(bytes);
    ACL_CHECK(aclnnHcPost(ws.ptr,bytes,ex,stream));
    ACL_CHECK(aclrtSynchronizeStream(stream));
    auto actual = output.read();
    auto expected = postgold(residual,doubles(comb),x,doubles(post),tokens,hc,hidden);
    size_t bad = 0;
    for (size_t i = 0; i < actual.size(); ++i)
        bad += actual[i] != bf16(static_cast<float>(expected[i])) && !(expected[i] == 0 && fp32(actual[i]) == 0);
    std::printf("correctness name=output count=%zu bad=%zu comparison=bf16_rne_exact\n",actual.size(),bad);
    if (bad) throw std::runtime_error("BF16 output differs from the FP64 reference rounded to BF16");
}

int main(int argc, char** argv) {
    setvbuf(stdout, nullptr, _IOLBF, 0);
    try {
        if (argc != 3) throw std::runtime_error("usage: host_probe TOKENS HIDDEN");
        int t = dim(argv[1]), h = dim(argv[2]);
        if (int64_t(t) * h > 1024 * 1024) throw std::runtime_error("development smoke size exceeded");
        Dl_info api{};
        if (!dladdr(reinterpret_cast<void*>(aclnnHcPost), &api)) throw std::runtime_error("missing HcPost symbol owner");
        std::printf("api=aclnnHcPost library=%s\n",api.dli_fname);
        ACL_CHECK(aclInit(nullptr));
        ACL_CHECK(aclrtSetDevice(0));
        std::printf("runtime_soc=%s\n",aclrtGetSocName());
        aclrtStream stream{};
        ACL_CHECK(aclrtCreateStream(&stream));
        test(stream,t,h);
        ACL_CHECK(aclrtDestroyStream(stream));
        ACL_CHECK(aclrtResetDevice(0));
        ACL_CHECK(aclFinalize());
        std::puts("SMOKE_PASS");
        return 0;
    } catch (const std::exception& e) {
        std::fprintf(stderr,"SMOKE_FAIL %s\n",e.what());
        return 1;
    }
}
