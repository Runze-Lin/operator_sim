// Same checked tensor and workspace ownership as the existing norm host.
#define main archived_norm_host
#include "../../../smoke/extension_norm_addmm.cpp"
#undef main
#include <aclnnop/aclnn_sigmoid.h>

static void sigmoid(aclrtStream stream, int rows, int heads) {
    const size_t n = size_t(rows) * heads;
    std::vector<float> input(n);
    for (size_t i = 0; i < n; ++i) input[i] = (int(i % 257) - 128)/16.0f;
    Tensor<float> x({rows,heads}, ACL_FLOAT, input);
    Tensor<float> y({rows,heads}, ACL_FLOAT, std::vector<float>(n,NAN));
    uint64_t bytes=0; aclOpExecutor* exe=nullptr;
    ACL_CHECK(aclnnSigmoidGetWorkspaceSize(x.tensor,y.tensor,&bytes,&exe));
    Buffer ws(bytes);
    ACL_CHECK(aclnnSigmoid(ws.ptr,bytes,exe,stream));
    ACL_CHECK(aclrtSynchronizeStream(stream));
    auto out = y.read();
    size_t bad = 0;
    for (size_t i=0; i<n; ++i) {
        const double expected=1/(1+std::exp(-double(input[i])));
        bad += !(std::abs(out[i]-expected) <= 2e-6);
    }
    std::printf("correctness count=%zu bad=%zu atol=2e-6\n", n,bad);
    if (bad) throw std::runtime_error("Sigmoid disagrees with CPU reference");
}

int main(int argc,char** argv) {
    setvbuf(stdout,nullptr,_IOLBF,0);
    try {
        if(argc!=3) throw std::runtime_error("Expected rows and local heads");
        int t=std::stoi(argv[1]),h=std::stoi(argv[2]);
        if(t<=0||h<=0||size_t(t)*h>67108864) throw std::runtime_error("Invalid test allocation");
        ACL_CHECK(aclInit(nullptr)); ACL_CHECK(aclrtSetDevice(0));
        std::printf("runtime_soc=%s\ncase=sigmoid_fp32 api=aclnnSigmoid T=%d H=%d\n",aclrtGetSocName(),t,h);
        aclrtStream stream=nullptr; ACL_CHECK(aclrtCreateStream(&stream));
        sigmoid(stream,t,h);
        ACL_CHECK(aclrtDestroyStream(stream)); ACL_CHECK(aclrtResetDevice(0)); ACL_CHECK(aclFinalize());
        std::puts("SMOKE_PASS"); return 0;
    } catch(const std::exception& e) {std::fprintf(stderr,"SMOKE_FAIL %s\n",e.what());return 1;}
}
