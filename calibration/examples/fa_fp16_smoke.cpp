// One dense FP16 SBH validation case for the installed 950 FlashAttentionScoreV2.
#include "../acl_tensor.hpp"
#include <aclnnop/aclnn_flash_attention_score.h>
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>

static std::vector<uint8_t> half_input(int count, int mul, int modulus) {
    std::vector<uint8_t> bytes(2 * count);
    for (int i = 0; i < count; ++i) {
        __fp16 value = float((i * mul + 3) % modulus - modulus / 2) / 16;
        std::memcpy(bytes.data() + 2 * i, &value, 2);
    }
    return bytes;
}

static float read_half(const std::vector<uint8_t>& bytes, int i) {
    __fp16 value;
    std::memcpy(&value, bytes.data() + 2 * i, 2);
    return value;
}

static void run(aclrtStream stream) {
    constexpr int rows = 9, keys = 257, dim = 64;
    auto q = half_input(rows * dim, 7, 17);
    auto k = half_input(keys * dim, 11, 19);
    auto v = half_input(keys * dim, 13, 23);
    Tensor qt({rows,1,dim},ACL_FLOAT16,q), kt({keys,1,dim},ACL_FLOAT16,k);
    Tensor vt({keys,1,dim},ACL_FLOAT16,v);
    std::vector<uint8_t> result(2 * rows * dim, 0xff);
    Tensor output({rows,1,dim},ACL_FLOAT16,result);
    std::vector<uint8_t> stats(rows * 8 * sizeof(float), 0xff);
    Tensor mx({1,1,rows,8},ACL_FLOAT,stats), sum({1,1,rows,8},ACL_FLOAT,stats);
    aclOpExecutor* executor = nullptr;
    uint64_t bytes = 0;
    char layout[] = "SBH";
    CHECK(aclnnFlashAttentionScoreV2GetWorkspaceSize(qt.desc,kt.desc,vt.desc,nullptr,nullptr,
        nullptr,nullptr,nullptr,nullptr,nullptr,0.125,1.0,2147483647,2147483647,1,layout,0,0,1,
        mx.desc,sum.desc,nullptr,output.desc,&bytes,&executor));
    Buffer workspace(bytes);
    CHECK(aclnnFlashAttentionScoreV2(workspace.ptr,bytes,executor,stream));
    CHECK(aclrtSynchronizeStream(stream));
    CHECK(aclrtMemcpy(result.data(),result.size(),output.buffer.ptr,result.size(),ACL_MEMCPY_DEVICE_TO_HOST));
    size_t bad = 0;
    double max_abs = 0;
    for (int i=0; i<rows; ++i) {
        std::vector<double> a(keys,0);
        for (int j=0;j<keys;++j)
            for (int t=0;t<dim;++t)
                a[j]+=0.125*read_half(q,i*dim+t)*read_half(k,j*dim+t);
        const double peak=*std::max_element(a.begin(),a.end());
        double denom=0;
        for (auto& x:a) {x=std::exp(x-peak); denom+=x;}
        for (int t=0;t<dim;++t) {
            double gold=0;
            for (int j=0;j<keys;++j) gold+=a[j]*read_half(v,j*dim+t)/denom;
            const double err=std::abs(read_half(result,i*dim+t)-gold);
            max_abs=std::max(max_abs,err);
            if (!std::isfinite(err) || err>0.002+0.01*std::abs(gold)) ++bad;
        }
    }
    std::printf("correctness output=FP16 elements=%d max_abs=%g bad=%zu\n",rows*dim,max_abs,bad);
    if (bad) throw std::runtime_error("FA output differs from dense CPU reference");
}

int main() {
    try {
        CHECK(aclInit(nullptr));
        CHECK(aclrtSetDevice(0));
        std::printf("runtime_soc=%s\n",aclrtGetSocName());
        aclrtStream stream;
        CHECK(aclrtCreateStream(&stream));
        run(stream);
        CHECK(aclrtDestroyStream(stream));
        CHECK(aclrtResetDevice(0));
        CHECK(aclFinalize());
        std::puts("SMOKE_PASS");
    } catch (const std::exception& exc) {
        std::fprintf(stderr,"SMOKE_FAIL: %s\n",exc.what());return 1;
    }
}
