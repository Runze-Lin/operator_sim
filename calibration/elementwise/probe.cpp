// Captured residual addition and per-row multiplication, with independent exact golden values.
#include "../acl_tensor.hpp"
#include <aclnnop/aclnn_add.h>
#include <aclnnop/aclnn_mul.h>
#include <cstdio>
#include <cstring>
#include <cstdlib>

static void encode(std::vector<uint8_t>& bytes, size_t i, float value, bool bf16) {
    if (bf16) {
        uint32_t bits; std::memcpy(&bits, &value, 4);
        uint16_t high = bits >> 16; // All test values and results are exact in BF16.
        std::memcpy(bytes.data()+2*i, &high, 2);
    } else {
        std::memcpy(bytes.data()+4*i, &value, 4);
    }
}

static void run(const std::string& route, int rows, int width, aclrtStream stream) {
    bool add = route=="add";
    if ((!add && route!="mul") || rows<=0 || width<=0) throw std::runtime_error("Invalid elementwise case");
    size_t n = size_t(rows)*width, b = add ? 2 : 4;
    std::vector<uint8_t> a(b*n), y(b*n,0xcd), golden(b*n), rhs(b*(add ? n : rows));
    for (size_t i=0; i<n; ++i) {
        float x = (int(i%13)-6)*0.5f;
        auto j = add ? i : i/width;
        float v = (int(j%7)-3)*0.25f;
        encode(a,i,x,add); encode(rhs,j,v,add); encode(golden,i,add ? x+v : x*v,add);
    }
    auto dtype=add ? ACL_BF16 : ACL_FLOAT;
    Tensor x({rows,width},dtype,a), other({rows,add ? width : 1},dtype,rhs), out({rows,width},dtype,y);
    uint64_t bytes=0; aclOpExecutor* exec=nullptr;
    float one=1;
    aclScalar* alpha=aclCreateScalar(&one,ACL_FLOAT);
    if (!alpha) throw std::runtime_error("alpha descriptor");
    if (add) CHECK(aclnnAddGetWorkspaceSize(x.desc,other.desc,alpha,out.desc,&bytes,&exec));
    else CHECK(aclnnMulGetWorkspaceSize(x.desc,other.desc,out.desc,&bytes,&exec));
    Buffer workspace(bytes);
    if (add) CHECK(aclnnAdd(workspace.ptr,bytes,exec,stream));
    else CHECK(aclnnMul(workspace.ptr,bytes,exec,stream));
    CHECK(aclrtSynchronizeStream(stream));
    CHECK(aclrtMemcpy(y.data(),y.size(),out.buffer.ptr,y.size(),ACL_MEMCPY_DEVICE_TO_HOST));
    aclDestroyScalar(alpha);
    size_t bad=0;
    for (size_t i=0;i<n;++i) {
        if (add) bad += std::memcmp(y.data()+i*b,golden.data()+i*b,b)!=0;
        else { // IEEE +/-0 compare equal.
            float actual, expected;
            std::memcpy(&actual,y.data()+i*b,b); std::memcpy(&expected,golden.data()+i*b,b);
            bad += actual!=expected;
        }
    }
    std::printf("case=%s rows=%d width=%d rhs_width=%d api=%s\ncorrectness Y count=%zu bad=%zu\n",
                route.c_str(),rows,width,add ? width : 1,add ? "aclnnAdd" : "aclnnMul",n,bad);
    if (bad) throw std::runtime_error("Numerical mismatch");
}

int main(int argc, char** argv) {
    std::setvbuf(stdout,nullptr,_IONBF,0);
    if (argc!=4) return 2;
    try {
        CHECK(aclInit(nullptr)); CHECK(aclrtSetDevice(0));
        std::printf("runtime_soc=%s\n",aclrtGetSocName());
        aclrtStream stream=nullptr; CHECK(aclrtCreateStream(&stream));
        run(argv[1],std::stoi(argv[2]),std::stoi(argv[3]),stream);
        CHECK(aclrtDestroyStream(stream)); CHECK(aclrtResetDevice(0)); CHECK(aclFinalize());
        std::puts("SMOKE_PASS"); std::_Exit(EXIT_SUCCESS);
    } catch (const std::exception& e) {std::fprintf(stderr,"SMOKE_FAIL %s\n",e.what());return 1;}
}
