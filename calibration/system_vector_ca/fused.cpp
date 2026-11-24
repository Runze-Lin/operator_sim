// Native development fixtures for split-channel SwiGLU and partial interleaved RoPE.
#include "../acl_tensor.hpp"
#include <aclnnop/aclnn_swi_glu.h>
#include <aclnnop/aclnn_inplace_partial_rotary_mul.h>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>

static uint16_t bf(float x) {
    uint32_t n;
    std::memcpy(&n,&x,4);
    return (n + 0x7fff + ((n >> 16) & 1)) >> 16;
}

static std::vector<uint8_t> enc(const std::vector<float>& a) {
    std::vector<uint8_t> data(2*a.size());
    for (size_t i=0; i<a.size(); ++i) {
        auto v=bf(a[i]);
        std::memcpy(data.data()+2*i,&v,2);
    }
    return data;
}

static void golden(Tensor& y, const std::vector<float>& g, float atol) {
    std::vector<uint16_t> got(g.size());
    CHECK(aclrtMemcpy(got.data(),got.size()*2,y.buffer.ptr,got.size()*2,ACL_MEMCPY_DEVICE_TO_HOST));
    size_t bad=0;
    for (size_t j=0;j<g.size();++j) {
        uint32_t raw=uint32_t(got[j])<<16; float x; std::memcpy(&x,&raw,4);
        bad+=!(std::abs(x-g[j])<=atol);
    }
    std::printf("correctness Y count=%zu bad=%zu\n",got.size(),bad);
    if(bad) throw std::runtime_error("incorrect output");
}

static void run(bool rope,int rows,int width,aclrtStream stream) {
    uint64_t ws=0; aclOpExecutor *exec=nullptr;
    if (!rope) {
        std::vector<float> x(rows*2*width),g(rows*width);
        for (int r=0;r<rows;r++) for(int c=0;c<width;c++) {
            float a=(c%9-4)*.25f,b=((c+r)%7-3)*.5f;
            x[r*2*width+c]=a; x[r*2*width+width+c]=b;
            g[r*width+c]=a*b/(1+std::exp(-a));
        }
        Tensor t({rows,2*width},ACL_BF16,enc(x)), y({rows,width},ACL_BF16,enc(g));
        CHECK(aclrtMemset(y.buffer.ptr,g.size()*2,0x99,g.size()*2));
        CHECK(aclnnSwiGluGetWorkspaceSize(t.desc,-1,y.desc,&ws,&exec));
        Buffer space(ws); CHECK(aclnnSwiGlu(space.ptr,ws,exec,stream)); CHECK(aclrtSynchronizeStream(stream));
        golden(y,g,.01f);
    } else {
        const int h=8, tail=64;
        std::vector<float> x(rows*h*width),g(x.size());
        for(size_t i=0;i<x.size();i++) x[i]=float(int(i%13)-6);
        g=x;
        for(size_t i=0;i<x.size();i+=width) for(int j=width-tail;j<width;j+=2) {
            g[i+j]=-x[i+j+1]; g[i+j+1]=x[i+j];
        }
        Tensor t({rows,1,h,width},ACL_BF16,enc(x)), c({rows,1,1,tail},ACL_BF16,enc(std::vector<float>(rows*tail,0))),
               s({rows,1,1,tail},ACL_BF16,enc(std::vector<float>(rows*tail,1)));
        const int64_t v[]={width-tail,width}; aclIntArray* slice=aclCreateIntArray(v,2);
        CHECK(aclnnInplacePartialRotaryMulGetWorkspaceSize(t.desc,c.desc,s.desc,1,slice,&ws,&exec));
        Buffer space(ws); CHECK(aclnnInplacePartialRotaryMul(space.ptr,ws,exec,stream));
        CHECK(aclrtSynchronizeStream(stream)); aclDestroyIntArray(slice);
        golden(t,g,0.f);
    }
}

int main(int argc,char** argv) {
    std::setvbuf(stdout,nullptr,_IONBF,0);
    if(argc!=4) return 2;
    try {
        CHECK(aclInit(nullptr)); CHECK(aclrtSetDevice(0));
        std::printf("runtime_soc=%s\n",aclrtGetSocName());
        aclrtStream stream=nullptr; CHECK(aclrtCreateStream(&stream));
        run(std::string(argv[1])=="rope",std::stoi(argv[2]),std::stoi(argv[3]),stream);
        CHECK(aclrtDestroyStream(stream)); CHECK(aclrtResetDevice(0)); CHECK(aclFinalize());
        std::puts("SMOKE_PASS"); std::_Exit(0);
    } catch(const std::exception& e){std::fprintf(stderr,"SMOKE_FAIL %s\n",e.what()); return 1;}
}
