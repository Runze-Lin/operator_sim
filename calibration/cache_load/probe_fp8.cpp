// Call the rowwise E4M3FN/FP32 -> BF16 implementation on a contiguous [R,D] view.
#include "../acl_tensor.hpp"
#include <aclnnop/aclnn_ascend_anti_quant.h>
#include <cmath>
#include <cstdio>
#include <cstring>

int main(int argc,char **argv) {
    std::setvbuf(stdout,nullptr,_IONBF,0);
    if (argc!=5 || std::string(argv[1])!="cache_load_fp8") return 2;
    const int b=std::stoi(argv[2]),s=std::stoi(argv[3]),d=std::stoi(argv[4]),rows=b*s;
    if (b<=0 || s<=0 || d<=0) return 2;
    try {
        CHECK(aclInit(nullptr)); CHECK(aclrtSetDevice(0)); std::printf("runtime_soc=%s\n",aclrtGetSocName());
        aclrtStream stream=nullptr; CHECK(aclrtCreateStream(&stream));
        const size_t count=size_t(rows)*d;
        std::vector<uint8_t> xb(count),scales(size_t(rows)*4);
        std::vector<uint16_t> observed(count),expected(count);
        const uint8_t codes[]={0x01,0x30,0x38,0x3c,0x44,0x7e,0xb0,0xbc,0xc4};
        const float numbers[]={1.0f/512,.5,1,1.5,3,448,-.5,-1.5,-3};
        for(int r=0;r<rows;++r) {
            const float scale=float(5+(r%19))/8;
            std::memcpy(scales.data()+4*r,&scale,4);
            for (int c=0;c<d;++c) {
                size_t i=size_t(r)*d+c,j=(c+r)%9;
                xb[i]=codes[j];
                float v=numbers[j]*scale; uint32_t bits;
                std::memcpy(&bits,&v,4); expected[i]=(bits+0x7fff+((bits>>16)&1))>>16;
            }
        }
        {
            Tensor x({rows,d},ACL_FLOAT8_E4M3FN,xb), scale({rows,1},ACL_FLOAT,scales);
            Tensor y({rows,d},ACL_BF16,std::vector<uint8_t>(count*2,0xcc));
            uint64_t bytes=0; aclOpExecutor* exe=nullptr;
            CHECK(aclnnAscendAntiQuantGetWorkspaceSize(x.desc,scale.desc,nullptr,ACL_BF16,false,y.desc,&bytes,&exe));
            std::printf("case=cache_load_fp8 api=aclnnAscendAntiQuant sqrt=false offset=null dst=bf16 shape=%dx%dx%d\n",b,s,d);
            Buffer tmp(bytes); CHECK(aclnnAscendAntiQuant(tmp.ptr,bytes,exe,stream)); CHECK(aclrtSynchronizeStream(stream));
            CHECK(aclrtMemcpy(observed.data(),count*2,y.buffer.ptr,count*2,ACL_MEMCPY_DEVICE_TO_HOST));
        }
        size_t bad=0; for(size_t i=0;i<count;++i) if(expected[i]!=observed[i]) {
            if(bad<3) std::printf("mismatch i=%zu expected=%04x got=%04x\n",i,expected[i],observed[i]); ++bad;
        }
        std::printf("correctness Y count=%zu bad=%zu\n",count,bad);
        if(bad) throw std::runtime_error("FP8 CPU golden mismatch");
        CHECK(aclrtDestroyStream(stream)); CHECK(aclrtResetDevice(0)); CHECK(aclFinalize());
        std::puts("SMOKE_PASS"); return 0;
    } catch(const std::exception &e) {std::fprintf(stderr,"SMOKE_FAIL %s\n",e.what());return 1;}
}
