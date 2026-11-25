#include "../acl_tensor.hpp"
#include <aclnnop/aclnn_layer_norm.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>

// Alternating 1 and 3, mean 2 and variance 1, gamma=2 and beta=1.
// Padding has a different value so that treating the input as dense fails.
static std::vector<uint8_t> filled(int n, uint16_t v) {
    std::vector<uint8_t> bytes(size_t(n)*2);
    for (int i=0;i<n;++i) std::memcpy(bytes.data()+i*2,&v,2);
    return bytes;
}

static void run(int rows, int width, int stride, aclrtStream stream) {
    if (rows<1 || width<2 || width%2 || stride<width) throw std::invalid_argument("invalid shape");
    auto data=filled(rows*stride,0x4100);
    for(int i=0;i<rows;++i) for(int j=0;j<width;++j){
        uint16_t value=(i+j)%2 ? 0x4040:0x3f80;
        std::memcpy(data.data()+2*(i*stride+j),&value,2);
    }
    Tensor storage({rows,stride},ACL_BF16,data);
    int64_t shape[]={rows,width}, strides[]={stride,1}, backing[]={rows,stride}, axis=width;
    auto x=aclCreateTensor(shape,2,ACL_BF16,strides,0,ACL_FORMAT_ND,backing,2,storage.buffer.ptr);
    auto normalized=aclCreateIntArray(&axis,1);
    if (!x || !normalized) throw std::runtime_error("descriptor creation failed");
    Tensor gamma({width},ACL_BF16,filled(width,0x4000)), beta({width},ACL_BF16,filled(width,0x3f80));
    auto output=filled(rows*width,0);
    Tensor y({rows,width},ACL_BF16,output);
    uint64_t size=0;
    aclOpExecutor *exec=nullptr;
    CHECK(aclnnLayerNormGetWorkspaceSize(x,normalized,gamma.desc,beta.desc,1e-6,y.desc,nullptr,nullptr,&size,&exec));
    Buffer workspace(size);
    CHECK(aclnnLayerNorm(workspace.ptr,size,exec,stream));
    CHECK(aclrtSynchronizeStream(stream));
    CHECK(aclrtMemcpy(output.data(),output.size(),y.buffer.ptr,output.size(),ACL_MEMCPY_DEVICE_TO_HOST));
    for(int i=0;i<rows;++i) for(int j=0;j<width;++j){
        uint16_t got;
        std::memcpy(&got,output.data()+2*(i*width+j),2);
        if(got!=((i+j)%2 ? 0x4040 : 0xbf80)) throw std::runtime_error("LayerNorm BF16 output mismatch");
    }
    aclDestroyTensor(x);
    aclDestroyIntArray(normalized);
    std::printf("case=layer_norm api=aclnnLayerNorm rows=%d width=%d stride=%d\ncorrectness Y count=%d bad=0\n",rows,width,stride,rows*width);
}

int main(int argc,char** argv) {
    if(argc!=5 || std::string(argv[1])!="layer_norm") return 2;
    try{
        CHECK(aclInit(nullptr)); CHECK(aclrtSetDevice(0));
        std::printf("runtime_soc=%s\n",aclrtGetSocName());
        aclrtStream stream; CHECK(aclrtCreateStream(&stream));
        run(std::stoi(argv[2]),std::stoi(argv[3]),std::stoi(argv[4]),stream);
        CHECK(aclrtDestroyStream(stream));CHECK(aclrtResetDevice(0));CHECK(aclFinalize());
        std::puts("SMOKE_PASS");std::fflush(stdout);std::_Exit(EXIT_SUCCESS);
    }catch(const std::exception &e){std::fprintf(stderr,"SMOKE_FAIL %s\n",e.what());return 1;}
}
