// Isolated BF16 -> E2M1 + E8M0 block32 dispatch. Count both nibbles in the golden.
#include "acl_tensor.hpp"
#include <aclnnop/aclnn_dynamic_mx_quant.h>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <memory>

static size_t compare(const Tensor &result, const std::vector<uint8_t> &gold, bool packed) {
    std::vector<uint8_t> values(gold.size());
    CHECK(aclrtMemcpy(values.data(), values.size(), result.buffer.ptr, values.size(), ACL_MEMCPY_DEVICE_TO_HOST));
    size_t bad=0;
    for (size_t i=0; i<values.size(); ++i) {
        uint8_t difference=gold[i]^values[i];
        if (packed) bad+=bool(difference & 0xf)+bool(difference >> 4);
        else bad+=bool(difference);
    }
    std::printf("correctness %s count=%zu bad=%zu\n", packed?"Y":"scale", gold.size()*(packed?2:1), bad);
    return bad;
}

static void run(int rows, int cols, int pitch, int offset, aclrtStream stream) {
    if (rows<1 || cols<32 || cols%2 || offset<0 || pitch<cols || offset>pitch-cols)
        throw std::runtime_error("Invalid logical shape, row pitch or storage offset");
    size_t count=size_t(rows)*cols;
    int scale_stride=2*((cols+63)/64);
    std::vector<uint8_t> xb(2*size_t(rows)*pitch,0xcd), yb(count/2), sb(size_t(rows)*scale_stride);
    // Each block starts with ±6, so its E8M0 exponent equals 'exponent' even for a short tail.
    const float numbers[]={6,-6,.5,-.5,1.5,-1.5,4,-4};
    const uint8_t codes[]={7,15,1,9,3,11,6,14};
    for (int r=0; r<rows; ++r) {
        for (int c=0; c<cols; ++c) {
            size_t i=size_t(r)*cols+c;
            int group=c/32, exponent=(r+group)%5-2;
            int index=(c%8)^((r+group)%2);
            float value=std::ldexp(numbers[index],exponent);
            uint32_t bits;
            std::memcpy(&bits,&value,sizeof(bits));
            uint16_t bf=bits>>16;
            std::memcpy(&xb[2*(size_t(r)*pitch+offset+c)],&bf,sizeof(bf));
            yb[i/2]|=codes[index]<<(4*(i%2));
            sb[size_t(r)*scale_stride+group]=127+exponent;
        }
    }
    Tensor x({rows,pitch},ACL_BF16,xb);
    std::unique_ptr<aclTensor,decltype(&aclDestroyTensor)> view(nullptr,&aclDestroyTensor);
    if (pitch!=cols || offset!=0) {
        int64_t logical[]={rows,cols}, storage[]={rows,pitch}, stride[]={pitch,1};
        view.reset(aclCreateTensor(logical,2,ACL_BF16,stride,offset,ACL_FORMAT_ND,storage,2,x.buffer.ptr));
        if (!view) throw std::runtime_error("Failed to construct strided tensor");
        std::printf("view_stride=%d,1 storage_offset=%d backing_bytes=%zu\n",pitch,offset,xb.size());
    }
    Tensor y({rows,cols},ACL_FLOAT4_E2M1,std::vector<uint8_t>(yb.size(),0xcc));
    Tensor s({rows,scale_stride/2,2},ACL_FLOAT8_E8M0,std::vector<uint8_t>(sb.size(),0xff));
    uint64_t bytes=0; aclOpExecutor *exe=nullptr;
    char mode[]="round";
    CHECK(aclnnDynamicMxQuantGetWorkspaceSize(view?view.get():x.desc,-1,mode,40,32,0,y.desc,s.desc,&bytes,&exe));
    std::printf("case=quant_mxfp4 rows=%d cols=%d api=aclnnDynamicMxQuant axis=-1 group=32 round=round scaleAlg=0\n",rows,cols);
    Buffer workspace(bytes);
    CHECK(aclnnDynamicMxQuant(workspace.ptr,bytes,exe,stream));
    CHECK(aclrtSynchronizeStream(stream));
    auto y_bad=compare(y,yb,true);
    auto s_bad=compare(s,sb,false);
    if (y_bad || s_bad) throw std::runtime_error("Output differs from OCP E2M1/E8M0 golden");
}

int main(int argc, char** argv) {
    std::setvbuf(stdout,nullptr,_IONBF,0);
    if ((argc!=4 && argc!=6) || std::string(argv[1])!="quant_mxfp4") return 2;
    try {
        CHECK(aclInit(nullptr)); CHECK(aclrtSetDevice(0));
        std::printf("runtime_soc=%s\n",aclrtGetSocName());
        aclrtStream stream=nullptr; CHECK(aclrtCreateStream(&stream));
        run(std::stoi(argv[2]),std::stoi(argv[3]),argc==6?std::stoi(argv[4]):std::stoi(argv[3]),
            argc==6?std::stoi(argv[5]):0,stream);
        CHECK(aclrtDestroyStream(stream)); CHECK(aclrtResetDevice(0)); CHECK(aclFinalize());
        std::puts("SMOKE_PASS"); return 0;
    } catch(const std::exception& e) { std::fprintf(stderr,"SMOKE_FAIL %s\n",e.what()); return 1; }
}
