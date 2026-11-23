// aclnnGroupedMatmulV5, grouped M, both inputs MXFP4, groupType=0, groupListType=1.
#include "../acl_tensor.hpp"
#include <aclnnop/aclnn_grouped_matmul_v5.h>
#include <algorithm>
#include <cstdio>
#include <cstring>
#include <numeric>
#include <sstream>

struct List {
    aclTensorList *value;
    explicit List(Tensor &t) {
        const aclTensor *items[]={t.desc};
        value=aclCreateTensorList(items,1);
        if (!value) throw std::runtime_error("aclCreateTensorList failed");
        t.desc=nullptr;  // List owns member descriptor.
    }
    ~List() { aclDestroyTensorList(value); }
};

static void view(Tensor &t, const std::vector<int64_t> &dims, const std::vector<int64_t> &strides,
                 const std::vector<int64_t> &storage, aclDataType dtype) {
    aclDestroyTensor(t.desc);
    t.desc=aclCreateTensor(dims.data(),dims.size(),dtype,strides.data(),0,ACL_FORMAT_ND,
                           storage.data(),storage.size(),t.buffer.ptr);
    if (!t.desc) throw std::runtime_error("aclCreateTensor view");
}

static uint16_t bf16(float v) {
    uint32_t bits;
    std::memcpy(&bits,&v,sizeof(bits));
    return (bits+0x7fff+((bits>>16)&1))>>16;
}

static void run(const std::vector<int64_t>& counts,int n,int k,aclrtStream stream) {
    const int g=counts.size(),m=std::accumulate(counts.begin(),counts.end(),0),p=(k+63)/64;
    if (m<1 || !g || n<1 || k<4 || k%2) throw std::runtime_error("invalid geometry");
    std::vector<uint8_t> av(size_t(m)*k/2),bv(size_t(g)*n*k/2),as(size_t(m)*2*p,127),
                         bs(size_t(g)*n*2*p,127),hist(sizeof(int64_t)*g);
    std::memcpy(hist.data(),counts.data(),hist.size());
    auto a_sign=[](int i,int t){return (i%7+t%5)%4 ? 1:-1;};
    auto w_sign=[](int e,int j,int t){return (e%3+j%5+t%7)%3 ? 1:-1;};
    auto a_exp=[](int i,int h){return (i+h)%2;};
    auto w_exp=[](int e,int j,int h){return (e+j+h/2)%2;};
    for (int i=0;i<m;++i) {
        for (int t=0;t<k;++t) av[(size_t(i)*k+t)/2]|=(a_sign(i,t)<0?10:2)<<((t%2)*4);
        for (int h=0;h<(k+31)/32;++h) as[size_t(i)*2*p+h]=127+a_exp(i,h);
    }
    for (int e=0;e<g;++e) for (int j=0;j<n;++j) {
        size_t row=size_t(e)*n+j;
        for (int t=0;t<k;++t) bv[(row*k+t)/2]|=(w_sign(e,j,t)<0?10:2)<<((t%2)*4);
        for (int h=0;h<(k+31)/32;++h) bs[row*2*p+h]=127+w_exp(e,j,h);
    }
    Tensor a({m,k},ACL_FLOAT4_E2M1,av),w({g,n,k},ACL_FLOAT4_E2M1,bv);
    Tensor sx({m,p,2},ACL_FLOAT8_E8M0,as),sw({g,n,p,2},ACL_FLOAT8_E8M0,bs);
    Tensor groups({g},ACL_INT64,hist),y({m,n},ACL_BF16,std::vector<uint8_t>(size_t(m)*n*2,0xff));
    view(w,{g,k,n},{int64_t(n)*k,1,k},{g,n,k},ACL_FLOAT4_E2M1);
    view(sw,{g,p,n,2},{int64_t(n)*p*2,2,2*p,1},{g,n,p,2},ACL_FLOAT8_E8M0);
    List xl(a),wl(w),sl(sw),ps(sx),ol(y);
    uint64_t ws=0; aclOpExecutor *ex=nullptr;
    CHECK(aclnnGroupedMatmulV5GetWorkspaceSize(xl.value,wl.value,nullptr,sl.value,nullptr,nullptr,
        nullptr,ps.value,groups.desc,nullptr,nullptr,nullptr,3,0,1,0,nullptr,ol.value,nullptr,nullptr,&ws,&ex));
    std::printf("api=aclnnGroupedMatmulV5 M=%d N=%d K=%d groups=",m,n,k);
    for (int i=0;i<g;++i) std::printf("%s%ld",i?",":"",long(counts[i]));
    std::printf(" group_list_type=1 split_item=3 storage=GNK dtype=MXFP4xMXFP4_to_BF16 scale=E8M0 workspace=%lu\n",ws);
    Buffer workspace(ws);
    CHECK(aclnnGroupedMatmulV5(workspace.ptr,ws,ex,stream)); CHECK(aclrtSynchronizeStream(stream));
    std::vector<uint16_t> out(size_t(m)*n);
    CHECK(aclrtMemcpy(out.data(),out.size()*2,y.buffer.ptr,out.size()*2,ACL_MEMCPY_DEVICE_TO_HOST));
    // Compute independent dots for each observed class, checking every output (including expert selection).
    int64_t bad=0; int start=0; std::vector<int> expected(6*14*10,INT32_MIN);
    for (int e=0;e<g;++e) {
        for (int i=start;i<start+counts[e];++i) for (int j=0;j<n;++j) {
            int& v=expected[(e%6)*140+(i%14)*10+j%10];
            if (v==INT32_MIN) {
                v=0;
                for (int t=0;t<k;++t) v+=a_sign(i,t)*w_sign(e,j,t)*(1<<(a_exp(i,t/32)+w_exp(e,j,t/32)));
            }
            if (out[size_t(i)*n+j]!=bf16(float(v))) {
                if (!bad) std::printf("mismatch row=%d col=%d e=%d got=%04x expected=%04x\n",i,j,e,out[size_t(i)*n+j],bf16(float(v)));
                ++bad;
            }
        }
        start+=counts[e];
    }
    std::printf("correctness count=%zu bad=%ld\n",out.size(),long(bad));
    if (bad) throw std::runtime_error("all-element golden failed");
}

int main(int argc,char**argv) {
    std::setvbuf(stdout,nullptr,_IONBF,0);
    if(argc!=4) return 2;
    try {
        std::vector<int64_t> groups; std::stringstream ss(argv[1]); std::string item;
        while (std::getline(ss,item,',')) {
            auto m=std::stol(item); if(m<0) throw std::runtime_error("negative count"); groups.push_back(m);
        }
        CHECK(aclInit(nullptr)); CHECK(aclrtSetDevice(0)); aclrtStream stream=nullptr;
        CHECK(aclrtCreateStream(&stream)); std::printf("runtime_soc=%s\n",aclrtGetSocName());
        run(groups,std::stoi(argv[2]),std::stoi(argv[3]),stream);
        CHECK(aclrtDestroyStream(stream)); CHECK(aclrtResetDevice(0)); CHECK(aclFinalize());
        std::puts("SMOKE_PASS"); return 0;
    } catch(const std::exception& e) { std::fprintf(stderr,"SMOKE_FAIL %s\n",e.what()); return 1; }
}
