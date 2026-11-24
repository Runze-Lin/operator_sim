#include "../acl_tensor.hpp"
#include <aclnnop/aclnn_causal_conv1d_update.h>
#include <aclnnop/aclnn_fused_infer_attention_score_v5.h>
#include <aclnnop/aclnn_lightning_indexer.h>
#include <aclnnop/aclnn_quant_lightning_indexer_v2.h>
#include <aclnnop/aclnn_quant_lightning_indexer_v2_metadata.h>
#include <algorithm>
#include <limits>
#include <cmath>
#include <cstdio>
#include <cstring>

template<class T> static std::vector<uint8_t> bytes(const std::vector<T>& v) {
    std::vector<uint8_t> data(v.size()*sizeof(T));
    std::memcpy(data.data(),v.data(),data.size());
    return data;
}
static uint16_t bf(float x) { uint32_t bits; std::memcpy(&bits,&x,4); return (bits+0x7fff+((bits>>16)&1))>>16; }
template<class T> static std::vector<T> read(Tensor& t,size_t n) {
    std::vector<T> v(n);
    CHECK(aclrtMemcpy(v.data(),n*sizeof(T),t.buffer.ptr,n*sizeof(T),ACL_MEMCPY_DEVICE_TO_HOST)); return v;
}

static void conv(aclrtStream s, int d) {
    constexpr int t=9,k=4,w=11;
    std::vector<uint16_t> input(t*d),weights(k*d,bf(.25)),state(w*d),out(t*d,bf(-7));
    for(int j=0;j<t*d;j++) input[j]=bf(1+(j%d)%2);
    for(int j=0;j<w*d;j++) state[j]=input[j%d];
    Tensor x({t,d},ACL_BF16,bytes(input)), kernel({k,d},ACL_BF16,bytes(weights));
    Tensor cache({1,w,d},ACL_BF16,bytes(state)),y({t,d},ACL_BF16,bytes(out));
    Tensor starts({2},ACL_INT32,bytes(std::vector<int32_t>{0,t})), indices({1},ACL_INT32,bytes(std::vector<int32_t>{0}));
    Tensor accepted({1},ACL_INT32,bytes(std::vector<int32_t>{1}));
    uint64_t space=0;aclOpExecutor* exec=nullptr;
    CHECK(aclnnCausalConv1dUpdateGetWorkspaceSize(x.desc,kernel.desc,cache.desc,nullptr,starts.desc,indices.desc,
           accepted.desc,nullptr,nullptr,"silu",-1,t,y.desc,&space,&exec));
    Buffer ws(space);
    CHECK(aclnnCausalConv1dUpdate(ws.ptr,space,exec,s)); CHECK(aclrtSynchronizeStream(s));
    out=read<uint16_t>(y,t*d);
    for(int j=0;j<t*d;j++){float a=1+j%2; if(std::abs(int(out[j])-int(bf(a/(1+std::exp(-a)))))>1) throw std::runtime_error("conv Y mismatch");}
    if(read<uint16_t>(cache,w*d)!=state) throw std::runtime_error("conv state mismatch");
}
static void fia(aclrtStream s, int64_t length) {
    constexpr int64_t t=9,h=12,g=2,d=64;
    std::vector<uint16_t> v(length*g*d);
    for(int j=0;j<length*g*d;j++) v[j]=bf(1+(j/d)%g + (j%d)%2);
    Tensor q({t,h,d},ACL_BF16,std::vector<uint8_t>(2*t*h*d)), k({length,g,d},ACL_BF16,std::vector<uint8_t>(length*g*d*2));
    Tensor value({length,g,d},ACL_BF16,bytes(v)), y({t,h,d},ACL_BF16,bytes(std::vector<uint16_t>(t*h*d,bf(-7))));
    Tensor lse({t,h,1},ACL_FLOAT,bytes(std::vector<float>(t*h,-7)));
    aclTensor* ks[]={k.desc}; aclTensor* vs[]={value.desc};
    aclTensorList* kl=aclCreateTensorList(ks,1);aclTensorList* vl=aclCreateTensorList(vs,1);
    auto ql=aclCreateIntArray(&t,1);auto kvl=aclCreateIntArray(&length,1);
    uint64_t space=0;aclOpExecutor* exec=nullptr;char layout[]="TND";
    CHECK(aclnnFusedInferAttentionScoreV5GetWorkspaceSize(q.desc,kl,vl,nullptr,nullptr,ql,kvl,
        nullptr,nullptr,nullptr,nullptr,nullptr,nullptr,nullptr,nullptr,nullptr,nullptr,nullptr,nullptr,nullptr,nullptr,
        nullptr,nullptr,nullptr,nullptr,nullptr,nullptr,nullptr,nullptr,nullptr,nullptr,
        h,0.125,2147483647,2147483647,layout,g,0,0,0,0,true,0,0,0,0,y.desc,lse.desc,&space,&exec));
    Buffer ws(space);
    CHECK(aclnnFusedInferAttentionScoreV5(ws.ptr,space,exec,s));CHECK(aclrtSynchronizeStream(s));
    auto out=read<uint16_t>(y,t*h*d);
    for(int j=0;j<t*h*d;j++) if(out[j]!=bf(1+(j/d%h)/(h/g)+(j%d)%2)) throw std::runtime_error("GQA output mismatch");
    for(auto x:read<float>(lse,t*h)) if(!std::isfinite(x)||std::abs(x-std::log(double(length)))>.01) throw std::runtime_error("LSE mismatch");
    aclDestroyIntArray(ql);aclDestroyIntArray(kvl);
    // Tensor lists own their constituent descriptors.
    aclDestroyTensorList(kl);aclDestroyTensorList(vl); k.desc=nullptr;value.desc=nullptr;
}
static void indexer(aclrtStream stream,int n) {
    constexpr int t=9,h=64,d=128,top=1024;
    std::vector<uint16_t> key(n*d),query(t*h*d),weight(t*h,bf(1));
    for(int i=0;i<t*h;++i) query[i*d]=bf(1);
    for(int j=0;j<n;++j) key[j*d]=bf(j<top ? 1 : -1);
    Tensor q({1,t,h,d},ACL_BF16,bytes(query)),k({1,n,1,d},ACL_BF16,bytes(key));
    Tensor w({1,t,h},ACL_BF16,bytes(weight)),out({1,t,1,top},ACL_INT32,std::vector<uint8_t>(4*t*top,0xfe));
    char layout[]="BSND";
    auto inf=std::numeric_limits<int64_t>::max();
    aclOpExecutor* ex=nullptr;uint64_t ws=0;
    CHECK(aclnnLightningIndexerGetWorkspaceSize(q.desc,k.desc,w.desc,nullptr,nullptr,nullptr,layout,layout,top,3,
                                              inf,inf,false,out.desc,nullptr,&ws,&ex));
    Buffer tmp(ws);CHECK(aclnnLightningIndexer(tmp.ptr,ws,ex,stream));CHECK(aclrtSynchronizeStream(stream));
    auto indices=read<int32_t>(out,t*top);
    for(int i=0;i<t;++i){
        auto first=indices.begin()+i*top;std::sort(first,first+top);
        for(int j=0;j<top;++j) if(first[j]!=j)throw std::runtime_error("Bad top-1024 selected IDs");
    }
}
static void mxindexer(aclrtStream s,int n,bool static_plan) {
    constexpr int qlen=1,h=32,d=128,top=2048;
    // Installed 9.2.1 wrapper reports a fixed 1024 words on both 950 SKUs.
    const int words=1024;
    // Runtime needs an explicit valid key length when preparing split metadata.
    Tensor usedq({1},ACL_INT32,bytes(std::vector<int32_t>{qlen}));
    Tensor usedk({1},ACL_INT32,bytes(std::vector<int32_t>{n}));
    // Diagnostic alternative: one AIC covers [b=0,b=1); no cross-core split-K merge.
    // Official 9.2.1 quant_lightning_indexer_v2_metadata.h indexes enable=0, BN2_END=4.
    std::vector<int32_t> table(words);
    if(static_plan){table[0]=1; table[4]=1;}
    Tensor md({words},ACL_INT32,bytes(table));
    char layout[]="BSND";uint64_t ws=0;aclOpExecutor* ex=nullptr;
    if(!static_plan){
    CHECK(aclnnQuantLightningIndexerV2MetadataGetWorkspaceSize(nullptr,nullptr,usedq.desc,usedk.desc,nullptr,h,1,d,
                               top,5,1,qlen,n,layout,layout,3,1,md.desc,&ws,&ex));
    Buffer mdws(ws);CHECK(aclnnQuantLightningIndexerV2Metadata(mdws.ptr,ws,ex,s));CHECK(aclrtSynchronizeStream(s));
    const auto fields=read<int32_t>(md,words);
    if(std::all_of(fields.begin(),fields.end(),[](int32_t i){return i==0;}))throw std::runtime_error("empty metadata");
    }
    std::vector<uint8_t> keys(n*d/2,0xaa);
    std::fill(keys.begin(),keys.begin()+top*d/2,0x22);
    Tensor query({1,qlen,h,d},ACL_FLOAT4_E2M1,std::vector<uint8_t>(h*d/2,0x22));
    Tensor key({1,n,1,d},ACL_FLOAT4_E2M1,keys);
    Tensor qs({1,qlen,h,2,2},ACL_FLOAT8_E8M0,std::vector<uint8_t>(h*4,127));
    Tensor ks({1,n,1,2,2},ACL_FLOAT8_E8M0,std::vector<uint8_t>(n*4,127));
    Tensor w({1,qlen,h},ACL_FLOAT,bytes(std::vector<float>(h,1))),out({1,qlen,1,top},ACL_INT32,std::vector<uint8_t>(4*top,0xfc));
    Tensor no_values({0},ACL_BF16,std::vector<uint8_t>(2));
    aclOpExecutor* kernel=nullptr; uint64_t work=0;
    CHECK(aclnnQuantLightningIndexerV2GetWorkspaceSize(query.desc,key.desc,w.desc,qs.desc,ks.desc,
          nullptr,nullptr,usedq.desc,usedk.desc,nullptr,nullptr,nullptr,md.desc,top,5,-1,layout,layout,3,1,0,
          out.desc,no_values.desc,&work,&kernel));
    Buffer workbuf(work);CHECK(aclnnQuantLightningIndexerV2(workbuf.ptr,work,kernel,s));CHECK(aclrtSynchronizeStream(s));
    auto indices=read<int32_t>(out,top);std::sort(indices.begin(),indices.end());
    for(int j=0;j<top;j++) if(indices[j]!=j)throw std::runtime_error("Bad MX top-k IDs");
}
int main(int argc,char** argv){
    setvbuf(stdout,nullptr,_IONBF,0);
    try{
        CHECK(aclInit(nullptr));CHECK(aclrtSetDevice(0));aclrtStream s;CHECK(aclrtCreateStream(&s));
        std::printf("runtime_soc=%s\n",aclrtGetSocName());
        if(argc==3 && std::string(argv[1])=="conv") conv(s,std::stoi(argv[2]));
        else if(argc==3 && std::string(argv[1])=="mxindexer") mxindexer(s,std::stoi(argv[2]),false);
        else if(argc==3 && std::string(argv[1])=="mxsingle") mxindexer(s,std::stoi(argv[2]),true);
        else if(argc==3 && std::string(argv[1])=="indexer") indexer(s,std::stoi(argv[2]));
        else if(argc==3 && std::string(argv[1])=="fia") fia(s,std::stoi(argv[2]));
        else return 2;
        CHECK(aclrtDestroyStream(s));CHECK(aclrtResetDevice(0));CHECK(aclFinalize());
        puts("GOLDEN_PASS");return 0;
    }catch(const std::exception& e){fprintf(stderr,"FAIL %s\n",e.what());return 1;}
}
