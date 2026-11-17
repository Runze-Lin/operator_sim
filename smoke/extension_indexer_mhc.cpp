#include <acl/acl.h>
#include <aclnnop/aclnn_mhc_pre.h>
#include <aclnnop/aclnn_mhc_pre_sinkhorn.h>
#include <aclnnop/aclnn_mhc_sinkhorn.h>
#include <aclnnop/aclnn_mhc_post.h>
#include <aclnnop/aclnn_lightning_indexer.h>
#include <limits>
#include <numeric>
#include <memory>
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <stdexcept>
#include <string>
#include <vector>

// Standalone callers of the installed CANN operators; no replacement kernels.
static void check(int code, const char* operation) {
    if (code) {
        const char* detail=aclGetRecentErrMsg();
        if(detail)std::fprintf(stderr,"ACL_RECENT_ERROR %s\n",detail);
        throw std::runtime_error(std::string(operation) + ": " + std::to_string(code));
    }
}
#define ACL_CHECK(expr) check((expr), #expr)

static uint16_t bf16(float x) {
    uint32_t bits;
    std::memcpy(&bits, &x, sizeof(bits));
    bits += 0x7fff + ((bits >> 16) & 1);
    return static_cast<uint16_t>(bits >> 16);
}
static float fp32(uint16_t x) {
    uint32_t bits = uint32_t(x) << 16;
    float result;
    std::memcpy(&result, &bits, sizeof(result));
    return result;
}

struct Buffer {
    void* ptr = nullptr;
    explicit Buffer(size_t bytes) {
        if (bytes) ACL_CHECK(aclrtMalloc(&ptr, bytes, ACL_MEM_MALLOC_HUGE_FIRST));
    }
    ~Buffer() { if (ptr) aclrtFree(ptr); }
    Buffer(const Buffer&) = delete;
    Buffer& operator=(const Buffer&) = delete;
};

template <class T> struct Tensor {
    Buffer memory;
    aclTensor* tensor = nullptr;
    size_t count;
    Tensor(std::vector<int64_t> shape, aclDataType dtype, const std::vector<T>& input)
        : memory(input.size() * sizeof(T)), count(input.size()) {
        int64_t elements = 1;
        for (auto d : shape) elements *= d;
        if (elements != int64_t(count)) throw std::runtime_error("Shape/data mismatch");
        std::vector<int64_t> strides(shape.size(), 1);
        for (int i = int(shape.size()) - 2; i >= 0; --i) strides[i] = strides[i+1] * shape[i+1];
        ACL_CHECK(aclrtMemcpy(memory.ptr, count*sizeof(T), input.data(), count*sizeof(T), ACL_MEMCPY_HOST_TO_DEVICE));
        tensor = aclCreateTensor(shape.data(), shape.size(), dtype, strides.data(), 0,
                                 ACL_FORMAT_ND, shape.data(), shape.size(), memory.ptr);
        if (!tensor) throw std::runtime_error("aclCreateTensor failed");
    }
    ~Tensor() { if (tensor) aclDestroyTensor(tensor); }
    std::vector<T> read() const {
        std::vector<T> out(count);
        ACL_CHECK(aclrtMemcpy(out.data(), count*sizeof(T), memory.ptr, count*sizeof(T), ACL_MEMCPY_DEVICE_TO_HOST));
        return out;
    }
};

static void compare(const char* name, const std::vector<double>& actual, const std::vector<double>& expected,
                    bool bfloat) {
    const double atol=bfloat?std::ldexp(1.,-6):std::ldexp(1.,-16);
    const double rtol=bfloat?std::ldexp(1.,-6):std::ldexp(1.,-10);
    const double limit=bfloat?1.:.01;
    size_t bad=0; double worst=0;
    for(size_t i=0;i<actual.size();++i) {
        double err=std::abs(actual[i]-expected.at(i));
        worst=std::max(worst,err);
        if(!std::isfinite(actual[i]) || err>atol+rtol*std::abs(expected[i])) ++bad;
    }
    std::printf("correctness name=%s count=%zu bad=%zu max_abs_error=%g atol=%g rtol=%g hard_limit=%g\n",
                name,actual.size(),bad,worst,atol,rtol,limit);
    // A small smoke checks every element, stricter than the skill's 99% matched ratio.
    if(bad || worst>limit) throw std::runtime_error("Correctness failed");
}
static std::vector<double> decode(const std::vector<uint16_t>& x) {
    std::vector<double> result; result.reserve(x.size());
    for(auto v:x) result.push_back(fp32(v));
    return result;
}
static std::vector<uint16_t> data(size_t count,int offset) {
    std::vector<uint16_t> x(count);
    for(size_t i=0;i<count;++i) x[i]=bf16(float(int((i+offset)%17)-8)/16.f);
    return x;
}
static std::vector<double> doubles(const std::vector<float>& v) {return {v.begin(),v.end()};}
static std::vector<double> sinkgold(std::vector<double> a,int t,int n,int iters) {
    constexpr double eps=1e-6;
    for(int b=0;b<t;++b) {
        double* z=a.data()+b*n*n;
        for(int i=0;i<n;++i) {
            double mx=*std::max_element(z+i*n,z+(i+1)*n),sum=0;
            for(int j=0;j<n;++j) {z[i*n+j]=std::exp(z[i*n+j]-mx);sum+=z[i*n+j];}
            for(int j=0;j<n;++j)z[i*n+j]=z[i*n+j]/sum+eps;
        }
        for(int step=1;step<2*iters;++step) {
            bool column=step%2;
            for(int aidx=0;aidx<n;++aidx) {
                double sum=eps;
                for(int j=0;j<n;++j)sum+=z[column?j*n+aidx:aidx*n+j];
                for(int j=0;j<n;++j)z[column?j*n+aidx:aidx*n+j]/=sum;
            }
        }
    }
    return a;
}
static void dosink(aclrtStream stream,Tensor<float>& x,Tensor<float>& y,int iters) {
    uint64_t bytes=0;aclOpExecutor* ex=nullptr;
    ACL_CHECK(aclnnMhcSinkhornGetWorkspaceSize(x.tensor,1e-6f,iters,y.tensor,nullptr,nullptr,&bytes,&ex));Buffer ws(bytes);
    ACL_CHECK(aclnnMhcSinkhorn(ws.ptr,bytes,ex,stream));ACL_CHECK(aclrtSynchronizeStream(stream));
}
static void dopost(aclrtStream stream,Tensor<uint16_t>& x,Tensor<float>& res,Tensor<uint16_t>& h,Tensor<float>& post,Tensor<uint16_t>& out) {
    uint64_t bytes=0;aclOpExecutor* ex=nullptr;
    ACL_CHECK(aclnnMhcPostGetWorkspaceSize(x.tensor,res.tensor,h.tensor,post.tensor,out.tensor,&bytes,&ex));Buffer ws(bytes);
    ACL_CHECK(aclnnMhcPost(ws.ptr,bytes,ex,stream));ACL_CHECK(aclrtSynchronizeStream(stream));
}
static std::vector<double> postgold(const std::vector<uint16_t>& x,const std::vector<double>& res,const std::vector<uint16_t>& h,const std::vector<double>& post,int t,int n,int d) {
    std::vector<double> out(t*n*d);
    for(int b=0;b<t;++b)for(int j=0;j<n;++j)for(int k=0;k<d;++k) {
        double val=fp32(h[b*d+k])*post[b*n+j];
        for(int i=0;i<n;++i)val+=res[(b*n+i)*n+j]*fp32(x[(b*n+i)*d+k]);
        out[(b*n+j)*d+k]=val;
    }
    return out;
}
static void pre(aclrtStream stream,int t,int n,int d,bool fused,bool chain) {
    const int p=n*n+2*n,k=n*d;
    auto xd=data(size_t(t)*k,3);std::vector<float> pd(size_t(p)*k),ad={.7f,1.1f,.9f},bd(p);
    // Exact binary fractions; within fused phi's +/-1/sqrt(n*d) requirement.
    for(int j=0;j<p;++j){bd[j]=float(j%11-5)/32;for(int i=0;i<k;++i)pd[j*k+i]=float((i*7+j*3)%17-8)/2048;}
    Tensor<uint16_t> x({t,n,d},ACL_BF16,xd),hin({t,d},ACL_BF16,std::vector<uint16_t>(t*d,0x7fc0));
    Tensor<float> phi({p,k},ACL_FLOAT,pd),alpha({3},ACL_FLOAT,ad),bias({p},ACL_FLOAT,bd);
    Tensor<float> post({t,n},ACL_FLOAT,std::vector<float>(t*n,NAN)),res(fused?std::vector<int64_t>{t,n*n}:std::vector<int64_t>{t,n,n},ACL_FLOAT,std::vector<float>(t*n*n,NAN));
    Tensor<float> inv(fused?std::vector<int64_t>{t,1}:std::vector<int64_t>{t},ACL_FLOAT,std::vector<float>(t,NAN)),mix({t,p},ACL_FLOAT,std::vector<float>(t*p,NAN)),hpre({t,n},ACL_FLOAT,std::vector<float>(t*n,NAN));
    // All eight public output arguments are required Tensor descriptors even when
    // needBackward=false makes the five auxiliary values semantically invalid.
    // Allocate placeholders without checking or using those invalid values.
    std::unique_ptr<Tensor<float>> sumAux, normAux;
    if(fused){
        sumAux=std::make_unique<Tensor<float>>(std::vector<int64_t>{40,t,n},ACL_FLOAT,std::vector<float>(40*t*n,NAN));
        normAux=std::make_unique<Tensor<float>>(std::vector<int64_t>{40,t,n,n},ACL_FLOAT,std::vector<float>(40*t*n*n,NAN));
    }
    uint64_t bytes=0;aclOpExecutor* ex=nullptr;
    if(fused)ACL_CHECK(aclnnMhcPreSinkhornGetWorkspaceSize(x.tensor,phi.tensor,alpha.tensor,bias.tensor,n,20,1e-6,1e-6,false,hin.tensor,post.tensor,res.tensor,hpre.tensor,mix.tensor,inv.tensor,sumAux->tensor,normAux->tensor,&bytes,&ex));
    else ACL_CHECK(aclnnMhcPreGetWorkspaceSize(x.tensor,phi.tensor,alpha.tensor,bias.tensor,nullptr,1e-6,1e-6,hin.tensor,post.tensor,res.tensor,inv.tensor,mix.tensor,hpre.tensor,&bytes,&ex));
    Buffer ws(bytes);
    std::printf("case=%s api=%s T=%d N=%d D=%d dtype=BF16 phi_dtype=FP32 num_iters=20 need_backward=false workspace=%llu\n",chain?"mhc":"mhc_pre",fused?"aclnnMhcPreSinkhorn":"aclnnMhcPre",t,n,d,(unsigned long long)bytes);
    if(fused)ACL_CHECK(aclnnMhcPreSinkhorn(ws.ptr,bytes,ex,stream));else ACL_CHECK(aclnnMhcPre(ws.ptr,bytes,ex,stream));
    ACL_CHECK(aclrtSynchronizeStream(stream));
    std::vector<double> gi(t),gm(t*p),gp(t*n),gpost(t*n),gr(t*n*n),gh(t*d);
    for(int b=0;b<t;++b){
        double sq=0;for(int i=0;i<k;++i){double v=fp32(xd[b*k+i]);sq+=v*v;}gi[b]=1/std::sqrt(sq/k+1e-6);
        for(int j=0;j<p;++j){double sum=0;for(int i=0;i<k;++i)sum+=double(fp32(xd[b*k+i]))*pd[j*k+i];gm[b*p+j]=sum;double v=sum*gi[b]*ad[j<n?0:(j<2*n?1:2)]+bd[j];if(j<n)gp[b*n+j]=1/(1+std::exp(-v))+1e-6;else if(j<2*n)gpost[b*n+j-n]=2/(1+std::exp(-v));else gr[b*n*n+j-2*n]=v;}
        for(int z=0;z<d;++z)for(int i=0;i<n;++i)gh[b*d+z]+=gp[b*n+i]*fp32(xd[(b*n+i)*d+z]);
    }
    if(fused)gr=sinkgold(gr,t,n,20);
    compare("hin",decode(hin.read()),gh,true);compare("hpost",doubles(post.read()),gpost,false);compare("hres",doubles(res.read()),gr,false);
    if(!fused){compare("inv_rms",doubles(inv.read()),gi,false);compare("hmix",doubles(mix.read()),gm,false);compare("hpre",doubles(hpre.read()),gp,false);}
    if(chain){
        Tensor<float> normalized({t,n,n},ACL_FLOAT,std::vector<float>(t*n*n,NAN));dosink(stream,res,normalized,20);
        auto sg=sinkgold(doubles(res.read()),t,n,20);compare("chain_sinkhorn",doubles(normalized.read()),sg,false);
        Tensor<uint16_t> out({t,n,d},ACL_BF16,std::vector<uint16_t>(t*n*d,0x7fc0));dopost(stream,x,normalized,hin,post,out);
        // Local-stage golden uses actual producer buffers, avoiding double-counting previous rounding.
        auto gold=postgold(xd,doubles(normalized.read()),hin.read(),doubles(post.read()),t,n,d);
        compare("chain_post",decode(out.read()),gold,true);
        // Additional end-to-end golden independently applies BF16 layer-boundary rounding.
        std::vector<uint16_t> rounded(gh.size());for(size_t z=0;z<gh.size();++z)rounded[z]=bf16(float(gh[z]));
        auto finalgold=postgold(xd,sinkgold(gr,t,n,20),rounded,gpost,t,n,d);
        compare("chain_end_to_end",decode(out.read()),finalgold,true);
    }
}
static void sink(aclrtStream stream,int t,int n,int iters) {
    std::vector<float> xd(t*n*n);for(size_t i=0;i<xd.size();++i)xd[i]=float(int((i*7+3)%29)-14)/8;
    Tensor<float> x({t,n,n},ACL_FLOAT,xd),out({t,n,n},ACL_FLOAT,std::vector<float>(xd.size(),NAN));
    std::printf("case=mhc_sinkhorn api=aclnnMhcSinkhorn T=%d N=%d iters=%d dtype=FP32\n",t,n,iters);
    dosink(stream,x,out,iters);compare("output",doubles(out.read()),sinkgold(doubles(xd),t,n,iters),false);
}
static void post(aclrtStream stream,int t,int n,int d) {
    auto xd=data(t*n*d,4),hd=data(t*d,7);std::vector<float> rd(t*n*n),pd(t*n);
    for(size_t i=0;i<rd.size();++i)rd[i]=float(int(i*3%13)-6)/16;
    for(size_t i=0;i<pd.size();++i)pd[i]=float(i%7+1)/8;
    Tensor<uint16_t> x({t,n,d},ACL_BF16,xd),h({t,d},ACL_BF16,hd),out({t,n,d},ACL_BF16,std::vector<uint16_t>(t*n*d,0x7fc0));
    Tensor<float> r({t,n,n},ACL_FLOAT,rd),p({t,n},ACL_FLOAT,pd);
    std::printf("case=mhc_post api=aclnnMhcPost T=%d N=%d D=%d dtype=BF16\n",t,n,d);
    dopost(stream,x,r,h,p,out);compare("output",decode(out.read()),postgold(xd,doubles(rd),hd,doubles(pd),t,n,d),true);
}
static void indexer(aclrtStream stream,int q,int l,int g) {
    constexpr int d=128,top=512;const int batch=1;
    std::vector<uint16_t> qd(q*g*d),kd(l*d),wd(q*g);
    for(int z=0;z<q;++z)for(int h=0;h<g;++h){wd[z*g+h]=bf16(float(h+1)/16);for(int j=0;j<d;++j)qd[(z*g+h)*d+j]=bf16((h%3==0?-1.f:1.f)*(j%2?float(h+1)/256:float(z+1)/32));}
    for(int z=0;z<l;++z)for(int j=0;j<d;++j)kd[z*d+j]=bf16(j%2?float(z%128)/1024:float(z/128+1)/8);
    Tensor<uint16_t> qt({batch,q,g,d},ACL_BF16,qd),kt({batch,l,1,d},ACL_BF16,kd),wt({batch,q,g},ACL_BF16,wd);
    Tensor<int32_t> out({batch,q,1,top},ACL_INT32,std::vector<int32_t>(q*top,-777));
    uint64_t bytes=0;aclOpExecutor* ex=nullptr;char layout[]="BSND";
    ACL_CHECK(aclnnLightningIndexerGetWorkspaceSize(qt.tensor,kt.tensor,wt.tensor,nullptr,nullptr,nullptr,layout,layout,top,3,std::numeric_limits<int64_t>::max(),std::numeric_limits<int64_t>::max(),false,out.tensor,nullptr,&bytes,&ex));
    Buffer ws(bytes);std::printf("case=lightning_indexer api=aclnnLightningIndexer B=1 Q=%d L=%d G=%d D=128 dtype=BF16 topk=512 sparse_mode=3 return_values=false workspace=%llu\n",q,l,g,(unsigned long long)bytes);
    ACL_CHECK(aclnnLightningIndexer(ws.ptr,bytes,ex,stream));ACL_CHECK(aclrtSynchronizeStream(stream));
    auto actual=out.read();size_t bad=0;
    for(int z=0;z<q;++z){
        int valid=l-q+z+1;std::vector<double> scores(valid);
        for(int j=0;j<valid;++j)for(int h=0;h<g;++h){double dot=0;for(int k=0;k<d;++k)dot+=double(fp32(qd[(z*g+h)*d+k]))*fp32(kd[j*d+k]);scores[j]+=std::max(0.,dot)*fp32(wd[z*g+h]);}
        std::vector<int> ids(valid);std::iota(ids.begin(),ids.end(),0);std::sort(ids.begin(),ids.end(),[&](int a,int b){return scores[a]>scores[b];});
        std::vector<int> expected(ids.begin(),ids.begin()+top),observed(actual.begin()+z*top,actual.begin()+(z+1)*top);
        std::sort(expected.begin(),expected.end());std::sort(observed.begin(),observed.end());for(int j=0;j<top;++j)bad+=expected[j]!=observed[j];
    }
    std::printf("correctness name=topk_set count=%zu bad=%zu exact_integer_set=true all_indices_checked=true\n",actual.size(),bad);
    if(bad)throw std::runtime_error("Indexer topk set mismatch");
}
static int dim(const char* arg) {size_t pos=0;long long n=std::stoll(arg,&pos);if(pos!=std::strlen(arg)||n<1||n>8192)throw std::runtime_error("Invalid dimension");return int(n);}
int main(int argc,char** argv) {
    aclrtStream stream=nullptr;bool initialized=false,device=false;int result=0;
    try{
        if(argc!=5)throw std::runtime_error("Usage: extension_indexer_mhc lightning_indexer Q L G | mhc_pre T N D | mhc_pre_sinkhorn T 4 4096 | mhc_sinkhorn T N iterations | mhc_post T N D | mhc T N D");
        std::string op=argv[1];int a=dim(argv[2]),b=dim(argv[3]),c=dim(argv[4]);
        if(int64_t(a)*b*c>8*1024*1024)throw std::runtime_error("Smoke size budget exceeded");
        if(op=="lightning_indexer"){if(b-a+1<512||c!=8)throw std::runtime_error("Smoke indexer requires >=512 visible keys and G8");}
        else if(b!=4)throw std::runtime_error("Smoke mHC requires N4");
        if(op=="mhc_pre_sinkhorn"&&c!=4096)throw std::runtime_error("Fused smoke requires D4096");
        ACL_CHECK(aclInit(nullptr));initialized=true;ACL_CHECK(aclrtSetDevice(0));device=true;
        std::printf("runtime_soc=%s\n",aclrtGetSocName());ACL_CHECK(aclrtCreateStream(&stream));
        if(op=="lightning_indexer")indexer(stream,a,b,c);
        else if(op=="mhc_pre")pre(stream,a,b,c,false,false);
        else if(op=="mhc_pre_sinkhorn")pre(stream,a,b,c,true,false);
        else if(op=="mhc")pre(stream,a,b,c,false,true);
        else if(op=="mhc_sinkhorn")sink(stream,a,b,c);
        else if(op=="mhc_post")post(stream,a,b,c);
        else throw std::runtime_error("Unknown operator");
        std::puts("SMOKE_PASS");
    }catch(const std::exception& e){std::fprintf(stderr,"SMOKE_FAIL %s\n",e.what());result=1;}
    if(stream)aclrtDestroyStream(stream);
    if(device)aclrtResetDevice(0);
    if(initialized)aclFinalize();
    return result;
}
