#include <aclnnop/aclnn_moe_gating_top_k.h>
#include <acl/acl.h>
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <stdexcept>
#include <string>
#include <vector>

static void check(int code, const char* operation) {
    if (code) throw std::runtime_error(std::string(operation) + ": " + std::to_string(code));
}
#define ACL_CHECK(expr) check((expr), #expr)

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

static void run(aclrtStream stream, int rows, int experts, int k) {
    std::vector<float> logits(rows * experts), bias(experts);
    for (int e = 0; e < experts; ++e) bias[e] = 0.1f * (e % 2);
    for (int i = 0; i < rows * experts; ++i)
        logits[i] = float((i % experts + 7 * (i / experts)) % experts) / experts;
    Tensor<float> x({rows, experts}, ACL_FLOAT, logits), b({experts}, ACL_FLOAT, bias);
    Tensor<float> y({rows,k}, ACL_FLOAT, std::vector<float>(rows*k, NAN));
    Tensor<int32_t> ids({rows,k}, ACL_INT32, std::vector<int32_t>(rows*k,-1));
    Tensor<float> aux({rows,experts}, ACL_FLOAT, std::vector<float>(rows*experts));
    uint64_t bytes = 0;
    aclOpExecutor* exec = nullptr;
    ACL_CHECK(aclnnMoeGatingTopKGetWorkspaceSize(x.tensor,b.tensor,k,1,1,1,0,1,false,1.0,1e-20,
                                                 y.tensor,ids.tensor,aux.tensor,&bytes,&exec));
    Buffer work(bytes);
    std::printf("case=moe_gating_top_k api=aclnnMoeGatingTopK T=%d E=%d K=%d dtype=FP32 norm_type=1 renorm=0 bias=true\n",rows,experts,k);
    ACL_CHECK(aclnnMoeGatingTopK(work.ptr,bytes,exec,stream));
    ACL_CHECK(aclrtSynchronizeStream(stream));
    auto weights = y.read();
    auto indices = ids.read();
    int bad_idx = 0, bad_weight = 0;
    for (int r=0;r<rows;++r) {
        std::vector<int> rank(experts);
        std::vector<double> score(experts);
        for (int e=0;e<experts;++e) {
            rank[e]=e;
            score[e]=1/(1+std::exp(-double(logits[r*experts+e])));
        }
        std::partial_sort(rank.begin(), rank.begin()+k, rank.end(), [&](int a, int b) {
            return score[a]+bias[a] > score[b]+bias[b];
        });
        double sum=0;
        for (int j=0;j<k;++j) sum += score[rank[j]];
        for (int j=0;j<k;++j) {
            int i=r*k+j;
            bad_idx += indices[i]!=rank[j];
            bad_weight += !(std::abs(weights[i]-score[rank[j]]/sum)<1e-5);
        }
    }
    std::printf("correctness ids count=%d bad=%d\ncorrectness weights count=%d bad=%d\n",rows*k,bad_idx,rows*k,bad_weight);
    if(bad_idx||bad_weight) throw std::runtime_error("Routing result mismatch");
}

int main(int argc,char** argv) {
    if (argc!=4) return 2;
    aclrtStream stream=nullptr;
    try {
        const int r=std::stoi(argv[1]),e=std::stoi(argv[2]),k=std::stoi(argv[3]);
        if (r<1||r>4096||e<2||e>2048||k<1||k>e) throw std::runtime_error("Invalid shape");
        ACL_CHECK(aclInit(nullptr)); ACL_CHECK(aclrtSetDevice(0)); ACL_CHECK(aclrtCreateStream(&stream));
        std::printf("runtime_soc=%s\n",aclrtGetSocName());
        run(stream,r,e,k);
        ACL_CHECK(aclrtDestroyStream(stream)); ACL_CHECK(aclrtResetDevice(0)); ACL_CHECK(aclFinalize());
        std::puts("SMOKE_PASS");
    } catch(const std::exception& e) { std::fprintf(stderr,"SMOKE_FAIL %s\n",e.what());return 1; }
    return 0;
}
