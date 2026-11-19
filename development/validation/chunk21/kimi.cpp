// Public CANN 9.2.1 ChunkKdaFwd with the vLLM Kimi prefill tensor contract.
#define main prior_smoke_main
#include "../../../smoke/extension_kda.cpp"
#undef main
#include <memory>

struct IntArray {
    aclIntArray *array;
    explicit IntArray(const std::vector<int64_t>& entries)
        : array(aclCreateIntArray(entries.data(), entries.size())) {
        if (!array) throw std::runtime_error("aclCreateIntArray failed");
    }
    ~IntArray() { aclDestroyIntArray(array); }
};

void prefill(aclrtStream stream, int tokens, int heads) {
    constexpr int dim = 128, chunk = 64;
    // A nonconstant gate and nonsymmetric state expose VK/KV layout mistakes.
    constexpr double lower = -0.03125;
    const double scale = 1 / std::sqrt(double(dim));
    const int elements = tokens * heads * dim;
    std::vector<uint16_t> q(elements), k(elements), v(elements);
    std::vector<float> raw(elements), beta(tokens*heads), alog(heads), bias(heads*dim);
    std::vector<float> state(heads*dim*dim);
    std::vector<double> out(elements), final(state.size());
    std::vector<double> aqk(heads*tokens*chunk), akk(aqk.size());
    for (int h = 0; h < heads; ++h) {
        Inputs input(tokens, dim, dim);
        alog[h] = 0.125f * h;
        for (int d = 0; d < dim; ++d) bias[h*dim+d] = ((h+d)%7-3)/32.f;
        for (int t = 0; t < tokens; ++t) {
            beta[t*heads+h] = input.beta[t];
            for (int d = 0; d < dim; ++d) {
                int i = (t*heads+h)*dim+d;
                raw[i] = ((t+3*d+h)%11-5)/8.f;
                double x = raw[i]+bias[h*dim+d];
                input.gate[t*dim+d] = lower / (1+std::exp(-std::exp(double(alog[h]))*x));
                q[i] = input.q[t*dim+d];
                k[i] = input.key[t*dim+d];
                v[i] = input.value[t*dim+d];
            }
        }
        auto gold = recurrent_gold(input);
        for (int t = 0; t < tokens; ++t)
            std::copy_n(gold.first.begin()+t*dim, dim, out.begin()+(t*heads+h)*dim);
        for (int d = 0; d < dim; ++d)
            for (int e = 0; e < dim; ++e) {
                state[(h*dim+e)*dim+d] = input.state[d*dim+e];
                final[(h*dim+e)*dim+d] = gold.second[d*dim+e];
            }
        for (int start = 0; start < tokens; start += chunk) {
            int n = std::min(chunk, tokens-start);
            std::vector<double> prefix(n*dim), triangular(n*n);
            for (int i = 0; i < n; ++i) {
                for (int d = 0; d < dim; ++d)
                    prefix[i*dim+d] = input.gate[(start+i)*dim+d]+(i ? prefix[(i-1)*dim+d] : 0);
                int row = (h*tokens+start+i)*chunk;
                for (int j = 0; j <= i; ++j) {
                    double qk = 0, kk = 0;
                    for (int d = 0; d < dim; ++d) {
                        double key = fp32(input.key[(start+j)*dim+d])*std::exp(prefix[i*dim+d]-prefix[j*dim+d]);
                        qk += fp32(input.q[(start+i)*dim+d])*key;
                        kk += fp32(input.key[(start+i)*dim+d])*key;
                    }
                    aqk[row+j] = scale*qk;
                    if (j < i) triangular[i*n+j] = input.beta[start+i]*kk;
                }
                for (int j = 0; j <= i; ++j) {
                    double x = i == j ? 1.0 : 0.0;
                    for (int z = j; z < i; ++z) x -= triangular[i*n+z]*akk[(h*tokens+start+z)*chunk+j];
                    akk[row+j] = x;
                }
            }
        }
    }
    Tensor<uint16_t> Q({1,tokens,heads,dim},ACL_BF16,q), K({1,tokens,heads,dim},ACL_BF16,k), V({1,tokens,heads,dim},ACL_BF16,v);
    Tensor<float> G({1,tokens,heads,dim},ACL_FLOAT,raw), B({1,tokens,heads},ACL_FLOAT,beta), A({heads},ACL_FLOAT,alog);
    Tensor<float> D({heads*dim},ACL_FLOAT,bias), I({1,heads,dim,dim},ACL_FLOAT,state);
    Tensor<uint16_t> O({1,tokens,heads,dim},ACL_BF16,std::vector<uint16_t>(elements,0x7fc0));
    Tensor<uint16_t> QK({1,heads,tokens,chunk},ACL_BF16,std::vector<uint16_t>(aqk.size(),0x7fc0)), KK({1,heads,tokens,chunk},ACL_BF16,std::vector<uint16_t>(akk.size(),0x7fc0));
    Tensor<float> F({1,heads,dim,dim},ACL_FLOAT,std::vector<float>(state.size(),NAN));
    std::vector<int64_t> indices;
    for (int start = 0; start < tokens; start += chunk) {
        indices.push_back(0);
        indices.push_back(start/chunk);
    }
    IntArray lengths({0,tokens}), chunks(indices);
    uint64_t bytes = 0;
    aclOpExecutor* executor = nullptr;
    ACL_CHECK(aclnnChunkKdaFwdGetWorkspaceSize(Q.tensor,K.tensor,V.tensor,G.tensor,B.tensor,A.tensor,D.tensor,I.tensor,
        lengths.array,chunks.array,"BSND",scale,chunk,true,lower,true,true,O.tensor,F.tensor,nullptr,QK.tensor,KK.tensor,
        nullptr,nullptr,nullptr,nullptr,nullptr,nullptr,&bytes,&executor));
    Buffer workspace(bytes);
    std::printf("api=aclnnChunkKdaFwd tokens=%d heads=%d workspace_bytes=%llu\n",tokens,heads,(unsigned long long)bytes);
    std::fflush(stdout);
    ACL_CHECK(aclnnChunkKdaFwd(workspace.ptr,bytes,executor,stream));
    ACL_CHECK(aclrtSynchronizeStream(stream));
    compare(decode(O.read()),out,true,"attention");
    compare(decode(QK.read()),aqk,true,"aqk");
    compare(decode(KK.read()),akk,true,"akk");
    auto f = F.read(), initial = I.read();
    compare({f.begin(),f.end()},final,false,"state_vk");
    if (initial != state) throw std::runtime_error("input VK state modified");
    std::puts("input_vk_unchanged=1");
}

int main(int argc, char** argv) {
    if (argc != 3) return 2;
    try {
        int tokens=std::stoi(argv[1]), heads=std::stoi(argv[2]);
        if (tokens<1 || tokens>1024 || heads<1 || heads>16) return 2;
        ACL_CHECK(aclInit(nullptr));
        ACL_CHECK(aclrtSetDevice(0));
        aclrtStream stream;
        ACL_CHECK(aclrtCreateStream(&stream));
        std::printf("runtime_soc=%s\n",aclrtGetSocName());
        prefill(stream,tokens,heads);
        ACL_CHECK(aclrtDestroyStream(stream));
        ACL_CHECK(aclrtResetDevice(0));
        ACL_CHECK(aclFinalize());
        std::puts("SMOKE_PASS");
        return 0;
    } catch (const std::exception& e) { std::fprintf(stderr,"SMOKE_FAIL %s\n",e.what()); return 1; }
}
