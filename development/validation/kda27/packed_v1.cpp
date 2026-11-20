// Host-only caller of the pinned FLA ABI. Device implementations come from the release.
#define main archived_smoke_main
#include "../../../smoke/extension_kda.cpp"
#undef main
#include "../kda_v23/precise_host.hpp"
#include <dlfcn.h>

template <typename T>
std::vector<T> read_file(const std::string& path, size_t n) {
    std::ifstream file(path, std::ios::binary);
    std::vector<T> data(n);
    file.read(reinterpret_cast<char*>(data.data()), n * sizeof(T));
    if (!file || file.peek() != EOF) throw std::runtime_error("wrong fixture size " + path);
    return data;
}

void run(aclrtStream stream, const std::string& dir) {
    constexpr size_t dim = 128, rows = 2, state_size = rows * dim * dim;
    const std::vector<int64_t> q_shape = {1, rows, 1, dim};
    const std::vector<int64_t> vk_shape = {rows, 1, dim, dim};
    const std::vector<int64_t> a_shape = {1, 1, rows, 64};
    Tensor<uint16_t> q(q_shape, ACL_BF16, read_file<uint16_t>(dir+"q.bf16", rows*dim));
    Tensor<uint16_t> k(q_shape, ACL_BF16, read_file<uint16_t>(dir+"k.bf16", rows*dim));
    Tensor<uint16_t> v(q_shape, ACL_BF16, read_file<uint16_t>(dir+"v.bf16", rows*dim));
    Tensor<uint16_t> g(q_shape, ACL_BF16, read_file<uint16_t>(dir+"gate.bf16", rows*dim));
    Tensor<float> beta({1,rows,1}, ACL_FLOAT, read_file<float>(dir+"beta.f32", rows));
    Tensor<float> alog({1}, ACL_FLOAT, {0.f});
    Tensor<float> bias({dim}, ACL_FLOAT, read_file<float>(dir+"dt_bias.f32",dim));
    auto initial = read_file<float>(dir+"initial.f32",state_size);
    Tensor<float> start(vk_shape, ACL_FLOAT, initial);
    Tensor<uint16_t> out(q_shape, ACL_BF16, std::vector<uint16_t>(rows*dim,0x7fc0));
    Tensor<uint16_t> aq(a_shape, ACL_BF16, std::vector<uint16_t>(rows*64,0x7fc0));
    Tensor<uint16_t> ak(a_shape, ACL_BF16, std::vector<uint16_t>(rows*64,0x7fc0));
    Tensor<float> state(vk_shape, ACL_FLOAT, std::vector<float>(state_size,NAN));
    const int64_t cu[] = {0,1,2}, indices[] = {0,0,1,0};
    auto* offsets = aclCreateIntArray(cu, 3);
    auto* chunks = aclCreateIntArray(indices, 4);
    if (!offsets || !chunks) throw std::runtime_error("metadata allocation failed");
    uint64_t bytes = 0;
    aclOpExecutor* executor = nullptr;
    ACL_CHECK(aclnnChunkKdaFwdGetWorkspaceSize(q.tensor,k.tensor,v.tensor,g.tensor,beta.tensor,alog.tensor,
        bias.tensor,start.tensor,offsets,chunks,"BSND",1./std::sqrt(double(dim)),64,true,-5.,true,true,
        out.tensor,state.tensor,nullptr,aq.tensor,ak.tensor,nullptr,nullptr,nullptr,nullptr,nullptr,nullptr,&bytes,&executor));
    Buffer workspace(bytes);
    std::printf("workspace_bytes=%llu cu=[0,1,2] chunk_indices=[0,0,1,0]\n",(unsigned long long)bytes);
    ACL_CHECK(aclnnChunkKdaFwd(workspace.ptr,bytes,executor,stream));
    ACL_CHECK(aclrtSynchronizeStream(stream));
    bool ok = true;
    auto check_array = [&](const char* name, const std::vector<double>& actual, bool bf) {
        try { compare(actual,readPreparedGolden(dir+name+".bin",actual.size()),bf,name); }
        catch (const std::runtime_error&) { ok=false; }
    };
    for (auto item : {std::make_pair("attention",&out),std::make_pair("aqk",&aq),std::make_pair("akk",&ak)}) {
        auto value = item.second->read(); writeActual(item.first,value);
        check_array(item.first,decode(value),true);
    }
    auto final = state.read();
    std::ofstream file("state_vk.f32",std::ios::binary);
    file.write(reinterpret_cast<const char*>(final.data()), final.size()*sizeof(float));
    if (!file) throw std::runtime_error("cannot archive state");
    check_array("state_vk",{final.begin(),final.end()},false);
    if (start.read()!=initial) throw std::runtime_error("initial VK changed");
    std::puts("input_vk_unchanged=1");
    aclDestroyIntArray(offsets); aclDestroyIntArray(chunks);
    if (!ok) throw std::runtime_error("four-output gate failed");
}
int main(int argc, char** argv) {
    if (argc!=2) return 2;
    std::setvbuf(stdout,nullptr,_IOLBF,0);
    try {
        for (auto name : {"aclnnChunkKdaFwd","aclnnChunkKdaFwdGetWorkspaceSize"}) {
            Dl_info info{}; const char* required=std::getenv("FLA_NPU_OP_API_LIB");
            if (!required || !dladdr(dlsym(RTLD_DEFAULT,name),&info) || std::string(info.dli_fname)!=required)
                throw std::runtime_error("unverified FLA binding");
            std::printf("bound_v1_api=%s library=%s\n",name,info.dli_fname);
        }
        ACL_CHECK(aclInit(nullptr)); ACL_CHECK(aclrtSetDevice(0));
        aclrtStream stream; ACL_CHECK(aclrtCreateStream(&stream));
        std::printf("runtime_soc=%s\n",aclrtGetSocName()); run(stream,std::string(argv[1])+"/");
        ACL_CHECK(aclrtDestroyStream(stream)); ACL_CHECK(aclrtResetDevice(0)); ACL_CHECK(aclFinalize());
        std::puts("SMOKE_PASS"); return 0;
    } catch (const std::exception& e) { std::fprintf(stderr,"SMOKE_FAIL %s\n",e.what()); return 1; }
}
