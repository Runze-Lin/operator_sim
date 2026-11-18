// Explicit simulation-only SKU selection. The stock msopprof injection library
// interposes these same runtime/HAL entrypoints; its profiler server cannot be
// used standalone with npusim. Always preserve underlying API failures.
#include <atomic>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <dlfcn.h>

static int32_t apply(int32_t status,char* buffer,uint32_t size) {
    if(status!=0)return status;
    const char* target=std::getenv("OPERATOR_SIM_TARGET_SOC");
    const char* model=std::getenv("soc_version");
    if(!target || std::strcmp(target,"Ascend950PR_957b") || !model || std::strcmp(model,"Ascend950PR"))return status;
    if(!buffer || size<=std::strlen(target))return 1;
    std::memcpy(buffer,target,std::strlen(target)+1);
    static std::atomic<bool> reported{false};
    if(!reported.exchange(true))std::fprintf(stderr,"SIM_SKU_ADAPTER target=%s physical_model=%s\n",target,model);
    return status;
}
extern "C" int32_t rtGetSocVersion(char* buffer,uint32_t size) {
    using Fn=int32_t(*)(char*,uint32_t);
    static Fn original=reinterpret_cast<Fn>(dlsym(RTLD_NEXT,"rtGetSocVersion"));
    if(!original)return 1;
    return apply(original(buffer,size),buffer,size);
}
extern "C" int32_t halGetSocVersion(uint32_t device,char* buffer,uint32_t size) {
    using Fn=int32_t(*)(uint32_t,char*,uint32_t);
    static Fn original=reinterpret_cast<Fn>(dlsym(RTLD_NEXT,"halGetSocVersion"));
    if(!original)return 1;
    return apply(original(device,buffer,size),buffer,size);
}
