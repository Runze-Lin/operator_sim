// Isolated compiler diagnostic. This is not the FLA operator.
#include "kernel_operator.h"
#if PIPE_EXPERIMENT >= 4
#include "catlass/arch/resource.hpp"
#endif

extern "C" __global__ __aicore__ void pipe_control() {
    AscendC::TPipe first;
#if PIPE_EXPERIMENT == 1
    // Same physical AIC, two simultaneously live constructors.
    AscendC::TPipe second;
#elif PIPE_EXPERIMENT == 2
    first.Destroy();
    AscendC::TPipe second;
#elif PIPE_EXPERIMENT == 3
    first.Reset();
#elif PIPE_EXPERIMENT == 4
    Catlass::Arch::Resource<Catlass::Arch::Ascend950> actual_wheel_resource;
#elif PIPE_EXPERIMENT == 5
    first.Destroy();
    Catlass::Arch::Resource<Catlass::Arch::Ascend950> actual_wheel_resource;
#endif
}
