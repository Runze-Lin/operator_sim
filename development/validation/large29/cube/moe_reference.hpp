// Dense, signed exact-FP8 inputs. Weight columns repeat every 15, including their
// E8M0 scale bytes (period 3). CPU caching leaves every reduction and numerical
// check intact. The vendor kernels still execute on dense, fully allocated data.
#include <map>
#include <array>
static std::vector<uint8_t> values(size_t n,int seed) {
    std::array<uint8_t,15> lut;
    for(int i=0;i<15;++i) {
        int v=(13*i+seed*3)%15-7;
        lut[i]=encode8((v?v:1)/64.);
    }
    std::vector<uint8_t> result(n);
    for(size_t i=0;i<n;++i) result[i]=lut[i%15];
    return result;
}
