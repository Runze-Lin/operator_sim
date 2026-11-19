// Used after stream synchronization by a test executable. No device-side code.
#include <cstdio>
#include <fstream>
#include <string>

static std::vector<double> readPreparedGolden(const std::string& path, size_t count) {
    std::ifstream input(path, std::ios::binary);
    std::vector<double> values(count);
    input.read(reinterpret_cast<char*>(values.data()), values.size()*sizeof(double));
    if (!input || input.peek()!=EOF) throw std::runtime_error("invalid prepared golden "+path);
    return values;
}

static void writeActual(const char *name, const std::vector<uint16_t>& values) {
    std::ofstream out(std::string(name)+".bf16", std::ios::binary);
    out.write(reinterpret_cast<const char*>(values.data()), values.size()*sizeof(uint16_t));
    if (!out) throw std::runtime_error("cannot preserve actual auxiliary matrix");
}

static void auxiliaryDiagnostic(const char* name,const char* reference,const std::vector<double>& actual,const std::vector<double>& truth,
                                int tokens) {
    size_t bad=0,index=0,relativeIndex=0;double worst=0,relative=0;bool finite=true;
    for(size_t i=0;i<actual.size();i++) {
        double err=std::abs(actual[i]-truth.at(i));
        if (!std::isfinite(err)) finite=false;
        if (err>2e-6+std::abs(truth[i])/64 || !std::isfinite(err)) ++bad;
        if(err>worst) {index=i;worst=err;}
        if(truth[i]!=0 && err/std::abs(truth[i])>relative) {relativeIndex=i;relative=err/std::abs(truth[i]);}
    }
    std::printf("auxiliary name=%s reference=%s count=%zu bad=%zu finite=%d max_abs=%.15g h=%zu t=%zu column=%zu got=%.15g expected=%.15g max_rel_nonzero=%.15g rel_index=%zu\n",
                name,reference,actual.size(),bad,finite,worst,index/(tokens*64),(index/64)%tokens,index%64,actual[index],truth[index],relative,relativeIndex);
}
