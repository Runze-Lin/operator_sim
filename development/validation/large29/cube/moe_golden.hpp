static std::vector<double> gmm(const std::vector<uint8_t>& x,const std::vector<uint8_t>& w,
 const std::vector<uint8_t>& xs,const std::vector<uint8_t>& ws,int m,int k,int n,const std::vector<int64_t>& counts) {
    std::vector<double> y(size_t(m)*n,NAN);
    int row=0,kg=(k+63)/64;
    for(size_t e=0;e<counts.size();++e) {
        std::map<std::vector<uint8_t>,std::array<double,15>> cache;
        for(int r=0;r<counts[e];++r,++row) {
            const auto xrow=x.begin()+size_t(row)*k;
            const auto srow=xs.begin()+size_t(row)*kg*2;
            std::vector<uint8_t> key(xrow,xrow+k);
            key.insert(key.end(),srow,srow+kg*2);
            auto found=cache.find(key);
            if(found==cache.end()) {
                std::array<double,15> sums{};
                for(int j=0;j<std::min(n,15);++j)
                    for(int q=0;q<k;++q)
                        sums[j]+=fp8(xrow[q])*fp8(w[(e*k+q)*n+j])*
                            s8(srow[q/32])*s8(ws[((e*kg+q/64)*n+j)*2+(q/32)%2]);
                found=cache.emplace(std::move(key),sums).first;
            }
            for(int j=0;j<n;++j) y[size_t(row)*n+j]=found->second[j%15];
        }
    }
    return y;
}
