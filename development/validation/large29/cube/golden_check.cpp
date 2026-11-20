// Verify the optimized reference against a full unrestricted reduction on CPU.
#define main large_moe_main
#include "../../../../runs/large29/cube/bin/moe.cpp"
#undef main
int main() {
    int m=35,k=128,n=192,e=2,kg=k/64;
    auto x=values(m*k,1),w=values(e*k*n,2),xs=scales(m*kg*2,0),ws=scales(e*kg*n*2,1);
    auto y=gmm(x,w,xs,ws,m,k,n,{17,18});
    for(int r=0;r<m;++r)for(int c=0;c<n;++c) {
        double sum=0;int expert=r>=17;
        for(int j=0;j<k;++j)sum+=fp8(x[r*k+j])*fp8(w[(expert*k+j)*n+c])*
            s8(xs[r*kg*2+j/32])*s8(ws[((expert*kg+j/64)*n+c)*2+j/32%2]);
        if (sum!=y[r*n+c])return 1;
    }
    std::puts("CPU_PERIOD_GOLDEN_PASS count=6720 two_experts=1 all_128_reduction_terms=1");return 0;
}
