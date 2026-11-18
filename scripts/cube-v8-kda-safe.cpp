// Reuse the same RAII, BF16 rounding, tolerances and independent state reference.
#define main legacy_kda_main
#include "../smoke/extension_kda.cpp"
#undef main

static void paired_chunk(aclrtStream stream,int s,int heads) {
    constexpr int k=128,v=128,c=64;constexpr double lower=-0.03125;
    const double scale=1./std::sqrt(double(k));
    std::vector<uint16_t> q(s*heads*k),key(q.size()),value(q.size());
    std::vector<float> gate(q.size()),beta(s*heads),alog(heads),initial(heads*k*v);
    std::vector<double> output(q.size()),final(initial.size()),gg(q.size()),ga(heads*s*c),gi(ga.size());
    for(int h=0;h<heads;++h){
        Inputs in(s,k,v);alog[h]=0.125f*h;
        for(int t=0;t<s;++t){
            beta[t*heads+h]=in.beta[t];
            for(int d=0;d<k;++d){
                const int i=t*k+d, packed=(t*heads+h)*k+d;
                gate[packed]=float((t+3*d+h)%11-5)/8;
                in.gate[i]=lower/(1.+std::exp(-std::exp(double(alog[h]))*gate[packed]));
                if(h%2)in.value[i]=bf16(-fp32(in.value[i]));
                q[packed]=in.q[i];key[packed]=in.key[i];value[packed]=in.value[i];
            }
        }
        std::copy(in.state.begin(),in.state.end(),initial.begin()+h*k*v);
        auto gold=recurrent_gold(in);std::copy(gold.second.begin(),gold.second.end(),final.begin()+h*k*v);
        for(int t=0;t<s;++t)std::copy_n(gold.first.begin()+t*v,v,output.begin()+(t*heads+h)*v);
        for(int start=0;start<s;start+=c){
            int n=std::min(c,s-start);std::vector<double> l(n*n);
            for(int i=0;i<n;++i){
                int row=h*s+start+i;
                for(int d=0;d<k;++d)gg[row*k+d]=in.gate[(start+i)*k+d]/std::log(2.)+(i?gg[(row-1)*k+d]:0.);
                for(int j=0;j<=i;++j){
                    const int col=h*s+start+j;double qk=0,kk=0;
                    for(int d=0;d<k;++d){
                        double kg=std::exp2(gg[row*k+d]-gg[col*k+d])*fp32(in.key[(start+j)*k+d]);
                        qk+=fp32(in.q[(start+i)*k+d])*kg;kk+=fp32(in.key[(start+i)*k+d])*kg;
                    }
                    ga[row*c+j]=scale*qk;if(j<i)l[i*n+j]=in.beta[start+i]*kk;
                }
                for(int j=0;j<=i;++j){double x=i==j?1.:0.;for(int z=j;z<i;++z)x-=l[i*n+z]*gi[(h*s+start+z)*c+j];gi[row*c+j]=x;}
            }
        }
    }
    Tensor<uint16_t> Q({1,s,heads,k},ACL_BF16,q),K({1,s,heads,k},ACL_BF16,key),V({1,s,heads,v},ACL_BF16,value);
    Tensor<float> G({1,s,heads,k},ACL_FLOAT,gate),B({1,s,heads},ACL_FLOAT,beta),A({heads},ACL_FLOAT,alog),I({1,heads,k,v},ACL_FLOAT,initial);
    Tensor<uint16_t> O({1,s,heads,v},ACL_BF16,std::vector<uint16_t>(output.size(),0x7fc0)),QK({1,heads,s,c},ACL_BF16,std::vector<uint16_t>(ga.size(),0x7fc0)),KK({1,heads,s,c},ACL_BF16,std::vector<uint16_t>(gi.size(),0x7fc0));
    Tensor<float> F({1,heads,k,v},ACL_FLOAT,std::vector<float>(final.size(),NAN)),GK({1,heads,s,k},ACL_FLOAT,std::vector<float>(gg.size(),NAN));
    uint64_t bytes=0;aclOpExecutor* executor=nullptr;
    ACL_CHECK(aclnnChunkKdaFwdGetWorkspaceSize(Q.tensor,K.tensor,V.tensor,G.tensor,B.tensor,A.tensor,nullptr,I.tensor,nullptr,nullptr,"BSND",scale,c,true,lower,true,false,O.tensor,F.tensor,GK.tensor,QK.tensor,KK.tensor,nullptr,nullptr,nullptr,nullptr,nullptr,nullptr,&bytes,&executor));
    Buffer ws(bytes);ACL_CHECK(aclnnChunkKdaFwd(ws.ptr,bytes,executor,stream));ACL_CHECK(aclrtSynchronizeStream(stream));
    compare(decode(O.read()),output,true,"output");compare(decode(QK.read()),ga,true,"aqk");compare(decode(KK.read()),gi,true,"akk");
    auto g=GK.read(),f=F.read(),i=I.read();compare({g.begin(),g.end()},gg,false,"gk");compare({f.begin(),f.end()},final,false,"state");compare({i.begin(),i.end()},{initial.begin(),initial.end()},false,"input_state");
}
int main(int argc,char** argv){
    if(argc!=3)return 1;
    try{int s=std::stoi(argv[1]),h=std::stoi(argv[2]);if(s<1||s>512||h<2||h>16||h%2)return 2;
        ACL_CHECK(aclInit(nullptr));ACL_CHECK(aclrtSetDevice(0));std::printf("runtime_soc=%s\n",aclrtGetSocName());aclrtStream stream;ACL_CHECK(aclrtCreateStream(&stream));paired_chunk(stream,s,h);ACL_CHECK(aclrtDestroyStream(stream));ACL_CHECK(aclrtResetDevice(0));ACL_CHECK(aclFinalize());std::puts("SMOKE_PASS");return 0;
    }catch(const std::exception& e){std::fprintf(stderr,"SMOKE_FAIL %s\n",e.what());return 1;}
}
