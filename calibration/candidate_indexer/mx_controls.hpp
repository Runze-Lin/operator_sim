#include <aclnnop/aclnn_quant_lightning_indexer_v2.h>

// Write the E2M1 value's four-bit code into contiguous, low-nibble-first ND.
static void put4(std::vector<uint8_t>& storage,size_t i,int code) {
    const int shift = 4*(i%2);
    storage[i/2] = (storage[i/2] & ~(15<<shift)) | (code<<shift);
}

static void mx_control(int queries,int blocks,int selected,aclrtStream stream) {
    constexpr int h=32,d=128,top=512;
    const int valid=selected*8,length=blocks*8;
    std::vector<int32_t> source_ids(selected), positions(length), gold(queries*top);
    std::iota(positions.begin(),positions.end(),0);
    for(int i=0;i<selected;++i) source_ids[i]=i*(blocks-1)/(selected-1);
    std::vector<int> compact_order(blocks,-1);
    for(int i=0;i<selected;++i) compact_order[source_ids[i]]=i;
    std::vector<uint8_t> q(queries*h*d/2),k(length*d/2),qs(queries*h*4,127),ks(length*4,127);
    std::vector<float> w(queries*h,0.f);
    // One shared ascending candidate list. Its last block is the latest;
    // the final Q<=8 positions remain consecutive in the compact KV table.
    for(int r=0;r<queries;++r) {
        const int visible=valid-queries+r+1;
        int count=0;
        for(int head=0;head<3;++head) put4(q,(r*h+head)*d+r+32*head,2); // +1
        qs[(r*h)*4]=126; // 0.5
        w[r*h]=.5f; w[r*h+1]=2.f;
        for(int j=0;j<length;++j) {
            const int ord=compact_order[j/8];
            const int compact=ord*8+j%8;
            const bool member=ord>=0 && compact<visible;
            const bool winner=member && (compact+37*r)%visible>=visible-top;
            const int h0=!member ? 7 : winner ? (j%2?7:10) : 5; // 6, -1 or 3
            const int h1=!member ? 7 : winner ? (j%2?10:1) : 10; // 6, -1 or 0.5
            put4(k,j*d+r,h0); put4(k,j*d+32+r,h1); put4(k,j*d+64+r,winner?0:7);
            ks[j*4+1]=128; // 2, second head uses this group
            if(winner) gold[r*top+count++]=j;
        }
        if(count!=top)throw std::runtime_error("Invalid CPU fixture");
    }

    Tensor ids({selected},ACL_INT32,bytes(source_ids));
    Tensor keys({blocks,8*d/2},ACL_UINT8,k), scales({blocks,8*4},ACL_UINT8,ks);
    Tensor rowmap({blocks,8},ACL_INT32,bytes(positions));
    Tensor packed({selected,8*d/2},ACL_UINT8,std::vector<uint8_t>(valid*d/2));
    Tensor packed_scales({selected,8*4},ACL_UINT8,std::vector<uint8_t>(4*valid));
    Tensor mapping({selected,8},ACL_INT32,std::vector<uint8_t>(4*valid));
    std::printf("mx_case Q=%d L=%d C=%d H=32 D=128 K=512 S=8\n",queries,length,selected);
    LAUNCH(aclnnGatherV2,keys.desc,0,ids.desc,packed.desc);
    LAUNCH(aclnnGatherV2,scales.desc,0,ids.desc,packed_scales.desc);
    LAUNCH(aclnnGatherV2,rowmap.desc,0,ids.desc,mapping.desc);
    Tensor query({1,queries,h,d},ACL_FLOAT4_E2M1,q);
    Tensor qscale({1,queries,h,2,2},ACL_FLOAT8_E8M0,qs);
    View key({1,valid,1,d},ACL_FLOAT4_E2M1,packed);
    View kscale({1,valid,1,2,2},ACL_FLOAT8_E8M0,packed_scales);
    Tensor weights({1,queries,h},ACL_FLOAT,bytes(w));
    Tensor usedq({1},ACL_INT32,bytes(std::vector<int32_t>{queries}));
    Tensor usedk({1},ACL_INT32,bytes(std::vector<int32_t>{valid}));
    // 9.2.1 ordinary QLI: enable one AIC for the entire document. Caller
    // supplies a static table because runtime metadata is an AI-CPU operation.
    std::vector<int32_t> slots(1024);slots[0]=1; slots[4]=1;
    Tensor md({1024},ACL_INT32,bytes(slots));
    Tensor ix({1,queries,1,top},ACL_INT32,std::vector<uint8_t>(4*queries*top,0xac));
    Tensor no_values({0},ACL_BF16,std::vector<uint8_t>(2));
    char layout[]="BSND";
    LAUNCH(aclnnQuantLightningIndexerV2,query.desc,key.desc,weights.desc,qscale.desc,kscale.desc,
           nullptr,nullptr,usedq.desc,usedk.desc,nullptr,nullptr,nullptr,md.desc,top,5,-1,layout,layout,3,1,0,
           ix.desc,no_values.desc);
    View flatmap({valid},ACL_INT32,mapping), flatix({queries*top},ACL_INT32,ix);
    Tensor global({queries*top},ACL_INT32,std::vector<uint8_t>(4*queries*top));
    LAUNCH(aclnnGatherV2,flatmap.desc,0,flatix.desc,global.desc);
    View global_rows({queries,top},ACL_INT32,global);
    Tensor sorted({queries,top},ACL_INT32,std::vector<uint8_t>(4*queries*top));
    Tensor order({queries,top},ACL_INT64,std::vector<uint8_t>(8*queries*top));
    LAUNCH(aclnnSort,global_rows.desc,false,-1,false,sorted.desc,order.desc);
    std::vector<int32_t> result(queries*top);
    CHECK(aclrtMemcpy(result.data(),result.size()*4,sorted.buffer.ptr,result.size()*4,ACL_MEMCPY_DEVICE_TO_HOST));
    int bad=0;
    for(size_t i=0;i<result.size();++i) if(result[i]!=gold[i]){
       if(bad<4) std::printf("difference i=%zu expected=%d got=%d\n",i,gold[i],result[i]);
       ++bad;
    }
    std::printf("mx_correctness ordered_global_positions=%zu bad=%d\n",result.size(),bad);
    if(bad)throw std::runtime_error("Wrong MX top-k positions");
}
