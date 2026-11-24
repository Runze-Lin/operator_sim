// Two executable component graphs for one last-token query with valid whole candidate blocks.
#include "../acl_tensor.hpp"
#include <aclnnop/aclnn_gather_v2.h>
#include <aclnnop/aclnn_lightning_indexer.h>
#include <aclnnop/aclnn_sort.h>
#include <aclnnop/aclnn_cast.h>
#include <aclnnop/aclnn_matmul.h>
#include <aclnnop/aclnn_masked_fill_scalar.h>
#include <aclnnop/aclnn_mul.h>
#include <aclnnop/aclnn_reduce_sum.h>
#include <aclnnop/aclnn_relu.h>
#include <aclnnop/aclnn_all.h>
#include <aclnnop/aclnn_ne_tensor.h>
#include <aclnnop/aclnn_topk.h>
#include <algorithm>
#include <cstdio>
#include <cstring>
#include <limits>
#include <numeric>

template<class T> static std::vector<uint8_t> bytes(const std::vector<T>& data) {
    std::vector<uint8_t> out(sizeof(T)*data.size());
    std::memcpy(out.data(),data.data(),out.size());
    return out;
}

static uint16_t bf(float x) {
    uint32_t bits;
    std::memcpy(&bits,&x,sizeof(bits));
    return bits>>16;
}

// Keep device data, give it the reshaped contiguous view needed by the next kernel.
struct View {
    aclTensor *desc;
    View(const std::vector<int64_t>& shape,aclDataType type,Tensor& base) {
        std::vector<int64_t> stride(shape.size(),1);
        for (size_t i=shape.size()-1;i>0;--i) stride[i-1]=stride[i]*shape[i];
        desc=aclCreateTensor(shape.data(),shape.size(),type,stride.data(),0,ACL_FORMAT_ND,
                             shape.data(),shape.size(),base.buffer.ptr);
        if (!desc) throw std::runtime_error("invalid view");
    }
    ~View(){ aclDestroyTensor(desc); }
};

static void gather(const aclTensor* table,const aclTensor* ids,Tensor& out,aclrtStream stream) {
    uint64_t ws=0;
    aclOpExecutor* exec=nullptr;
    CHECK(aclnnGatherV2GetWorkspaceSize(table,0,ids,out.desc,&ws,&exec));
    Buffer scratch(ws);
    CHECK(aclnnGatherV2(scratch.ptr,ws,exec,stream));
    CHECK(aclrtSynchronizeStream(stream));
}

#define LAUNCH(op, ...) do { \
    uint64_t workspace=0; aclOpExecutor* executor=nullptr; \
    std::printf("launch=%s\n", #op); \
    CHECK(op##GetWorkspaceSize(__VA_ARGS__, &workspace, &executor)); \
    Buffer scratch(workspace); CHECK(op(scratch.ptr,workspace,executor,stream)); \
    CHECK(aclrtSynchronizeStream(stream)); \
} while (false)

// Literal full-cache score then mask. The three BF16 input tensors already
// contain the values at the system graph boundary (after any upstream fake quant).
static void dense(int blocks,int selected,int heads,Tensor& query,Tensor& key,
                  Tensor& weight,Tensor& ids,Tensor& sorted,aclrtStream stream) {
    const int d=128, rows=blocks*8, top=512;
    View q({heads,d},ACL_BF16,query), k({rows,d},ACL_BF16,key), w({heads,1},ACL_BF16,weight);
    Tensor qf({heads,d},ACL_FLOAT,std::vector<uint8_t>(heads*d*4));
    Tensor kf({rows,d},ACL_FLOAT,std::vector<uint8_t>(rows*d*4));
    Tensor wf({heads,1},ACL_FLOAT,std::vector<uint8_t>(heads*4));
    LAUNCH(aclnnCast,q.desc,ACL_FLOAT,qf.desc);
    LAUNCH(aclnnCast,k.desc,ACL_FLOAT,kf.desc);
    LAUNCH(aclnnCast,w.desc,ACL_FLOAT,wf.desc);

    const int64_t dims[]={d,rows}, strides[]={1,d}, storage[]={rows,d};
    aclTensor* kt=aclCreateTensor(dims,2,ACL_FLOAT,strides,0,ACL_FORMAT_ND,storage,2,kf.buffer.ptr);
    if (!kt) throw std::runtime_error("invalid transposed key view");
    Tensor dots({heads,rows},ACL_FLOAT,std::vector<uint8_t>(heads*rows*4));
    LAUNCH(aclnnMatmul,qf.desc,kt,dots.desc,0);
    aclDestroyTensor(kt);
    LAUNCH(aclnnInplaceRelu,dots.desc);
    LAUNCH(aclnnInplaceMul,dots.desc,wf.desc);
    const int64_t axis=0;
    auto* reduce_axes=aclCreateIntArray(&axis,1);
    Tensor scores({rows},ACL_FLOAT,std::vector<uint8_t>(rows*4));
    LAUNCH(aclnnReduceSum,dots.desc,reduce_axes,false,ACL_FLOAT,scores.desc);
    aclDestroyIntArray(reduce_axes);

    // Compute one exclusion bit per compressed block. This general ACLNN
    // membership lowering compares every history block to every supplied ID.
    // It prices this smoke implementation only; an optimized set need not do so.
    std::vector<int32_t> block_range(blocks);
    std::iota(block_range.begin(),block_range.end(),0);
    Tensor block_col({blocks,1},ACL_INT32,bytes(block_range));
    View id_row({1,selected},ACL_INT32,ids);
    Tensor different({blocks,selected},ACL_BOOL,std::vector<uint8_t>(blocks*selected));
    Tensor mask({blocks},ACL_BOOL,std::vector<uint8_t>(blocks));
    LAUNCH(aclnnNeTensor,block_col.desc,id_row.desc,different.desc);
    const int64_t id_axis=1;
    auto* id_axes=aclCreateIntArray(&id_axis,1);
    LAUNCH(aclnnAll,different.desc,id_axes,false,mask.desc);
    aclDestroyIntArray(id_axes);
    View block_scores({blocks,8},ACL_FLOAT,scores), reject({blocks,1},ACL_BOOL,mask);
    float neg_inf=-std::numeric_limits<float>::infinity();
    auto* fill=aclCreateScalar(&neg_inf,ACL_FLOAT);
    LAUNCH(aclnnInplaceMaskedFillScalar,block_scores.desc,reject.desc,fill);
    aclDestroyScalar(fill);
    Tensor topvals({top},ACL_FLOAT,std::vector<uint8_t>(top*4));
    Tensor topids({top},ACL_INT64,std::vector<uint8_t>(top*8));
    LAUNCH(aclnnTopk,scores.desc,top,-1,true,false,topvals.desc,topids.desc);
    Tensor sorted64({top},ACL_INT64,std::vector<uint8_t>(top*8));
    Tensor order64({top},ACL_INT64,std::vector<uint8_t>(top*8));
    LAUNCH(aclnnSort,topids.desc,false,-1,false,sorted64.desc,order64.desc);
    LAUNCH(aclnnCast,sorted64.desc,ACL_INT32,sorted.desc);
}

static void run(bool full_scan,int blocks,int selected,int heads,aclrtStream stream) {
    const int lanes=128, block=8, top=512;
    if (selected<65 || selected>=blocks || blocks>512 || (heads!=32 && heads!=8))
        throw std::runtime_error("expected 65<=C<L/8<=512; H=8 or 32");
    const int rows=blocks*block, valid=selected*block;
    std::vector<int32_t> block_ids(selected),row_ids(rows),gold;
    std::iota(row_ids.begin(),row_ids.end(),0);
    // Spread the retained blocks across history, always include the last.
    // Reverse order so a missing remap or final chronology sort is detectable.
    for(int j=0;j<selected;++j) {
        int id=(selected-j-1)*(blocks-1)/(selected-1);
        block_ids[j]=id;
        for(int offset=0;offset<block;++offset) gold.push_back(block*id+offset);
    }
    std::vector<uint16_t> q(heads*lanes,0),k(rows*lanes,0),w(heads,bf(1));
    for(int h=0;h<heads;++h) {
        // One dominant positive head, one negative head and zero-weight heads.
        q[lanes*h]=q[lanes*h+1]=q[lanes*h+2]=bf(h%2 ? -1.f:1.f);
        w[h]=bf(h<2 ? .5f:0.f);
    }
    // At larger score magnitudes, a BF16 sorting key can tie across the
    // selection boundary. Separate the two sets without changing their order.
    // Global IDs still require both candidate filtering and position remapping.
    std::sort(gold.begin(),gold.end());
    const int cutoff=gold[gold.size()-top];
    for(int i=0;i<rows;++i) {
        k[i*lanes]=bf(float(i/128)); k[i*lanes+1]=bf(float(i%128)/128);
        if (blocks>256 && i>=cutoff) k[i*lanes+2]=bf(64);
    }
    Tensor ids({selected},ACL_INT32,bytes(block_ids));
    Tensor key({blocks,block*lanes},ACL_BF16,bytes(k));
    Tensor rowmap({blocks,block},ACL_INT32,bytes(row_ids));
    Tensor packed({selected,block*lanes},ACL_BF16,std::vector<uint8_t>(valid*lanes*2));
    Tensor mapping({selected,block},ACL_INT32,std::vector<uint8_t>(valid*4));
    Tensor query({1,1,heads,lanes},ACL_BF16,bytes(q)),weight({1,1,heads},ACL_BF16,bytes(w));
    Tensor ix({1,1,1,top},ACL_INT32,std::vector<uint8_t>(top*4,0xd3));
    Tensor position({top},ACL_INT32,std::vector<uint8_t>(top*4,0xad));
    Tensor sorted({top},ACL_INT32,std::vector<uint8_t>(top*4,0xfa));
    Tensor order({top},ACL_INT64,std::vector<uint8_t>(top*8,0xaf));
    std::printf("case=%s blocks=%d selected=%d heads=%d block_size=8 D=128 Q=1 top=512\n",full_scan ? "dense" : "compact",blocks,selected,heads);
    if (full_scan) {
        dense(blocks,selected,heads,query,key,weight,ids,sorted,stream);
    } else {
    gather(key.desc,ids.desc,packed,stream);
    gather(rowmap.desc,ids.desc,mapping,stream);
    View packview({1,valid,1,lanes},ACL_BF16,packed);
    char layout[]="BSND";
    const auto inf=std::numeric_limits<int64_t>::max();
    aclOpExecutor* li=nullptr;
    uint64_t ws=0;
    CHECK(aclnnLightningIndexerGetWorkspaceSize(query.desc,packview.desc,weight.desc,nullptr,nullptr,nullptr,
              layout,layout,top,3,inf,inf,false,ix.desc,nullptr,&ws,&li));
    Buffer tmp(ws);
    CHECK(aclnnLightningIndexer(tmp.ptr,ws,li,stream)); CHECK(aclrtSynchronizeStream(stream));
    View flatmap({valid},ACL_INT32,mapping),flatix({top},ACL_INT32,ix);
    gather(flatmap.desc,flatix.desc,position,stream);
    aclOpExecutor* sort=nullptr;
    CHECK(aclnnSortGetWorkspaceSize(position.desc,false,-1,false,sorted.desc,order.desc,&ws,&sort));
    Buffer scratch(ws);
    CHECK(aclnnSort(scratch.ptr,ws,sort,stream)); CHECK(aclrtSynchronizeStream(stream));
    }
    std::vector<int32_t> values(top);
    CHECK(aclrtMemcpy(values.data(),top*4,sorted.buffer.ptr,top*4,ACL_MEMCPY_DEVICE_TO_HOST));
    // All candidate blocks are valid for this last-token query. Their scores
    // grow strictly with global row; full-scan + candidate mask yields the top rows.
    size_t bad=0;
    for(int j=0;j<top;++j) bad+=values[j]!=gold[gold.size()-top+j];
    std::printf("correctness ordered_global_positions=%d bad=%zu\n",top,bad);
    if(bad) throw std::runtime_error("incorrect top-k global positions");
}

#include "ordinary_fixture.hpp"
#include "mx_controls.hpp"

int main(int argc,char** argv) {
    std::setvbuf(stdout,nullptr,_IONBF,0);
    bool controls = argc==2 && std::string(argv[1])=="controls";
    bool mx = argc==2 && std::string(argv[1])=="mx-controls";
    if(!controls && !mx && (argc!=5 || (std::string(argv[1])!="compact" && std::string(argv[1])!="dense"))) return 2;
    try {
        CHECK(aclInit(nullptr)); CHECK(aclrtSetDevice(0)); aclrtStream s=nullptr;CHECK(aclrtCreateStream(&s));
        std::printf("runtime_soc=%s\n",aclrtGetSocName());
        if(mx) {
            mx_control(1,128,81,s);
            mx_control(1,257,81,s);
            mx_control(4,257,81,s);
        } else if(controls) {
            for(int blocks : {128,257}) {
                run(true,blocks,81,32,s);
                ordinary_control(1,blocks*8,s);
            }
            ordinary_control(5,2056,s);
        } else {
            run(std::string(argv[1])=="dense",std::stoi(argv[2]),std::stoi(argv[3]),std::stoi(argv[4]),s);
        }
        CHECK(aclrtDestroyStream(s)); CHECK(aclrtResetDevice(0)); CHECK(aclFinalize());
        std::puts("SMOKE_PASS");std::_Exit(EXIT_SUCCESS);
    } catch(const std::exception& e) {std::fprintf(stderr,"SMOKE_FAIL %s\n",e.what());return 1;}
}
