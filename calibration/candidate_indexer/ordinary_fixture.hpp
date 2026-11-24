// Native, candidate-free LI control. Each query selects a different known set
// of 512 causal positions out of a full BF16 table. CPU sorts returned IDs
// only for comparison because the ordinary LI contract returns ranked IDs.
static void ordinary_control(int queries, int length, aclrtStream stream) {
    constexpr int heads = 32, depth = 128, top = 512;
    if (queries < 1 || 3*queries > depth || length-queries < top) {
        throw std::runtime_error("Unsupported ordinary LI fixture");
    }
    std::vector<uint16_t> q(queries*heads*depth), key(length*depth), w(queries*heads);
    std::vector<std::vector<int32_t>> expected(queries);
    for(int r=0; r<queries; ++r) {
        int visible = length-queries+r+1;
        for(int h=0; h<3; ++h) q[(r*heads+h)*depth+3*r+h]=bf(1);
        w[r*heads] = bf(.5f);
        w[r*heads+1] = bf(2.f);
        // Nonuniform non-unit weights and a third zero-weight distractor.
        // A winner has score 3 (first head) or 2 (second head). A valid
        // loser has score 1.5. Its negative second-head dot must be rectified.
        for(int j=0; j<length; ++j) {
            int shifted = (j+37*r)%visible;
            bool winner = j<visible && shifted>=visible-top;
            bool future = j>=visible;
            key[j*depth+3*r] = bf(future ? 10.f : winner ? (j%2 ? 6.f : -8.f) : 3.f);
            key[j*depth+3*r+1] = bf(future ? 10.f : winner ? (j%2 ? -8.f : 1.f) : -2.f);
            key[j*depth+3*r+2] = bf(winner ? 0.f : 64.f);
            if(winner) expected[r].push_back(j);
        }
    }
    Tensor query({1,queries,heads,depth},ACL_BF16,bytes(q));
    Tensor kv({1,length,1,depth},ACL_BF16,bytes(key));
    Tensor weights({1,queries,heads},ACL_BF16,bytes(w));
    Tensor indices({1,queries,1,top},ACL_INT32,std::vector<uint8_t>(queries*top*4,0xab));
    char layout[]="BSND";
    auto inf=std::numeric_limits<int64_t>::max();
    std::printf("li_case Q=%d L=%d H=%d D=%d K=%d\n",queries,length,heads,depth,top);
    LAUNCH(aclnnLightningIndexer,query.desc,kv.desc,weights.desc,nullptr,nullptr,nullptr,
           layout,layout,top,3,inf,inf,false,indices.desc,nullptr);
    std::vector<int32_t> ids(queries*top);
    CHECK(aclrtMemcpy(ids.data(),ids.size()*4,indices.buffer.ptr,ids.size()*4,ACL_MEMCPY_DEVICE_TO_HOST));
    size_t wrong=0;
    for(int r=0;r<queries;++r) {
        auto row=ids.begin()+r*top;
        std::sort(row,row+top);
        if(expected[r].size()!=top) throw std::runtime_error("invalid ordinary CPU fixture");
        for(int j=0;j<top;++j) wrong+=row[j]!=expected[r][j];
    }
    std::printf("li_correctness sorted_for_set_check=%d bad=%zu\n",queries*top,wrong);
    if(wrong) throw std::runtime_error("ordinary LI causality, weights or top-k mismatch");
}
