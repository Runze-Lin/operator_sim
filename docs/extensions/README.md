# 新增算子建模入口与校准状态

## Summary

已加入RMSNorm、Addmm、LightningIndexer、mHC、KDA、GMM1–SwiGLU–量化及后续GMM2组成链的**结构建模入口**：核对真实950库接口，计算有依据的数学工作量、张量shape、依赖和部分源码tiling候选。它们还没有通用时延校准，所有 `predicted_us` / 利用率为 `null`；不能与v0.5 FA的准确率混算。

RMSNorm和Addmm另有真实ACLNN调用器，主线程各安排一个极小CA case验证正确性和dispatch，结果及成本见 [调用验证记录](smoke-950dt.json)。一个实测点只是后续校准的起点，不构成一个泛化时延模型。其他族本轮不新增CA。

| 族 / 入口 | shape约定 | 真实库及接口 | 当前可返回 / 主要边界 |
|---|---|---|---|
| `rmsnorm` | R,C | ops-nn / aclnnRmsNorm | 归约/乘法/rsqrt工作、UB与分核候选；不把归约算成一次普通乘加 |
| `addmm` | M,N,K | ops-nn / aclnnAddmm | self广播、alpha/beta和融合候选；不能直接取Matmul时延 |
| `lightning_indexer` | B,Q,L,G,D | ops-transformer / aclnnLightningIndexer | QK→ReLU→head加权归约→TopK；TopK算法和tiling未测 |
| `mhc_pre`, `mhc_pre_sinkhorn`, `mhc_post` | T,N,D | ops-transformer / 对应aclnnMhc* | 投影、归一化、混合、后处理工作；融合与拆分路径互斥 |
| `mhc_sinkhorn` | T,N | ops-transformer / aclnnMhcSinkhorn | FP32归一化迭代；VectorCore，不能当Cube矩阵乘 |
| `mhc` | T,N,D | 上述接口组合 | `implementation=fused/split`；不包含外部Attention/MLP主体耗时 |
| `kda_recurrent` | B,S,Hq,Hv,K,V | ops-transformer / aclnnRecurrentKda | 有状态递推链；单序列S≤8，主模型Q=9返回unsupported |
| `kda_chunk` | B,S,Hq,Hv,K,V | ops-transformer / aclnnChunkKdaFwd | chunk及state链；5阶段在同一kernel内部，不计5次launch |
| `grouped_matmul_swiglu_quant_v2` | M,K,F,E | ops-transformer / aclnnGroupedMatmulSwigluQuantV2 | GMM1→dequant→SwiGLU→MX quant；只覆盖第一融合 |
| `groupedmatmulswiglugmm` / `moe_ffn` | M,K,F,E | 上述融合 + aclnnGroupedMatmulV5候选 | 第二GMM数学工作；quant-scale ABI串联未验证，非一个同名单kernel |

KDA的Hq是Q/K heads，Hv是value heads；MoE的F是SwiGLU之后的宽度，E是本rank专家数，**不是EP大小**。MoE必须给出本rank实际token分配直方图，不能只从batch和EP直接算出专家负载。

## 使用

```bash
python3 -m operator_model.extensions --list
python3 -m operator_model.extensions --op rmsnorm --shape 9 7168
python3 -m operator_model.extensions --op addmm --shape 9 128 256 \
  --attrs-json '{"self_shape":[128],"alpha":1,"beta":1}'
python3 -m operator_model.extensions --op lightning_indexer --shape 1 9 131072 8 128
python3 -m operator_model.extensions --op mhc --shape 9 4 7168 \
  --attrs-json '{"implementation":"fused"}'
python3 -m operator_model.extensions --op kda_recurrent --shape 1 9 4 4 128 128
python3 -m operator_model.extensions --op kda_chunk --shape 1 2048 4 4 128 128
python3 -m operator_model.extensions --op groupedmatmulswiglugmm --shape 8 128 128 4 \
  --attrs-json '{"expert_counts":[1,0,7,0]}'
```

其中RecurrentKda Q9示例预期返回 `unsupported`，不会静默拆成8+1。`lightening_indexer`兼容用户原拼写，内部归一为`lightning_indexer`。

```python
from operator_model.extensions import plan
result = plan('rmsnorm', (9, 7168))
assert result['predicted_us'] is None
```

字典shape也可通过 `--shape-json` 提供给Indexer/mHC/KDA/MoE，键名严格遵循文档。调用器与模型接口分离；纯结构查询不启动CANN或CA。

## 三组详细方法、证据和限制

- [RMSNorm / Addmm](norm-addmm.md)：公式、源码分核/UB、广播和融合门控、真实库smoke。
- [LightningIndexer / mHC](indexer-mhc.md)：头数/稀疏模式、TopK未知部分、Pre/Sinkhorn/Post两条组成路径和Vector实现。
- [KDA / MoE](kda-moe.md)：recurrent/chunk公式及state依赖、专家直方图、MXFP8融合边界和GMM2接口缺口。

实际库二进制路径和SHA在各组evidence.json；它们是安装包候选身份，只有真实采集才有选中entry/核数证据。知识卡、固定开源源码和安装包版本不同的地方均明确保留，不伪造对应关系。

特别需要运行配置才能确定的场景：普通950 Indexer只接受G∈{8,16,24,32,64}、D=128。若配置32个index heads被假设TP8直接切成G4，则该路径不满足约束；需要vLLM调用点证据说明是否复制heads或改用量化/其他路径。模型config能给候选shape，但不足以证明单卡实际dispatch。

## 后续校准顺序

优先用已完成的小case确认完整kernel序列与tiling，再按机制选择少量未见shape。RMSNorm覆盖小/长归约与UB分界；Addmm分别覆盖bias和完整矩阵self。Indexer要单独识别TopK/合并开销；mHC比较融合/拆分时不可重复计数；KDA要分开decode递推、prefill chunk和Q9校验调度；MoE必须覆盖偏斜专家负载，而不是只扫均匀M。

在这些证据到位前，不把结构FLOPs除以峰值当作已经校准的耗时。每族后续都沿用本项目的“策略公式 + 带Summary的逐shape误差报告”交付规则。
