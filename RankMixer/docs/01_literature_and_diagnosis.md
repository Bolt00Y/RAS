# RankMixer 文献综述与基线诊断

## 1. 研究对象与边界

本文以当前已经完成的 RankMixer 论文基线为唯一主对照：

```text
1234 fields × 17-d embedding
        ↓ flatten + concat
[B, 20978]
        ↓ ordered Autosplit
16 segments
        ↓ 16 independent projectors
X0: [B, 16, 768]
        ↓ 2 × RankMixer Block
XL: [B, 16, 768]
        ↓ Mean Pooling
h: [B, 768]
        ↓ CVR head
logit / probability
```

当前连续维度 Autosplit 与 RankMixer 的公式实现一致，因此保留为 `RM-B0`。后续提出的随机划分、Global Token、显式交叉和动态门控均属于研究变量，而不是对基线实现的修复。

本文只讨论 dense interaction backbone 及 CVR 目标层。Embedding 表的分片、参数服务器、负样本生成和线上召回不在当前范围内。

---

## 2. 当前模型的参数与计算基准

设：

- token 数 `T=16`；
- 隐藏维度 `D=768`；
- block 数 `L=2`；
- PFFN 扩张倍数为 `k`；
- 输入展开维度 `F=20978`。

输入投影参数量近似为：

```math
P_{tokenizer}=F\times D=20,978\times768=16,111,104.
```

每个 RankMixer block 的 PFFN 参数量近似为：

```math
P_{PFFN/block}\approx 2kTD^2.
```

如果 `k=4`，则：

```text
PFFN / block ≈ 75.50M
2 blocks      ≈ 150.99M
Tokenizer     ≈ 16.11M
Dense total   ≈ 167.11M
```

这里不计 bias、LayerNorm 和任务头。Token Mixing 本身是参数无关的 reshape / split / concat 操作。

这个基准非常重要：任何“加一个小模块”的方案都必须同时报告：

1. 总参数量；
2. 激活参数量；
3. 每样本 FLOPs；
4. 实测训练吞吐；
5. 线上 P50/P95/P99 延迟；
6. MFU 和 HBM 峰值。

在工业推荐模型中，参数很少并不代表延迟很低。大量碎片化的小算子可能是 memory-bound，并显著降低 MFU。

---

## 3. RankMixer 原论文给出的核心归纳偏置

RankMixer 的两个核心模块分别解决不同问题：

### 3.1 Multi-head Token Mixing

每个 token 被切成 `H` 个 channel head，同一个 head index 的切片在不同 token 之间重新拼接。论文设置 `H=T`，从而保持 mixing 前后 token 数相同并支持残差连接。

该操作具有：

- 无参数；
- 高并行度；
- 不依赖异构 token 之间的内积相似度；
- 适合大规模 grouped GEMM；
- 固定、与样本无关的 mixing pattern。

最后一点既是效率优势，也是后续工作的主要改进方向：固定 permutation 并不一定是当前数据上最优的交互图。

### 3.2 Per-token FFN

每个 token 拥有独立 FFN 参数，避免高频特征空间长期支配低频或长尾特征空间。它同时带来很强的容量扩展能力，但也带来以下潜在问题：

- token 分组质量直接影响参数利用率；
- 某些低变化 token 的 FFN 可能训练不足；
- 深层 FFN 可能导致表示秩收缩；
- dense PFFN 的活跃计算随容量同步增长；
- 不同样本使用相同参数路径，缺乏条件计算。

### 3.3 当前残差的语义问题

原始 RankMixer 将 mixing 后的 token 与 mixing 前同位置 token 直接相加。虽然张量形状相同，但位置语义已经发生了重新组合。TokenMixer-Large 将其称为 residual semantic misalignment，并通过 Mixing & Reverting 恢复原 token 布局后再进行跨层残差。

对于当前 `L=2`，该问题不一定造成明显训练不稳定；但它会限制继续向 `L=4/6/8` 扩展时的收益。

---

## 4. 直接后续文献

### 4.1 RankMixer

**J. Zhu et al. RankMixer: Scaling Up Ranking Models in Industrial Recommenders, 2025.**  
https://arxiv.org/abs/2507.15551

与当前研究最相关的结论：

- Token Mixing + Per-token FFN 是硬件友好的统一交互结构；
- 原论文已验证 dense scaling 和 Sparse MoE；
- 论文在 trillion-scale 生产数据上研究参数、FLOPs 与效果的关系；
- 论文的 Sparse MoE 使用 ReLU routing 和 dense-training/sparse-inference 思路；
- 论文没有证明固定 Autosplit、固定 mixing 或 Post-Norm 是不可改进的最优结构。

### 4.2 TokenMixer-Large

**Y. Jiang et al. TokenMixer-Large: Scaling Up Large Ranking Models in Industrial Recommenders, 2026.**  
https://arxiv.org/abs/2602.06563  
https://arxiv.org/html/2602.06563

它是 RankMixer/TokenMixer 最直接的结构升级，提出：

- Global Token；
- Mixing & Reverting；
- Per-token SwiGLU；
- Pre-RMSNorm；
- 深层 inter-residual；
- lower-layer auxiliary loss；
- Sparse-Pertoken MoE；
- shared expert、gate value scaling 和 down-matrix small initialization；
- sparse train + sparse inference；
- Token Parallel、FP8 和算子融合。

论文特别指出原始 RankMixer 的三个限制：

1. mixing 前后残差位置语义不一致；
2. 浅层结构容易训练，但直接加深时梯度和收敛会恶化；
3. ReLU-MoE 的激活专家数量动态，线上预算不够可预测。

该论文的实验条件与当前场景高度接近：其主要离线数据约 4 亿条/天，batch size 2048，并同时评估 CTR 和 CVR；当前数据约 5.5 亿条/天，batch size 同样为 2048。

需要注意：TokenMixer-Large 也主张在超大规模模型中移除 DCN、LHUC 等碎片化算子，以提高纯 backbone 的 MFU。因此，将 DCN 接到当前模型上必须通过低维、可融合的支路实现，并以端到端延迟为准，而不能只比较参数量。

### 4.3 RankUp

**J. Chen et al. RankUp: Towards High-rank Representations for Large Scale Advertising Recommender Systems, 2026.**  
https://arxiv.org/abs/2604.17878  
https://arxiv.org/html/2604.17878

这是与当前任务匹配度最高的文献之一：

- 任务是工业广告 CVR；
- 每天约 2000 万样本；
- 超过 1200 个 sparse features；
- 基线是两层 RankMixer；
- 当前任务同样是 CVR、两层 RankMixer，并有 1234 个 sparse fields。

RankUp 观察到 RankMixer token 表示的 effective rank 随深度出现阻尼振荡，在深层甚至下降。它提出：

- Randomized Permutation Splitting；
- Multi-embedding；
- Global Token；
- 预训练 user/item embedding 的乘法 cross token；
- Task-specific Token；
- PreNorm 与 SwiGLU。

论文在三个 CVR 子任务上报告的独立消融包括：

```text
Randomized Permutation Split: +0.06% / +0.06% / +0.08% Realtime AUC
Global Token + Multi-Emb:     +0.21% / +0.18% / +0.13%
Cross Embedding:              +0.22% / +0.10% / +0.03%
Task Token:                   +0.09% / +0.02% / +0.02%
Full RankUp:                  +0.41% / +0.23% / +0.25%
```

这些数值是论文自身环境中的相对提升，不能直接外推，但说明“输入表示秩和 token 冗余”是值得优先验证的研究问题。

### 4.4 UniMixer

**M. Ha et al. UniMixer: A Unified Architecture for Scaling Laws in Recommendation Systems, 2026.**  
https://arxiv.org/abs/2604.00590  
https://arxiv.org/html/2604.00590

UniMixer 将固定 TokenMixer 视为一个巨大的 permutation matrix，并对其进行结构化参数化。主要意义是：

- mixing pattern 可以由数据学习；
- 不再强制 `H=T`；
- 可以同时表达局部和全局 mixing；
- UniMixing-Lite 使用 basis-composed local matrices 和低秩 global matrix 控制参数/FLOPs；
- 通过 Sinkhorn-Knopp 约束 mixing matrix 接近双随机矩阵，并使用温度退火获得稀疏模式。

它为“在不退化为完整 attention 的前提下，使 RankMixer mixing 可学习”提供了直接依据。

---

## 5. 补充交互模块文献

### 5.1 DCN V2

**R. Wang et al. DCN V2: Improved Deep & Cross Network and Practical Lessons for Web-scale Learning to Rank Systems, WWW 2021.**  
https://arxiv.org/abs/2008.13535

DCNv2 显式学习 bounded-degree feature crosses，并通过低秩和 mixture 结构控制成本。它与 RankMixer 的互补性在于：

- RankMixer 主要依赖固定 token mixing + 非线性 PFFN 隐式形成交互；
- DCNv2 直接引入 `x0 ⊙ f(xl)` 形式的乘法交叉；
- 显式低阶交叉可能更快学习 user × item、item × creative、price × preference 等强结构信号；
- 低秩 CrossNet 可以在很小的附加参数下实现。

但不能直接在 20,978 维原始向量上使用 full-rank matrix：单层矩阵约为 `20,978² ≈ 440M` 参数。建议先将 16 个 token 压缩到 512 维左右，再运行低秩 CrossNet。

### 5.2 DCN²

**B. Škrlj et al. DCN²: Interplay of Implicit Collision Weights and Explicit Cross Layers for Large-Scale Recommendation, 2025.**  
https://arxiv.org/abs/2506.21624

该工作说明显式 cross layer 在现代大规模系统中仍可能有效，并研究了 lookup 权重、显式 cross 和 pairwise similarity 的组合。它支持“显式交叉并未因为大模型出现而失去价值”，但不代表任意将 DCN 串接到 RankMixer 都会有正收益。

### 5.3 DHEN

**B. Zhang et al. DHEN: A Deep and Hierarchical Ensemble Network for Large-Scale Click-Through Rate Prediction, 2022.**  
https://arxiv.org/abs/2203.11014

DHEN 的核心观察是：不同 feature interaction operator 即使声称建模相同阶数，实际捕获的信息也可能不重合。该结论为“RankMixer 主干 + 小型显式交叉支路”提供合理性。

同时，DHEN 也提示必须进行模块级消融；否则无法判断收益来自哪一种 interaction operator。

---

## 6. 动态特征选择与乘法门控文献

### 6.1 FiBiNET

**T. Huang et al. FiBiNET: Combining Feature Importance and Bilinear Feature Interaction for CTR Prediction, RecSys 2019.**  
https://arxiv.org/abs/1905.09433

FiBiNET 使用 SENET 动态学习不同 field 对当前样本的重要性，并使用 bilinear function 建模细粒度交互。对当前 RankMixer 的启示是：

- Per-token FFN 是 token-specific，但不是 instance-specific；
- 同一个 token 在所有请求中使用固定的特征强度；
- 推广搜索 CVR 中，不同 query、user intent、价格区间和 creative 类型可能导致完全不同的有效特征组合；
- 轻量级 token/channel gate 可以为每个样本动态调节 RankMixer 输入。

不建议直接复现 field-level FiBiNET。1234 个字段共有：

```math
\binom{1234}{2}=760,761
```

个字段对。若显式保留每个 pair 的 17 维交互，batch 2048 的中间张量不可接受。应把 FiBiNET 思路迁移到 `T=16` 的 token 层，或使用低秩 factorized gate。

### 6.2 MaskNet

**Z. Wang et al. MaskNet: Introducing Feature-Wise Multiplication to CTR Ranking Models by Instance-Guided Mask, 2021.**  
https://arxiv.org/abs/2102.07619

MaskNet 指出纯加性 FFN 对常见乘法交互的建模效率有限，并使用 instance-guided mask 将加性和乘法交互结合。该思路适合用作 RankMixer block 内的低成本 residual adapter，但必须：

- identity initialization；
- 限制 gate 范围；
- 防止长尾 token 被持续压低；
- 避免生成完整 `[B,16,768]` 的大 MLP mask。

推荐使用 token gate 与 channel gate 的外积作为低秩 mask。

---

## 7. MoE 文献

### 7.1 RankMixer Sparse MoE

RankMixer 原论文已将每个 token 的 dense PFFN 扩展为 Sparse MoE，并采用 ReLU routing，让高信息 token 可以激活更多专家。

优点：

- 动态分配计算；
- 扩大容量；
- 与 per-token 参数隔离理念一致。

缺点：

- 激活专家数量动态，线上尾延迟难预算；
- 原方案 dense train + sparse inference，训练成本没有同步下降；
- 每个 token 独立专家可能导致专家数量和训练不均衡问题。

### 7.2 ReMoE

**Z. Wang et al. ReMoE: Fully Differentiable Mixture-of-Experts with ReLU Routing, 2024.**  
https://arxiv.org/abs/2412.14711

ReMoE 用 ReLU 替代不连续的 Top-k + Softmax，并通过正则控制稀疏度与负载。它适合允许动态计算预算的训练或离线任务。

### 7.3 TokenMixer-Large Sparse-Pertoken MoE

TokenMixer-Large 改用固定 top-k 的 per-token sub-experts，并强调：

- 先放大 dense model，再 sparsify；
- sparse train + sparse inference；
- 每个 token 有自己的 expert set，形成强 routing prior；
- 每个 token 设置 shared expert；
- gate value scaling 与 sparsity ratio 对齐；
- down matrix 使用约 0.01 倍小初始化。

对于严格线上延迟，Sparse-Pertoken MoE 比动态 ReLU expert count 更容易控制，应作为首选 MoE 方案。

---

## 8. CVR 目标文献

### 8.1 ESMM

**X. Ma et al. Entire Space Multi-Task Model: An Effective Approach for Estimating Post-Click Conversion Rate, SIGIR 2018.**  
https://arxiv.org/abs/1804.07931

如果当前模型只在点击样本上训练、却在全曝光空间推理，则存在：

- sample selection bias；
- conversion label 极度稀疏；
- 点击与转化的序列依赖没有被利用。

ESMM 在全曝光空间同时建模 CTR 和 CTCVR：

```math
p_{CTCVR}=p_{CTR}\cdot p_{CVR}.
```

这不是单纯的 backbone 变化，而是改变估计问题本身。如果数据具备曝光、点击和转化标签，目标层升级通常应与 backbone 改造并行研究。

### 8.2 PLE 与 Task-specific Token

**H. Tang et al. Progressive Layered Extraction, RecSys 2020.**  
https://doi.org/10.1145/3383313.3412236

PLE 明确分离 shared experts 和 task-specific experts，以缓解多任务负迁移和 seesaw。RankUp 则直接在 token 层加入 task-specific tokens。

对当前模型更轻量的做法是保留共享 RankMixer backbone，但为 CTR、CVR 使用独立的 token pooling 与小型任务塔，而不是立刻引入完整 PLE。

---

## 9. 当前基线真正值得验证的能力缺口

| 能力缺口 | 当前 RankMixer 表现 | 对应文献 | 可验证方案 |
|---|---|---|---|
| 输入 token 冗余/低秩 | 固定有序 Autosplit | RankUp | 固定随机 feature permutation、Global Token |
| 残差语义错位 | mixing 后直接与原位置相加 | TokenMixer-Large | Mixing & Reverting |
| 深度扩展困难 | 当前只有 2 层，直接加深风险高 | TokenMixer-Large、UniMixer | PreNorm、SwiGLU、inter-residual、aux loss |
| Mixing pattern 固定 | reshape/permutation 不依赖数据 | UniMixer | 结构化可学习 mixing adapter |
| 显式乘法交叉不足 | 主要依赖 PFFN 隐式学习 | DCNv2、DHEN | 低秩 CrossNet 支路 |
| 样本级特征选择不足 | token-specific 但非 instance-specific | FiBiNET、MaskNet | factorized token-channel gate |
| 容量与活跃 FLOPs绑定 | Dense PFFN 全量激活 | RankMixer MoE、TokenMixer-Large | Sparse-Pertoken MoE |
| CVR 样本偏差 | 取决于是否 clicked-only | ESMM、PLE | Entire-space multi-task + task pooling |

---

## 10. 不建议的“为了改变而改变”

### 10.1 不建议完整字段级 Self-Attention

1234 fields 的 attention score 为 `1234²≈1.52M` 个/样本，batch 2048 下仅 score 张量就非常大，并且异构 ID 空间的内积相似度未必合理。若研究 attention，应在 16 token 层进行，并与固定 mixing 做严格 FLOPs 匹配。

### 10.2 不建议在 20,978 维上使用 full-rank DCNv2

单层权重约 440M 参数，且大矩阵后仍需逐元素乘法。它既不符合低风险试验，也不符合 RankMixer 的硬件友好目标。

### 10.3 不建议一次性串联 SENet + DCN + Attention + MoE

这种实验即使获得收益，也无法确定因果来源；若失败，也无法定位冲突。首轮必须一次只改变一个建模假设。

### 10.4 不建议每个 batch 重新随机划分字段

Per-token FFN 依赖稳定 token identity。Randomized Permutation Splitting 应在配置生成阶段固定一次并持久化，训练、评估和推理使用完全相同映射。

### 10.5 不建议直接把 block 从 2 层堆到 6 层

TokenMixer-Large 与 RankUp 都指出深层 RankMixer 存在梯度、残差语义和 effective rank 问题。必须先升级 residual/norm/FFN，再研究深度 scaling。

### 10.6 不建议只比较参数量

一个 1M 参数的碎片化 side module 可能比数千万参数的 fused grouped GEMM 更慢。最终取舍必须基于端到端吞吐和线上延迟。

---

## 11. 文献导出的优先结论

1. **最贴近当前数据和任务的改进是 RankUp-Lite。** 它使用 CVR、两层 RankMixer、超过 1200 个稀疏特征，实验条件与当前设置高度相似。
2. **最直接的 block 升级是 TokenMixer-Large Lite。** Mixing & Reverting 和 per-token SwiGLU 是其消融中贡献最大的组件，但必须做计算量匹配版本。
3. **DCNv2 只能以低维支路的形式尝试。** 显式乘法交互有科学互补性，但必须防止 MFU 被小算子拖累。
4. **SENET/MaskNet 应迁移为 token/channel 低秩门控。** 不应复现 1234 fields 的完整 pairwise 结构。
5. **MoE 适合在数据量和容量足够后进入。** 当前 5.5 亿条/天具备训练专家的样本基础，但路由、Grouped GEMM 和分布式通信是主要工程风险。
6. **CVR 的样本空间优先于 backbone 花样。** 如果存在 clicked-only 训练、entire-space 推理，ESMM 类目标升级应进入同一研究计划。
