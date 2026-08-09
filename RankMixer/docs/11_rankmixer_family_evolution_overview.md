# RankMixer 方法谱系总览：TokenMixer-Large、RankUp、MixFormer 与 UniMixer

> 本文不是单篇论文复述，而是把 RankMixer 及其后续代表工作放在统一的技术坐标系中，解释每篇论文到底修复了什么问题、哪些方向互补、哪些结论不能直接混用。

## 1. 方法谱系

```text
RankMixer
│
├── TokenMixer-Large
│     修复 residual 语义、深层优化与 MoE，继续扩大 dense capacity
│
├── RankUp
│     修复输入冗余与表示坍缩，提高 effective rank
│
├── MixFormer
│     把静态特征交互与行为序列建模统一到同一 backbone
│
└── UniMixer
      把固定 Token Mixing 改成结构化可学习 mixing
```

这些工作不是严格的前后替代关系。它们分别处理 RankMixer 的四个不同瓶颈：

| 论文 | 主要瓶颈 | 核心改造 |
|---|---|---|
| RankMixer | 工业排序模型难以高 MFU 扩展 | 固定 Token Mixing + Per-token FFN |
| TokenMixer-Large | residual 错位、深层梯度、MoE 不一致 | Mixing & Reverting、pSwiGLU、Pre-RMSNorm、SP-MoE |
| RankUp | 参数增长未转化为有效表示维度 | Random Split、Multi-embedding、Global/Cross/Task Tokens |
| MixFormer | dense 与 sequence 分离，计算预算相互竞争 | Query Mixer + 每层 Cross-Attention + UI decoupling |
| UniMixer | fixed mixing 不可学习，且受 $H=T$ 约束 | parameterized global-local mixing + UniMixing-Lite |

---

## 2. 统一起点：RankMixer 的基本计算图

RankMixer 把输入特征压缩为：

$$\mathbf X_0 \in \mathbb R^{B\times T\times D}.$$

一个 block 由：

$$\mathbf M_l = \operatorname{Mix}(\mathbf X_{l-1}),$$

$$\mathbf X_l = \operatorname{PFFN}(\mathbf M_l) + \text{residual / norm}$$

组成。

它建立了整个谱系的三个基础假设：

1. 推荐特征可以先组织为有限数量 tokens；
2. 跨 token 交换不一定需要动态 Self-Attention；
3. 大容量主要放在独立 Per-token FFN 中，适合 grouped GEMM。

后续工作并没有完全推翻这三点，而是分别质疑：

- fixed Mix 是否足够；
- residual 是否语义对齐；
- tokenizer 是否产生高秩表示；
- static tokens 是否应该和 sequence 分开；
- Mean Pooling 与任务塔是否充分利用 tokens。

---

## 3. 演进时间线

| 时间 | 方法 | 主要贡献 |
|---|---|---|
| 2025-07 | RankMixer | 统一硬件友好 ranking backbone，在线部署 1B dense model |
| 2026-02 | TokenMixer-Large | 修复 block 与 MoE，在线扩展到约 7B，离线到约 15B |
| 2026-02 | MixFormer | 统一 dense feature interaction 与行为序列 co-scaling |
| 2026-04 | UniMixer | 建立 attention / TokenMixer / FM 的统一 mixing 框架 |
| 2026-04 | RankUp | 从 effective rank 角度修复表示坍缩与 token 冗余 |

时间接近并不意味着技术上只有一条主线。TokenMixer-Large 与 MixFormer来自相似的 RankMixer 工程背景，但分别向“更大 dense model”和“dense-sequence unified model”扩展；RankUp 与 UniMixer则从表示理论和 mixing operator 角度提出替代方向。

---

## 4. 五种方法的结构图对照

### 4.1 RankMixer

<p align="center">
  <img src="https://arxiv.org/html/2507.15551v3/x1.png" width="86%" alt="RankMixer architecture">
</p>

```text
Autosplit tokens
    -> fixed Token Mixing
    -> Per-token FFN
    -> Mean Pooling
```

### 4.2 TokenMixer-Large

<p align="center">
  <img src="https://arxiv.org/html/2602.06563v2/x1.png" width="86%" alt="TokenMixer-Large architecture">
</p>

```text
Semantic tokens + Global Token
    -> Mix
    -> mixed-layout pSwiGLU
    -> Revert
    -> original-layout pSwiGLU
    -> aligned residual
```

### 4.3 RankUp

<p align="center">
  <img src="https://arxiv.org/html/2604.17878v3/x1.png" width="86%" alt="RankUp architecture">
</p>

```text
Randomized sparse tokens
+ multi-embedding
+ Global / Cross / Sequence / Task tokens
    -> MetaFormer / RankMixer backbone
```

### 4.4 MixFormer

<p align="center">
  <img src="https://arxiv.org/html/2602.14110v2/x1.png" width="86%" alt="MixFormer architecture">
</p>

```text
Non-sequential query heads
    -> HeadMixing + pSwiGLU
    -> Cross-Attention over sequence
    -> per-head Output Fusion
```

### 4.5 UniMixer

<p align="center">
  <img src="https://arxiv.org/html/2604.00590v1/x2.png" width="86%" alt="UniMixer architecture">
</p>

```text
Learnable Global Mixing
× Block-specific Local Mixing
    -> pSwiGLU / SP-MoE
```

---

## 5. Tokenization 维度对比

| 方法 | token 如何产生 | 是否保留字段完整性 | 是否加入新 token 类型 |
|---|---|---|---|
| RankMixer | embedding concat 后连续 Autosplit | 不强制 | 通常无 |
| TokenMixer-Large | semantic group-wise tokenizer | 通常更强调语义组 | Global Token |
| RankUp | 字段级随机排列后分组投影 | 是 | Global、Cross、Seq、Task 等 |
| MixFormer | 非序列向量连续切成 query heads | 不强制 | sequence 作为 K/V，不一定作为普通 token |
| UniMixer | 可沿用任意 tokenizer | 由前端决定 | 重点不在新增 token，而在 mixing 参数化 |

### 5.1 关键区别

RankMixer 与 MixFormer 的 tokenizer 更接近“连续向量切片”；RankUp 则明确在 field level 随机分组；TokenMixer-Large 强调 semantic group；UniMixer 对 tokenizer 相对中立。

因此，比较不同论文时必须避免把 tokenizer 和 backbone 同时改变后只归因于 mixer。

---

## 6. Mixing operator 对比

### 6.1 RankMixer / MixFormer Query Mixer

固定 reshape mixing：

$$\mathbf X \rightarrow \operatorname{reshape} \rightarrow \operatorname{transpose} \rightarrow \widetilde{\mathbf X}.$$

优点：无参数、无 score、规则高效。缺点：对所有样本和所有训练阶段固定。

### 6.2 TokenMixer-Large

Mixing pattern 基本不变，但增加 inverse operation：

$$\mathbf X \xrightarrow{\operatorname{Mix}} \mathbf M \xrightarrow{\operatorname{Nonlinear}} \mathbf M' \xrightarrow{\operatorname{Revert}} \mathbf R.$$

重点是语义对齐，而不是学习新 pattern。

### 6.3 RankUp

RankUp 不直接修改 Token Mixing，而是修改进入 mixer 的表示分布。它相信：

$$\text{better input basis} \Rightarrow \text{higher effective rank after mixing}.$$

### 6.4 MixFormer

在 fixed HeadMixing 外增加动态序列 Cross-Attention：

$$\operatorname{softmax} \left( \frac{QK^\top}{\sqrt d} \right)V.$$

因此非序列 head mixing 固定，序列读取按样本动态。

### 6.5 UniMixer

将 fixed permutation 改为可训练矩阵：

$$\mathbf y = \mathbf x \mathbf W_{\mathrm{mix}}.$$

再用低秩、basis 和 Sinkhorn 约束保持计算与结构可控。

---

## 7. FFN 与参数隔离对比

| 方法 | 非线性模块 | 参数隔离方式 |
|---|---|---|
| RankMixer | GELU Per-token FFN | 每 token 独立 |
| TokenMixer-Large | 两组 Per-token SwiGLU | mixed layout 与 original layout 独立 |
| RankUp | Per-token SwiGLU | 保留 token-specific 参数 |
| MixFormer | Per-head SwiGLU | Query Mixer 与 Output Fusion 独立 |
| UniMixer | Per-token SwiGLU / SP-MoE | local block / token-specific |

所有后续工作都没有回到“所有 token 共享一个普通 FFN”。这说明 token-specific nonlinear capacity 是 RankMixer 谱系中最稳定的共识之一。

TokenMixer-Large、MixFormer、RankUp 和 UniMixer普遍使用 SwiGLU，反映另一个共识：乘法 gate 比纯 GELU FFN 更适合超大 ranking backbone。

---

## 8. 归一化与 residual 对比

| 方法 | Norm | Residual 重点 |
|---|---|---|
| RankMixer | Post-LayerNorm | 简单相邻 residual |
| TokenMixer-Large | Pre-RMSNorm | Revert 后语义对齐，增加 inter-residual |
| RankUp | PreNorm | 保护高秩表示与深层稳定性 |
| MixFormer | Pre-RMSNorm | Query、Cross-Attention、Fusion 多子层 residual |
| UniMixer | SiameseNorm | 协调 global/local mixing 与 residual 统计 |

Post-Norm 在初始 RankMixer 中足以支撑两层模型，但后续工作普遍转向 Pre-Norm 或定制 Norm。这说明加深与增大后，优化稳定性成为主要瓶颈。

---

## 9. 表示容量问题：参数量并不等于有效秩

RankMixer 的 dense 参数近似为：

$$P \approx 2kLTD^2.$$

这一公式只描述参数数量，不描述参数是否产生互补表示。

RankUp 使用 effective rank：

$$\operatorname{erank}(\mathbf H) = \exp \left( - \sum_i p_i\log p_i \right),$$

$$p_i = \frac{\sigma_i}{\sum_j\sigma_j}.$$

它发现 RankMixer 常出现：

```text
Token Mixer 后 effective rank 上升
Per-token FFN 后 effective rank 下降
```

UniMixer 则从 mixing matrix 表达力解释相似问题：固定 permutation 可能无法随数据学习最优交互。

两者分别给出两个互补诊断：

- RankUp：检查表示最终使用了多少有效方向；
- UniMixer：检查 mixing operator 是否具有足够自由度。

---

## 10. Scaling 目标对比

| 方法 | 主要 scaling 轴 | 论文关注的 ROI |
|---|---|---|
| RankMixer | $T,D,L,E$ | 参数 / FLOPs 增长时保持低延迟和高 MFU |
| TokenMixer-Large | depth、dense width、SP-MoE | 扩到 7B/15B 并稳定训练 |
| RankUp | representation rank、token diversity | 参数增加真正转化为表示容量 |
| MixFormer | dense width + sequence length | 统一预算下 co-scaling |
| UniMixer | mixing capacity + depth | 提高参数与 FLOPs scaling exponent |

因此“谁 scaling 更好”必须先说明 scaling 的对象：

- 是 dense 参数？
- active FLOPs？
- 行为序列长度？
- effective rank？
- mixing pattern capacity？

不同论文的 scaling curves 不能只看横轴数值直接比较。

---

## 11. 代表性实验结果横向整理

### 11.1 RankMixer

- 约 107M 模型在核心任务上取得约 +0.64% 到 +1.33% 相对提升；
- 约 1.1B 模型取得约 +0.95% 到 +1.82%；
- MFU 从约 4.5% 提升到约 45%；
- 1B 模型在线延迟与旧小模型接近。

### 11.2 TokenMixer-Large

- 电商约 500M 模型相对增益约 +0.94%；
- 约 4B 为 +1.14%；
- 约 7B 为 +1.20%；
- SP-MoE 以约 2.3B active 参数接近 4B dense 效果；
- 线上订单、GMV、广告和直播指标均有提升。

### 11.3 RankUp

- Randomized Split 在三个任务上约 +0.06% 到 +0.08% Realtime AUC；
- Full RankUp 约 +0.23% 到 +0.41%；
- 三个微信广告场景 GMV 显著提升；
- 新广告和高价值 Order 任务收益更明显。

### 11.4 MixFormer

- Medium 模型在 Finish / Skip AUC 与 UAUC 上显著优于解耦基线；
- UI-MixFormer 约减少 36% FLOPs，并有超过 30% serving speedup；
- Douyin / Douyin Lite 活跃、时长、互动指标提升。

### 11.5 UniMixer

- UniMixer-Lite 用约 38.2M 参数达到 0.752327 AUC，高于约 135.5M RankMixer 的 0.749329；
- 参数与 FLOPs scaling exponent 均高于 RankMixer；
- 4/8 blocks 继续改善，而 RankMixer 从 2 到 4 blocks 出现下降；
- 快手长期留存在线指标获得明显提升。

这些结果来自不同数据、不同指标、不同基线，不能按数值大小直接排名。其价值在于验证各自提出的机制。

---

## 12. 五种方法的核心消融证据

| 方法 | 最关键消融 | 论文内结论 |
|---|---|---|
| RankMixer | 去掉 Token Mixing，约 -0.50% | 固定跨 token 交换是核心 |
| RankMixer | Per-token 改 shared，约 -0.31% | token-specific FFN 必要 |
| TokenMixer-Large | 去掉 Mixing & Reverting，约 -0.27% | residual 语义对齐是主要升级 |
| TokenMixer-Large | pSwiGLU 改 shared，约 -0.21% | 参数隔离继续重要 |
| RankUp | Random Split / Global+Multi-Emb / Cross / Task Token | 多种输入扩秩组件分别有效，但存在重叠 |
| MixFormer | Per-head Output Fusion 改 shared，约 -0.06% | sequence 证据需要 head-specific 融合 |
| UniMixer | 去 temperature，约 -0.1645% | mixing pattern 的优化轨迹关键 |
| UniMixer | 去 warm-up，约 -0.0856% | 不能过早离散化 mixing |

---

## 13. 哪些组件可以组合

### 13.1 RankMixer + TokenMixer-Large block

这是最直接组合：保留当前 tokenizer 和任务头，只替换 block。

### 13.2 RankUp tokenizer + TokenMixer-Large block

理论上高度互补：

```text
RankUp 改善输入 basis 与 token diversity
TokenMixer-Large 改善 block residual 与深层优化
```

应先分别验证，再做 2×2 factorial。

### 13.3 RankUp tokenizer + UniMixer

也具有互补性：

```text
Random Split 降低初始冗余
Learnable Mixing 学习后续交互模式
```

但若两者都改变 token geometry，可能存在重叠，需要比较 ordered field split、random split 和 learned mixing 的独立贡献。

### 13.4 TokenMixer-Large + MixFormer

可把 MixFormer Query Mixer 替换为 Mixing & Reverting block，并把 pSwiGLU、Pre-RMSNorm 和 SP-MoE引入统一 dense-sequence backbone。论文尚未完整验证这一组合。

### 13.5 UniMixer + MixFormer

可把固定 Query HeadMixing 替换为 learnable structured mixing，同时保留 Cross-Attention。风险是：

- query mixing 更复杂；
- 与序列 attention 同时学习可能增加优化不稳定；
- UI mask 需要映射到结构化 mixing matrix 中。

---

## 14. 哪些组件不应一次性堆叠

不建议首轮直接构建：

```text
Random Split
+ Global Token
+ Multi-embedding
+ Mixing & Reverting
+ UniMixing
+ Cross-Attention
+ SP-MoE
+ Attention Pooling
```

原因不是这些组件理论上冲突，而是：

- 无法归因；
- 失败时无法定位；
- 参数、FLOPs、训练周期同时变化；
- token 数、语义和 residual 坐标系可能一起改变；
- 当前 Base 的 SENet/DCNv2 能力还未被公平控制。

正确方法是先构造最小可证伪实验，再组合两个已独立获益的模块。

---

## 15. 面向当前电商推广搜 CVR 的适配判断

当前已知输入：

$$\begin{aligned} \mathbf E_u &\in \mathbb R^{B\times385\times17},\\ \mathbf E_i &\in \mathbb R^{B\times835\times17},\\ \mathbf E_c &\in \mathbb R^{B\times14\times17}. \end{aligned}$$

当前两种 RankMixer：

$$(T,D,L) \in \{(16,768,2),(32,1536,2)\}.$$

强 Base 为：

```text
bucket-wise BN
-> hierarchical SENet
-> DCNv2
-> MLP
```

### 15.1 最适合立即研究

1. TokenMixer-Large 的 Mixing & Reverting；
2. 计算匹配 Per-token SwiGLU；
3. Pre-RMSNorm 与 down small init；
4. RankUp fixed random field split；
5. Global Adapter 或保持总 token 数的 Global Token；
6. 简单 learned token mixing，再决定是否完整实现 UniMixer；
7. Base-preserving residual hybrid。

### 15.2 有条件研究

MixFormer 需要当前已有的用户行为序列。若训练样本本来就包含可用序列，可进入完整实验；若没有，不应新增虚构 sequence。

UI-MixFormer 的单向 user-item mask 即使没有序列也可以借鉴，因为它与 Base 的层级条件 SENet方向一致。

### 15.3 暂不优先

- 全字段 Multi-embedding；
- SP-MoE；
- 8 层以上深模型；
- 完整 UniMixer + MixFormer 组合；
- 在未证明 dense capacity 有效前继续增大参数。

---

## 16. 推荐研究路线

### Phase A：确认 RankMixer 基础块问题

```text
A0  原始 RankMixer
A1  + Mixing & Reverting
A2  + compute-matched pSwiGLU
A3  + Pre-RMSNorm
A4  + down projection small init
```

### Phase B：确认输入表示问题

```text
B0  ordered Autosplit
B1  ordered field-aligned split
B2  fixed random field split × 3 seeds
B3  domain-preserving tokens
B4  local + global token
```

### Phase C：确认 fixed mixing 问题

```text
C0  fixed Token Mixing
C1  learned T×T matrix
C2  low-rank learned mixing
C3  Sinkhorn + temperature
C4  UniMixing-Lite
```

### Phase D：恢复当前业务归纳偏置

```text
D0  Base BN + RankMixer
D1  + hierarchical SENet
D2  + low-rank DCNv2 adapter
D3  Base + zero-init RankMixer residual
```

### Phase E：有序列时研究 MixFormer

```text
E0  sequence tower -> RankMixer
E1  parallel sequence + RankMixer
E2  MixFormer Query Mixer + Cross-Attention
E3  UI-MixFormer request-level reuse
```

---

## 17. 统一评价维度

### 17.1 预测指标

```text
AUC / GAUC / UAUC
LogLoss
PR-AUC
CVR calibration bias
ECE / Brier Score
```

### 17.2 表示指标

```text
sample-token effective rank
batch-channel effective rank
minimum / mean / std token erank
pairwise token cosine similarity
mutual information
pooling weight entropy
```

### 17.3 优化指标

```text
per-token gradient norm
parameter update-to-weight ratio
activation RMS
NaN / Inf counter
gradient clipping frequency
expert load balance
```

### 17.4 系统指标

```text
examples per second
MFU
peak HBM
kernel count
active FLOPs
P50 / P95 / P99 latency
request-level candidate reuse rate
```

每个论文提出的机制都应有对应指标。例如：

- RankUp 必须看 effective rank 与 redundancy；
- TokenMixer-Large 必须看梯度、稳定性和深度收益；
- MixFormer 必须看 sequence length scaling 与 request-level cost；
- UniMixer 必须看 mixing sparsity、temperature 和 scaling exponent。

---

## 18. 最终横向结论

| 问题 | 最直接的方法 |
|---|---|
| 当前 block residual 语义错位 | TokenMixer-Large |
| 模型加深后不稳定 | TokenMixer-Large |
| 参数很多但 token 表示低秩 | RankUp |
| token 之间冗余严重 | RankUp |
| 固定 mixing pattern 不适配数据 | UniMixer |
| $H=T$ 限制配置空间 | UniMixer / TokenMixer-Large |
| 静态特征与行为序列割裂 | MixFormer |
| 多候选重复计算 | UI-MixFormer |
| 需要最高硬件效率的静态 ranking backbone | RankMixer / TokenMixer-Large |
| 需要保留当前强 Base | Base-preserving RankMixer residual |

---

## 19. 阅读导航

- [RankMixer 论文详解](07_rankmixer_paper_detailed_review.md)
- [TokenMixer-Large 论文详解](08_tokenmixer_large_paper_detailed_review.md)
- [MixFormer 论文详解](09_mixformer_paper_detailed_review.md)
- [UniMixer 论文详解](10_unimixer_paper_detailed_review.md)
- [RankUp 图解与复现分析](04_rankup_paper_walkthrough.md)

---

## 20. 一句话总结

RankMixer 方法谱系不是围绕一个模块不断堆叠，而是在四个正交方向演进：

> TokenMixer-Large 让 block 更大、更深、更稳；RankUp 让表示真正高秩；MixFormer 让静态与序列共同扩展；UniMixer 让固定 mixing 变成结构化可学习。

对当前电商推广搜 CVR，最合理的研究方式是先确定实际瓶颈属于哪一个方向，再选择对应方法，而不是把所有后续组件一次性加入。
