# RankUp 论文详解：让 RankMixer 的参数增长转化为高秩表示

> 论文：**RankUp: Towards High-rank Representations for Large Scale Advertising Recommender Systems**  
> 作者：Jin Chen, Shangyu Zhang, Bin Hu, Chao Zhou, Junwei Pan, Gengsheng Xue, Wentao Ning, Gengyu Weng, Wang Zheng, Shaohua Liu, Zeen Xu, Chengyuan Mai, Tingyu Jiang, Lifeng Wang, Shudong Huang, Chengguo Yin, Haijie Gu, Jie Jiang  
> 初始提交：2026-04-20；本文按 arXiv v3 阅读  
> 原文：[arXiv Abstract](https://arxiv.org/abs/2604.17878) · [HTML](https://arxiv.org/html/2604.17878v3) · [PDF](https://arxiv.org/pdf/2604.17878v3)  
> 更长的逐图复现讨论见：[RankUp 论文图解与复现分析](04_rankup_paper_walkthrough.md)

## 1. 论文定位

RankUp 不认为 RankMixer 的主要问题是“参数不够多”，而是提出：

> 参数量增长后，token representation 使用的有效方向未必同步增加；Per-token FFN 甚至可能把 Token Mixer 扩展出的表示重新压缩。

因此，RankUp 研究的是 **representation capacity utilization**，而不是单纯的 parameter capacity。

它在 RankMixer / MetaFormer 主干外主要增加：

1. Randomized Permutation Splitting；
2. Multi-embedding；
3. Global Token；
4. Crossed Pre-trained Embedding Token；
5. Task-Specific Token；
6. PreNorm 与 Per-token SwiGLU。

这些组件分别针对输入冗余、几何视角不足、全局信息缺失、外部匹配先验不足和多任务负迁移。

---

## 2. Effective rank

设表示矩阵为：

$$
\mathbf H
\in
\mathbb R^{m\times n}.
$$

奇异值为：

$$
\sigma_1,\sigma_2,\ldots,\sigma_k,
\qquad
k=\min(m,n).
$$

归一化奇异值：

$$
p_i
=
\frac{\sigma_i}
{\sum_{j=1}^{k}\sigma_j}.
$$

Effective rank 定义为：

$$
\operatorname{erank}(\mathbf H)
=
\exp
\left(
-
\sum_{i=1}^{k}
p_i\log p_i
\right).
$$

### 2.1 为什么普通 rank 不够

两组奇异值：

$$
[100,1,1,1,1]
$$

与：

$$
[20,20,20,20,20]
$$

普通 rank 都为 5，但第一组几乎被单一方向支配，第二组均匀使用 5 个方向。Effective rank 能区分这种差异。

### 2.2 两种统计口径

单样本 token matrix：

$$
\mathbf H_b
\in
\mathbb R^{T\times D}.
$$

其 effective rank 上限是：

$$
\min(T,D).
$$

固定 token 跨 batch 的表示矩阵：

$$
\mathbf E_t
\in
\mathbb R^{B\times D}.
$$

它衡量一个 token 在样本维度上的 channel diversity。复现时应分开记录：

```text
sample_token_erank
batch_channel_erank
```

---

## 3. Figure 1：整体架构

<p align="center">
  <img src="https://arxiv.org/html/2604.17878v3/x1.png" width="94%" alt="RankUp Figure 1">
</p>

*图源：RankUp Figure 1。*

Figure 1 展示了多种 token 来源：

```text
Original sparse features
├── Global Token view
├── Randomized sparse-token view
└── Predefined interaction view

Dense pre-trained features
├── Dense Token
└── Dense Cross Token

User sequence
└── Seq Token

Tasks
└── Task-Specific Tokens
```

所有 token 被投影到统一维度后进入 MetaFormer / RankMixer blocks。

---

## 4. Randomized Permutation Splitting

设有 $M$ 个 sparse fields：

$$
\mathcal F
=
\{f_1,f_2,\ldots,f_M\}.
$$

生成字段级排列：

$$
\mathcal F_{\sigma}
=
\{f_{\sigma(1)},f_{\sigma(2)},\ldots,f_{\sigma(M)}\}.
$$

将排列后字段均衡分成 $T$ 组：

$$
G_1,G_2,\ldots,G_T.
$$

第 $t$ 组完整拼接字段 embeddings：

$$
\mathbf u_t
=
\operatorname{Concat}
\left(
\{\mathbf e_j:j\in G_t\}
\right).
$$

独立投影为 token：

$$
\mathbf x_t
=
\mathbf W_t\mathbf u_t+\mathbf b_t.
$$

### 4.1 与 RankMixer Autosplit 的区别

| 维度 | RankMixer | RankUp Random Split |
|---|---|---|
| 打乱粒度 | 通常不打乱，连续向量切分 | 字段索引固定随机排列 |
| embedding 完整性 | 边界可穿过 embedding | 每个 field embedding 保持完整 |
| token 组成 | 常受原特征顺序影响 | user/item/context 更均匀混合 |
| 目标 | 规则、硬件友好 tokenization | 降低 token 冗余、平衡有效秩 |

### 4.2 固定还是动态排列

论文没有充分披露 permutation 重采样周期。对于 Per-token FFN，最合理工程实现是：

- 模型版本级生成一次；
- 固定完整 permutation 数组；
- 训练、验证、推理使用同一映射；
- 不按 batch 或 epoch 动态打乱；
- 多个预注册 mapping seeds 用于验证稳定性。

动态打乱会使同一 token-specific projector / FFN 每一步接收不同字段子空间，破坏参数身份。

---

## 5. Figure 2：Mutual Information 差分

<p align="center">
  <img src="https://arxiv.org/html/2604.17878v3/figs/mi_different_matrix_cluster_48.png" width="48%" alt="RankUp MI 48 clusters">
  <img src="https://arxiv.org/html/2604.17878v3/figs/mi_different_matrix_cluster_64.png" width="48%" alt="RankUp MI 64 clusters">
</p>

*图源：RankUp Figure 2。*

连续 token 表示先用 K-means 离散，再估计互信息：

$$
I(X_i;X_j)
=
\sum_{a,b}
P(a,b)
\log
\frac{P(a,b)}{P_i(a)P_j(b)}.
$$

差分矩阵：

$$
\Delta\mathbf M
=
\mathbf M_{\mathrm{Random}}
-
\mathbf M_{\mathrm{Semantic}}.
$$

图中大面积蓝色表示随机分组后 token pair 的 MI 更低，支持“整体冗余下降”的假设。

但不能过度解释为每一对 token 都更独立，因为：

- 仍存在红色区域；
- K-means 估计依赖 cluster 数；
- 初始化、抽样和标准化都会影响 MI；
- 低 MI 不自动等于高任务价值。

---

## 6. Figure 3：Token effective rank

<p align="center">
  <img src="https://arxiv.org/html/2604.17878v3/figs/token_effective_rank.png" width="88%" alt="RankUp token effective rank">
</p>

*图源：RankUp Figure 3。*

随机分组后的 sparse tokens 整体 rank 更高、更平滑；语义分组中部分 token rank 明显偏低。

论文的解释是：

- 长尾、低覆盖字段可能被集中到一个语义 token；
- 高度相关字段形成共线性；
- 某些 token 大部分样本接近默认状态；
- token-specific FFN 获得的有效梯度不均衡。

随机分组把高频、低频、高基数和长尾字段分散到不同 tokens，使最低 token rank 上升，并降低 token 间 rank 方差。

真正应观察的不是最大 rank，而是：

```text
mean token erank
minimum token erank
std token erank
pairwise redundancy
```

---

## 7. Multi-embedding

传统单 embedding mapping：

$$
\psi:
\mathcal F
\rightarrow
\mathbb R^d.
$$

RankUp 为同一个字段提供多个独立表示视角：

$$
\mathbf e_j
=
\{\psi_k(f_j):\psi_k\in\mathcal K_j\}.
$$

例如同一个 item ID 可以拥有：

```text
local sparse-token embedding
Global Token embedding
interaction-token embedding
```

这些不是复制，而是独立参数空间。

### 7.1 为什么不等价于单表加宽

单表加宽仍由同一路径、同一 token 和同一梯度共同训练；多表可以：

- 进入不同 token；
- 接收不同下游梯度；
- 学习不同几何基底；
- 减少所有 token 共享同一 embedding manifold 的耦合。

### 7.2 成本

主要成本可能不在 dense backbone，而在：

- embedding table memory；
- optimizer state；
- parameter-server bandwidth；
- lookup latency；
- checkpoint 与 warm-start。

因此当前业务更适合 Selective Multi-embedding，而不是复制全部 1234 个字段。

---

## 8. Global Token

论文定义：

$$
\mathbf g
=
\operatorname{func}
\left(
\operatorname{Pool}
\left(
\{\operatorname{Embed}(f_i)\}_{i=1}^{M}
\right)
\right).
$$

初始序列：

$$
\mathbf H_0
=
[\mathbf g;\mathbf x_1;\ldots;\mathbf x_T].
$$

Global Token 与最终 Mean Pooling 不同：

- Global Token 在 backbone 前生成；
- 参与每一层 Token Mixing；
- 可以反复向 local tokens 广播全局信息；
- Mean Pooling 只在 backbone 后压缩输出。

论文把 Global Token 与 Multi-embedding 在部分表格中联合报告，因此两者独立贡献不能完全分离。

---

## 9. Crossed Pre-trained Embedding Token

若已有预训练 user/item dense embeddings：

$$
\mathbf z_u,
\mathbf z_i
\in
\mathbb R^{d_p}.
$$

构造逐元素交互：

$$
\mathbf e_{\mathrm{cross}}
=
\operatorname{Proj}
\left(
\mathbf z_u\odot\mathbf z_i
\right).
$$

它提供 Factorization-Machine 风格的匹配先验：只有 user 和 item 在同一 latent dimension 同时激活时，乘积才显著。

相对 RankMixer，Cross Token 的价值是：

- 不要求 backbone 从 sparse IDs 重新学习全部长期匹配语义；
- 把外部检索或预训练空间中的 user-item compatibility 直接注入；
- 对新广告、长尾 item 和高价值转化任务可能更有帮助。

没有外部预训练向量时，用当前 sparse tokens 构造 Hadamard cross 只能称为 RankUp-inspired，不能宣称严格复现。

---

## 10. Task-Specific Token

对于 $K$ 个任务，增加可学习 tokens：

$$
\{\mathbf t_1,\mathbf t_2,\ldots,\mathbf t_K\}.
$$

它们和共享 tokens 一起进入 backbone，但每个 task tower 只读取自己的 task token：

$$
\widehat y_k
=
\operatorname{Tower}_k
\left(
\mathbf t_k^{(L)},
\operatorname{Pool}(\mathbf H_L)
\right).
$$

Task Token 在表示空间中提供任务私有汇聚位置，与 MMoE / PLE 的参数路径解耦不同。

### 10.1 Figure 5

<p align="center">
  <img src="https://arxiv.org/html/2604.17878v3/x3.png" width="86%" alt="RankUp task token mutual information">
</p>

*图源：RankUp Figure 5。*

论文将 task representation 聚类后计算 cluster assignment $Z$ 与标签 $Y$ 的互信息：

$$
I(Z;Y)
=
\sum_{z,y}
P(z,y)
\log
\frac{P(z,y)}{P(z)P(y)}.
$$

带 Task Token 的表示在多个 cluster 粒度下与任务标签具有更高 MI，支持任务私有表示的假设。

---

## 11. Figure 4：层间 effective-rank 动态

<p align="center">
  <img src="https://arxiv.org/html/2604.17878v3/x2.png" width="88%" alt="RankUp layer-wise effective rank">
</p>

*图源：RankUp Figure 4。*

典型轨迹：

```text
Block 1 Token Mixer：rank 上升
Block 1 FFN：rank 下降
Block 2 Token Mixer：rank 再上升
Block 2 FFN：rank 再下降
```

论文称之为 damped oscillation。各消融曲线说明：

- Semantic Group：输入冗余传播到后层；
- Single Embedding：输入几何自由度不足；
- Subset Features：缺少全局信息；
- w/o Cross：缺少预训练匹配先验；
- Full RankUp：最终层仍保持更高 effective rank。

这张图要求实验不能只看最终 AUC，还要记录每个 mixer 和 FFN 后的 rank。

---

## 12. Table 1：组件消融

| 组件 | Order | Book | Add Service |
|---|---:|---:|---:|
| Randomized Permutation Split | +0.06% | +0.06% | +0.08% |
| Global Token + Multi-Embedding | +0.21% | +0.18% | +0.13% |
| Cross Embedding | +0.22% | +0.10% | +0.03% |
| Task Token | +0.09% | +0.02% | +0.02% |
| Full RankUp | +0.41% | +0.23% | +0.25% |

重要结论：

1. 不同任务受益组件不同；
2. 单项收益不能简单相加；
3. 组件存在能力重叠或负交互；
4. Global Token 与 Multi-embedding 没有被完全拆开；
5. Full RankUp 必须通过单组件消融后再组合。

---

## 13. 线上结果

论文在 Weixin Video Accounts、Official Accounts 和 Moments 上部署。代表结果包括：

| 场景 | Realtime AUC | CTCVR | GMV |
|---|---:|---:|---:|
| Video Accounts | +0.367% | +1.41% | +3.41% |
| Official Accounts | +0.331% | +0.21% | +4.81% |
| Moments | +0.269% | +0.87% | +2.12% |

新广告 GMV：

| 场景 | New Ads GMV |
|---|---:|
| Video Accounts | +5.83% |
| Official Accounts | +9.67% |
| Moments | +2.84% |

Order 任务 GMV：

| 场景 | Order Task GMV |
|---|---:|
| Video Accounts | +5.18% |
| Official Accounts | +7.18% |
| Moments | +4.79% |

冷启动和高价值任务收益更大，与“更丰富输入基底和外部匹配先验更能帮助弱监督场景”的解释一致，但论文没有给出新广告流量上的组件级消融。

---

## 14. 与 RankMixer 的逐项对比

| 维度 | RankMixer | RankUp |
|---|---|---|
| 研究重点 | 高效参数 / FLOPs scaling | 有效表示容量 scaling |
| Sparse tokenization | 连续 Autosplit | 字段级随机分组 |
| Embedding | 通常单视角 | Multi-embedding |
| Global context | 依赖 mixing / pooling | 显式 Global Token |
| 外部匹配先验 | 无专门设计 | Crossed Pre-trained Token |
| 多任务解耦 | 普通多塔 | Task-Specific Tokens |
| FFN | Per-token GELU FFN | Per-token SwiGLU |
| Norm | Post-LN | PreNorm |
| 诊断指标 | AUC、MFU、latency | effective rank、MI、AUC |
| 主要风险 | fixed mixing / residual / pooling | embedding 与 token 类型成本增加 |

---

## 15. 对当前电商推广搜 CVR 的适配

当前 1234 个字段分为 16 tokens 时：

$$
1234
=
2\times78
+
14\times77.
$$

分为 32 tokens 时：

$$
1234
=
18\times39
+
14\times38.
$$

推荐实验：

```text
RU-0  当前连续 Autosplit
RU-1  ordered field-aligned split
RU-2a/b/c  fixed random field split，3 个 mapping seeds
RU-3  stratified balanced split
RU-4  15 local + 1 global，或 31 local + 1 global
RU-5  selective multi-embedding
```

必须同时记录：

- minimum / mean / std token erank；
- pairwise cosine 与 MI；
- 每层 mixer / FFN 后 sample-token erank；
- per-token gradient norm；
- AUC、LogLoss、calibration；
- throughput 与 P99。

Random Split 若只在一个 mapping seed 上提升，不能视为结构性结论。

---

## 16. 局限与复现注意事项

1. 随机排列更新周期没有充分公开；
2. Figure 3 与 Figure 4 的 rank 统计口径需要谨慎区分；
3. Global Token 与 Multi-embedding 的独立收益不完全清楚；
4. Multi-embedding 的真实成本可能主要在 sparse tables；
5. Cross Token 依赖预训练 dense embeddings；
6. Task Token 只在多任务条件下成立；
7. MI 依赖 K-means 粒度和初始化；
8. 随机 split 缺少多 mapping seed 误差条；
9. Full RankUp 的组件收益存在重叠，不能简单累加；
10. 论文环境与当前 CVR Base 的 SENet/DCNv2 归纳偏置不同。

---

## 17. 一句话总结

RankUp 的核心贡献可以概括为：

> 不再只问 RankMixer 有多少参数，而是测量 token 表示真正使用了多少有效方向；再通过随机字段分组、多 embedding 视角、Global/Cross/Task Tokens 提高并保护表示的 effective rank。

它是 RankMixer 发展谱系中“从参数规模转向有效表示容量”的分支。
