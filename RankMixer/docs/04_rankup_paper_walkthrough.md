# RankUp 论文图解：从 RankMixer 的表示坍缩到高秩表示学习

> 论文：**RankUp: Towards High-rank Representations for Large Scale Advertising Recommender Systems**  
> 作者：Jin Chen, Shangyu Zhang, Bin Hu, Chao Zhou, et al.，Tencent Inc.  
> 阅读版本：arXiv v3，2026-05-12  
> 原文：[arXiv HTML](https://arxiv.org/html/2604.17878v3) · [PDF](https://arxiv.org/pdf/2604.17878v3)  
> 本文定位：结合论文 Figure 1-5 与 Table 1-4，对 RankUp 的动机、结构、实验和复现边界进行逐图分析。

## 1. 一句话理解 RankUp

RankUp 的核心观点不是“RankMixer 的 mixing 不够复杂”，而是：

> **模型参数变多，不等于模型真正使用了更多表示维度。**

RankMixer 的 Token Mixing 会阶段性扩展 token 表示的秩，但 Per-token FFN 往往再次压缩表示；随着层数增加，effective rank 呈现“上升—下降—再上升—再下降”的阻尼振荡。RankUp 因此不把主要精力放在设计更复杂的 Mixer，而是从输入表示、全局上下文、外部先验和多任务解耦四个方向扩大并保护 latent space：

1. Randomized Permutation Splitting：降低 token 间冗余；
2. Multi-embedding：增加同一特征的几何自由度；
3. Global Token：给所有 local token 提供全局上下文；
4. Cross Pre-trained Embedding Token：注入 user-item 交互先验；
5. Task-Specific Token：减少多任务梯度相互挤压；
6. PreNorm + SwiGLU：提高深层训练稳定性和 FFN 表达能力。

这些组件最终仍运行在 MetaFormer / RankMixer 风格的 Token Mixer + Per-token FFN 主干上。

---

## 2. RankUp 要解决的不是代数秩，而是 effective rank

### 2.1 为什么普通 rank 不够

设某层单个样本的 token 矩阵为：

```math
\mathbf{H}_l\in\mathbb{R}^{T\times D}.
```

普通矩阵秩只要遇到极小噪声就可能变成满秩，无法反映“大部分信息是否被少数主方向支配”。RankUp 使用奇异值分布的熵定义 effective rank：

```math
p_i=\frac{\sigma_i}{\sum_{j=1}^{k}\sigma_j},\qquad
\operatorname{erank}(\mathbf{H}_l)
=\exp\left(-\sum_{i=1}^{k}p_i\log p_i\right),
\qquad k=\min(T,D).
```

直观上：

- 若只有一个奇异值占绝对优势，effective rank 接近 1；
- 若多个奇异值较均匀，effective rank 接近其可用上限；
- effective rank 越高，表示越不容易集中在少数相似方向上。

### 2.2 一个对当前模型非常重要的上限

论文在层间动态分析中，对每个样本的 $T\times D$ token 矩阵计算 effective rank，因此它的理论上限是 $\min(T,D)$，而不是隐藏维度 $D$。

你的当前基线为 $T=16$、$D=768$，所以单样本层间 effective rank 的上限只有 16。论文 Figure 4 中的值接近 40-48，是因为论文系统包含 32 个 sparse tokens 以及 dense、sequence、cross、task 等非 sparse tokens，总 token 数明显大于 16。**论文的绝对数值不能直接和你的 16-token 模型横向比较。**

建议同时报告归一化指标：

```math
\operatorname{nerank}(\mathbf{H}_l)
=\frac{\operatorname{erank}(\mathbf{H}_l)}{\min(T,D)}.
```

### 2.3 Figure 3 与 Figure 4 的 rank 口径不要混用

Figure 4 明确使用单样本 token 矩阵 $\mathbf{H}_b\in\mathbb{R}^{T\times D}$；Figure 3 则描述“每个 token 的 embedding matrix”，图中数值超过 100，显然不可能来自一个 token 维度为 1 的单样本矩阵。更合理的理解是：Figure 3 对固定 token 在 batch / 样本维度上形成的矩阵计算 effective rank，例如 $\mathbf{E}_t\in\mathbb{R}^{B\times D}$。

论文没有把 Figure 3 的精确计算张量写得像 Figure 4 那样清楚，因此复现时应把两类指标分开命名：

- `sample_token_erank`：每个样本上的 $T\times D$；
- `batch_channel_erank`：每个 token 跨样本的 $B\times D$。

---

## 3. Figure 1：RankUp 整体架构逐层拆解

![Figure 1: Overall Framework of RankUp](https://arxiv.org/html/2604.17878v3/x1.png)

*图源：RankUp Figure 1，arXiv v3。图片加载失败时可打开[原图](https://arxiv.org/html/2604.17878v3/x1.png)。*

Figure 1 应从下往上阅读。

### 3.1 Input Features：三类输入源

底层输入分为三大类：

1. **Original Features**：user、item、context 等大规模 sparse fields；
2. **Dense Features**：例如外部预训练 user/item embeddings；
3. **User sequence behaviors**：由独立序列编码器建模的历史行为。

RankUp 并没有要求把所有输入都强行变成同一种 sparse token。序列特征仍可由 sequence encoder 产生 Seq Token，dense 先验也可以单独产生 Dense Token / Dense Cross。

### 3.2 Embedding Layers：不同 token 使用不同表示来源

图中至少包含三套 embedding 视角：

- **Embedding Table 1**：服务 Global Token Group，面向全局聚合；
- **Embedding Table 2**：服务 Shuffle & Grouping，形成随机划分后的 sparse tokens；
- **Embedding Table 3**：服务部分特征或预定义交互，提供额外几何视角；
- Dense Features 直接形成 Dense Token，并通过显式 cross 形成 Dense Cross；
- Sequential Modelling 形成 Seq Token。

这张图传达的关键不是“必须恰好三张表”，而是**不同 token 不必共享同一个低维 embedding 视角**。同一原始信号可以通过独立参数空间进入不同 token，从而扩大初始表示矩阵的自由度。

### 3.3 Tokens：RankUp 扩展了 token 的类型

图中的 token 序列包括：

- Global Token；
- Sparse Token 1 到 Sparse Token N；
- Predefined Feature Interactions；
- Dense Token；
- Dense Cross；
- Seq Token；
- Task Token 1 到 Task Token N。

其中真正属于 RankUp 主要贡献的是随机 sparse token、multi-embedding、global token、cross token 和 task token。Seq Token 更像对现有序列塔输出的统一接入方式。

### 3.4 Stacked Blocks：主干仍是 MetaFormer

每个 block 采用 PreNorm 风格：

```math
\mathbf{H}'_l
=\operatorname{TokenMixer}(\operatorname{Norm}(\mathbf{H}_{l-1}))
+\mathbf{H}_{l-1},
```

```math
\mathbf{h}_{l,i}
=\operatorname{SwiGLU}_i(\operatorname{Norm}(\mathbf{h}'_{l,i}))
+\mathbf{h}'_{l,i}.
```

需要注意两点：

- RankUp 没有用 Self-Attention 替换 RankMixer 的 Token Mixing；
- FFN 改为 Per-token SwiGLU，并把 Norm 放在子层之前，以改善优化稳定性。

### 3.5 Task Towers：共享信息与任务私有信息同时进入塔

每个任务塔既读取共享 backbone 的 pooled representation，也读取自己的 task token：

```math
y^{(k)}
=\operatorname{Tower}^{(k)}\left(
\mathbf{x}^{(k)}_{\mathrm{task}},
\operatorname{Pool}(\mathbf{H}'_L)
\right).
```

这不是把每个任务完全拆成独立模型，而是在共享 backbone 内保留一条任务私有的表示通道。

---

## 4. Randomized Permutation Splitting：为什么随机分组可能优于语义分组

### 4.1 方法

给定 $M$ 个 sparse fields：

```math
\mathcal{F}=\{f_1,f_2,\ldots,f_M\},
```

先生成一个随机排列：

```math
\mathcal{F}_{\sigma}
=\{f_{\sigma(1)},f_{\sigma(2)},\ldots,f_{\sigma(M)}\},
```

再按顺序均匀分成 $T$ 组，每组完整拼接字段 embedding 并独立投影成 token。

### 4.2 科学假设

语义分组把高度相关、低基数或长尾字段放进同一个 token，可能造成：

- token 内特征共线；
- 不同 token 的信息量非常不均衡；
- 某些 token 的 PFFN 获得大量有效梯度，另一些 token 长期低活跃；
- 初始 token 矩阵的几何基底不充分。

随机分组并不是认为业务语义没有价值，而是用“打散相关性”换取更均衡的 token 信息密度。

### 4.3 论文没有说明的关键实现细节

论文使用 stochastic permutation operator 描述方法，但没有清楚说明 permutation 是：

- 模型初始化时生成一次并永久固定；
- 每次训练任务重新生成；
- 每个 batch 动态重采样。

对于 RankMixer 的 Per-token FFN，token identity 必须稳定，因此工程上更合理的首版是：**固定字段级 permutation、固定 seed、训练与推理共享同一映射**。每 batch 动态打乱会让第 $i$ 个 PFFN 持续接收不同语义子空间，破坏 token-specific 参数的意义。

---

## 5. Figure 2：MI 差分热力图证明了什么

<p align="center">
  <img src="https://arxiv.org/html/2604.17878v3/figs/mi_different_matrix_cluster_48.png" width="48%" alt="Figure 2a: MI difference matrix with 48 clusters">
  <img src="https://arxiv.org/html/2604.17878v3/figs/mi_different_matrix_cluster_64.png" width="48%" alt="Figure 2b: MI difference matrix with 64 clusters">
</p>

*图源：RankUp Figure 2(a)(b)，arXiv v3。原图：[48 clusters](https://arxiv.org/html/2604.17878v3/figs/mi_different_matrix_cluster_48.png) · [64 clusters](https://arxiv.org/html/2604.17878v3/figs/mi_different_matrix_cluster_64.png)。*

### 5.1 MI 的估计流程

对 batch 表示 $\mathbf{E}\in\mathbb{R}^{B\times T\times D}$，论文将每个 token 的连续表示用 K-means 离散成 $K$ 个 cluster，再计算 token $i$ 与 token $j$ 的离散互信息：

```math
M_{ij}
=\sum_{a=1}^{K}\sum_{b=1}^{K}
P(a,b)\log\frac{P(a,b)}{P_i(a)P_j(b)}.
```

比较矩阵定义为：

```math
\Delta\mathbf{M}
=\mathbf{M}_{\mathrm{Randomized}}
-\mathbf{M}_{\mathrm{Semantic}}.
```

因此：

- 蓝色：随机分组的 MI 更低，即 token 对更少冗余；
- 红色：随机分组的 MI 更高；
- 接近白色：差异较小。

### 5.2 如何读图

论文系统中 Token ID 0-31 是 32 个 sparse tokens，Token ID 32-46 是其他 dense、sequence、cross 或 task tokens。

图中最明显的现象是：

- sparse-sparse 的下三角区域大面积为蓝色；
- sparse 与非 sparse token 的交叉区域也出现较多蓝色；
- 非 sparse tokens 的构造在两种方法中没有改变，所以对应区域差异相对小；
- $K=48$ 与 $K=64$ 时图案相似，说明结论不完全依赖某一个聚类粒度。

### 5.3 不应过度解读

这张图不能证明“随机分组让每一对 token 都更独立”。热力图中仍有红色区域，说明部分 token 对的 MI 上升。它支持的是**整体冗余下降和分布更均衡**，而不是逐元素单调改善。

此外，MI 是通过 K-means 离散化估计的，会受以下因素影响：

- cluster 数 $K$；
- K-means 初始化；
- 抽样规模；
- 表示标准化方式；
- 高频与长尾样本比例。

复现时至少应使用多个 K-means seed，并同时报告均值与方差。

---

## 6. Figure 3：随机分组如何改善 token 级 effective rank

![Figure 3: Effective rank comparison of token embeddings](https://arxiv.org/html/2604.17878v3/figs/token_effective_rank.png)

*图源：RankUp Figure 3，arXiv v3。图片加载失败时可打开[原图](https://arxiv.org/html/2604.17878v3/figs/token_effective_rank.png)。*

### 6.1 主要现象

- 蓝色的 Random Splitting 在 32 个 sparse tokens 上整体更高、更平滑；
- 红色的 Semantic Splitting 波动很大；
- 论文特别指出 Token 12、29 和 31 的 semantic rank 低于 20；
- 随机分组减少了“一个 token 被一批相似长尾字段占满”的情况。

### 6.2 为什么长尾字段集中会降秩

假设某个语义组主要由低覆盖、低基数、强相关字段组成。跨 batch 观察这个 token 时，大量样本的激活模式接近，token 表示会集中在少数主方向上。即使 token 的维度很高，真正被使用的方向仍很少。

随机分组将这些字段分散到多个 token，并与更高覆盖、更高变化的字段混合，使各 token 的 batch-level 表示更丰富。

### 6.3 证据强度与缺口

Figure 3 对 RankUp 的输入设计提供了直接诊断证据，但论文没有报告：

- 多个 random permutation seed 的误差条；
- 不同字段覆盖率分桶下的 rank；
- random split 是否影响线上特征增删和 warm-start；
- token projector / PFFN 的梯度均衡性。

因此你的复现不能只跑一个随机 seed。至少应使用 3 个固定 mapping，并检查收益是否具有一致方向。

---

## 7. Multi-embedding：扩大输入表示的“基底数量”

传统单 embedding 表可写为：

```math
\psi:\mathcal{F}\rightarrow\mathbb{R}^{d}.
```

RankUp 对部分或全部特征引入 $K$ 套独立 embedding 表：

```math
\mathbf{e}_j
=\left\{\psi_k(f_j)\mid \psi_k\in\mathcal{K}_j\right\}.
```

同一个离散 ID 因而拥有多个独立几何视角。它和“把一张 embedding 从 17 维加宽到 34 维”并不完全等价：独立表可以进入不同 token、接受不同下游参数和梯度，从而减少输入阶段过早耦合。

### 7.1 为什么它可能有效

- 增加初始表示自由度；
- 为稀有信号提供多条梯度路径；
- 使不同 token 不必从同一个 embedding manifold 出发；
- Figure 4 中 Single Embedding 版本在后层 effective rank 最低，说明输入瓶颈会沿深度传播。

### 7.2 为什么它是高成本组件

如果完整复制 user_id、item_id 等超大词表，新增成本主要来自：

- embedding 参数与 optimizer state；
- 参数服务器或分片显存；
- lookup 带宽；
- checkpoint 与 warm-start；
- online cache 命中率。

因此在你的系统中，首版更适合 **Selective Multi-embedding**：只为 user_id、item_id、shop_id、category_id、query_id、creative_id 等 anchor fields 增加第二视角，而不是复制全部 1234 个字段。

---

## 8. Global Token：给 local tokens 提供全局摘要

论文定义：

```math
\mathbf{g}
=A(f_1,f_2,\ldots,f_M)
=\operatorname{func}\left(
\operatorname{Pool}\left(\{\operatorname{Embed}(f_i)\}_{i=1}^{M}\right)
\right).
```

`func` 可以是 MLP，也可以是 FM、FFM 或 DCNv2 等显式交互模块。Global Token 被追加到初始 token 序列：

```math
\mathbf{H}^{(0)}
=[\mathbf{g},\mathbf{e}_1,\ldots,\mathbf{e}_T].
```

### 8.1 Global Token 与 mean pooling 不同

- Global Token 在 backbone **之前**生成并参与每一层 mixing；
- Mean Pooling 通常只在 backbone **之后**聚合输出；
- Global Token 可以在多层中把全局信息反复广播给 local tokens；
- 它不是单纯换一个最终 pooling。

### 8.2 Global Token 的风险

如果 global branch 太强，所有 token 都可能依赖同一个摘要，反而提高 token 间相似度。应监控：

- Global Token 与各 local token 的 cosine similarity；
- 去掉 global token 后的 rank 动态；
- global branch 输出范数；
- global token 是否压制 creative / tail item token。

---

## 9. Cross Integration：把预训练相似度先验变成显式交互 token

预训练 user/item embeddings 通常来自 two-tower 检索模型，其目标偏向距离或相似度。RankUp 不只做拼接，而是构造逐元素乘积：

```math
\mathbf{e}_{\mathrm{cross}}
=\operatorname{Proj}\left(
\mathbf{z}_{ue}\odot\mathbf{z}_{ie}
\right).
```

这个操作具有 Factorization Machine 风格的乘法归纳偏置：某个维度只有在 user 与 item 同时激活时才产生大值。

### 9.1 它补足了什么

- 原始 sparse tokens 主要表达离散字段组合；
- 预训练 embeddings 包含跨场景、跨任务或长周期语义；
- Hadamard product 把“两个向量各自是什么”转化为“这对 user-item 在哪些隐空间方向上匹配”。

### 9.2 没有预训练向量时不能声称复现原方案

若系统没有独立训练的 user/item dense embeddings，可以尝试：

```math
\widetilde{\mathbf{e}}_{\mathrm{cross}}
=\operatorname{Proj}\left(
\operatorname{Pool}(\mathbf{H}_{user})
\odot
\operatorname{Pool}(\mathbf{H}_{item})
\right),
```

但这只是 RankUp-inspired 的内部 cross token，不是论文中的 Cross Pre-trained Embedding。实验报告中必须明确区分。

---

## 10. Figure 4：层间 effective-rank 动态是整篇论文最关键的证据

![Figure 4: Layer-wise Effective Rank Evolution](https://arxiv.org/html/2604.17878v3/x2.png)

*图源：RankUp Figure 4，arXiv v3。图片加载失败时可打开[原图](https://arxiv.org/html/2604.17878v3/x2.png)。*

横轴依次为：

```text
block1-TM -> block1-FFN -> block2-TM -> block2-FFN
```

### 10.1 图中的共同模式

所有曲线都表现出：

1. Token Mixer 后 effective rank 较高；
2. FFN 后明显下降；
3. 下一层 Token Mixer 再次抬升；
4. 最后一层 FFN 再次压缩。

这就是论文所说的 damped oscillation：Mixer 扩秩，FFN 缩秩，而且后层 FFN 的压缩往往更严重。

### 10.2 各消融曲线的角色

- **RankUp**：最终 FFN 后仍保持最高 effective rank；
- **Semantic Group**：说明取消 randomized split 后，输入冗余会传播到后层；
- **Single Embedding**：最终下降最明显，说明 multi-embedding 主要解决初始表示自由度不足；
- **Subset Features**：论文用它说明缺少全局信息时，深层 rank 更容易持续衰减；
- **w/o Cross**：去掉预训练 cross 先验会在较早阶段损失表示丰富度。

### 10.3 这张图给工程实验的直接启示

只记录最终 pooled vector 的 AUC 不足以判断 RankUp 是否按预期工作。至少要在以下位置记录：

```text
tokenizer output
block1 Token Mixer + residual
block1 FFN + residual
block2 Token Mixer + residual
block2 FFN + residual
```

如果某组件提升 AUC，却没有改善它声称要解决的 rank / redundancy 指标，应考虑收益是否来自单纯参数增加、优化变化或数据随机性。

---

## 11. Task-Specific Token：多任务环境中的表示解耦

共享多任务模型中，不同目标会把共享表示拉向不同方向。高频或梯度更大的任务可能主导 latent space，使弱任务只能使用被压缩后的子空间。

RankUp 为每个任务增加可学习 token：

```math
\left\{\mathbf{x}^{(k)}_{\mathrm{task}}\right\}_{k=1}^{K},
```

这些 token 与共享 tokens 一起进入 backbone，但最终只送给对应任务塔。

### 11.1 它与 MMoE / PLE 的区别

- MMoE / PLE 在 expert 参数路径上解耦任务；
- Task Token 在 token 表示空间中解耦任务；
- Task Token 的额外参数非常小，但仍能形成任务私有的信息汇聚位置；
- 两者可以组合，但论文只证明了 token-level decoupling 的收益。

---

## 12. Figure 5：Task Token 是否真的携带更多任务信息

![Figure 5: Mutual Information across Different Cluster Granularities](https://arxiv.org/html/2604.17878v3/x3.png)

*图源：RankUp Figure 5，arXiv v3。图片加载失败时可打开[原图](https://arxiv.org/html/2604.17878v3/x3.png)。*

论文先用 K-means 将连续表示离散为 $K$ 个 cluster，再计算 cluster assignment $Z$ 与任务标签 $Y$ 的互信息：

```math
I(Z;Y)
=\sum_{z,y}P(z,y)
\log\frac{P(z,y)}{P(z)P(y)}.
```

### 12.1 图中结论

- 在 Book 和 Order 两个任务上，带 task token 的曲线都高于不带 task token；
- 从 8 到 64 clusters，结论方向保持一致；
- cluster 越细，差距总体越明显，论文据此认为 task token 能保存更细粒度的任务结构。

### 12.2 需要保留的审慎解释

更高的 $I(Z;Y)$ 说明表示和标签更相关，但不等价于：

- 一定有更好的校准；
- 一定对所有任务都无负迁移；
- 一定具有更好的 out-of-distribution 泛化；
- 一定是 task token 本身而不是额外参数量带来的提升。

因此仍需同时看任务 AUC、LogLoss、calibration、梯度冲突和任务间 seesaw。

---

## 13. Table 1：Realtime AUC 消融如何解读

| 方案 | Order | Book | Add Service |
|---|---:|---:|---:|
| Randomized Permutation Split | +0.06% | +0.06% | +0.08% |
| Global Token + Multi-Emb | +0.21% | +0.18% | +0.13% |
| Cross Embedding | +0.22% | +0.10% | +0.03% |
| Task Token | +0.09% | +0.02% | +0.02% |
| Full RankUp | +0.41% | +0.23% | +0.25% |

### 13.1 三个重要观察

1. **不同任务的主要受益组件不同。** Order 对 Cross Embedding 最敏感，Add Service 对 Random Split 相对更敏感。
2. **Global Token 与 Multi-embedding 被捆绑报告。** 表中没有把二者完全拆开，因此无法从 Table 1 判断各自独立贡献。
3. **收益不是简单相加。** Order 的单项增益总和约为 +0.58%，Full RankUp 为 +0.41%；Book 的单项总和约为 +0.36%，Full 为 +0.23%。这说明组件之间存在能力重叠、优化耦合或负交互。Add Service 的单项总和约为 +0.26%，与 Full 的 +0.25% 接近，任务依赖非常明显。

因此复现时必须先做单组件消融，再使用 2×2 factorial design 研究组合，不能一次堆叠全部模块后只看最终结果。

---

## 14. 线上部署结果：效果与成本需要一起看

### 14.1 论文部署条件

- 18 个月生产 user-ad 日志；
- 3 个微信广告场景；
- 32 个联合优化任务；
- 超过 1000 个 sparse fields；
- 2 层 MetaFormer backbone；
- 每场景从约 10M 扩展到约 100M 参数；
- 推理 batch size 为 300；
- 每 batch 约 70 GFLOPs；
- 报告 MFU 为 23%；
- 20% 流量 A/B，持续 14 天；
- 最终全量部署。

### 14.2 Table 2：整体线上提升

| 场景 | Realtime AUC | CTCVR | GMV |
|---|---:|---:|---:|
| Weixin Video Accounts | +0.367% | +1.41% | +3.41% |
| Weixin Official Accounts | +0.331% | +0.21% | +4.81% |
| Weixin Moments | +0.269% | +0.87% | +2.12% |

### 14.3 Table 3：新广告冷启动

| 场景 | New Ads GMV |
|---|---:|
| Weixin Video Accounts | +5.83% |
| Weixin Official Accounts | +9.67% |
| Weixin Moments | +2.84% |

冷启动提升大于总体提升，与论文假设一致：当历史行为监督不足时，更丰富的输入几何视角和外部预训练先验更有价值。但这仍是相关性解释，论文没有单独给出“新广告流量上的组件级消融”。

### 14.4 Table 4：高价值 Order 任务

| 场景 | Order Task GMV |
|---|---:|
| Weixin Video Accounts | +5.18% |
| Weixin Official Accounts | +7.18% |
| Weixin Moments | +4.79% |

Order 任务对购买意图的细粒度区分要求更高，Cross Embedding 和高秩表示可能因此更有价值。

### 14.5 版本差异提醒

早期 arXiv 摘要或第三方索引中可能看到 Weixin Moments GMV 为 2.21%；本文统一采用 v3 正文 Table 2 与摘要中的 **2.12%**。引用结果时应标注论文版本。

---

## 15. RankUp 的科学贡献与局限

### 15.1 贡献

- 把“参数规模”与“有效表示容量”明确区分；
- 用 effective rank 和 token MI 建立结构诊断闭环；
- 改进点与诊断问题一一对应，而不是任意堆模块；
- 数据规模、字段数、任务类型和 RankMixer 基线与工业 CVR 高度匹配；
- 同时给出组件消融、表示分析和线上业务结果。

### 15.2 局限

1. **代码和关键超参数未公开。** embedding table 分配、token 数、隐藏维度、随机映射更新策略等难以严格复现。
2. **深层动机与两层实验之间存在张力。** 论文强调深层表示坍缩，但主要公平对照固定为 2-layer backbone。
3. **Random Split 缺少多 seed 误差条。** 无法排除某个随机 mapping 偶然较优。
4. **Global Token + Multi-Emb 没有完全拆分。** 两者独立贡献不够清晰。
5. **MI 是离散化估计。** 结果依赖 K-means 粒度和初始化。
6. **计算成本披露有限。** 论文给出参数量级、70 GFLOPs/batch 和 23% MFU，但没有完整 P50/P95/P99、显存、lookup 带宽和组件级成本。
7. **统计信息不完整。** 文中称线上结果具有统计显著性，但未给置信区间、p-value 或实验单元定义。
8. **预印本元数据仍有占位内容。** v3 PDF 中仍可见 conference year / DOI 模板字段，因此应按 arXiv preprint 引用。

---

## 16. 与当前 RankMixer 基线的匹配度

| 维度 | RankUp 论文 | 当前系统 | 匹配判断 |
|---|---|---|---|
| 任务 | 广告 CVR，多任务 | 推广搜 CVR | 高 |
| sparse fields | 超过 1200 | 1234 | 极高 |
| 日样本 | 约 2000 万 | 约 5.5 亿 | 当前数据更充足 |
| backbone | 2 层 RankMixer / MetaFormer | 2 层 RankMixer | 极高 |
| sparse token 数 | 32 | 16 | 有差异 |
| 总 token 类型 | sparse + dense + seq + cross + task | 当前仅 16 个 autosplit tokens | 有明显差异 |
| multi-task | 32 tasks | 当前描述为单 CVR | Task Token 条件不满足或需重新定义 |
| 预训练 user/item embedding | 有 | 未说明 | Cross Token 取决于数据条件 |

这是 RankUp 值得优先研究的主要原因：它不是一个与现有系统完全不同的序列生成架构，而是对两层 RankMixer、千级 sparse fields、工业 CVR 的直接改造。

---

## 17. 面向当前配置的可执行复现方案

当前输入为：

```math
385\times17+835\times17+14\times17=20{,}978,
```

当前 backbone 为 $T=16$、$D=768$、$L=2$。

### 17.1 先建立诊断基线 RU-D0

保持模型完全不变，增加离线统计：

- tokenizer 输出的 sample-token normalized erank；
- block1-TM、block1-FFN、block2-TM、block2-FFN 的 erank；
- 每个 token 跨 batch 的 channel erank；
- token 两两 cosine similarity；
- token MI difference matrix；
- 每个 projector / PFFN 的梯度范数与更新比。

只有确认当前模型也存在“TM 扩张、FFN 收缩”或 token rank 极不均衡，才能判断 RankUp 的机制是否对症。

### 17.2 RU-R1：固定字段级 Random Split

在 1234 个完整 field 上做固定 permutation，再均衡分成 16 组：

```math
2\text{ groups}:\ 78\text{ fields},\qquad
14\text{ groups}:\ 77\text{ fields}.
```

保持：

- $T=16$；
- $D=768$；
- 两层 RankMixer；
- PFFN 参数与训练配置；
- 每个字段完整的 17 维 embedding。

运行至少 3 个 mapping seed。该实验成本最低，最能直接验证 Figure 2-3 的假设。

### 17.3 RU-G1：15 Local + 1 Global

为保持 $T=16$ 和 $H=T$，首版不额外增加第 17 个 token，而是：

```text
15 local sparse tokens + 1 global token
```

Global Token 可以先由 local token 的 mean / RMS 统计经过小 MLP 构造。若该版本获益，再测试基于低秩 DCNv2 的 global branch。

需要增加一个 16-local-token 宽化对照，排除收益仅来自新增 global MLP 参数。

### 17.4 RU-M1：Selective Multi-embedding

只对 anchor fields 增加第二 embedding view：

```text
user_id, item_id, shop_id, category_id, query_id, creative_id
```

主 view 与 auxiliary view 应进入不同 token，不能在 lookup 后立即相加。必须单独报告：

- embedding 参数与 optimizer state；
- PS / lookup QPS；
- HBM 与通信；
- cold-start 分桶；
- 长尾 field 分桶。

### 17.5 RU-C1：Cross Pre-trained Embedding Token

仅当存在独立预训练 user/item embeddings 时运行：

```math
\mathbf{e}_{\mathrm{cross}}
=\operatorname{Proj}(\mathbf{z}_{u}\odot\mathbf{z}_{i}).
```

若保持总 token 数为 16，可采用：

```text
14 local sparse + 1 global + 1 cross
```

这会减少 local token 数，因此还需要一个“14 local + 2 learned dummy / control tokens”的控制实验，区分 token budget 改变与 cross 信息本身的贡献。

### 17.6 RU-T1：Task Token 只在多任务条件下运行

若当前只有一个 CVR 标签，单个 task token 更接近 learned pooling token，不能验证论文的“任务冲突解耦”假设。

只有在存在 CTR、CVR、CTCVR、order、book、add-service 等多任务目标时，才建议增加独立 task tokens，并监控：

- 每任务 AUC / LogLoss / calibration；
- task-token 与 label 的 MI；
- 任务梯度 cosine；
- seesaw 与长尾任务收益。

### 17.7 RU-O1：PreNorm + Per-token SwiGLU

该组件改变的是优化和 FFN 表达，不应与 Random Split 在首轮同时加入。建议使用参数 / FLOPs 匹配的 SwiGLU hidden size，并分别消融：

```text
PreNorm only
SwiGLU only
PreNorm + SwiGLU
```

### 17.8 是否应该把 token 数从 16 提高到 32

论文对 Random Split 的直接证据来自 32 sparse tokens，而你的模型只有 16 tokens。扩大 token 数可能提高 rank 上限并减少每个 token 的输入压缩，但会同时改变：

- Token Mixing 的 head 数和 head dimension；
- Per-token FFN 参数量；
- grouped GEMM 形状；
- 总 token budget；
- 线上延迟。

$D=768$ 可以被 24、32 和 48 整除，因此这些 token 数在结构上可行，但必须做参数 / FLOPs 匹配。建议先验证 $T=16$ 的 RankUp-Lite，再将 token 数作为独立变量研究，而不是把扩 token 与所有 RankUp 组件一起上线。

---

## 18. 推荐实验顺序

```text
Phase 0  RU-D0  原 RankMixer + rank/MI/gradient 诊断
Phase 1  RU-R1  固定 Random Split，3 个 mapping seeds
Phase 2  RU-G1  15 Local + 1 Global
Phase 3  RU-M1  Selective Multi-embedding
Phase 4  RU-C1  Cross Pre-trained Embedding Token（条件满足时）
Phase 5  RU-O1  PreNorm / SwiGLU 独立优化消融
Parallel RU-T1  Task Token（仅多任务）
Phase 6  对两个独立获益组件做 2×2 factorial 组合
Phase 7  单独研究 T=24 / 32 的 token scaling
```

每一步都应同时回答三个问题：

1. 效果是否超过基线 seed 波动？
2. 它声称改善的 rank / MI 指标是否同步改善？
3. 收益是否覆盖 embedding、FLOPs、吞吐和 P99 延迟成本？

---

## 19. 最终结论

RankUp 最有价值的地方不是提供了五个可以任意堆叠的模块，而是给出了一套研究范式：

```text
发现表示坍缩
  -> 用 effective rank / MI 定位
  -> 为不同坍缩来源设计不同组件
  -> 做组件级消融
  -> 检查层间表示动态
  -> 再验证线上业务收益
```

对当前两层、16-token、1234-field、5.5 亿样本/天的 RankMixer CVR 系统，最优先的不是直接复刻完整 RankUp，而是：

1. 建立与论文一致但口径明确的 rank / MI 诊断；
2. 先验证固定 Random Split；
3. 再验证保持总 token 数不变的 Global Token；
4. 在有外部预训练向量和多任务标签时分别引入 Cross Token 与 Task Token；
5. Multi-embedding 必须从少量 anchor fields 开始，并把 lookup 成本纳入结论。

只有当“结构收益、表示诊断和工程成本”三者同时成立，才能说明 RankUp 在当前推广搜 CVR 场景中被科学地复现，而不是仅仅增加了模型参数。

---

## 参考资料

1. Jin Chen et al. [RankUp: Towards High-rank Representations for Large Scale Advertising Recommender Systems](https://arxiv.org/abs/2604.17878), arXiv:2604.17878v3, 2026.
2. Jie Zhu et al. [RankMixer: Scaling Up Ranking Models in Industrial Recommenders](https://arxiv.org/abs/2507.15551), 2025.
3. Olivier Roy and Martin Vetterli. [The Effective Rank: A Measure of Effective Dimensionality](https://ieeexplore.ieee.org/document/4140265), 2007.
4. Ruoxi Wang et al. [DCN V2: Improved Deep & Cross Network and Practical Lessons for Web-scale Learning to Rank Systems](https://arxiv.org/abs/2008.13535), 2021.
