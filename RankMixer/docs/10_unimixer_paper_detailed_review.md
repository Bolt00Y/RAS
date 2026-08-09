# UniMixer 论文详解：从固定 Token Mixing 到统一可学习特征混合

> 论文：**UniMixer: A Unified Architecture for Scaling Laws in Recommendation Systems**  
> 作者：Mingming Ha, Guanchen Wang, Linxun Chen, Xuan Rao, Yuexin Shi, Tianbao Ma, Zhaojie Liu, Yunqian Fan, Zilong Lu, Yanan Niu, Han Li, Kun Gai  
> 初始提交：2026-04-01  
> 原文：[arXiv Abstract](https://arxiv.org/abs/2604.00590) · [HTML](https://arxiv.org/html/2604.00590) · [PDF](https://arxiv.org/pdf/2604.00590)

## 1. 论文定位

UniMixer 关注 RankMixer 体系中的另一个根本问题：

> RankMixer 的 Token Mixing 虽然高效，但 mixing pattern 是人工固定的；Self-Attention 虽然可学习，但动态 $QK^\top$ 计算成本高且训练可能出现过尖或过平的注意力分布。

论文把推荐系统中常见的可扩展特征交互方法概括为三类：

1. Attention-based methods；
2. TokenMixer-based methods；
3. Factorization-Machine-based methods。

这些方法看似结构完全不同，但 UniMixer 试图用统一的“全局 mixing + 局部 mixing”框架描述它们，并构造一个可学习、可压缩、可控制稀疏性的 mixing 模块。

它相对 RankMixer 的核心变化是：

- 把固定 permutation 解释为一个矩阵算子；
- 将这个算子参数化并可训练；
- 解除 $H=T$ 的强约束；
- 同时表示全局 token 交换和局部 channel 变换；
- 通过低秩、basis composition 与 Sinkhorn 约束提高 scaling ROI。

---

## 2. RankMixer Token Mixing 的矩阵解释

RankMixer 输入：

$$
\mathbf X
\in
\mathbb R^{T\times D}.
$$

将其 flatten：

$$
\mathbf x
=
\operatorname{vec}(\mathbf X)
\in
\mathbb R^{L},
\qquad
L=TD.
$$

RankMixer 的 reshape、transpose 和 concat 可以等价写成一个置换矩阵：

$$
\operatorname{vec}
\left(
\operatorname{TokenMix}(\mathbf X)
\right)
=
\mathbf x
\mathbf W_{\mathrm{perm}},
$$

其中：

$$
\mathbf W_{\mathrm{perm}}
\in
\{0,1\}^{L\times L}.
$$

置换矩阵具有：

- 每行只有一个 1；
- 每列只有一个 1；
- 行和列都为 1；
- 不改变向量范数；
- mixing pattern 完全固定。

当 $H=T$ 时，这个置换还可以表达为更小 mixing matrix 与单位矩阵的 Kronecker 结构：

$$
\mathbf W_{\mathrm{perm}}
=
\mathbf G
\otimes
\mathbf I.
$$

这里 $\mathbf G$ 描述 token/head 级交换，$\mathbf I$ 保留局部 channel 子块。

UniMixer 的出发点是：既然固定 TokenMixer 可以写成矩阵，就可以把矩阵从人工规则改造成可学习参数。

---

## 3. Figure 2：UniMixer 整体框架

<p align="center">
  <img src="https://arxiv.org/html/2604.00590v1/x2.png" width="94%" alt="UniMixer Figure 2">
</p>

*图源：UniMixer Figure 2。若图片加载失败，请打开论文 [HTML](https://arxiv.org/html/2604.00590) 或 [PDF 第 5 页](https://arxiv.org/pdf/2604.00590#page=5)。*

Figure 2 中，一个 UniMixer block 主要包含：

```text
UniMixing / UniMixing-Lite
        ↓
Per-token SwiGLU
        ↓
SiameseNorm
        ↓
可选 Sparse-Pertoken MoE
```

UniMixing 自身又被拆成：

```text
Global Mixing
        ×
Block-specific Local Mixing
```

这里的 global/local 不是业务意义上的 user/global token，而是矩阵作用范围：

- Global Mixing 决定不同 block 或 token 子空间怎样交换；
- Local Mixing 决定每个 block 内部的 channel 怎样变换。

---

## 4. Block granularity

将长度为 $L=TD$ 的 flatten 向量按 block size $B$ 切分：

$$
\mathbf x
=
[\mathbf x_1;\mathbf x_2;\ldots;\mathbf x_M],
$$

其中：

$$
M
=
\frac{L}{B},
\qquad
\mathbf x_i
\in
\mathbb R^B.
$$

UniMixer 不再强制 block 必须等于原 token，也不强制 head 数等于 token 数。$B$ 控制 mixing 粒度：

- $B=D$ 时，一个 block 近似对应一个 token；
- $B=D/H$ 时，一个 block 近似对应一个 token head；
- 更小 $B$ 提供更细粒度 mixing；
- 更大 $B$ 降低 global mixing 的维度。

这使 UniMixer 能在 attention、TokenMixer 和 FM 风格交互之间调整结构。

---

## 5. Global Mixing

设有 $M$ 个 blocks。Global Mixing 使用矩阵：

$$
\mathbf W_G
\in
\mathbb R^{M\times M}.
$$

它控制输入 block $i$ 对输出 block $j$ 的贡献。

若先把每个 block 看成一个向量，global mixing 可写为：

$$
\widetilde{\mathbf x}_j
=
\sum_{i=1}^{M}
W_{G,ij}
\mathbf x_i.
$$

与 RankMixer 固定 permutation 相比：

- RankMixer 的 $W_{G,ij}$ 只有 0 或 1；
- 每个输出 block 只读取一个确定输入 block；
- UniMixer 的权重可连续学习；
- 一个输出 block 可以聚合多个输入 blocks；
- 训练后可通过温度得到接近稀疏置换的模式。

与 attention 相比，$\mathbf W_G$ 是参数矩阵，不需要对每个样本计算 $QK^\top$，因此推理路径更稳定、规则。

---

## 6. Block-specific Local Mixing

Global Mixing 决定 block 间交换，Local Mixing 决定每个 block 内部的线性变换。

第 $i$ 个 block 使用：

$$
\mathbf W_B^{(i)}
\in
\mathbb R^{B\times B}.
$$

局部变换：

$$
\mathbf z_i
=
\mathbf x_i
\mathbf W_B^{(i)}.
$$

如果所有 blocks 共用同一个 $\mathbf W_B$，模型会失去不同特征子空间的独立性。UniMixer 因此保留 block-specific local matrices。

完整 mixing 可以抽象为：

$$
\operatorname{UniMixing}(\mathbf X)
=
\operatorname{reshape}
\left(
\mathbf W_G
\begin{bmatrix}
\mathbf x_1\mathbf W_B^{(1)}\\
\mathbf x_2\mathbf W_B^{(2)}\\
\vdots\\
\mathbf x_M\mathbf W_B^{(M)}
\end{bmatrix}
ight).
$$

这一公式把“跨块交互”和“块内异构变换”明确拆开。

---

## 7. Figure 3：从固定置换到可学习 mixing

<p align="center">
  <img src="https://arxiv.org/html/2604.00590v1/x3.png" width="94%" alt="UniMixer Figure 3 mixing matrices">
</p>

*图源：UniMixer Figure 3。*

Figure 3 的核心信息是：

- RankMixer 对应规则、稀疏、离散的 permutation pattern；
- Self-Attention 对应样本相关、稠密的 dynamic pattern；
- UniMixer 学习一个结构化、可控稀疏度的 static parameterized pattern；
- Sinkhorn 约束后，权重可以接近 doubly stochastic matrix；
- 温度降低时，matrix 从平滑逐渐变尖锐。

这使 UniMixer 位于固定 TokenMixer 和动态 attention 之间：

```text
固定规则置换
        ← UniMixer →
样本动态注意力
```

---

## 8. 统一视角：Attention、TokenMixer 与 FM

论文用 global/local mixing pattern 统一多类方法。

### 8.1 Self-Attention

Self-Attention 可写为：

$$
\mathbf A(\mathbf X)
=
\operatorname{softmax}
\left(
\frac{
\mathbf Q\mathbf K^\top
}{\sqrt d}
\right).
$$

Global Mixing 由输入动态决定，local value transform 通常共享。

### 8.2 Heterogeneous Attention

若不同 token 使用不同 $Q/K/V$ 参数，相当于 global mixing 和 local transform 都具有异构性，但计算与参数更高。

### 8.3 TokenMixer

TokenMixer 的 global mixing 是固定 permutation，local mixing 通常是固定 identity / slicing。

### 8.4 Factorization Machine

FM 的 pairwise product 可以被视为特定局部交互与全局聚合模式：

$$
\sum_{i<j}
\langle
\mathbf v_i,
\mathbf v_j
\rangle
x_ix_j.
$$

UniMixer 通过 block-specific local matrices 和 global combination，提供一个更一般的线性 mixing 框架，再由后续 SwiGLU 完成非线性表达。

### 8.5 统一框架的意义

它并不是证明所有方法数学上完全相同，而是指出它们可以从两个问题比较：

1. 谁决定跨 token / block 的 mixing 权重？
2. 每个局部子空间使用共享还是独立变换？

这为设计 scaling block 提供了统一坐标系。

---

## 9. UniMixing-Lite：为什么需要轻量化

完整 $\mathbf W_G$ 和大量 $\mathbf W_B^{(i)}$ 仍可能很大。UniMixer-Lite 使用两种压缩。

### 9.1 Basis-composed Local Mixing

不为每个 block 保存完整独立矩阵，而是学习 $b$ 个 basis：

$$
\{\mathbf Z_1,\mathbf Z_2,\ldots,\mathbf Z_b\},
\qquad
\mathbf Z_j
\in
\mathbb R^{B\times B}.
$$

第 $i$ 个 local matrix 由 basis 线性组合：

$$
\mathbf W_B^{(i)}
=
\sum_{j=1}^{b}
\omega_j^{(i)}
\mathbf Z_j.
$$

这样既保留 block-specific 差异，又把参数从 $MB^2$ 降到：

$$
bB^2+Mb.
$$

当 $b\ll M$ 时，节省明显。

### 9.2 Low-rank Global Mixing

Global matrix 使用低秩分解：

$$
\mathbf W_G
=
\mathbf A_G
\mathbf B_G,
$$

其中：

$$
\mathbf A_G
\in
\mathbb R^{M\times r},
\qquad
\mathbf B_G
\in
\mathbb R^{r\times M}.
$$

参数从 $M^2$ 降为：

$$
2Mr.
$$

低秩 global mixing 假设跨 block 交互主要由少量潜在 mixing patterns 组成。

---

## 10. Sinkhorn-Knopp 约束

RankMixer permutation matrix 是 doubly stochastic：每行、每列和都为 1。UniMixer 希望可学习矩阵在保留柔性的同时继承这种稳定结构。

对任意非负矩阵 $\mathbf A$，交替进行行归一化和列归一化：

$$
A_{ij}
\leftarrow
\frac{A_{ij}}
{\sum_kA_{ik}},
$$

$$
A_{ij}
\leftarrow
\frac{A_{ij}}
{\sum_kA_{kj}}.
$$

多次迭代后得到近似 doubly stochastic matrix：

$$
\sum_jA_{ij}\approx1,
\qquad
\sum_iA_{ij}\approx1.
$$

其作用是：

- 避免某些输出 block 吸收过多总权重；
- 避免某些输入 block 被所有输出重复使用；
- 改善 mixing 负载均衡；
- 让学习结果接近置换矩阵的结构先验；
- 控制 activation scale。

论文消融中，取消相应结构约束会导致效果下降。

---

## 11. Temperature 与 warm-up

在 Sinkhorn 前对 logits 使用温度：

$$
\mathbf A
=
\exp
\left(
\frac{\mathbf Z}{\tau}
\right).
$$

- 高温度 $\tau$ 使权重平滑；
- 低温度使权重尖锐、接近稀疏置换；
- 训练早期过低温度可能导致 mixing pattern 过早离散化；
- warm-up 先用较高温度探索，再逐步降低。

论文代表策略可理解为：

```text
warm-up 阶段：τ 约 1
后续阶段：τ 逐渐降到约 0.05
```

消融结果：

| 消融 | 相对 AUC 变化 |
|---|---:|
| 去掉 temperature 控制 | -0.1645% |
| 去掉 symmetry / structural constraint | -0.0573% |
| 去掉 block-specific local mixing | -0.0436% |
| 去掉 warm-up | -0.0856% |
| SiameseNorm 改为 Post-Norm | -0.0273% |

temperature 和 warm-up 是影响最大的两项，说明 mixing matrix 的优化轨迹非常重要。

---

## 12. SiameseNorm

UniMixer 引入 SiameseNorm，以适配 mixing 前后的双分支表示。其思想是对相互对应的路径使用协调的归一化，避免：

- global mixing 与 local mixing 输出尺度失配；
- residual 分支和主分支统计差异过大；
- 可学习 mixing matrix 的权重尺度被 activation scale 干扰。

虽然论文具体实现应以原始公式和代码为准，但从消融看，SiameseNorm 相比 Post-Norm 有稳定小幅收益。

对于迁移实验，应把 Norm 设计作为独立变量，不要与可学习 mixing 同时一次性替换。

---

## 13. Per-token SwiGLU 与 Sparse-Pertoken MoE

UniMixer block 仍保留 TokenMixer-Large 证明有效的组件：

- Per-token SwiGLU；
- Sparse-Pertoken MoE；
- token-specific parameterization。

这说明 UniMixer 并不是完全抛弃 RankMixer 体系，而是主要替换 mixing operator：

```text
RankMixer fixed mixing
        ↓
UniMixer learnable structured mixing
```

后续非线性容量仍由 per-token gated FFN / MoE 提供。

---

## 14. Table 2：主结果

论文公开数据上的代表结果如下：

| 模型 | 参数量 | AUC | 主要 FLOPs |
|---|---:|---:|---:|
| RankMixer | 约 135.5M | 0.749329 | — |
| TokenMixer-Large | 约 103.3M | 0.748410 | — |
| UniMixer-Lite，4 blocks | 约 38.2M | 0.752327 | 约 1.26T |
| UniMixer-Lite，较大配置 | 约 84.5M | 0.752718 | 约 4.24T |

关键观察：

- UniMixer-Lite 用更少参数超过 RankMixer；
- 优势来自更高效的 mixing 表达，而不只是 FFN 扩容；
- TokenMixer-Large 在该数据与配置上未必优于 RankMixer，说明不同论文内部的工程结论不能无条件跨数据迁移；
- 可学习 mixing 提升了参数 / FLOPs 的 scaling ROI。

---

## 15. Figure 4：Scaling exponent

<p align="center">
  <img src="https://arxiv.org/html/2604.00590v1/x4.png" width="88%" alt="UniMixer Figure 4 scaling curves">
</p>

*图源：UniMixer Figure 4。*

论文拟合效果随参数量和 FLOPs 的 scaling 曲线。代表 exponent 为：

| 模型 | 参数 scaling exponent | FLOPs scaling exponent |
|---|---:|---:|
| RankMixer | 0.116043 | 0.116635 |
| UniMixer | 0.131973 | 0.125702 |
| UniMixer-Lite | 0.141903 | 0.135327 |

更高 exponent 表示增加资源时，指标改善速度更快。UniMixer-Lite 在参数和 FLOPs 两个维度都显示更高 slope。

需要注意：

- exponent 依赖数据、模型区间和拟合方式；
- 不应把一个数据集上的 exponent 当作普适常数；
- 但相同实验条件下，它可用于比较 scaling ROI。

---

## 16. Table 4：basis、rank 与深度

### 16.1 Local basis 数量

从 2 个 basis 增加到 4 个时，相对 AUC 约提升 0.1002%；增加到 8 个时约提升 0.1055%。

这表明：

- 太少 basis 会限制 local heterogeneity；
- 4 个 basis 已捕获大部分收益；
- 继续增加的边际收益下降。

### 16.2 Global low-rank 维度

global rank 从 2 增加到 256，累计相对 AUC 提升约 0.0971%。这说明跨 block mixing 不完全是极低秩，但也不一定需要完整 $M^2$ 参数。

### 16.3 深度扩展

| 模型 | 2 blocks 到 4 blocks | 到 8 blocks |
|---|---:|---:|
| RankMixer | 约 -0.1066% | 未显示稳定增益 |
| UniMixer-Lite | 约 +0.1575% | 约 +0.1647% |

这张表是 UniMixer 的重要证据：固定 Token Mixing 在加深时可能退化，而可学习、结构化 mixing 更能利用深度。

---

## 17. Figure 5 与 Figure 6：学习到的 mixing pattern

<p align="center">
  <img src="https://arxiv.org/html/2604.00590v1/x5.png" width="48%" alt="UniMixer Figure 5 learned patterns">
  <img src="https://arxiv.org/html/2604.00590v1/x6.png" width="48%" alt="UniMixer Figure 6 temperature patterns">
</p>

*图源：UniMixer Figure 5–6。*

这些可视化说明：

- 训练早期 mixing 较平滑；
- 温度降低后形成更稀疏、更结构化的连接；
- 不同 blocks 学到不同 local mixing；
- 学习结果既不是纯 identity，也不是完全无结构的 dense matrix；
- 可学习 pattern 可能发现人工语义分组未覆盖的交互。

可视化不能单独证明因果，但与消融中的 temperature、warm-up 和 local-specific 权重结果相互支持。

---

## 18. 线上实验

论文在快手 CAD 场景进行在线实验，目标涉及用户留存，训练数据约 0.7B，覆盖约一年。

论文报告在 D1 到 D30 留存指标上平均获得超过 15% 的相对改善。该结果说明 UniMixer 的可学习 mixing 不只提升离线 AUC，也可能改善长期目标。

但需要谨慎：

- 线上指标、基线和业务定义与当前 CVR 不同；
- 留存是长期用户目标，CVR 是候选转化目标；
- 结果不能直接换算成当前电商推广搜收益。

---

## 19. 与 RankMixer 的逐项对比

| 维度 | RankMixer | UniMixer / UniMixer-Lite |
|---|---|---|
| Mixing pattern | 固定 permutation | 可学习 structured matrix |
| 样本依赖 | 固定 | 参数固定但训练可学习，不按样本动态 |
| $H=T$ 约束 | 需要 | 可解除 |
| Global mixing | 人工规则 | 学习矩阵 / 低秩分解 |
| Local mixing | slicing / identity | block-specific matrices |
| 参数压缩 | 无 mixing 参数 | low-rank + basis composition |
| 稀疏性 | 天然置换稀疏 | temperature + Sinkhorn 学习稀疏 |
| Norm | RankMixer Post-LN | SiameseNorm |
| FFN | Per-token GELU FFN | Per-token SwiGLU / SP-MoE |
| 深度扩展 | 可能退化 | 论文中 4/8 blocks 继续获益 |
| 解释性 | mixing 规则明确 | 可查看学习到的 mixing matrix |
| 工程复杂度 | 极低 | 更高，需要矩阵约束与自定义实现 |

---

## 20. 与 Self-Attention 的对比

| 维度 | Self-Attention | UniMixer |
|---|---|---|
| mixing 权重 | 每个样本动态计算 | 训练后固定参数 |
| 计算 | $QK^\top$ + softmax | 结构化矩阵乘法 |
| token 数复杂度 | 通常二次 | 取决于低秩和 block 设计 |
| 输入适应性 | 强 | 不按样本动态 |
| 稳定性 | 可能过尖或过平 | Sinkhorn + temperature 控制 |
| serving 可预测性 | 较复杂 | 更规则 |
| 参数异构性 | 常用共享 QKV | block-specific local mixing |

UniMixer 不是 attention 的完全替代，而是面向工业 ranking 的中间解：比固定 TokenMixer 灵活，比动态 attention 更规则。

---

## 21. 对当前电商推广搜 CVR 的意义

当前 RankMixer 使用 $T=16,D=768$ 或 $T=32,D=1536$。UniMixer 最值得验证的问题是：

> 当前固定 Autosplit + 固定 Token Mixing 是否把模型限制在错误的跨 token 交互模式上？

建议分阶段消融。

### 21.1 最简单可学习 mixing 对照

先不直接实现完整 UniMixer，只加入一个 token-level matrix：

$$
\widetilde{\mathbf X}
=
\mathbf A
\mathbf X,
$$

其中：

$$
\mathbf A
\in
\mathbb R^{T\times T}.
$$

使用 identity initialization：

$$
\mathbf A\big|_{\mathrm{init}}
=
\mathbf I.
$$

这回答最基础问题：一个可学习 $T\times T$ mixing 是否优于固定 HeadMixing。

### 21.2 低秩 global adapter

$$
\mathbf A
=
\mathbf U\mathbf V^\top,
$$

$$
\mathbf U,\mathbf V
\in
\mathbb R^{T\times r}.
$$

在 $T=16$ 或 $32$ 时，这个模块参数非常小，适合作为低成本实验。

### 21.3 UniMixing-Lite

只有简单 matrix 和低秩 adapter 已有增益后，再加入：

- block-specific local mixing；
- basis composition；
- Sinkhorn；
- temperature warm-up；
- SiameseNorm。

否则无法判断复杂约束是否必要。

### 21.4 与 Base 归纳偏置的关系

可学习 mixing 可能发现 user-item-creative 交互，但它不会自动等价于：

- Base 的层级条件 SENet；
- DCNv2 显式乘法交叉；
- creative-aware pooling。

因此 UniMixer 更适合解决 RankMixer 内部 mixing 表达不足，不应被当作 Base 全部能力的替代。

---

## 22. 推荐实验矩阵

```text
UM-0  原始 RankMixer
UM-1  learned T×T mixing，identity init
UM-2  low-rank token mixing，rank 2/4/8
UM-3  UM-2 + doubly stochastic normalization
UM-4  UM-3 + temperature warm-up
UM-5  UniMixing-Lite with 4 local bases
UM-6  UM-5 + SiameseNorm
UM-7  最优 2-block 版本扩展到 4 blocks
```

必要对照：

- 同参数量 MLP adapter；
- 固定随机 orthogonal mixing；
- RankUp fixed random field split；
- Self-Attention token mixer；
- 训练吞吐、MFU 和 P99。

只有可学习 mixing 稳定优于固定随机矩阵与同参数 MLP，才能证明收益来自 learned interaction pattern。

---

## 23. 局限与复现注意事项

1. UniMixer 发布时间较新，公开复现和跨数据验证仍有限；
2. Sinkhorn、temperature 和 basis mixing 增加实现复杂度；
3. 可学习 static matrix 仍不能针对每个样本动态调整；
4. 低秩假设是否成立取决于 tokenization；
5. mixing matrix 可解释性不等于业务因果解释；
6. 公开实验数据与当前推广搜 CVR 分布不同；
7. 在线留存提升不能直接推导 CVR 提升；
8. 需要防止 Sinkhorn 与低温度造成梯度不稳定；
9. 自定义 kernel 和 fused implementation 对实际 ROI 很重要；
10. 应先做简单 learned mixing，再逐步引入完整 UniMixing-Lite。

---

## 24. 一句话总结

UniMixer 的核心贡献可以概括为：

> 把 RankMixer 的固定 Token Mixing 写成结构化矩阵，并将其升级为可学习的 global-local mixing；再通过低秩、basis composition、Sinkhorn 与温度调度，在保留工业计算规则性的同时提高 mixing 表达力和 scaling ROI。

它是 RankMixer 发展谱系中“让 mixing pattern 从人工固定走向结构化可学习”的分支。
