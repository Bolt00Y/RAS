# 大参数 RankMixer 若仍弱于 Base：系统消融实验方案

> 适用条件：大参数 RankMixer 采用与当前小模型相同的论文基线结构，仅将 token 数和隐藏维度扩大为 32 与 1536；训练数据仍仅为现有 user、item、creative 稀疏特征及 CVR 标签，不引入新的行为序列、外部预训练向量、新标签或额外数据源。
>
> 当前已知结果：小参数 RankMixer 的 CVR AUC 为 0.86188，强 Base 的 CVR AUC 为 0.86865。本文讨论的触发条件是：大参数 RankMixer 训练后仍显著弱于 Base。

## 1. 执行摘要

如果大参数 RankMixer 仍弱于 Base，不应立刻得出“RankMixer 不适合当前业务”或“继续扩大到 MoE 就会更好”的结论。当前小模型与 Base 并不是单变量对照：Base 同时拥有 bucket-wise BatchNorm、层级条件 SENet、DCNv2 显式交叉和 MLP 任务头，而 RankMixer 使用连续 Autosplit、两层原始 RankMixer block、Mean Pooling 和较弱的任务头。大模型仅扩大相同结构，无法自动补回这些已被当前数据证明有效的归纳偏置。

最优先的研究顺序是：

1. 先对 Base 做反向消融，量化 BN、三级 SENet、DCNv2 和 MLP head 分别贡献多少；
2. 将 token 数、隐藏宽度、PFFN 容量和 pooling 梯度效应拆开，不再只比较 16×768 与 32×1536；
3. 优先验证任务头、pooling、归一化和初始化等低成本因素；
4. 再迁移 Base 已验证有效的条件门控与显式交叉；
5. 随后研究 RankUp tokenization 与 TokenMixer-Large block；
6. 只有 dense 扩容已经稳定获益后，才进入 Sparse-Pertoken MoE。

生产上最稳健的候选不是让 RankMixer 立即完全替代 Base，而是以 Base 为性能锚点，通过零初始化 residual branch 让 RankMixer 只学习 Base 尚未覆盖的残差。

---

## 2. 大模型并不是“小模型的等比例增强”

设展开输入维度为：

```math
F=20{,}978.
```

RankMixer 的主要量可以写为：

```math
\begin{aligned}
d_{\mathrm{seg}} &\approx \frac{F}{T},\\
r_{\mathrm{proj}} &\approx \frac{D}{F/T}=\frac{DT}{F},\\
d_{\mathrm{head}} &= \frac{D}{T},\\
P_{\mathrm{PFFN}} &\approx 2kLTD^2.
\end{aligned}
```

其中：

- $d_{\mathrm{seg}}$ 是每个输入 segment 的近似宽度；
- $r_{\mathrm{proj}}$ 是 segment 投影到 token 后的宽度比；
- $d_{\mathrm{head}}$ 是 Token Mixing 中每个 head 的 channel 宽度；
- $k$ 是 PFFN 扩张倍数；
- $L=2$ 是 block 数。

假设 $k=4$，两种模型的主要几何与参数如下。

| 配置 | segment 宽度 | 投影宽度比 | mixing head 宽度 | Tokenizer 权重 | PFFN 权重 |
|---|---:|---:|---:|---:|---:|
| 16 tokens，768 hidden | 前 15 个为 1311，最后为 1313 | 0.586 | 48 | 16.11M | 151.0M |
| 32 tokens，1536 hidden | 前 31 个为 655，最后为 673 | 2.343 | 48 | 32.22M | 1.208B |

### 2.1 四个容易被忽略的事实

第一，单个 mixing head 的宽度没有增加：

```math
\frac{768}{16}=\frac{1536}{32}=48.
```

因此大模型没有扩大每个 head 携带的 channel 数；它主要增加 token 粒度和每-token PFFN 容量。

第二，输入投影从压缩变成了明显的过完备扩张：

```math
0.586\longrightarrow 2.343.
```

大模型中约 655 维 segment 被映射到 1536 维 token。额外维度不会凭空增加输入信息，如果 projector 输出高度相关，后续巨大的 PFFN 可能只是在低有效秩表示上增加参数。

第三，PFFN 约扩大 8 倍：

```math
\frac{32\times1536^2}{16\times768^2}=8.
```

若大模型仍不如 Base，这会强烈削弱“只是参数量不够”的解释。

第四，Mean Pooling 对每个 token 的直接梯度会随 token 数增加而下降。若：

```math
\mathbf h=\frac{1}{T}\sum_{t=1}^{T}\mathbf x_t,
```

则：

```math
\frac{\partial \mathcal L}{\partial \mathbf x_t}
=\frac{1}{T}\frac{\partial \mathcal L}{\partial \mathbf h}.
```

从 16 个 token 增加到 32 个 token 后，来自预测头的直接 pooling 梯度系数减半。若不同 token 的信息量本来就不均衡，Mean Pooling 可能成为大模型的额外瓶颈。

### 2.2 creative 信号的特殊风险

在当前连续 Autosplit 中，全部 238 维 creative 特征仍只出现在最后一个输入 segment。

小模型最后一个 segment 包含：

```math
1075\ \text{维 item}+238\ \text{维 creative}.
```

大模型最后一个 segment 包含：

```math
435\ \text{维 item}+238\ \text{维 creative}.
```

虽然 creative 在最后 segment 中的占比提高，但该 token 在 Mean Pooling 中的先验权重从 $1/16$ 降到 $1/32$。因此，大模型更需要验证 creative-aware tokenization 或 task-aware pooling，而不能只依赖参数扩张。

---

## 3. 大模型结果出来后先分类，不要统一处理

| 观测结果 | 更可能的原因 | 首选消融方向 |
|---|---|---|
| 大模型明显优于小模型，但仍低于 Base | scaling 有效，但缺少 Base 的条件选择、显式交叉或强任务头 | Base 反向消融、SENet/DCN 迁移、双分支融合 |
| 大模型与小模型基本持平 | 容量没有转化为有效表示；tokenizer、pooling、rank collapse 或优化不足 | 规模轴拆分、RankUp tokenization、pooling、有效秩诊断 |
| 大模型低于小模型 | 过度参数化、学习率不匹配、Post-Norm 不稳定、Mean Pooling 梯度稀释 | 优化与初始化、减小 PFFN、Pre-RMSNorm、任务头 |
| 大模型训练 AUC 也低 | 欠优化、梯度传播或结构利用率低 | 学习率、warmup、初始化、归一化、参数更新比 |
| 大模型训练 AUC 高而验证 AUC 低 | 过拟合或时间泛化差 | 减小 $k$、weight decay、shared/private FFN、distillation |
| Base 与大模型 blend 超过 Base | 两者存在互补信息 | zero-init residual hybrid、双流融合 |
| Blend 也不能超过 Base | RankMixer 主要在重复但更差地表达 Base 信号 | 优先迁移 Base 归纳偏置，不急于继续扩容 |

---

## 4. 文献证据如何映射到当前问题

### 4.1 RankMixer：扩容轴必须拆开

[RankMixer](https://arxiv.org/html/2507.15551v3) 将 token 数、隐藏宽度、网络深度和专家数视为四个可独立扩展的轴，并给出 dense 参数近似与 $LTD^2$ 成正比。当前从 16×768 直接跳到 32×1536，同时改变了 token granularity、projector 宽度比和 PFFN 容量，无法判断真正有效的轴。

### 4.2 TokenMixer-Large：原始 block 本身存在可消融缺陷

[TokenMixer-Large](https://arxiv.org/html/2602.06563v2) 指出原始 RankMixer 存在 mixing 前后 residual 语义错位、深层梯度不足和 Post-Norm 数值不稳定等问题。在其 4B 模型消融中：

- 去掉 Mixing & Reverting，AUC 相对下降 0.27%；
- 去掉 residual，下降 0.15%；
- Per-token SwiGLU 换回 Per-token FFN，下降 0.10%；
- Per-token SwiGLU 换成所有 token 共享的 SwiGLU，下降 0.21%；
- 去掉 Global Token，下降 0.02%。

这些数值来自其生产数据，不能直接外推，但它们表明 block、FFN 类型和 residual 设计必须作为独立变量，而不能假设“原始 RankMixer block 加大即可”。

该论文还报告：Post-Norm 曾短暂获得约 0.01% 的提升但最终出现 NaN，Pre-Norm 更稳定；将 SwiGLU 的 down projection 初始化尺度设为 up/gate 的 0.01 倍，在其消融中获得约 0.03% 的提升。

### 4.3 RankUp：参数增加不等于有效表示维度增加

[RankUp](https://arxiv.org/html/2604.17878v3) 从 effective rank 角度指出，RankMixer 表示在 Token Mixer 后扩秩、在 FFN 后降秩，并可能随深度出现阻尼振荡。其 Randomized Permutation Split 在三个任务上报告 0.06% 到 0.08% 的相对 Realtime AUC 增益，并使 token effective rank 更高、更均匀。

这支持对当前 Autosplit 做字段级有序、随机和分层随机消融，但也说明 Random Split 更可能是增量优化，而不是独自闭合当前 0.00677 的绝对 AUC 差距。

### 4.4 Base 中的模块具有直接业务证据

[FiBiNET](https://arxiv.org/abs/1905.09433) 使用 SENET 动态学习 field importance，并结合 bilinear interaction；[MaskNet](https://arxiv.org/abs/2102.07619) 通过 instance-guided multiplicative mask 说明乘法门控可补充普通 FFN；[DCNv2](https://arxiv.org/abs/2008.13535) 在 web-scale 排序系统中验证了低秩 mixture cross network；[DHEN](https://arxiv.org/abs/2203.11014) 指出不同交互算子学习的信息并不完全重合；[FinalMLP](https://arxiv.org/html/2304.00902) 则说明双流模型、stream-specific feature gating 和高阶 stream fusion 可以形成互补表示。

对当前场景而言，最强证据不是公开数据上的平均结果，而是：同一批 user、item、creative 特征上，BN、层级 SENet、DCNv2 与 MLP 组成的 Base 已经达到 0.86865。

---

# 5. 消融组 A：先反向拆解强 Base

## 5.1 为什么必须先拆 Base

如果不知道 Base 的 0.86865 分别由哪些模块贡献，就无法判断 RankMixer 需要补什么。建议先做两类分析。

### 便宜但非因果的 inference knockout

在已训练 Base 上临时执行：

- 将 SENet gate 强制为 1；
- 将某一级 user、item 或 creative gate 强制为 1；
- 将 DCNv2 residual 输出置 0；
- 将最终 MLP 的某一支路置 0。

它可以快速显示模型敏感位置，但由于模块被移除后其余参数没有重新适应，只能用于排序优先级，不能当作正式因果结论。

### 正式的 retrained ablation

| ID | Base 变体 | 回答的问题 |
|---|---|---|
| BA-0 | 完整 Base | 性能锚点 |
| BA-1 | 去掉 bucket-wise BN | 输入尺度对齐贡献多少 |
| BA-2 | 去掉全部 SENet | 动态特征选择总贡献多少 |
| BA-3 | 只去掉 user SENet | user 自条件门控贡献多少 |
| BA-4 | 只去掉 conditional item SENet | user-item 条件化贡献多少 |
| BA-5 | 只去掉 conditional creative SENet | creative 条件化贡献多少 |
| BA-6 | 去掉 DCNv2，仅保留同规模 MLP | 显式交叉贡献多少 |
| BA-7 | 保留 DCNv2，去掉层级 SENet | SENet 与 DCN 是否互补 |
| BA-8 | 将 hierarchical SENet 换成单一 global SENet | 收益来自一般 gating 还是条件方向 |
| BA-9 | 缩浅最终 MLP | 任务头容量贡献多少 |

### 5.2 决策规则

- 若 BA-6 降幅最大，RankMixer 应优先增加 DCNv2 adapter 或并行 Base cross branch；
- 若 BA-4 或 BA-5 降幅大，说明推广搜 CVR 的非对称条件关系是核心，优先迁移 hierarchical SENet；
- 若 BA-9 降幅明显，先做 head 对齐再比较 backbone；
- 若各模块单独贡献都不大，但完整 Base 显著更强，说明模块组合存在互补，应优先研究双分支与融合。

---

# 6. 消融组 B：拆分 token 数、宽度和 PFFN 容量

## 6.1 最小 2×2 网格

| ID | token 数 | hidden 维度 | 主要目的 |
|---|---:|---:|---|
| SC-0 | 16 | 768 | 当前小模型 |
| SC-1 | 32 | 768 | 只提高 token granularity |
| SC-2 | 16 | 1536 | 只提高 token width 与 head width |
| SC-3 | 32 | 1536 | 当前大模型 |

这四个实验能分离：

- 更多 token 是否有价值；
- 更宽 token 是否有价值；
- 二者是否存在正交增益或负交互；
- 保持 mixing head 宽度 48 的扩容线是否合理。

### 6.2 关键派生量

| 配置 | 投影宽度比 | head 宽度 | 相对 PFFN 规模 |
|---|---:|---:|---:|
| SC-0 | 0.586 | 48 | 1× |
| SC-1 | 1.172 | 24 | 2× |
| SC-2 | 1.172 | 96 | 4× |
| SC-3 | 2.343 | 48 | 8× |

SC-1 与 SC-2 的投影宽度比相同，但 mixing head 宽度和 PFFN 规模不同，是非常有价值的结构对照。

## 6.3 增加一个中间点

可选运行：

| ID | token 数 | hidden 维度 | head 宽度 | 相对 PFFN 规模 |
|---|---:|---:|---:|---:|
| SC-4 | 24 | 1152 | 48 | 3.38× |

SC-0、SC-4、SC-3 都保持 head 宽度为 48，可观察沿当前 scaling path 增加 token 与 width 时，性能是否单调改善。

## 6.4 PFFN 扩张倍数消融

大模型每个 token、每个 block 的普通 PFFN 主要权重约为：

```math
2kD^2.
```

当 $D=1536$ 且 $k=4$ 时，每个 token、每个 block 约有 18.87M 权重；两个 block 合计每 token 约 37.75M。建议运行：

| ID | 大模型 PFFN 扩张倍数 | 目的 |
|---|---:|---|
| FF-0 | 4 | 当前大模型 |
| FF-1 | 2 | 判断大模型是否过度参数化或难优化 |
| FF-2 | 1 | 低容量诊断下限 |
| FF-3 | 共享 FFN | 诊断 token-specific 参数隔离是否真正有益 |
| FF-4 | shared trunk + token-private low-rank adapter | 在共享统计强度与 token 专用能力间折中 |

共享 FFN 不一定是生产候选。TokenMixer-Large 的结果支持 per-token 参数隔离，但在你的数据上，它仍是判断 1.2B PFFN 是否被有效训练的重要诊断对照。

---

# 7. 消融组 C：优化、归一化和初始化

大模型如果沿用小模型全部优化超参数，失败原因可能是训练动力学，而不是表达能力。

## 7.1 必须分开学习率组

建议至少区分：

```text
embedding learning rate
segment projector learning rate
RankMixer block learning rate
prediction head learning rate
```

在同一结构上先搜索 dense backbone 学习率相对于小模型配置的倍率：

```math
\{0.25,\ 0.5,\ 1.0,\ 2.0\}.
```

embedding 学习率保持一致，避免同时改变稀疏表训练。

## 7.2 warmup 与梯度稳定性

建议比较：

- 原 warmup；
- warmup 步数扩大 2 倍；
- warmup 步数扩大 4 倍；
- 有无 global gradient clipping；
- projector、PFFN up/down matrix 和 head 的梯度范数分开记录。

## 7.3 归一化位置

| ID | 归一化 | 目的 |
|---|---|---|
| OP-0 | 当前 Post-LayerNorm | 论文基线 |
| OP-1 | Pre-LayerNorm | 只改 norm 位置 |
| OP-2 | Pre-RMSNorm | 同时验证稳定性与吞吐 |
| OP-3 | Sandwich Norm | 完整对照，不作为首选 |

TokenMixer-Large 报告 Post-Norm 在其环境中出现梯度爆炸风险，而 Pre-Norm 更稳定；RMSNorm 原论文也强调其重缩放不变性与较低计算开销。[RMSNorm](https://arxiv.org/abs/1910.07467)

## 7.4 residual 小初始化

若采用 SwiGLU 或新的 residual block，建议只缩小最后 down projection 的初始化尺度：

```math
\operatorname{scale}(W_{\mathrm{down}})
\in\{1,\ 0.1,\ 0.01\}.
```

不要把 up、gate、down 全部同时缩小。TokenMixer-Large 的消融显示只将 down projection 缩小到 0.01 更优，而所有矩阵都缩小反而下降；这一设计与 [ReZero](https://arxiv.org/abs/2003.04887) 让 residual 分支初始接近恒等映射的思想一致。

## 7.5 需要记录的利用率指标

```text
parameter update norm / parameter norm
per-token gradient norm
per-token activation RMS
GELU or SwiGLU gate distribution
projector output effective rank
PFFN hidden effective rank
NaN / Inf counter
loss scale and clipping frequency
```

若大模型大量参数的 update ratio 长期接近 0，应先减小 PFFN 或改善优化，而不是继续增参。

---

# 8. 消融组 D：Pooling 与任务头

## 8.1 先将任务头与 Base 对齐

| ID | Pooling | Head |
|---|---|---|
| PH-0 | Mean | 当前 RankMixer head |
| PH-1 | Mean | 与 Base 层数、激活和宽度相近的 MLP head |
| PH-2 | Mean | 与 PH-1 同参数量但更深/更窄的 MLP |

PH-1 是最重要的公平性对照之一。若它显著缩小差距，说明当前结果不能归因于 backbone。

## 8.2 保持输出尺度稳定的聚合

Mean Pooling 可替换为：

```math
\mathbf h_{\mathrm{sqrt\text{-}sum}}
=\operatorname{RMSNorm}
\left(
\frac{1}{\sqrt{T}}
\sum_{t=1}^{T}\mathbf x_t
\right).
```

它将每个 token 的直接梯度系数从 $1/T$ 提高到 $1/\sqrt{T}$，再通过 RMSNorm 控制输出尺度。

| ID | 聚合方式 | 初始化 |
|---|---|---|
| PH-3 | sqrt-sum + RMSNorm | 无额外 gate |
| PH-4 | learned scalar weighted pooling | 权重初始化为 $1/T$ |
| PH-5 | Mean + Attention residual pooling | residual 系数初始化为 0 |
| PH-6 | Mean、Max、RMS 三种统计拼接后接 MLP | 直接训练 |

Attention residual 形式为：

```math
\begin{aligned}
\mathbf h_{\mathrm{att}} &= \sum_{t=1}^{T}a_t\mathbf W_v\mathbf x_t,\\
\mathbf h &= \mathbf h_{\mathrm{mean}}
+\beta
\left(
\mathbf h_{\mathrm{att}}-\mathbf h_{\mathrm{mean}}
\right),\\
\beta\big|_{\mathrm{init}}&=0.
\end{aligned}
```

这样训练起点与 Mean Pooling 完全一致。

## 8.3 creative-aware pooling

使用现有 creative embeddings 生成 pooling query，不增加任何新数据：

```math
\mathbf q_c
=\operatorname{MLP}
\left(
\operatorname{Pool}(\mathbf E_c)
\right).
```

再由 $\mathbf q_c$ 计算 32 个 token 的权重。该方案应在普通 learned pooling 已有收益后再运行，以避免将收益错误归因于 creative 条件化。

---

# 9. 消融组 E：Tokenization 与表示秩

## 9.1 大模型字段级分组尺寸

1234 个完整字段分成 32 组时：

```math
1234=18\times39+14\times38.
```

因此字段对齐版本使用 18 个 39-field group 和 14 个 38-field group，每个字段的 17 维 embedding 保持完整。

## 9.2 最小必要实验

| ID | Tokenizer | 目的 |
|---|---|---|
| TK-0 | 当前连续维度 Autosplit | 论文基线 |
| TK-1 | ordered field-aligned split | 隔离“字段完整性”影响 |
| TK-2a/b/c | fixed random field split，3 个预注册 seed | 验证 RankUp 随机划分与 seed 方差 |
| TK-3 | stratified balanced random split | 平衡覆盖率、基数或字段域 |
| TK-4 | domain-preserving split | 保留 user/item/creative 边界 |
| TK-5 | 31 local + 1 global | 保持总 token 数 32 的 Global Token 对照 |

TK-1 与 TK-2 的比较最关键，因为二者都保持完整字段，主要变量仅是字段顺序。

## 9.3 domain-preserving 的具体配置

按照当前三类输入维度占比，可采用：

```text
10 user tokens
21 item tokens
1 creative token
```

具体分组为：

```math
\begin{aligned}
385 &= 5\times39+5\times38,\\
835 &= 16\times40+5\times39,\\
14 &= 1\times14.
\end{aligned}
```

该设计给 creative 一个独立 token，避免其只作为最后 segment 的附属部分。它不是 RankMixer 原论文 Autosplit，也不是 RankUp 的全局随机划分，必须单独命名为 Domain-Preserving Tokenization。

## 9.4 保持 32 tokens 的 Global Token

将 1234 个字段分成 31 个 local groups：

```math
1234=25\times40+6\times39.
```

得到 31 个 local tokens，再由 local token 的 Mean 与 RMS 生成 1 个 global token：

```math
\begin{aligned}
\boldsymbol\mu &= \frac{1}{31}\sum_{t=1}^{31}\mathbf x_t,\\
\mathbf r &= \sqrt{\frac{1}{31}\sum_{t=1}^{31}\mathbf x_t^2+\epsilon},\\
\mathbf g &= \operatorname{MLP}([\boldsymbol\mu;\mathbf r]).
\end{aligned}
```

最终仍为 32 个 token，因此保持 $H=T=32$ 和 $D=1536$ 不变。

## 9.5 机制诊断

每个 tokenizer 必须同时记录：

```text
每个 token 跨 batch 的 effective rank
最小、平均和标准差 token erank
每样本 32×1536 token matrix 的 normalized erank
pairwise cosine similarity
K-means 离散后的 token mutual information
每个 projector 与 PFFN 的 gradient norm
字段缺失率或默认值占比在 token 间的分布
```

理想的 Random Split 结果不是只提高最大 token rank，而是提高最低 rank、降低 token 间 rank 方差并降低冗余。

---

# 10. 消融组 F：逐级迁移 Base 的条件特征选择

输入仍为现有 embeddings：

```math
\mathbf E_u\in\mathbb R^{B\times385\times17},\quad
\mathbf E_i\in\mathbb R^{B\times835\times17},\quad
\mathbf E_c\in\mathbb R^{B\times14\times17}.
```

## 10.1 逐级实验

| ID | RankMixer 前端 | 目的 |
|---|---|---|
| BI-0 | 无 Base 前端 | 大模型锚点 |
| BI-1 | 同一套 bucket-wise BN | 对齐输入尺度 |
| BI-2 | BI-1 + user SENet | user 自条件选择 |
| BI-3 | BI-2 + conditional item SENet | user-item 条件选择 |
| BI-4 | BI-3 + conditional creative SENet | user-item-creative 条件选择 |
| BI-5 | token-level instance mask | 低成本乘法门控对照 |

层级 gate 可写为：

```math
\begin{aligned}
\mathbf g_u &= \operatorname{SE}_u(\widetilde{\mathbf E}_u),\\
\mathbf g_i &= \operatorname{SE}_i([\widetilde{\mathbf E}_u;\widetilde{\mathbf E}_i]),\\
\mathbf g_c &= \operatorname{SE}_c([\widetilde{\mathbf E}_u;\widetilde{\mathbf E}_i;\widetilde{\mathbf E}_c]),\\
\widehat{\mathbf E}_s &= \widetilde{\mathbf E}_s\odot\mathbf g_s.
\end{aligned}
```

若重新实现 gate，使用 identity initialization：

```math
\mathbf g=2\sigma(\mathbf a),
\qquad
\mathbf a\big|_{\mathrm{init}}=0.
```

首选直接复用 Base 中已经稳定的实现与超参数，减少变量。

## 10.2 如何解释结果

- BI-1 显著提升：主要问题是输入尺度和训练条件不公平；
- BI-3 显著提升：user-item 条件化是主要业务先验；
- BI-4 显著提升：creative 信号确实被当前 tokenizer 或 pooling 稀释；
- BI-5 优于完整 field-level SENet：token-level 乘法门控可能是更高效的工程折中。

---

# 11. 消融组 G：显式交叉与异构双流

## 11.1 低秩 token-level DCNv2 adapter

对大模型 token 做独立压缩：

```math
\mathbf X\in\mathbb R^{B\times32\times1536}
\longrightarrow
\mathbf Z\in\mathbb R^{B\times32\times32}
\longrightarrow
\mathbf z_0\in\mathbb R^{B\times1024}.
```

使用 3 层 rank-128 low-rank CrossNet：

```math
\begin{aligned}
\mathbf a_l &= \mathbf V_l^\top\mathbf z_l,\\
\mathbf c_l &= \mathbf U_l\phi(\mathbf a_l)+\mathbf b_l,\\
\mathbf z_{l+1} &= \mathbf z_l+\mathbf z_0\odot\mathbf c_l.
\end{aligned}
```

再映射回 1536 维，与 RankMixer pooled representation 进行 zero-init residual 融合。主要附加权重约为：

```math
32\times1536\times32
+3\times2\times1024\times128
+1024\times1536
\approx3.93\,\mathrm{M}.
```

## 11.2 必须有同参数量对照

| ID | 方案 | 目的 |
|---|---|---|
| CX-0 | 大模型 RankMixer | 锚点 |
| CX-1 | + low-rank DCNv2 adapter | 显式乘法交叉 |
| CX-2 | + 同参数量 MLP adapter | 排除只是增加参数 |
| CX-3 | + Base 完整 DCNv2 并行分支 | 直接复用已验证能力 |
| CX-4 | + MaskBlock/instance mask | 乘法门控替代交叉网络 |

只有 CX-1 显著优于 CX-2，才能证明 DCNv2 的结构性价值。

## 11.3 如何看待“大模型会吸收 DCN”

TokenMixer-Large 在其数据上报告 DCN 增益随规模从 150M 的 0.09%、500M 的 0.04% 降到 700M 的 0。但这不能直接推导当前约 1.2B PFFN 的原始 RankMixer 已经吸收 DCN，因为：

- 它使用的是升级后的 TokenMixer-Large block；
- 业务特征与当前推广搜 CVR 不同；
- 当前 Base 已在同一数据上给出 DCNv2 有效的直接证据。

因此正确做法是实测 CX-1/CX-2，而不是根据参数规模预先删除 DCN。

---

# 12. 消融组 H：升级 RankMixer block

## 12.1 Mixing & Reverting

原始 RankMixer 将 mixing 后的新 token 与 mixing 前同位置 token 相加。两者形状相同，但位置语义已经重新组合。Mixing & Reverting 先在 mixed layout 中建模，再恢复 original-token layout 后做跨 block residual。

| ID | Block 变体 | 目的 |
|---|---|---|
| BL-0 | 原始 RankMixer block | 锚点 |
| BL-1 | 只增加 Mixing & Reverting | 验证 residual 语义错位 |
| BL-2 | BL-1 + compute-matched Per-token SwiGLU | 验证 gated FFN |
| BL-3 | BL-2 + Pre-RMSNorm | 改善稳定性 |
| BL-4 | BL-3 + down projection 0.01 small init | 让 residual 初始接近恒等 |
| BL-5 | 最优两层 block 扩展到 4 层 | 只在前面获益后运行 |

## 12.2 计算量匹配的 SwiGLU

原普通 PFFN 主要参数约为：

```math
2kD^2.
```

Mixing & Reverting block 中若包含两个 Per-token SwiGLU，每个 hidden size 为 $h$，主要参数约为：

```math
6Dh.
```

令二者近似匹配：

```math
h\approx\frac{kD}{3}.
```

当 $k=4$ 时：

```math
\begin{aligned}
D=768&:\quad h\approx1024,\\
D=1536&:\quad h\approx2048.
\end{aligned}
```

因此 BL-2 不应直接使用更大的 hidden size，否则无法区分结构收益和 FLOPs 增加。[GLU Variants Improve Transformer](https://arxiv.org/abs/2002.05202) 为 SwiGLU 相对普通 ReLU/GELU FFN 的有效性提供了基础证据。

## 12.3 深度相关模块的顺序

当前只有两层时，不应优先加入 inter-residual 和 auxiliary loss。它们主要用于更深网络。正确顺序是：

```text
两层 Mixing & Reverting 获益
↓
两层 Pre-RMSNorm + pSwiGLU 获益
↓
扩展到四层
↓
再消融 interval residual 与 auxiliary loss
```

---

# 13. 消融组 I：Base 与 RankMixer 的互补性、融合和蒸馏

## 13.1 零训练成本的 logit blend

在同一验证集上保存：

```math
z_b,\qquad z_r.
```

分别做仿射校准后搜索：

```math
z_{\mathrm{blend}}
=\alpha\widetilde z_b+(1-\alpha)\widetilde z_r,
\qquad
\alpha\in[0,1].
```

若 blend 超过 Base，说明 RankMixer 即使单模型较弱也包含互补排序信号。

## 13.2 Base-preserving residual hybrid

最安全的训练结构为：

```text
shared existing embeddings
        ↓
Base branch: BN → hierarchical SENet → DCNv2 → MLP → base logit
        ↓
RankMixer residual branch: tokenizer → RankMixer → pooling → residual logit
        ↓
base logit + gated residual
```

```math
z=z_{\mathrm{base}}+\gamma r_{\mathrm{rm}},
\qquad
\gamma\big|_{\mathrm{init}}=0.
```

消融：

| ID | 方案 |
|---|---|
| HY-0 | Base only |
| HY-1 | Base frozen + zero-init RankMixer residual |
| HY-2 | HY-1 + 解冻 Base 最终 MLP，小学习率联合训练 |
| HY-3 | Base DCN representation 与 RM representation concat + MLP |
| HY-4 | Base 与 RM 的 multi-head bilinear fusion |
| HY-5 | Base + 同参数量 residual MLP control |

HY-5 用于证明收益是否来自 RankMixer，而不是任何额外残差网络。FinalMLP 关于双流差异化输入和 bilinear stream fusion 的研究，为 HY-3/HY-4 提供了方法依据。

## 13.3 Base teacher distillation

若最终部署必须保持单分支 RankMixer，可以在相同训练样本上使用 Base 作为 teacher，不增加新数据：

```math
\begin{aligned}
p_t^{(\tau)} &= \sigma(z_t/\tau),\\
p_s^{(\tau)} &= \sigma(z_s/\tau),\\
\mathcal L &= (1-\lambda)\operatorname{BCE}(y,\sigma(z_s))\\
&\quad+\lambda\tau^2
\operatorname{KL}
\left(
\operatorname{Bern}(p_t^{(\tau)})
\parallel
\operatorname{Bern}(p_s^{(\tau)})
\right).
\end{aligned}
```

先做 pointwise logit distillation；只有现有训练样本已经天然保留同请求候选集合时，才研究 listwise distillation。[CLID](https://arxiv.org/abs/2312.08727) 提醒 listwise 蒸馏必须同时保护 ranking 与 calibration，不能直接套用普通 listwise loss。

---

# 14. 推荐的分阶段实验矩阵

## Phase 0：在大模型训练期间就必须记录

```text
每日 train AUC、validation AUC、LogLoss
AUC 相对 seen examples 与 wall-clock 的学习曲线
32 个 token 的 gradient norm 与 activation RMS
Tokenizer、每层 Mixing、每层 PFFN 后的 effective rank
Mean Pooling 前后的表示范数
每个参数组的 update-to-weight ratio
NaN、Inf 与 gradient clipping 频率
```

缺少这些信息，训练结束后只能看到一个 AUC，无法区分优化失败与结构失败。

## Phase 1：最低成本原因定位

```text
P1-01  Base 与大模型的 calibrated logit blend
P1-02  Base inference knockout：SENet、DCN、head
P1-03  SC-1：32 tokens，768 hidden
P1-04  SC-2：16 tokens，1536 hidden
P1-05  FF-1：大模型 PFFN expansion 2
P1-06  PH-1：Base-matched MLP head
P1-07  PH-3：sqrt-sum + RMSNorm pooling
P1-08  BI-1：Base bucket-wise BN
```

## Phase 2：迁移当前数据上已验证的能力

```text
P2-01  BI-2：BN + user SENet
P2-02  BI-3：再加 conditional item SENet
P2-03  BI-4：再加 conditional creative SENet
P2-04  CX-1：low-rank DCNv2 adapter
P2-05  CX-2：same-parameter MLP adapter
P2-06  HY-1：Base frozen + zero-init RM residual
```

## Phase 3：升级 RankMixer 自身

```text
P3-01  TK-1：ordered field-aligned split
P3-02  TK-2a/b/c：fixed random split，3 个 seed
P3-03  TK-4：10 user + 21 item + 1 creative token
P3-04  TK-5：31 local + 1 global
P3-05  BL-1：Mixing & Reverting
P3-06  BL-2：compute-matched pSwiGLU
P3-07  BL-3/4：Pre-RMSNorm + down small init
```

## Phase 4：组合与压缩

仅组合两个已经独立获益的模块，并使用 2×2 factorial design。例如：

| 实验 | SENet 前端 | Mixing & Reverting |
|---|---|---|
| F-00 | 无 | 无 |
| F-10 | 有 | 无 |
| F-01 | 无 | 有 |
| F-11 | 有 | 有 |

这样可以判断二者是相加、互补还是冲突。最终若双分支效果最好但推理成本过高，再通过 Base teacher distillation 压回单分支。

---

# 15. 训练预算与晋级规则

## 15.1 三级预算

```text
Stage S0：1% 数据，验证 shape、吞吐、数值稳定与指标管线
Stage S1：5% 数据，筛除明显无效结构
Stage S2：10%～20% 数据，确认学习曲线和机制指标
Stage S3：完整窗口，只给通过前述阶段的候选
```

对于大模型，必须同时比较：

```text
相同 seen examples
相同 wall-clock
各自训练至收敛
```

否则高吞吐模型与低吞吐模型的结论会混淆。

## 15.2 Gap recovery 指标

定义候选相对 RankMixer 的 Base 差距恢复率：

```math
R_{\mathrm{recover}}
=
\frac{\mathrm{AUC}_{\mathrm{candidate}}-\mathrm{AUC}_{\mathrm{RM}}}
{\mathrm{AUC}_{\mathrm{Base}}-\mathrm{AUC}_{\mathrm{RM}}}.
```

它只用于描述，不替代绝对 AUC 和置信区间。

建议晋级条件：

1. 与同阶段 RankMixer 锚点相比，paired AUC difference 的 95% CI 不跨明显负增益；
2. LogLoss 和 calibration 不显著恶化；
3. 方案声称解决的机制指标同步改善；
4. 收益在至少两个初始化或 mapping seed 上方向一致；
5. 训练吞吐、显存和线上 P99 位于可接受 Pareto 前沿。

不要从大量随机 split seed 中挑最好的一个再报告，这会产生 seed selection bias。

---

# 16. 统一评价指标

## 16.1 预测指标

```text
AUC
GAUC 或 UAUC（若现有管线已有）
LogLoss
PR-AUC
CVR calibration bias
ECE 或 Brier Score
```

## 16.2 结构指标

```text
sample-token effective rank
batch-channel effective rank
minimum / mean / std token erank
pairwise token cosine similarity
per-token gradient norm
activation RMS
parameter update ratio
pooling weight entropy
Base/RM representation correlation
Base/RM error overlap
```

## 16.3 系统指标

```text
examples per second
steps per second
MFU
peak HBM
forward / backward time
kernel count
small-op proportion
P50 / P95 / P99 inference latency
```

## 16.4 统计要求

- 使用完全相同的绝对时间窗口和 label maturation 规则；
- 训练与评估样本权重、负采样和 checkpoint 选择规则必须一致；
- 在当前已有的最稳定业务单元上做 paired bootstrap；若只能使用样本级 bootstrap，应明确其相关性局限；
- 同时报绝对 AUC 差、相对差、95% CI 和业务定义的最小有意义增益；
- 不仅看最终 checkpoint，还要看 AUC 对 seen examples 与 wall-clock 的完整曲线。

---

# 17. 专家优先级总评

| 方向 | 优先级 | 若大模型仍弱时的价值 | 成本 | 结论 |
|---|---|---|---|---|
| Base 反向消融 | P0 | 最高 | 中 | 先确定强 Base 到底强在哪里 |
| 规模轴 2×2 拆分 | P0 | 最高 | 中高 | 判断 token、width 和过完备投影哪个失效 |
| Base-matched head + pooling | P0 | 高 | 低 | 32-token Mean Pooling 是明确风险点 |
| Base BN + hierarchical SENet | P0 | 高 | 低到中 | 当前数据上已有直接证据 |
| Base-preserving residual hybrid | P0 | 高 | 中高 | 最可能保住并超过 Base |
| low-rank DCNv2 adapter | P0/P1 | 中高 | 中 | 必须与同参数 MLP 对照 |
| Mixing & Reverting + pSwiGLU | P1 | 中高 | 中 | 大模型不应只用原始 RankMixer block |
| RankUp field split / Global Token | P1 | 中 | 低到中 | 解决表示秩，不太可能独自闭合大差距 |
| Base teacher distillation | P1 | 中 | 训练期中 | 适合最终部署单分支 |
| 减小 PFFN 或 shared/private FFN | P1 | 中 | 中 | 诊断 1.2B PFFN 是否真正被利用 |
| Sparse-Pertoken MoE | P3 | 低，除非 dense scaling 已赢 | 高 | 遵循 first enlarge, then sparse |
| 全字段 Multi-embedding | P3 | 未知 | 很高 | embedding memory 和 lookup 成本未知 |

---

# 18. 最终推荐决策树

### 情况一：大模型明显优于小模型，但仍低于 Base

优先执行：

```text
Base reverse ablation
↓
Base-matched head / pooling
↓
BN + hierarchical SENet
↓
DCNv2 adapter 或 Base residual hybrid
↓
TokenMixer-Large block
```

说明 scaling 有价值，但还缺业务归纳偏置。

### 情况二：大模型与小模型持平

优先执行：

```text
SC-1 / SC-2 拆分
↓
PFFN expansion 2
↓
Pooling ablation
↓
field-aligned / random split
↓
effective-rank 与 update-ratio 诊断
```

说明参数没有转化为有效表示容量。

### 情况三：大模型低于小模型

优先执行：

```text
学习率与 warmup
↓
Pre-RMSNorm
↓
down projection small init
↓
减小 PFFN expansion
↓
Base-matched head / sqrt-sum pooling
```

先解决优化与过度参数化，不要继续加深或加 MoE。

### 情况四：任何单分支 RankMixer 都无法追平 Base，但 blend 有增益

直接进入：

```text
Base frozen + zero-init RankMixer residual
↓
小学习率联合微调
↓
必要时 Base teacher distillation 压回单分支
```

这是当前业务风险最低、科学解释最清楚的路线。

---

# 19. 不建议的实验方式

1. 不建议在大模型失败后立刻加 Sparse MoE；TokenMixer-Large 明确采用 first enlarge, then sparse，稀疏化首先是效率手段，不是修复无效 dense 架构的手段。
2. 不建议一次同时加入 SENet、DCN、Random Split、Global Token、SwiGLU 和新 pooling；即使效果提升也无法归因。
3. 不建议只比较最终 AUC，不记录 train/validation 曲线、effective rank、gradient 和 update ratio。
4. 不建议用单个 Random Split seed 得出结论。
5. 不建议先换 Focal Loss 或 AUC surrogate；Base 与 RankMixer 使用同一标签，首要矛盾仍是表示、交互和优化结构。
6. 不建议只按参数量匹配；还要报告活跃 FLOPs、wall-clock、MFU、HBM 和 P99 延迟。
7. 不建议把 Base 中的模块称为“旧模块”并机械删除；同数据结果比跨场景论文结论更有决策权。

---

# 20. 参考文献

1. [RankMixer: Scaling Up Ranking Models in Industrial Recommenders](https://arxiv.org/abs/2507.15551)
2. [TokenMixer-Large: Scaling Up Large Ranking Models in Industrial Recommenders](https://arxiv.org/abs/2602.06563)
3. [RankUp: Towards High-rank Representations for Large Scale Advertising Recommender Systems](https://arxiv.org/abs/2604.17878)
4. [DCN V2: Improved Deep & Cross Network and Practical Lessons for Web-scale Learning to Rank Systems](https://arxiv.org/abs/2008.13535)
5. [FiBiNET: Combining Feature Importance and Bilinear Feature Interaction for CTR Prediction](https://arxiv.org/abs/1905.09433)
6. [MaskNet: Introducing Feature-Wise Multiplication to CTR Ranking Models by Instance-Guided Mask](https://arxiv.org/abs/2102.07619)
7. [DHEN: A Deep and Hierarchical Ensemble Network for Large-Scale Click-Through Rate Prediction](https://arxiv.org/abs/2203.11014)
8. [FinalMLP: An Enhanced Two-Stream MLP Model for CTR Prediction](https://arxiv.org/abs/2304.00902)
9. [GLU Variants Improve Transformer](https://arxiv.org/abs/2002.05202)
10. [Root Mean Square Layer Normalization](https://arxiv.org/abs/1910.07467)
11. [ReZero is All You Need: Fast Convergence at Large Depth](https://arxiv.org/abs/2003.04887)
12. [Calibration-compatible Listwise Distillation of Privileged Features for CTR Prediction](https://arxiv.org/abs/2312.08727)
