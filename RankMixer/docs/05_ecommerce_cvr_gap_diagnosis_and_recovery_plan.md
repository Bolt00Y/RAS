# 电商推广搜 CVR：RankMixer 与强 Base 的差距诊断及改进路线

> 适用场景：仅使用当前已经存在的 user、item、creative 稀疏特征及 CVR 标签，不引入新的行为序列、外部预训练向量、新标签或额外数据源。
>
> 当前结论基于以下已知事实：每天约 5.5 亿条训练样本，batch size 为 2048；小模型 RankMixer 为 $T=16$、$D=768$、$L=2$；大模型 RankMixer 为 $T=32$、$D=1536$、$L=2$；强 Base 为 `BN -> hierarchical SENet -> DCNv2 -> MLP`。

## 1. 执行摘要

当前小模型 RankMixer 的 CVR AUC 为 $0.86188$，同期 Base 为 $0.86865$，绝对差值为：

```math
\Delta \mathrm{AUC}
=0.86865-0.86188
=0.00677.
```

若以 $1-\mathrm{AUC}$ 作为一个粗略的排序错误尺度，则 RankMixer 相对 Base 的错误量增幅约为：

```math
\frac{1-0.86188}{1-0.86865}-1
\approx 5.15\%.
```

这个差距不能被视为普通随机波动。RankMixer 原论文在自身工业数据上将 $0.0001$ AUC 视为可信显著变化的量级；虽然该阈值不能直接迁移到当前业务，但当前差值是这一量级的 $67.7$ 倍，足以说明必须先做结构诊断，而不是简单等待更久或直接把模型放大。[RankMixer](https://arxiv.org/html/2507.15551v3)

若每天 5.5 亿条数据都完整训练一次，则 10 天约对应：

```math
N_{\mathrm{sample}}\approx 5.5\times 10^9,
\qquad
N_{\mathrm{step}}\approx
\frac{5.5\times 10^9}{2048}
\approx 2.69\times 10^6.
```

这只是数量级估算，但它意味着“小模型仅仅因为数据不足而落后”的解释优先级不高；除非训练曲线显示 RankMixer 的验证 AUC 仍持续快速上升，否则继续原样训练很难自然弥合 $0.00677$ 的差距。

### 核心决策

1. **暂不建议直接给原始大模型 RankMixer 完整训练预算。** 在相同层数和 PFFN 扩张率下，PFFN 主体参数及 FLOPs 近似正比于 $TD^2$，因此 $T=32,D=1536$ 相对 $T=16,D=768$ 放大约 8 倍，但没有补回 Base 已证明有效的归纳偏置。
2. **第一优先级不是“让 RankMixer 单独替代 Base”，而是“保留 Base 的强能力，让 RankMixer 学习剩余误差”。** 最安全的生产研究路线是 Base-preserving residual hybrid。
3. **当前实验不是纯粹的 backbone 对比。** 两个模型同时在输入归一化、动态特征选择、显式交叉、输出聚合和任务头上存在差异，不能把全部差距归因于 Token Mixing。
4. **最可能造成差距的三个因素是：** 缺失层级条件 SENet、缺失 DCNv2 显式乘法交叉、Mean Pooling 加弱任务头导致信息稀释。
5. **RankUp 与 TokenMixer-Large 值得做，但它们更适合解决表示秩与可扩展性问题，单独使用不太可能保证闭合当前 $0.00677$ 的巨大差距。**

---

## 2. 当前两个系统到底差在哪里

### 2.1 当前 RankMixer

```text
user / item / creative sparse embeddings
        -> flatten and concat
        -> [B, 20978]
        -> ordered Autosplit
        -> T independent projectors
        -> [B, T, D]
        -> 2 RankMixer blocks
        -> Mean Pooling
        -> CVR prediction head
```

小模型：

```math
T=16,\qquad D=768,\qquad L=2.
```

大模型：

```math
T=32,\qquad D=1536,\qquad L=2.
```

### 2.2 当前 Base

```text
user / item / creative sparse embeddings
        -> bucket-wise BatchNorm
        -> hierarchical SENet
             user gate      <- user
             item gate      <- [user, item]
             creative gate  <- [user, item, creative]
        -> gated user / item / creative embeddings
        -> concat
        -> DCNv2
        -> MLP
        -> CVR prediction
```

### 2.3 这不是单变量实验

| 能力 | Base | 当前 RankMixer | 可能影响 |
|---|---|---|---|
| 异构输入尺度对齐 | bucket-wise BN | 未说明具有同等处理 | 影响投影层与梯度尺度 |
| 样本级特征选择 | 层级条件 SENet | 无 | 影响 user/item/creative 的动态重要性 |
| user-item-creative 非对称关系 | 明确编码 | 只通过固定 mixing 间接学习 | 推广搜 CVR 中可能非常关键 |
| 显式乘法交叉 | DCNv2 | 无 | 强 user × item、item × creative 信号可能学习更慢 |
| 隐式高阶交互 | MLP | PFFN | 两者均有，但参数共享结构不同 |
| 输出聚合 | DCN 输出进入 MLP | Mean Pooling | Mean Pooling 可能稀释强 token |
| 任务头 | 多层 MLP | 当前描述未体现同等 MLP | 可能形成明显容量差异 |
| 输入分组 | user/item/creative 结构保留 | 连续 Autosplit | 可能损失层级条件先验 |

因此，科学上正确的问题不是：

> RankMixer 为什么不如 Base？

而是：

> 当归一化、条件选择、显式交叉和任务头保持可比时，RankMixer backbone 是否还能贡献额外的交互能力？

---

## 3. 根因假设：按优先级排序

## 3.1 H1：缺失层级条件特征选择

**优先级：最高。**

Base 的 SENet 不是普通的全局静态特征权重，而是具有明确的条件方向：

```text
user importance      depends on user
item importance      depends on user + item
creative importance  depends on user + item + creative
```

这与电商推广搜 CVR 的因果结构非常吻合：

- 同一 item 对不同 user 的价值不同；
- 同一 creative 对不同 user-item 组合的影响不同；
- creative 字段只有 14 个，若没有显式重标定，容易被 385 个 user 字段和 835 个 item 字段淹没。

FiBiNET 通过 SENET 动态学习 field importance，并将其与显式双线性交互结合；MaskNet 进一步表明，实例条件的乘法 mask 可以补充纯加性 FFN 对乘法关系建模效率不足的问题。[FiBiNET](https://arxiv.org/abs/1905.09433) · [MaskNet](https://arxiv.org/abs/2102.07619)

**可证伪实验：** 在 RankMixer tokenizer 之前复用 Base 的同一套 BN 与层级 SENet，保持后续 RankMixer 不变。若 AUC 大幅回升，则差距主要来自输入条件化，而不是 Token Mixing 本身。

## 3.2 H2：缺失 DCNv2 显式乘法交叉

**优先级：最高。**

DCNv2 直接学习 bounded-degree 显式交叉，并在 web-scale 学习排序系统中验证了低秩和 mixture 结构的有效性。[DCNv2](https://arxiv.org/abs/2008.13535)

RankMixer 的固定 Token Mixing 本身是 reshape / split / concat；真正的非线性交互主要由 PFFN 完成。它理论上能够拟合乘法关系，但在有限层数下未必能像 DCNv2 一样高效学习：

```text
user purchase power × item price
user category preference × item category
user-item compatibility × creative type
```

当前 Base 在完全相同数据上明显领先，是 DCNv2 对当前数据分布有效的直接业务证据。DHEN 也指出，不同 interaction operator 即使声称学习相同阶数，实际捕获的信息仍可能不重合，因此异构交互模块的并行组合具有科学合理性。[DHEN](https://arxiv.org/abs/2203.11014)

**可证伪实验：** 在同一 RankMixer 上增加一个受控的 DCNv2 分支，并设置同参数量宽化 PFFN 对照。若 DCN 分支显著优于宽化 PFFN，则收益来自显式交叉，而不是单纯增加参数。

## 3.3 H3：Mean Pooling 与任务头过弱

**优先级：高。**

Mean Pooling 隐含假设是所有 token 对当前样本同等重要：

```math
\mathbf h_{\mathrm{mean}}
=\frac{1}{T}\sum_{t=1}^{T}\mathbf x_t.
```

但当前 16 个 token 的信息量很可能不均衡，最后一个 token 还包含全部 creative 信息。均值池化可能把少数高价值信号平均掉。Base 则在 DCNv2 后使用 MLP，对交叉表示进行任务相关非线性压缩。

FinalMLP 的工业与公开实验表明，两路表示、特征选择和 interaction aggregation 可以优于单一普通 MLP；这支持“输出融合方式本身就是重要建模变量”，而不是把所有能力都交给 backbone。[FinalMLP](https://arxiv.org/abs/2304.00902)

**可证伪实验：** 保持 RankMixer backbone 完全不变，只比较 Mean Pooling、可学习加权池化、Mean + Attention 残差池化，以及与 Base 同规模的 MLP head。

## 3.4 H4：Autosplit 的表示秩与 token 信息失衡

**优先级：中。**

当前 Autosplit 严格符合 RankMixer 原始公式，不是实现错误。但 RankUp 指出，固定 Autosplit 或语义分组可能让强相关字段集中，造成 token 冗余和有效秩不足。RankUp 通过字段级随机排列、Multi-embedding 和 Global Token 提高初始表示多样性，并在工业广告任务上获得一致增益。[RankUp](https://arxiv.org/html/2604.17878v3)

这可能解释为什么参数很多却没有转化成有效表示容量。不过 RankUp 中单项 Random Split 的 Realtime AUC 增益约为 $0.06\%$ 到 $0.08\%$，远小于当前绝对 AUC 差距，因而它更像重要补充，而不是唯一解法。

## 3.5 H5：原始 RankMixer block 的残差与归一化设计限制效果

**优先级：中。**

TokenMixer-Large 指出，原始 RankMixer 在 mixing 前后直接做位置对应残差，存在 token 语义错位；其后续工作通过 Mixing & Reverting、Per-token SwiGLU、Pre-RMSNorm、inter-residual 和辅助损失改善训练。4B 模型消融中，去掉 Mixing & Reverting、Residual 或将 Per-token SwiGLU 换回 Per-token FFN 都会造成 AUC 下降。[TokenMixer-Large](https://arxiv.org/html/2602.06563v2)

但当前只有 2 层，因此深层梯度问题不太可能独自造成 $0.00677$ 的全部差距。该方向应在补回 Base 关键能力后实施。

## 3.6 H6：优化配置或训练仍未收敛

**优先级：未知，必须通过曲线判断。**

需要同时看：

- 训练 AUC 与验证 AUC；
- 每日增益斜率；
- LogLoss；
- projector、PFFN 和任务头梯度范数；
- activation RMS；
- 不同 token 的参数更新比。

若 RankMixer 的训练 AUC 也低于 Base，说明更像欠拟合、优化困难或归纳偏置不合适；若训练 AUC 高而验证 AUC低，则更像过拟合、校准或分布泛化问题。

## 3.7 H7：小模型容量不足

**当前优先级：低。**

RankMixer 的 PFFN 参数近似为：

```math
P_{\mathrm{PFFN}}
\approx 2kLTD^2.
```

因此大模型与小模型的 PFFN 比例为：

```math
\frac{32\times1536^2}{16\times768^2}=8.
```

若 $k=4$、$L=2$，仅 PFFN 权重约为：

```math
P_{\mathrm{small,PFFN}}\approx 151\,\mathrm{M},
\qquad
P_{\mathrm{large,PFFN}}\approx 1.208\,\mathrm{B}.
```

Tokenizer 权重则从约 $16.11\,\mathrm{M}$ 增加到 $32.22\,\mathrm{M}$。这是一笔很大的预算。如果小模型的主要问题是缺少条件选择和显式交叉，8 倍扩容只会放大不匹配的结构。

---

## 4. Phase 0：在改模型前必须完成的诊断

## 4.1 训练管线一致性审计

Base 与 RankMixer 必须确认以下项目完全一致：

```text
absolute train / validation / test time windows
label maturation window
sample weights
positive / negative sampling
embedding table definitions and initialization policy
feature order and missing-value policy
optimizer and learning-rate schedule
checkpoint selection rule
precision and gradient clipping
```

最重要的是增加两个桥接对照：

```text
R-BN:     current RankMixer + the same bucket-wise BN as Base
R-BN-H:   R-BN + the same prediction head depth as Base
```

如果这两个简单对照已经显著缩小差距，说明当前实验存在 preprocessing/head 不公平，而不是 RankMixer backbone 本身失效。

## 4.2 预测互补性测试

在同一个验证集上保存 Base 与 RankMixer logits：

```math
z_b,\qquad z_r.
```

先分别做只使用验证集拟合的仿射校准：

```math
\widetilde z_b=a_bz_b+c_b,
\qquad
\widetilde z_r=a_rz_r+c_r.
```

再搜索简单 logit blend：

```math
z_{\mathrm{blend}}
=\alpha\widetilde z_b+(1-\alpha)\widetilde z_r,
\qquad \alpha\in[0,1].
```

判断：

- 若最优 blend 明显超过 Base，说明 RankMixer 虽然单模型较弱，但包含互补信号；应优先做并行 residual hybrid。
- 若 blend 无法超过 Base，说明 RankMixer 主要是在重复但更差地表达 Base 已有信号；应先改输入条件化和显式交叉。

同时报告：

```text
Pearson / Spearman score correlation
Base-correct RankMixer-wrong samples
Base-wrong RankMixer-correct samples
positive and negative score distributions
calibration bias
```

这是决定是否值得做双分支的最低成本实验。

## 4.3 表示与梯度诊断

对每层 token matrix：

```math
\mathbf H_b^{(l)}\in\mathbb R^{T\times D},
```

计算 effective rank：

```math
p_i=\frac{\sigma_i}{\sum_j\sigma_j},
\qquad
\operatorname{erank}(\mathbf H_b^{(l)})
=\exp\left(-\sum_i p_i\log(p_i+\epsilon)\right).
```

建议采样：

```text
tokenizer output
block 1 mixing output
block 1 PFFN output
block 2 mixing output
block 2 PFFN output
```

并记录：

```text
pairwise token cosine similarity
per-token batch variance
per-token projector gradient norm
per-token PFFN gradient norm
parameter update ratio
activation RMS
```

若只有少数 token 高秩、高梯度，RankUp-style split 和动态 gating 的优先级上升。

---

# 5. 方案一：Base-Preserving Residual RankMixer

## 5.1 结论

**这是最推荐、最有希望至少守住 Base 并进一步获得增益的方案。**

不要让 RankMixer 从零重新学习 Base 已经很好地掌握的条件选择与显式交叉，而是让它只学习 Base 的残差错误。

## 5.2 架构

```text
shared sparse embeddings
        |
        +--> existing Base branch
        |       BN -> hierarchical SENet -> DCNv2 -> MLP -> z_base
        |
        +--> RankMixer residual branch
                BN / SENet output or raw embeddings
                -> tokenizer
                -> RankMixer / TokenMixer-Large
                -> task-aware pooling
                -> residual head -> r_rm

final logit = z_base + gated residual
```

定义：

```math
z=z_{\mathrm{base}}+\gamma\,r_{\mathrm{rm}}.
```

最安全的初始化是：

```math
\gamma=0
```

或者将 residual head 最后一层权重 zero-init，使模型起点严格等价于 Base。

也可使用样本条件 gate：

```math
\gamma(\mathbf x)
=\gamma_{\max}\,\sigma(\operatorname{MLP}(\mathbf c)),
```

但首版应先用全局标量，避免 gate 本身引入混淆。

## 5.3 训练方式

### BR-1：冻结 Base，训练 residual branch

```text
load converged Base
freeze Base parameters
train only RankMixer residual branch and gamma
```

优点：

- 不会破坏 Base 已有能力；
- 直接验证 RankMixer 是否具有互补信息；
- 训练稳定，失败时可以安全回退。

### BR-2：小学习率联合微调

当 BR-1 确认有增益后：

```text
Base branch LR = residual branch LR × 0.05~0.2
```

只解冻 Base 的 DCN 顶层与 MLP，embedding 和早期 gating 可继续冻结或使用更小学习率。

## 5.4 必要消融

```text
BR-0: Base only
BR-1: Base + zero-init RankMixer residual, Base frozen
BR-2: BR-1 + unfreeze Base MLP
BR-3: BR-1 + unfreeze top DCNv2 layers
BR-4: Base + same-parameter residual MLP control
```

`BR-4` 用于判断收益是否来自 RankMixer 结构，而不是简单增加一个残差网络。

## 5.5 专家评估

| 维度 | 评估 |
|---|---|
| 闭合当前差距的概率 | 高 |
| 超过 Base 的潜力 | 高，前提是预测互补性存在 |
| 实现风险 | 中 |
| 推理成本 | 高于单模型，需要后续压缩 |
| 科学解释性 | 强，初始等价 Base，增益可归因于 residual branch |

---

# 6. 方案二：复用 Base 的 BN 与层级 SENet 作为 RankMixer 前端

## 6.1 目标

在不改变数据的前提下，把 Base 已验证有效的动态特征选择能力迁移到 RankMixer。

输入：

```math
\mathbf E_u\in\mathbb R^{B\times385\times17},
\quad
\mathbf E_i\in\mathbb R^{B\times835\times17},
\quad
\mathbf E_c\in\mathbb R^{B\times14\times17}.
```

经过与 Base 完全相同的 bucket-wise BN：

```math
\widetilde{\mathbf E}_u,
\quad
\widetilde{\mathbf E}_i,
\quad
\widetilde{\mathbf E}_c.
```

层级 gate：

```math
\begin{aligned}
\mathbf g_u &= \operatorname{SE}_u(\widetilde{\mathbf E}_u),\\
\mathbf g_i &= \operatorname{SE}_i([\widetilde{\mathbf E}_u;\widetilde{\mathbf E}_i]),\\
\mathbf g_c &= \operatorname{SE}_c([\widetilde{\mathbf E}_u;\widetilde{\mathbf E}_i;\widetilde{\mathbf E}_c]).
\end{aligned}
```

重标定：

```math
\begin{aligned}
\widehat{\mathbf E}_u &= \widetilde{\mathbf E}_u\odot\mathbf g_u,\\
\widehat{\mathbf E}_i &= \widetilde{\mathbf E}_i\odot\mathbf g_i,\\
\widehat{\mathbf E}_c &= \widetilde{\mathbf E}_c\odot\mathbf g_c.
\end{aligned}
```

再按当前 RankMixer Autosplit 生成 tokens。

## 6.2 初始化

若新写 gate，应使初始 gate 为 1：

```math
\mathbf g=2\sigma(\mathbf a),
\qquad
\mathbf a\big|_{\mathrm{init}}=0.
```

但由于 Base 已有成熟模块，首选直接复用同一实现与超参数，减少不必要变量。

## 6.3 消融顺序

```text
HS-0: current RankMixer
HS-1: + Base bucket-wise BN
HS-2: HS-1 + user SENet
HS-3: HS-2 + conditional item SENet
HS-4: HS-3 + conditional creative SENet
```

不能首轮直接只跑完整 HS-4，否则无法判断主要收益来自哪一级条件化。

## 6.4 专家评估

| 维度 | 评估 |
|---|---|
| 闭合差距概率 | 中高 |
| 成本 | 低到中，已有成熟实现 |
| 线上风险 | 低 |
| 最大价值 | 直接迁移当前数据上已经被验证的归纳偏置 |

---

# 7. 方案三：RankMixer + DCNv2 显式交叉

## 7.1 首选：复用现有 Base DCNv2 的并行分支

```text
BN + hierarchical SENet output
        |
        +--> existing DCNv2 -> base-style MLP representation h_dcn
        |
        +--> tokenizer -> RankMixer -> pooled representation h_rm

fusion -> final MLP -> CVR
```

融合可先用 concatenation：

```math
\mathbf h
=\operatorname{MLP}_{\mathrm{fusion}}([\mathbf h_{\mathrm{dcn}};\mathbf h_{\mathrm{rm}}]).
```

更安全的是 residual fusion：

```math
\mathbf h
=\mathbf h_{\mathrm{dcn}}
+\alpha\operatorname{Proj}(\mathbf h_{\mathrm{rm}}),
\qquad \alpha\big|_{\mathrm{init}}=0.
```

## 7.2 低延迟版本：token-level low-rank DCNv2 adapter

小模型中，将每个 token 从 768 维压到 32 维：

```math
\mathbf X\in\mathbb R^{B\times16\times768}
\rightarrow
\mathbf Z_0\in\mathbb R^{B\times16\times32}
\rightarrow
\mathbf z_0\in\mathbb R^{B\times512}.
```

使用 3 层 rank-64 low-rank CrossNet：

```math
\begin{aligned}
\mathbf a_l &= \mathbf V_l^\top\mathbf z_l,\\
\mathbf c_l &= \mathbf U_l\phi(\mathbf a_l)+\mathbf b_l,\\
\mathbf z_{l+1} &= \mathbf z_l+\mathbf z_0\odot\mathbf c_l.
\end{aligned}
```

再投影到 768 维并 zero-init residual 融合。主要参数约为：

```math
16\times768\times32
+3\times2\times512\times64
+512\times768
\approx 0.98\,\mathrm{M}.
```

大模型可使用每 token 压缩到 32 维、CrossNet rank 128，总附加参数约数百万，远小于大模型 PFFN。

## 7.3 必须有的对照

```text
DC-0: RankMixer
DC-1: RankMixer + low-rank DCNv2
DC-2: RankMixer + same-parameter wider PFFN
DC-3: Base-DCN branch + RankMixer residual
```

若 DC-1 优于 DC-2，才能证明显式交叉的结构价值。

## 7.4 专家评估

| 维度 | 评估 |
|---|---|
| 闭合差距概率 | 中高 |
| 理论依据 | 强，Base 已提供同数据业务证据 |
| 工程风险 | 中，注意小算子与 MFU |
| 长期策略 | 小模型保留 DCN；超大模型再评估是否被纯 backbone 吸收 |

TokenMixer-Large 报告 DCN 对 150M 模型仍有正收益，但随着模型增长到 700M，其增益逐渐趋近于零；这说明当前应实测，不应机械地永久保留或机械地删除 DCN。[TokenMixer-Large Appendix A.3](https://arxiv.org/html/2602.06563v2)

---

# 8. 方案四：任务相关池化与强任务头

## 8.1 与 Base 对齐的 MLP head

先做最简单的公平对照：

```text
Mean Pooling -> MLP head -> logit
```

MLP 层数、激活和中间维度尽量与 Base 的最终 MLP 对齐。该实验成本很低，但可能显著缩小“backbone 后任务压缩能力”的差异。

## 8.2 Mean + Attention 残差池化

定义可学习 task query：

```math
a_t
=\operatorname{softmax}
\left(
\frac{\mathbf q^\top\mathbf W_k\mathbf x_t}{\sqrt{d_k}}
\right),
```

```math
\mathbf h_{\mathrm{att}}
=\sum_{t=1}^{T}a_t\mathbf W_v\mathbf x_t.
```

使用零初始化插值：

```math
\mathbf h
=\mathbf h_{\mathrm{mean}}
+\beta(\mathbf h_{\mathrm{att}}-\mathbf h_{\mathrm{mean}}),
\qquad \beta\big|_{\mathrm{init}}=0.
```

这样初始行为与 Mean Pooling 完全一致，收益可以归因于任务相关聚合。

## 8.3 creative-aware pooling

由于 creative 只有 14 个字段，可从原始 creative embedding 产生一个 query context：

```math
\mathbf q_c
=\operatorname{MLP}
\left(
\operatorname{Pool}(\mathbf E_c)
\right),
```

再用于 token pooling。该方案不需要新数据，但应作为普通 task-query pooling 获益后的二阶段实验。

## 8.4 消融

```text
PH-0: Mean Pooling + current head
PH-1: Mean Pooling + Base-matched MLP head
PH-2: learned weighted pooling + Base-matched head
PH-3: Mean + Attention residual pooling
PH-4: creative-aware pooling
```

## 8.5 专家评估

| 维度 | 评估 |
|---|---|
| 成本 | 很低 |
| 优先级 | P0 |
| 单独闭合全部差距 | 不确定，但值得最先验证 |
| 风险 | attention 过拟合少数 token，需监控权重熵 |

---

# 9. 方案五：RankUp-style 输入表示升级

## 9.1 字段级固定随机划分

当前 1234 个完整字段先形成：

```math
\mathbf E\in\mathbb R^{B\times1234\times17}.
```

对字段索引生成模型版本级固定 permutation，再均衡分组。

### 小模型 $T=16$

```math
1234=2\times78+14\times77.
```

因此：

```text
2 groups with 78 complete fields
14 groups with 77 complete fields
```

### 大模型 $T=32$

```math
1234=18\times39+14\times38.
```

因此：

```text
18 groups with 39 complete fields
14 groups with 38 complete fields
```

必须同时运行：

```text
RU-F0: ordered field-aligned split
RU-R1: fixed random split, seed 1
RU-R2: fixed random split, seed 2
RU-R3: fixed random split, seed 3
RU-S1: stratified balanced random split
```

`RU-F0` 用于隔离“保持完整字段”与“随机化”两个因素。

## 9.2 保持 token 数不变的 Global Adapter

由于当前 RankMixer 要求 $H=T$，直接增加一个 token 会改变结构。可先使用保持 $T$ 不变的全局上下文注入：

```math
\mathbf g
=\operatorname{MLP}
\left(
[\operatorname{Mean}(\mathbf X_0);
\operatorname{RMS}(\mathbf X_0)]
\right),
```

```math
\widehat{\mathbf x}_t
=\mathbf x_t+\alpha_t\mathbf g,
\qquad \alpha_t\big|_{\mathrm{init}}=0.
```

这不是严格的 RankUp Global Token，但能用最小结构改动验证全局上下文是否有价值。

严格版本可采用：

```text
T=16: 15 local + 1 global
T=32: 31 local + 1 global
```

## 9.3 Multi-embedding 的优先级

不建议立即复制全部 1234 个字段的 embedding tables。当前稀疏表基数未知，而 Multi-embedding 的主要成本可能来自 embedding memory、optimizer state 和 lookup 带宽，而不是 dense 参数。

只有 Random Split 与 Global Adapter 已有稳定收益后，才对少量 anchor fields 做 Selective Multi-embedding。

## 9.4 专家评估

| 维度 | 评估 |
|---|---|
| 机制价值 | 中高，改善表示秩与 token 均衡性 |
| 单独闭合当前差距 | 低到中 |
| 工程成本 | Random Split 低，Multi-embedding 高 |
| 推荐顺序 | Random Split -> Global Adapter -> selective Multi-embedding |

---

# 10. 方案六：升级为 TokenMixer-Large Block

## 10.1 为什么大模型应优先改 block，而不是原样放大

TokenMixer-Large 指出原始 RankMixer 存在 residual semantic misalignment，并在工业消融中发现 Mixing & Reverting 和 Per-token SwiGLU 是影响最大的组件之一。[TokenMixer-Large](https://arxiv.org/html/2602.06563v2)

首版 block：

```text
X
 -> Pre-RMSNorm
 -> Mix
 -> per-token SwiGLU in mixed layout
 -> Revert
 -> per-token SwiGLU in original layout
 -> residual
```

## 10.2 计算量匹配

假设原 PFFN 扩张倍数为 $k$，单个普通 PFFN 的主要参数为：

```math
2kD^2.
```

Mixing & Reverting block 包含两个 pSwiGLU。若每个 pSwiGLU hidden size 为 $h$，总主要参数约为：

```math
6Dh.
```

为了计算匹配：

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

这样可以区分结构收益与单纯增加 FLOPs。

## 10.3 实验顺序

```text
TM-0: original RankMixer block
TM-1: Mixing & Reverting + compute-matched GELU/PFFN
TM-2: TM-1 + compute-matched per-token SwiGLU
TM-3: TM-2 + Pre-RMSNorm + small down initialization
TM-4: only after TM-3 wins, increase depth to L=4
```

当前 $L=2$ 时，不要优先加入 inter-residual 和辅助损失；它们主要面向深层训练。

## 10.4 专家评估

| 维度 | 评估 |
|---|---|
| 小模型收益预期 | 中 |
| 大模型必要性 | 高 |
| 单独闭合全部差距 | 低到中 |
| 工程风险 | 中，需要高效 revert 与 grouped GEMM |

---

# 11. 方案七：利用强 Base 做知识蒸馏

该方案不增加新数据，只使用同一样本上的 Base 预测或中间表示。

## 11.1 Pointwise logit distillation

Teacher 为已收敛 Base，Student 为 RankMixer：

```math
p_t^{(\tau)}=\sigma(z_t/\tau),
\qquad
p_s^{(\tau)}=\sigma(z_s/\tau).
```

总损失：

```math
\mathcal L
=(1-\lambda)\operatorname{BCE}(y,\sigma(z_s))
+\lambda\tau^2
\operatorname{KL}
\left(
\operatorname{Bern}(p_t^{(\tau)})
\parallel
\operatorname{Bern}(p_s^{(\tau)})
\right).
```

建议把 $\lambda$、$\tau$ 作为小规模搜索变量，并严格检查 calibration。

## 11.2 中间表示蒸馏

若能读取 Base 的 DCNv2/MLP 隐层表示 $\mathbf h_t$，可增加：

```math
\mathcal L_{\mathrm{repr}}
=\left\|
\operatorname{Proj}(\mathbf h_s)-\operatorname{StopGrad}(\mathbf h_t)
\right\|_2^2.
```

但中间表示维度和几何可能不同，首版先做 logit distillation。

## 11.3 可选 listwise distillation

只有当当前样本已经包含同请求候选集合时，才研究 listwise distillation；这不需要新数据，但不能假设请求分组一定可用。CTR 工业研究指出，listwise 蒸馏需要同时保护 ranking 与 calibration，不能直接套用普通 listwise loss。[CLID](https://arxiv.org/abs/2312.08727)

## 11.4 专家评估

| 维度 | 评估 |
|---|---|
| 成本 | 训练期中等，推理期无额外成本 |
| 闭合差距概率 | 中 |
| 最适用场景 | 希望最终仍部署单分支 RankMixer |
| 风险 | Student 复制 Teacher 偏差，校准可能恶化 |

---

# 12. 大模型训练策略：拆开 $T$ 与 $D$

当前从 $T=16,D=768$ 直接跳到 $T=32,D=1536$，同时改变了 token granularity 和 width，无法判断哪个轴有效。

应先做四格实验：

| ID | Token 数 | 宽度 | 目的 |
|---|---:|---:|---|
| SC-0 | 16 | 768 | 当前小模型 |
| SC-1 | 32 | 768 | 只增加 token granularity |
| SC-2 | 16 | 1536 | 只增加 width |
| SC-3 | 32 | 1536 | 当前计划的大模型 |

保持：

```text
same data windows
same seen examples
same optimizer policy
same L=2
same PFFN expansion ratio
same output head
```

同时报告：

```text
same-seen-examples result
same-wall-clock result
train-to-convergence result
```

## 12.1 分阶段预算

在完整长周期训练前，先在相同时间窗口与数据分布上做：

```text
1% pipeline correctness
5% architecture screening
10% confirmation
full-scale only for survivors
```

每个阶段都绘制 AUC/LogLoss 相对 seen examples 和 wall-clock 的学习曲线。

## 12.2 大模型进入全量训练的条件

至少满足：

1. 相同 seen examples 下，验证 AUC 的增益方向稳定；
2. 多个初始化 seed 的置信区间不与明显负增益重叠；
3. LogLoss 与 calibration 不恶化；
4. 表示 effective rank 或 token redundancy 指标与方案假设一致；
5. 训练吞吐、HBM 与线上 P99 处于可接受 Pareto 前沿。

如果大模型只在训练 AUC 上提高、验证 AUC 没有改善，应停止继续放大并优先处理归纳偏置与正则化。

---

# 13. 推荐实验矩阵

## Phase A：低成本原因定位

```text
A0  Base reproducibility, at least 3 seeds/checkpoints
A1  Current RankMixer reproducibility
A2  RankMixer + Base BN
A3  RankMixer + Base-matched MLP head
A4  Mean + Attention residual pooling
A5  calibrated Base/RankMixer logit blend
```

**目标：** 判断差距是否主要来自管线、任务头和预测互补性。

## Phase B：迁移 Base 的关键能力

```text
B1  BN + user SENet + RankMixer
B2  B1 + conditional item SENet
B3  B2 + conditional creative SENet
B4  Base + zero-init RankMixer residual
B5  RankMixer + low-rank DCNv2 adapter
B6  Base DCNv2 branch + RankMixer branch
```

**目标：** 找到能否守住 Base 的桥接结构。

## Phase C：RankMixer 自身升级

```text
C1  ordered field-aligned split
C2  fixed random split, 3 mapping seeds
C3  global adapter
C4  Mixing & Reverting
C5  compute-matched per-token SwiGLU + Pre-RMSNorm
C6  Base-teacher logit distillation
```

**目标：** 在桥接结构上提升表示秩、block 质量和单分支能力。

## Phase D：规模扩展

```text
D1  T=32, D=768
D2  T=16, D=1536
D3  T=32, D=1536
D4  best architecture at L=4
D5  Sparse-Pertoken MoE only after dense enlargement wins
```

---

# 14. 方案优先级总评

| 方向 | 优先级 | 单独闭合差距可能性 | 工程成本 | 专家结论 |
|---|---|---|---|---|
| Base-preserving residual hybrid | P0 | 高 | 中高 | 最安全、最符合当前业务证据 |
| Base BN + hierarchical SENet 前端 | P0 | 中高 | 低中 | 应最先迁移的归纳偏置 |
| Base-matched MLP head + task pooling | P0 | 中 | 低 | 低成本必要公平对照 |
| DCNv2 并行/低秩 adapter | P0/P1 | 中高 | 中 | Base 已证明有效，必须受控验证 |
| Base teacher distillation | P1 | 中 | 训练期中 | 适合最终压回单分支 |
| RankUp Random Split / Global Adapter | P1 | 低中 | 低中 | 改善有效秩，但不应被当作唯一解 |
| TokenMixer-Large block | P1 | 低中 | 中 | 大模型前应完成的结构升级 |
| 原样训练 $T=32,D=1536$ | P2 | 未知且当前偏低 | 很高 | 不建议立即给完整预算 |
| Sparse MoE | P3 | 当前不明确 | 高 | 先证明 dense capacity scaling 有效 |
| 全字段 Multi-embedding | P3 | 未知 | 很高 | embedding 成本未知，不宜首轮采用 |

---

# 15. 不建议的做法

## 15.1 不建议直接完整训练当前大模型

小模型已经显著落后 Base，而大模型只扩大相同结构。必须先用 SC-1/SC-2 拆分 token 数和宽度影响，并完成短周期学习曲线。

## 15.2 不建议一次加入 SENet、DCN、Random Split、Global Token、SwiGLU 和 MoE

即使效果变好，也无法确定因果来源；若失败，也无法定位冲突。首轮必须单变量或使用 2×2 factorial design。

## 15.3 不建议用 Focal Loss 或 AUC surrogate 作为第一补救措施

Base 与 RankMixer使用同一 CVR 标签，当前差距首先指向表示与交互结构。贸然换损失可能损害 calibration，并掩盖架构问题。

## 15.4 不建议在 1234 fields 上做完整 Self-Attention

字段级 attention score 每样本约为：

```math
1234^2\approx1.52\times10^6,
```

batch size 2048 时开销过大，也偏离当前硬件友好目标。

## 15.5 不建议根据单个 Random Split seed 做结论

至少使用 3 个预先登记的固定 mapping seeds，避免 seed selection bias。

---

# 16. 统一评价协议

主指标：

```text
AUC / GAUC or UAUC if already available
LogLoss
PR-AUC
CVR calibration bias
ECE / Brier Score
```

系统指标：

```text
examples per second
steps per second
MFU
HBM peak
forward / backward time
P50 / P95 / P99 inference latency
kernel count and small-op proportion
```

结构指标：

```text
sample-token effective rank
batch-channel effective rank
pairwise token cosine similarity
per-token gradient norm
activation RMS
DCN and RankMixer representation correlation
gate distributions
```

统计要求：

- 所有模型使用完全相同的绝对时间窗口；
- Base 和候选在同一 bootstrap sample 上计算 paired difference；
- 以 user、request 或 day 中当前已经可用的最合适业务单元重采样；
- 报告绝对 AUC 变化、相对变化和 95% CI；
- 大规模样本下不能只看 p-value，还要定义最小业务有意义增益；
- checkpoint 选择规则必须预先固定。

---

# 17. 最终推荐路线

从行业实践角度，最合理的路线不是让 RankMixer 立即完全替代 Base，而是：

```text
Step 1
验证输入 BN、任务头和 pooling 的公平性

Step 2
做 Base/RankMixer calibrated blend，判断互补性

Step 3
复用 Base 的 hierarchical SENet 作为 RankMixer 前端

Step 4
以 Base 为主干，加入 zero-init RankMixer residual branch

Step 5
在残差分支内尝试 DCNv2 adapter、RankUp split 和 TokenMixer-Large block

Step 6
若双分支有效，再通过 distillation 压缩为单分支模型

Step 7
只有最佳结构已经稳定超过 Base，才扩展到 T=32、D=1536 或 Sparse MoE
```

最重要的判断是：

> 当前 Base 的领先不是一个需要被绕开的障碍，而是当前数据分布已经提供的最强结构证据。新模型应该先继承它的有效归纳偏置，再利用 RankMixer 的高容量和硬件效率学习增量信息。

---

# 18. 参考文献

1. Jie Zhu et al. [RankMixer: Scaling Up Ranking Models in Industrial Recommenders](https://arxiv.org/html/2507.15551v3), 2025.
2. Jin Chen et al. [RankUp: Towards High-rank Representations for Large Scale Advertising Recommender Systems](https://arxiv.org/html/2604.17878v3), 2026.
3. Yuchen Jiang et al. [TokenMixer-Large: Scaling Up Large Ranking Models in Industrial Recommenders](https://arxiv.org/html/2602.06563v2), 2026.
4. Ruoxi Wang et al. [DCN V2: Improved Deep & Cross Network and Practical Lessons for Web-scale Learning to Rank Systems](https://arxiv.org/abs/2008.13535), WWW 2021.
5. Tongwen Huang et al. [FiBiNET: Combining Feature Importance and Bilinear Feature Interaction for CTR Prediction](https://arxiv.org/abs/1905.09433), RecSys 2019.
6. Zhiqiang Wang et al. [MaskNet: Introducing Feature-Wise Multiplication to CTR Ranking Models by Instance-Guided Mask](https://arxiv.org/abs/2102.07619), DLP-KDD 2021.
7. Buyun Zhang et al. [DHEN: A Deep and Hierarchical Ensemble Network for Large-Scale Click-Through Rate Prediction](https://arxiv.org/abs/2203.11014), 2022.
8. Kelong Mao et al. [FinalMLP: An Enhanced Two-Stream MLP Model for CTR Prediction](https://arxiv.org/abs/2304.00902), AAAI 2023.
9. Xiaoqiang Gui et al. [Calibration-compatible Listwise Distillation of Privileged Features for CTR Prediction](https://arxiv.org/abs/2312.08727), 2023.
