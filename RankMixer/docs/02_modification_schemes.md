# RankMixer 详细改进方案

## 0. 统一记号

```text
B = 2048
F = 20978
T = 16
D = 768
L = 2
X0 = [B, T, D]
```

当前基线：

$$
\begin{aligned}
S_l &= \mathrm{LN}(\mathrm{Mix}(X_{l-1}) + X_{l-1}),\\
X_l &= \mathrm{LN}(\mathrm{PFFN}(S_l) + S_l),\\
h &= \mathrm{MeanPool}(X_L).
\end{aligned}
$$

所有方案都必须从该基线独立出发，不允许首轮一次组合多个新增模块。

---

# 方案一：RankUp-Lite 高秩输入表示

## 1.1 研究问题

当前有序 Autosplit 是否使某些 token 聚集了高度相关、低变化或长尾特征，从而产生 token 冗余和低 effective rank？

## 1.2 科学依据

RankUp 在两层 RankMixer、广告 CVR、超过 1200 个 sparse features 的工业环境中发现：

- Randomized Permutation Splitting 能降低 token 间 mutual information；
- 随机划分后的 token effective rank 更高、更均匀；
- Global Token 与 Multi-embedding 能进一步扩展初始表示空间；
- 这些组件分别带来独立的 Realtime AUC 收益。

这与当前“1234 fields + 两层 RankMixer + CVR”的设置高度匹配。

## 1.3 方案 1A：固定随机 feature permutation

不要在 20,978 个连续维度上逐维随机，而应在 1234 个 field 级别生成固定 permutation：

```python
perm = torch.randperm(1234, generator=fixed_generator)
```

然后将完整的 17 维 field embedding 按 permutation 顺序分到 16 组：

```text
2 groups × 78 fields -> 1326 input dims
14 groups × 77 fields -> 1309 input dims
```

每组仍使用独立投影：

$$
x_i=W_i\operatorname{Concat}(e_j:j\in G_i)+b_i,\quad x_i\in\mathbb{R}^{768}.
$$

最终仍为：

```text
X0_random: [B, 16, 768]
```

因此 backbone、`T=H=16`、PFFN 参数和输出头完全不变。

### 实现约束

- permutation 只生成一次；
- seed、field list hash 和映射必须进入模型配置；
- 训练、离线评估和线上推理必须使用同一映射；
- 增删 field 时生成新版本映射，不能静默漂移；
- 不允许每个 batch 动态打乱，否则 per-token FFN 的 token identity 失效。

### 可选的频率平衡版本

若已知每个 field 的覆盖率、基数和非零率，可增加一个“分层随机”对照：

1. 按 `log(cardinality)`、coverage、历史梯度范数分桶；
2. 每个桶内部随机；
3. round-robin 分配到 16 个 token。

该版本不是 RankUp 的纯随机复现，而是为了防止偶然形成全长尾 token。必须与纯随机版本分别报告。

## 1.4 方案 1B：15 Local Tokens + 1 Global Token

直接在 16 个 local tokens 之外追加 global token 会使 `T=17`，而 `768` 不能被 `17` 整除，也会破坏原始 `H=T` 约束。

推荐保持总 token 数为 16：

```text
15 local tokens + 1 global token
```

1234 个 fields 分成 15 个 local groups：

```text
4 groups × 83 fields -> 1411 dims
11 groups × 82 fields -> 1394 dims
```

得到：

```text
X_local: [B, 15, 768]
```

使用 local token 的一阶与二阶统计构造 global token：

$$
\begin{aligned}
\mu &= \frac{1}{15}\sum_{i=1}^{15}x_i,\\
r &= \sqrt{\frac{1}{15}\sum_{i=1}^{15}x_i^2+\epsilon},\\
u &= [\mu;r]\in\mathbb{R}^{1536},\\
g &= W_2\operatorname{SiLU}(W_1u)\in\mathbb{R}^{768}.
\end{aligned}
$$

首版设置：

```text
W1: 1536 -> 192
W2: 192  -> 768
```

附加参数约：

```text
1536 × 192 + 192 × 768 = 442,368
```

最终：

```text
X0 = concat([g, X_local], dim=1) -> [B, 16, 768]
```

### 为什么不是 raw flatten -> global MLP

直接将 `[B,20978]` 投影为 768 维会再增加约 16.1M 参数和很大的输入 GEMM，几乎复制一次 tokenizer。基于 local token 统计的 global token 更适合作为低风险首版。

## 1.5 方案 1C：Selective Multi-Embedding

RankUp 使用多个独立 embedding table 增加输入几何视角。但当前 embedding 表基数未知，不能无条件复制全部 1234 张表。

仅建议对少量 anchor fields 研究第二 embedding view，例如：

```text
user_id / item_id / shop_id / category_id / query_id / creative_id
```

两种可控方式：

1. 为选中 field 建立第二个独立 17 维表；
2. 建立 8 或 16 维的 hashed auxiliary table，控制额外显存。

主 embedding 和 auxiliary embedding 分别进入不同 local token，避免在输入处立即相加而失去多视角意义。

该方案必须单独报告 embedding memory、PS/QPS 和 lookup latency，不可只统计 dense 参数。

## 1.6 推荐消融

```text
RM-R0: Ordered Autosplit baseline
RM-R1: Fixed random feature permutation, T=16
RM-R2: Ordered 15 local + 1 global
RM-R3: Random 15 local + 1 global
RM-R4: RM-R3 + selective multi-embedding
```

## 1.7 诊断指标

对每层输出 `H ∈ R^[B,T,D]`，按样本计算：

$$
\operatorname{erank}(H_b)=\exp\left(-\sum_i p_i\log p_i\right),
\quad p_i=\frac{\sigma_i}{\sum_j\sigma_j}.
$$

由于 `T=16`，每个样本只需对 `16×768` 矩阵做 SVD；训练时每隔固定步数抽样计算即可。

同时记录：

- token 两两 cosine redundancy；
- 每个 token 的 batch variance；
- 每个 token projector 与 PFFN 的梯度范数；
- effective rank 在 tokenizer、Mixing 后和 PFFN 后的变化；
- AUC、LogLoss 与 effective rank 的相关性。

## 1.8 可行性结论

- 代码复杂度：低；
- 线上开销：Random Split 几乎为零，Global Token 很低；
- 论文匹配度：非常高；
- 推荐优先级：P0；
- 主要风险：随机 seed 偶然性、warm-start 困难、减少一个 local token 后的信息压缩。

Random Split 至少运行 3 个固定 seed。若只有一个 seed 获益，不应得出结构性结论。

---

# 方案二：TokenMixer-Large Block Lite

## 2.1 研究问题

原始 RankMixer mixing 后的 token 语义已经变化，却直接与 mixing 前同位置 token 相加。该残差是否限制了效果，并导致继续增加深度时收益下降？

## 2.2 核心结构：Mixing & Reverting

对输入：

```text
X: [B, 16, 768]
```

先按原 RankMixer 方式 mixing：

```text
H = Mix(X): [B, 16, 768]
```

在 mixed layout 中进行 per-token SwiGLU，然后将 channel/token 布局 revert 回原 token 语义位置：

```text
H1 = H + pSwiGLU_mix(RMSNorm(H))
R  = Revert(H1)                  # [B,16,768]
X1 = X + pSwiGLU_orig(RMSNorm(R))
```

可在 block 末尾再做 RMSNorm，或采用完整 PreNorm 形式。关键是跨 block residual 对齐到原 token layout，而不是把 mixed token 与原 token 直接逐位置相加。

## 2.3 Per-token SwiGLU

$$
\operatorname{pSwiGLU}(x)=W_{down}\left(\operatorname{SiLU}(W_{gate}x)\odot W_{up}x\right).
$$

每个 token 拥有独立的 `W_gate/W_up/W_down`。

### 参数匹配

如果基线是 `k=4` 的 GELU FFN：

```text
D -> 4D -> D
```

单 token 参数约：

```text
2 × 768 × 3072 = 4,718,592
```

单个参数匹配 SwiGLU 的 hidden size 为 2048：

```text
3 × 768 × 2048 = 4,718,592
```

但一个 TokenMixer-Large block 有 mixed-space 和 original-space 两个 pSwiGLU。建议同时跑两种预算：

### 计算量匹配版

```text
pSwiGLU_mix hidden  = 1024
pSwiGLU_orig hidden = 1024
```

两个模块合计：

```text
2 × 3 × 768 × 1024 = 4,718,592 / token / block
```

与一个 `k=4` 基线 PFFN 参数和主要 GEMM FLOPs 近似一致。

### 容量增强版

```text
pSwiGLU_mix hidden  = 2048
pSwiGLU_orig hidden = 2048
```

PFFN 参数/FLOPs 约为基线 2 倍，用来验证收益究竟来自结构还是单纯计算增加。

## 2.4 Norm 与初始化

推荐：

- Post-LayerNorm -> Pre-RMSNorm；
- `FC_down` 权重标准差设为 `FC_up/FC_gate` 的约 0.01 倍；
- residual branch 最后线性层可进一步 zero-init 做安全 warm-start；
- BF16 训练，loss 和校准统计保持 FP32。

## 2.5 深层版本

先验证 `L=2` 的 block 替换，再研究：

```text
L = 4, 6
```

当 `L>=4` 时：

- 每 2 层增加一次 inter-residual；
- 最后一层不接低层 inter-residual；
- 在中间层增加 auxiliary CVR head；
- 总损失：

$$
\mathcal{L}=\mathcal{L}_{final}+\lambda_{aux}\mathcal{L}_{mid}.
$$

建议搜索：

```text
lambda_aux ∈ {0.05, 0.1, 0.2}
```

辅助 head 仅用于训练，推理时删除。

## 2.6 推荐消融

```text
RM-T0: original RankMixer block
RM-T1: Mixing & Reverting, GELU FFN, compute matched
RM-T2: RM-T1 + pSwiGLU, compute matched
RM-T3: RM-T2 + Pre-RMSNorm + small down init
RM-T4: RM-T3, L=4, inter-residual
RM-T5: RM-T4 + auxiliary loss
RM-T6: RM-T3, L=2, capacity enhanced
```

## 2.7 可行性与风险

优点：

- RankMixer 最直接的后续结构；
- 与当前 batch size、数据量和 CVR 环境高度匹配；
- 为深度 scaling 建立正确残差路径；
- 全部核心计算仍可使用 grouped GEMM。

风险：

- 一个 block 内两次 pSwiGLU，若不做预算匹配会把结构收益和 FLOPs 收益混在一起；
- 需要高效 revert kernel；
- `L=2` 时深层优化组件可能收益很小；
- block 改动大于 side module，warm-start 难度较高。

推荐优先级：P0，但首版必须使用计算量匹配配置。

---

# 方案三：UniMixer-inspired Learnable Mixing Adapter

## 3.1 研究问题

RankMixer 的 mixing 是固定 permutation。是否可以在保留结构化、低开销和近似 permutation 约束的前提下，让数据学习更合适的 mixing pattern？

## 3.2 为什么不直接换 Self-Attention

16 token 的 self-attention 在算力上并非完全不可接受，但它会：

- 引入异构 token 内积相似度假设；
- 将收益同时归因于 Q/K/V 投影和动态 attention；
- 偏离 RankMixer 的硬件设计；
- 很难判断固定 mixing 的问题究竟来自“不可学习”还是“缺少样本条件”。

因此先研究受约束的 learnable adapter。

## 3.3 结构化 adapter

将：

```text
X: [B,16,768]
```

展平为：

```text
v: [B,12288]
```

以 channel block size `Bc=48` 分块：

```text
M = 12288 / 48 = 256 blocks
V: [B,256,48]
```

### Local mixing

定义 `b=4` 个 basis matrices：

```text
Z_l: [48,48], l=1..4
```

每个 block 有系数 `omega_i ∈ R^4`：

$$
W_{local}^{(i)}=\operatorname{Sinkhorn}\left(\sum_{l=1}^{4}\omega_l^{(i)}Z_l\right).
$$

### Global mixing

使用 rank `r=16` 的低秩矩阵：

```text
A: [256,16]
B: [16,256]
W_global = Sinkhorn(A @ B): [256,256]
```

先做 local mixing，再在 256 个 blocks 之间做 global mixing，最后 reshape 回 `[B,16,768]`。

### 安全残差

$$
X_{adapt}=X+\gamma\cdot\operatorname{UniMixLite}(X).
$$

`gamma` 初始化为 0，或使用逐层可学习标量且初始值 `1e-3`，保证训练开始时严格接近基线。

## 3.4 参数与计算估算

配置 `Bc=48, M=256, b=4, r=16`：

```text
Local bases:        4 × 48 × 48 = 9,216
Block coefficients:256 × 4      = 1,024
Global low-rank:    2 × 256 ×16 = 8,192
Total:                            18,432 parameters
```

主要矩阵计算约为每样本 7.5M FLOPs 量级，不含 Sinkhorn 迭代和 kernel overhead。相对于 `k=4`、两层 RankMixer 的 PFFN 主计算较小，但必须实测，因为 256×256 mixing 可能成为 memory-bound 小算子。

## 3.5 训练策略

- 温度从 `1.0` 线性或 cosine 退火到 `0.05`；
- 前 5%~10% steps 只训练 adapter/final head 或使用更低 LR；
- Sinkhorn 迭代 3~5 次起步；
- 保持 symmetry/doubly-stochastic 约束；
- 在训练完成后，推理可预计算归一化后的 mixing matrices；
- 首版只在第一个 RankMixer block 前放一个 adapter。

## 3.6 必须包含的简单对照

```text
RM-U0: fixed RankMixer mixing
RM-U1: learned 16×16 token matrix A @ X，仅 residual + zero-init
RM-U2: low-rank 16×16 learned token matrix
RM-U3: structured UniMixing-Lite adapter
RM-U4: RM-U3 without temperature annealing
RM-U5: RM-U3 without Sinkhorn constraints
```

如果简单的 `16×16` token mixing 已获得同等收益，应优先选择简单版本，不应为了论文形式保留更复杂 adapter。

## 3.7 可行性与风险

- 参数开销：极低；
- 代码复杂度：中；
- 训练数值风险：中；
- 线上风险：取决于 Sinkhorn 是否离线固化、算子是否融合；
- 推荐优先级：P1/P2；
- 主要失败模式：mixing 退化为密集平均、破坏 token 异质性、温度过低导致早期离散化、训练吞吐下降。

---

# 方案四：RankMixer + 低秩 DCNv2 显式交叉支路

## 4.1 研究问题

RankMixer 的固定 mixing 与 PFFN 是否需要较多数据/层数才能学习强乘法关系？增加一个低成本 explicit cross branch，能否更高效捕获 user × item、item × creative 等交互？

## 4.2 设计原则

不在 `[B,20978]` 上使用 full-rank DCNv2，也不把大型 DCN 串联在每个 RankMixer block 中。

先从 tokenizer 输出压缩：

```text
X0: [B,16,768]
C_i: 768 -> 32
Z0: [B,16,32]
z0: [B,512]
```

每个 token 可使用独立压缩矩阵，以保留 feature-subspace heterogeneity。

## 4.3 低秩 CrossNet

设置 `m=512, r=64, N_cross=3`：

$$
\begin{aligned}
a_l &= V_l^Tz_l,\quad a_l\in\mathbb{R}^{64},\\
c_l &= U_l\phi(a_l)+b_l,\quad c_l\in\mathbb{R}^{512},\\
z_{l+1} &= z_l+z_0\odot c_l.
\end{aligned}
$$

其中 `U_l,V_l ∈ R^[512,64]`，`phi` 可先使用 identity 或 GELU，二者分别消融。

## 4.4 与 RankMixer 融合

主干输出：

```text
h_rm = mean(XL, dim=1): [B,768]
```

Cross 输出：

```text
g_cross = W_o LayerNorm(z3): [B,768]
```

使用安全 gated residual：

$$
\alpha=\sigma(w^T[h_{rm};g_{cross}]+b),\qquad
h=h_{rm}+\alpha g_{cross}.
$$

首版可直接将 `W_o` zero-init，保证初始模型等价于基线。

## 4.5 参数估算

```text
Token compression: 16 × 768 × 32        = 393,216
3 low-rank layers: 3 × 2 × 512 × 64     = 196,608
Output projection: 512 × 768             = 393,216
Total ≈ 983,040 parameters
```

不到 `k=4` 基线 dense 参数的 1%。主要每样本计算约 2M FLOPs 量级，但实际效率取决于 kernel fusion。

## 4.6 CrossNet-Mix 后续版本

只有在单一低秩 CrossNet 获益后，才增加 `E=4` experts：

$$
c_l=\sum_{e=1}^{E}g_e(z_l)U_{l,e}\phi(V_{l,e}^Tz_l).
$$

需要记录 expert 使用分布，防止 4 个 experts 退化为相同矩阵。

## 4.7 推荐消融

```text
RM-D0: baseline
RM-D1: low-rank CrossNet, 2 layers, r=32
RM-D2: low-rank CrossNet, 3 layers, r=64
RM-D3: RM-D2 + scalar gated fusion
RM-D4: RM-D2 + vector gated fusion
RM-D5: CrossNet-Mix, E=4
RM-D6: RM-D2 but raw 20978 low-rank input（仅研究，不推荐线上）
```

## 4.8 科学可行性

支持理由：

- DCNv2 在大规模排序系统中验证过显式 bounded-degree crosses；
- DHEN 表明异构交互模块可能捕获非重叠信息；
- RankUp 允许使用 DCNv2 生成 Global Token。

反对理由：

- TokenMixer-Large 观察到超大 backbone 可以吸收小型 DCN 收益，并指出碎片化算子降低 MFU；
- 当前 RankMixer 已有很强的 PFFN 容量，显式支路可能冗余；
- 低秩压缩可能丢失 field-level 精细交互。

因此该方案只有在“端到端 AUC/LogLoss 收益大于吞吐和延迟损失”时才成立。推荐优先级 P1。

---

# 方案五：Factorized Token-Channel Recalibration

## 5.1 研究问题

Per-token FFN 能区分 feature subspaces，但所有样本都走相同强度的 token/channel 路径。不同 user/query/item/creative 请求是否需要动态调整各 token 的贡献？

## 5.2 低秩门控结构

输入：

```text
X: [B,16,768]
```

构造全局上下文：

$$
\begin{aligned}
\mu &= \operatorname{Mean}_{token}(X)\in\mathbb{R}^{768},\\
r &= \operatorname{RMS}_{token}(X)\in\mathbb{R}^{768},\\
c &= \operatorname{SiLU}(W_c[\mu;r])\in\mathbb{R}^{64}.
\end{aligned}
$$

生成 token gate 和 channel gate：

$$
\begin{aligned}
a &= 1+\rho\tanh(W_tc)\in\mathbb{R}^{16},\\
b &= 1+\rho\tanh(W_dc)\in\mathbb{R}^{768},\\
M &= a\otimes b\in\mathbb{R}^{16\times768},\\
\hat X &= X\odot M.
\end{aligned}
$$

推荐 `rho`：

```text
初始上限 0.1
稳定后搜索 {0.1, 0.2, 0.3}
```

将 `W_t` 和 `W_d` zero-init，使初始 `M=1`。

## 5.3 参数量

```text
Context: 1536 × 64         = 98,304
Token gate: 64 × 16        = 1,024
Channel gate: 64 × 768     = 49,152
Total ≈ 148,480 parameters
```

这是对完整 `[16,768]` mask 的 rank-1 factorization。若 rank-1 不足，可扩展为 `R=4` 个外积之和，但必须单独消融。

## 5.4 插入位置

首轮只测试：

1. tokenizer 后、block 1 前；
2. block 1 后、block 2 前。

不要一开始在每个子层都加 gate，否则会增加 kernel fragmentation，并可能导致反复缩放。

## 5.5 可选 gated pooling

在主干不变时单独研究：

$$
\alpha_i=\operatorname{softmax}(q^Tx_i),\qquad
h=\sum_i\alpha_ix_i.
$$

它解决的是输出聚合问题，不应与输入 gate 在同一首轮实验中同时加入。

## 5.6 正则与诊断

使用轻量 identity regularization：

$$
\mathcal{L}_{gate}=\lambda_g\|M-1\|_2^2.
$$

建议从 `lambda_g ∈ {0, 1e-5, 1e-4}` 搜索。

必须监控：

- 每个 token gate 的均值、方差和分位数；
- 正负样本 gate 差异；
- user/item/creative 对应 token 是否长期接近下界；
- gate entropy；
- 长尾样本和低转化率分桶效果；
- 梯度是否因 gate 过小而被截断。

## 5.7 推荐消融

```text
RM-G0: baseline
RM-G1: token-only SE gate
RM-G2: channel-only gate
RM-G3: rank-1 token × channel gate
RM-G4: rank-4 factorized gate
RM-G5: gated pooling only
RM-G6: RM-G3 without identity initialization
```

## 5.8 可行性结论

- 代码复杂度：低；
- 参数/FLOPs：很低；
- 论文依据：FiBiNET + MaskNet；
- 推荐优先级：P1；
- 主要风险：过度抑制长尾 token、收益过小、碎片化算子降低 MFU。

若 RM-G1/G2 已与 RM-G3 相当，应保留更简单版本。

---

# 方案六：Sparse-Pertoken MoE

## 6.1 研究问题

当前 dense PFFN 的容量与活跃 FLOPs 同步增长。能否在保持每样本主要 FFN 计算近似不变时，提高模型总容量和条件建模能力？

## 6.2 为什么采用 per-token expert set

不建议所有 token 共享同一组标准 MoE experts。RankMixer 的优势来自不同 token 的参数隔离；标准共享 MoE 会弱化这种先验，并让路由器在训练早期同时学习 token identity 和样本条件。

采用：

```text
每个 token 独立一组 sub-experts
其中 1 个 shared expert 对该 token 永远激活
其余 experts 在该 token 内 top-k 路由
```

这里的 shared expert 是“该 token expert set 内的 shared expert”，不是 16 个 token 全局共享。

## 6.3 计算匹配配置

基线 `k=4` GELU PFFN 的单 token 参数/活跃计算：

```text
2 × 768 × 3072 = 4,718,592
```

建议首版：

```text
E = 4 experts / token
hidden per expert = 1024
active = 1 always-on shared + 1 top-1 routed
```

每个 SwiGLU expert 参数：

```text
3 × 768 × 1024 = 2,359,296
```

单 token 总容量：

```text
4 × 2,359,296 = 9,437,184
```

单 token活跃计算：

```text
2 × 2,359,296 = 4,718,592
```

因此：

- 总 FFN 容量约为基线 2 倍；
- 主要活跃 GEMM FLOPs 与基线近似相同；
- 每个 block 总容量约 150.99M；
- 每个 block 活跃参数约 75.50M。

不计 router、bias 和 dispatch 成本。

## 6.4 路由

对 token `t`：

```text
router_t: R^768 -> R^3
```

3 个 routed experts 中选择 top-1，再加 always-on shared expert。

$$
y_t=E_{shared,t}(x_t)+\alpha\,g_{j^*}(x_t)E_{j^*,t}(x_t).
$$

激活比例为 1:2 时，TokenMixer-Large 的经验建议 `alpha≈2`。必须同时测试 `alpha ∈ {1,2}`，不能直接假设论文最优值可迁移。

增加 load-balance loss，并记录每个 token 内 3 个 routed experts 的分配比例。

## 6.5 First Enlarge, Then Sparse

推荐三阶段：

1. `Dense pSwiGLU hidden=2048`：与基线参数/FLOPs匹配，验证激活函数和门控本身；
2. `Dense pSwiGLU hidden=4096`：约 2 倍计算，验证额外容量是否真的带来收益；
3. 将 4096 hidden 切成 `4×1024` experts，激活 shared+top1，把活跃计算降回基线。

只有阶段 2 明确优于阶段 1，阶段 3 的 sparsification 才有科学意义。否则“增加不可用容量”没有价值。

## 6.6 初始化

- `FC_down` 使用约 0.01 倍小初始化；
- router bias 初始设为均匀；
- shared expert 可以从 dense pSwiGLU 的一部分权重初始化；
- routed experts 由 dense 权重分块并加入小噪声，避免完全对称；
- 训练初期提高 load-balance 权重，随后衰减。

## 6.7 数据量可行性

batch 2048 时，每个固定 token 每 step 有 2048 个样本。3 个 routed experts 均衡时，单 expert 每 step 平均约 683 个 routed 样本，样本量足够。

每天 5.5 亿条样本也为专家专门化提供了训练基础。真正瓶颈更可能是：

- grouped GEMM kernel；
- dispatch/permute/unpermute；
- optimizer state 和显存；
- 多卡 expert/token parallel 通信；
- 小 batch 在线推理的 memory bandwidth。

## 6.8 推荐消融

```text
RM-M0: dense GELU PFFN
RM-M1: dense pSwiGLU hidden=2048
RM-M2: dense pSwiGLU hidden=4096
RM-M3: E=4, shared+top1, h=1024, alpha=1
RM-M4: E=4, shared+top1, h=1024, alpha=2
RM-M5: RM-M4 without shared expert
RM-M6: RM-M4 without small down init
RM-M7: ReLU routing with dynamic expert count
```

RM-M7 用于研究，不建议直接作为严格延迟线上首版。

## 6.9 可行性结论

- 算法依据：强；
- 数据规模：满足；
- 工程复杂度：高；
- 线上风险：高；
- 推荐优先级：P2；
- 进入条件：dense pSwiGLU 的容量放大实验已经证明模型处于 capacity-limited 区域。

---

# 方案七：条件方案——ESMM + 任务特定 Token Pooling

## 7.1 使用条件

仅当训练数据包含全曝光空间的：

```text
impression label / click label / conversion label
```

且当前 CVR 存在“clicked-only 训练、entire-space 推理”时采用。

如果当前所有样本已经严格定义为点击后的 CVR 样本，且线上也只在点击条件下使用，则不要为了多任务而增加 ESMM。

## 7.2 共享 RankMixer + 任务特定池化

保留：

```text
XL: [B,16,768]
```

分别为 CTR 和 CVR 学习 pooling query：

$$
\begin{aligned}
\alpha_{ctr} &= \operatorname{softmax}(X_Lq_{ctr}/\sqrt D),\\
h_{ctr} &= \sum_i\alpha_{ctr,i}X_{L,i},\\
\alpha_{cvr} &= \operatorname{softmax}(X_Lq_{cvr}/\sqrt D),\\
h_{cvr} &= \sum_i\alpha_{cvr,i}X_{L,i}.
\end{aligned}
$$

独立任务塔：

$$
p_{CTR}=\sigma(f_{ctr}(h_{ctr})),\qquad
p_{CVR}=\sigma(f_{cvr}(h_{cvr})).
$$

遵循 ESMM：

$$
p_{CTCVR}=p_{CTR}\cdot p_{CVR}.
$$

损失：

$$
\mathcal{L}=\mathrm{BCE}(click,p_{CTR})+
\lambda\,\mathrm{BCE}(click\times conversion,p_{CTCVR}).
$$

标准 ESMM 不需要给未点击样本构造伪 CVR label。

## 7.3 为什么使用任务特定池化

共享 Mean Pooling 强迫 CTR 和 CVR 使用完全相同的 token 聚合方式。Task-specific pooling 只增加两个 query 和两个小塔，却允许：

- CTR 更关注曝光吸引力、creative、位置等信号；
- CVR 更关注 item 价值、价格、用户购买倾向等信号；
- 减少两个任务在最终表示上的直接冲突。

若发现明显 seesaw，再升级为 PLE/shared-private experts；不建议首版直接引入完整 PLE。

## 7.4 标签与评估注意事项

- 使用成熟转化窗口，避免尚未回流的正例被当作负例；
- 按 label delay 分桶评估；
- 同时报告 CTR AUC、CTCVR AUC、CVR AUC、LogLoss 和 calibration；
- 监控 `sum(pred)/sum(label)` 的 CVR bias；
- 线上排序分数究竟使用 pCVR 还是 pCTR×pCVR 必须由业务目标决定；
- 不能用未来转化信息构造训练时特征。

## 7.5 可行性结论

- 算法价值：在满足条件时非常高；
- backbone 开销：很低；
- 数据依赖：高；
- 推荐优先级：条件 P0；
- 最大风险：标签定义和延迟反馈错误，而不是模型结构本身。

---

# 8. 组合策略

首轮只做单模块实验。组合只允许发生在两个模块均独立获益之后。

## 8.1 推荐组合

### 组合 A：RankUp-Lite + TokenMixer-Large Lite

```text
固定随机/Global Token
        ↓
Mixing & Reverting + pSwiGLU
```

分别解决输入表示秩和 block 残差问题，结构互补性最强。

### 组合 B：RankUp-Lite + Sparse-Pertoken MoE

Randomized Split 使不同 token 的信息量更均匀，可能改善 per-token expert 的训练平衡。但必须先验证 MoE 单独收益。

### 组合 C：DCNv2 Cross Branch + Factorized Gate

显式交叉和动态选择可能互补，但两者都属于小型 side operator，组合后需特别关注 MFU 和 kernel fragmentation。

### 组合 D：ESMM + 最优 backbone

ESMM 改变学习目标，应在选定单任务最优 backbone 后重新进行完整对照；不能把单任务 AUC 与 ESMM 输出直接无条件比较。

## 8.2 不推荐组合

- TokenMixer-Large 深层版 + DCN + Gate + MoE 一次上线；
- Random Split、Global Token、Multi-Embedding、Cross Token 同时加入却不做拆分消融；
- learnable mixing 与 self-attention 同时加入；
- ESMM、PLE 和 task token 同时加入。

---

# 9. 最终优先路线

```text
Phase 1：低风险表示研究
  RM-R1 Fixed Random Split
  RM-R2 15 Local + 1 Global
  RM-G1 Token-only Gate

Phase 2：核心 block 升级
  RM-T1 Mixing & Reverting compute-matched
  RM-T2 + pSwiGLU / Pre-RMSNorm
  RM-T4 L=4 with inter-residual

Phase 3：互补交互
  RM-D2 Low-rank DCNv2 branch
  RM-U1/U3 Learnable Mixing Adapter

Phase 4：容量扩展
  RM-M1 Dense pSwiGLU
  RM-M2 Dense enlarge
  RM-M3 Sparse-Pertoken MoE

Parallel conditional track：
  ESMM + task-specific pooling
```

最值得首先实现的三个方案是：

1. **RankUp-Lite：Random Split / Global Token**；
2. **TokenMixer-Large Lite：Mixing & Reverting + 计算量匹配 pSwiGLU**；
3. **低秩 DCNv2 并行显式交叉支路**。

Factorized Gate 是低成本快速试验；UniMixing 和 Sparse-Pertoken MoE 更适合作为后续中高风险研究。