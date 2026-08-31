# RankMixer 模块消融、Scaling Law 与三类 Mixer 公平对比实验设计

> 适用任务：搜索首次转化率 `fst_CVR`<br>
> 代码范围：`src/models/rankmixer` 及现有 Base `src/models/seq_model/cvr_bn_senet_dcnm_fst.py`<br>
> 设计目标：在尽量少的正式训练任务下，同时完成 RankMixer 模块归因、局部 Scaling Law 验证，以及 RankMixer / TokenMixer-Large / UniMixer 的公平比较。<br>
> 设计日期：2026-08-31

---

## 1. 结论先行

不建议继续把当前 `v1 → v10` 当作一条消融链。它们同时改变了 Tokenization、Token 数、隐藏宽度、FFN、Norm、Global Token、显式 Cross、Readout 和任务头输入宽度，版本之间的 AUC 差不能归因到单一模块。

建议新增一个统一实验骨架：

```text
相同 1,234 个字段
→ 相同三桶 Input BN
→ 相同 Hierarchical SENet
→ 相同 32 组语义 Tokenizer（10 common + 21 item + 1 creative）
→ 三选一、且仅此处不同：
   RankMixer / TokenMixer-Large Dense Core / UniMixer-Lite Core
→ 相同固定顺序 PureFlat
→ 相同 Base 任务头 2048 → 2048 → 256 → 1
→ fst_CVR
```

推荐正式实验分两波：

1. **第一波 10 个配置**：Base、NoMixer、三个 Mixer 主模型、五个 RankMixer 核心消融；先回答“模块是否有效”和“三个方案谁更好”。
2. **第二波最多新增 6 个配置**：仅当 RankMixer 主模型相对 NoMixer 有效时，再跑 4 个宽度点和 2 个等参数方向点，验证局部 Scaling Law。

因此：

- 最差情况下只跑第一波 **10 个单 seed 正式任务**即可停止；
- 完整单 seed 闭环共 **16 个唯一配置**；
- 只对 Base 和第一波选出的 Top-2 候选补到 3 seeds，增加 6 个任务，推荐总量为 **22 个冷启动任务**；
- 若第三名与第二名处于等价区，再给第三名补 2 个 seed，最多 **24 个冷启动任务**。

三个主模型在 `T=32, D=512, L=2` 下通过 FFN 容量配平，预计 Dense 参数分别约为：

| 主模型 | Dense 参数 | 相对最大值差异 |
|---|---:|---:|
| RankMixer | 150,438,757 | -0.212% |
| TokenMixer-Large Dense Core | 150,582,117 | -0.116% |
| UniMixer-Lite Core | 150,757,733 | 0 |

三者参数差小于 0.22%，已经足以把主比较解释为“同输入、同输出、近等参数预算下的交互模块比较”，而不是大模型天然占优。

---

## 2. 要回答的研究问题

### 2.1 目标一：RankMixer 详细模块消融

需要回答以下五个问题：

1. 固定 Multi-head Token Mixing 是否真的提供跨 Token 信息交换？
2. Per-token FFN 是否优于参数共享 FFN？
3. 两条残差是否是有效性或可训练性的必要条件？
4. LayerNorm 是否是有效性或可训练性的必要条件？
5. 业务语义分组是否优于同预算的确定性随机分组？

此外，用一组宽度扩展点和两个近等参数方向点回答：

6. 随 Dense 参数增大，测试 NLL/AUC 是否呈稳定、可外推的局部 scaling 趋势？
7. 近似相同参数量下，增加宽度、增加深度、改变 Token 数能否得到近似相同效果？

### 2.2 目标二：三类 Mixer 只替换中间模块

需要严格区分两个问题：

- **本实验回答**：在本搜索 CVR 的相同 SENet、相同 Tokenizer、相同 PureFlat 和相同 Base MLP 下，哪一种 Mixer Core 最有效、最稳定、成本最好？
- **本实验不回答**：三篇论文各自完整原生系统谁最好。完整 TokenMixer-Large 还包含 Global Token、跨层残差、辅助损失、SP-MoE 和系统优化；当前 UniMixer 实现使用的是 UniMixing-Lite，而不是完整稠密 UniMixing。

这一边界必须写进实验名称和结论，避免把局部 Core 对比误报成论文整套方案复现。

---

## 3. 为什么不能直接比较当前 v1–v10

当前版本的主要问题如下。

| 现有方案 | 关键变化 | 是否保留为正式对照 | 建议用途 |
|---|---|---|---|
| v1 / v1_lrfix | 裸维度切 16 Token、无有效语义边界、未保留 SENet 主路径 | 否 | 仅保留为失败案例；不要重跑 |
| v2 | 字段安全分组、SENet、Gated Pool、Bucket Cross、`k=2` 同时变化 | 否 | 参考生命周期和字段拆分 |
| v3 | 16 个业务语义 Token、原始 RankMixer block、Gated Pool、Bucket Cross | 否 | **复用原始 RankMixer block 和 16-token 分组** |
| v4 | v3 + Query→Item Low-rank Cross | 否 | 本轮不研究额外交叉 |
| v5 | 31 Local + 1 Global、`D=1024`、TML 风格双 pSwiGLU、增强 Readout | 否 | 参考 Global Token 和增强读出，不重跑 |
| v6 | v5 型结构，`D=512`、语义均衡分组 | 否 | 参考 TML Core 与 SENet 接口 |
| v7 | v3 主干 + 深任务头，但仍含 16 Token、Gated Pool、Bucket Cross | 否 | 参考 Base 深任务头实现 |
| v8 | TML Core 前叠加 Masked Low-rank DCN | 否 | 本轮明确排除显式 Cross |
| v9 | Base DCNM + Raw/Cross 双视图 + Shortcut | 否 | 后续产品混合方案候选，不属于纯 Mixer 对比 |
| v6-E2 / v6-E3 / v10 | 31 Local + 1 Global、TML Core、PureFlat；Norm 有差异 | 否 | **复用 PureFlat、任务头和 TML block 代码** |
| UniMixer v1 | 32 语义 Token、UniMixing-Lite、SiameseNorm、PureFlat 深头 | 不直接作为主对照 | **复用 32 组字段 ABI 和 UniMixer block** |
| Base | SENet + 2×DCNM500 + Base MLP | 是 | 业务绝对锚点，不是单模块对照 |

关键判断：

- v5/v6/v8/v9/v10 的 Mixer block 已经是 TokenMixer-Large 风格，不应继续统称为“原始 RankMixer”。
- v9 同时保留 DCNM、双视图和 Shortcut，不能用于判断纯 TokenMixer 是否有效。
- v10 与 UniMixer v1 虽然都用 PureFlat 深头，但 Token 定义、投影、Norm 和 block 都不同，不能直接把其差值解释为 Mixer 差值。

---

## 4. 统一实验骨架

### 4.1 相同输入和 SENet

所有正式对照固定使用：

| 项目 | 固定值 |
|---|---:|
| Common 字段 | 385 |
| Item 字段 | 835 |
| Creative 字段 | 14 |
| 总字段数 | 1,234 |
| 单字段 Embedding | 17 |
| SENet 后展平宽度 | 20,978 |
| Input BN | 开启，沿用 Base 实现 |
| Hierarchical SENet | 开启，`hidden=128` |
| SENet 内 BN | 开启 |
| 其他输入 | 不加入 sequence、DIN、gattr、coupon 或新特征 |

三桶 Input BN 和 Hierarchical SENet 应直接复用 Base 的同一份函数，不在三个分支中复制三份。所有相关参数、初始化、BN decay、Riemann BN 开关必须完全一致。

### 4.2 相同 32-token Tokenizer

主比较统一复用 `cvr_bn_unimixer_v1.py` 中已经冻结并校验过的 32 个业务语义组：

```text
10 common + 21 item + 1 creative = 32 local semantic tokens
```

每组使用相同外部适配器：

```text
组内字段 Embedding concat
→ token-specific Linear(input_dim_t → D)
→ token-specific BatchNorm
→ 不加投影激活
→ stack 为 [B,T,D]
```

选择这个口径的原因：

1. 每个字段恰好出现一次，Token 边界不切断 17 维字段；
2. 32 Token 可以直接满足原始 RankMixer 的 `H=T`；
3. 当前 UniMixer v1 已有完整字段表、覆盖校验和静态测试；
4. Linear + Token-BN 对三个方案完全相同，不把 GELU、Global Token 或投影层差异混入 Mixer 对比。

必须新增三类静态约束：

- 每桶字段集合与 FeatureConfig 严格一致，无缺失、重复、跨桶字段；
- Token 顺序、每组字段顺序和 SHA256 固定；
- 所有模型的 Tokenizer AST/配置完全一致，只有 `mixer_type` 后面的图分支不同。

### 4.3 相同 Readout 和 Base MLP

三个主模型均使用：

```text
[B,T,D]
→ 按固定 Token 顺序 PureFlat 为 [B,T×D]
→ FC 2048 + Base BatchNorm + GELU2
→ FC 2048 + Base BatchNorm + GELU2
→ FC 256  + Base BatchNorm + GELU2
→ FC 1
→ sigmoid
```

主比较时 `T=32,D=512`，因此任务头输入统一为 16,384。

“沿用 Base 的 2048、2048、256 MLP”指隐藏层宽度、BN、激活、初始化、正则和输出层实现完全相同。Base 的 DCNM 输出宽度为 20,978，而 Mixer 的 PureFlat 输入为 16,384，因此 Base 与 Mixer 的第一层权重形状不可能相同；这不影响三个 Mixer 之间的严格公平性，但 Base 只能作为端到端业务锚点。

### 4.4 Base 的位置

Base 保持原样：

```text
Input BN → Hierarchical SENet → 2×DCNM500 → 2048→2048→256→1
```

Base 不参加“只替换 Mixer Core”的因果比较，只回答最终候选是否值得替换当前生产强基线。

另外增加 NoMixer 对照：

```text
Input BN → SENet → 相同 Tokenizer → PureFlat → Base MLP
```

NoMixer 是所有 Mixer 的直接局部锚点，用于判断收益究竟来自 Tokenizer + 大 MLP，还是来自 Mixer block。

---

## 5. 三个主 Mixer 的精确定义

### 5.1 RankMixer：原论文 Dense Core

主配置：

```text
T=32, H=32, D=512, L=2, PFFN expansion k=3
Post-LayerNorm
GELU2 Per-token FFN
```

每个 block：

```text
S = LN(FixedMix(X) + X)
Y = LN(PerTokenFFN(S) + S)
```

其中 `FixedMix` 只做 reshape/transpose/reshape，不含可学习参数；每个 Token 有独立的两层 FFN。

### 5.2 TokenMixer-Large Dense Core

主配置：

```text
T=32, H=32, D=512, L=2
每套 pSwiGLU hidden M=D=512
Pre-RMSNorm
down projection init scale=0.01
```

每个 block：

```text
M  = FixedMix(X)
M' = M + pSwiGLU(RMSNorm(M))
R  = Revert(M')
Y  = X + pSwiGLU(RMSNorm(R))
```

最后增加当前 v6 风格 Final RMSNorm，再 PureFlat。

主表必须命名为 `TML_DENSE_CORE`，而不是“完整 TokenMixer-Large”，因为为满足“只更换中间模块”，主表故意不加入：

- TML 专属 Global Token；
- interval/inter-layer residual；
- auxiliary prediction loss；
- Sparse-Pertoken MoE；
- FP8、Token Parallel、Grouped Kernel 等系统优化。

`L=2` 时跨层残差和辅助头本身也难以形成有意义的深度优势；若 TML Core 获胜，再把完整深层方案作为下一阶段，而不是混进本轮主比较。

### 5.3 UniMixer-Lite Core

主配置沿用当前 `cvr_bn_unimixer_v1.py` 的核心：

```text
T=32, D=512, L=2
block_size q=32
global low-rank r=128
local basis K=8
pSwiGLU expansion=2
tau: 1.0 → 0.05，linear annealing
Sinkhorn iterations=10
SiameseNorm
```

每个 block 使用 UniMixing-Lite 的可学习局部/全局近双随机 Mixing，再接一套 Per-token SwiGLU 和 SiameseNorm 双流更新。

主表必须命名为 `UNIMIXER_LITE_CORE`。当前仓库确实实现了完整 `unimixing` 算子，但正式 32×512 方案使用的是 Lite；本轮不额外增加 Full UniMixing，以免多一个模型却不帮助回答两个核心目标。

### 5.4 近等参数配平

三个方案的 FFN 主参数统一为每 block 约 `6TD²`：

- RankMixer PFFN：`2kTD²`，取 `k=3`，得到 `6TD²`；
- TML：两套 pSwiGLU，每套约 `3TDM`，取 `M=D`，合计 `6TD²`；
- UniMixer-Lite：一套 expansion=2 的 pSwiGLU，约 `6TD²`，额外 Mixing 参数很小。

在当前字段数和公共头下，静态估算为：

| 方案 | Dense 参数 | 推理前向 FLOPs/样本（估算） | 说明 |
|---|---:|---:|---|
| NoMixer | 49,640,293 | 0.0993G | 公共外壳，无 Mixer block |
| RankMixer | 150,438,757 | 0.3021G | fixed mixing + PFFN |
| TML Dense Core | 150,582,117 | 0.3012G | 含 Final RMSNorm |
| UniMixer-Lite Core | 150,757,733 | 0.3371G | Mixing matmul 使 FLOPs 约高 12% |

FLOPs 只作排期估算，正式报告必须同时给出图静态 FLOPs和真实 step time、examples/s、峰值内存。UniMixer 的 Sinkhorn、矩阵重构和算子碎片可能使真实耗时偏离 FLOPs。

---

## 6. 第一波：10 个配置完成主比较与 RankMixer 模块消融

### 6.1 主模型与直接锚点

| ID | 配置 | 预计 Dense 参数 | 直接对照 | 回答的问题 |
|---|---|---:|---|---|
| `M00_BASE_DCNM` | 现有 Base | 90,341,785 | — | 当前业务强基线 |
| `M01_NO_MIXER` | SENet + 32T Tokenizer + PureFlat + Base MLP | 49,640,293 | Base（仅端到端） | Tokenizer + MLP 本身能达到什么水平 |
| `M02_RM_FULL` | 统一外壳 + RankMixer | 150,438,757 | M01 | RankMixer block 的总净收益 |
| `M03_TML_CORE` | 统一外壳 + TML Dense Core | 150,582,117 | M01、M02 | 对齐残差和双 pSwiGLU 是否更优 |
| `M04_UM_LITE` | 统一外壳 + UniMixer-Lite Core | 150,757,733 | M01、M02、M03 | 可学习 Mixing + SiameseNorm 是否更优 |

预注册主对比：

```text
F1 = M02 − M01   RankMixer Core 总收益
F2 = M03 − M01   TML Core 总收益
F3 = M04 − M01   UniMixer-Lite Core 总收益
F4 = M03 − M02   TML 相对原始 RankMixer
F5 = M04 − M02   UniMixer-Lite 相对原始 RankMixer
F6 = M04 − M03   可学习 Mixing 相对 Mixing/Reverting
F7 = Top-1 Mixer − M00   最终业务位置，只作确认性比较
```

### 6.2 RankMixer 五个核心消融

所有消融只相对 `M02_RM_FULL` 解释。

| ID | 唯一改动 | 预计 Dense 参数 | 直接对照 | 允许结论 |
|---|---|---:|---|---|
| `A01_RM_NO_MIX` | `FixedMix` 分支置零，使第一步成为 `LN(X)`；PFFN、残差、Norm 不变 | 150,438,757 | M02 | 固定跨 Token Mixing 的净贡献 |
| `A02_RM_SHARED_FFN` | 32 套 PFFN 改为一套共享 PFFN；仍逐 Token 执行 | 52,794,213 | M02 | Token 参数隔离 + 额外参数容量的组合贡献 |
| `A03_RM_NO_SKIP` | 去掉两个 Add：`LN(Mix(X)) → LN(PFFN(S))` | 150,438,757 | M02 | 两条残差的组合贡献与稳定性作用 |
| `A04_RM_NO_NORM` | 去掉 block 内两处 LN，保留两个 Add | 150,434,661 | M02 | Post-LN 的组合贡献与稳定性作用 |
| `A05_RM_HASH_TOKEN` | 语义字段映射改为确定性分层 hash；每个 Token 的字段数与原位置完全相同 | 150,438,757 | M02 | 语义分组相对随机分组的净贡献 |

实现细节与解释限制：

1. `A01` 不能把 Mixing 替换成 `Identity` 后仍做 `LN(X+X)`，应真正旁路 Mixing 分支，使残差输入只出现一次。
2. `A02` 的逐样本 FFN FLOPs与 M02 基本相同，但参数显著减少。若 A02 下降，不能仅凭这一对就断言“语义专属权重”是唯一原因；需结合 scaling 曲线判断其下降是否超出低参数量的预期。
3. `A03/A04` 保持相同学习率；若出现 NaN 或梯度爆炸，这本身就是“该组件对当前训练协议必要”的结果，不应事后只为消融模型单独调参并覆盖主结论。
4. `A05` 的随机性只来自预先固定的字段分组 seed，例如 `group_seed=20260831`；每个桶内先稳定 shuffle，再按语义方案原有的组大小切分，从而保持精确参数量。若 A05 与 M02 差异落在灰区，再补两个 `group_seed`，不要一开始就跑三套随机分组。

为了控制任务数，本轮不额外跑以下方案：

- “完全移除 PFFN”：M01 已给出无 block 下界，A02 已回答参数共享；
- Mean/Gated Pool：主目标要求后端固定，改变 Readout 会破坏三模型公平性；
- Self-Attention 替换：不是当前两个业务目标的必要对照，且原论文已有相关消融；
- MoE：在 Dense Core 尚未超过 Base 前，不应扩大稀疏专家搜索空间。

---

## 7. 第二波：最小 Scaling Law 实验

只有当 `M02_RM_FULL` 相对 `M01_NO_MIXER` 至少非劣，且训练没有明显不收敛时，才启动第二波。

### 7.1 宽度 Scaling：4 个新增点 + 1 个已存在端点

固定：

```text
T=32, H=32, L=2, k=3
相同 32-token ABI
相同 SENet、PureFlat、Base MLP、优化器、数据量和训练步数
```

| ID | T | D | L | 预计 Dense 参数 | 角色 |
|---|---:|---:|---:|---:|---|
| `S01_RM_D128_L2` | 32 | 128 | 2 | 22,707,301 | 最小尺度 |
| `S02_RM_D192_L2` | 32 | 192 | 2 | 36,131,557 | 中小尺度 |
| `S03_RM_D256_L2` | 32 | 256 | 2 | 52,701,541 | 中尺度 |
| `S04_RM_D384_L2` | 32 | 384 | 2 | 95,278,693 | 拟合最大点 |
| `M02_RM_FULL` | 32 | 512 | 2 | 150,438,757 | **预注册外推 holdout** |

`D` 均可被 `H=32` 整除，对应 head_dim 为 4、6、8、12、16。

拟合时先隐藏 M02 的结果，只用 S01–S04 拟合，再预测 M02。M02 同时是三模型主比较点，因此无需为 holdout 新增训练任务。

### 7.2 两个近等参数方向点

仅做两个额外点，就能判断“总参数决定效果”是否在本任务成立。

| ID | T | D | L | 预计 Dense 参数 | 直接对照 | 参数差 |
|---|---:|---:|---:|---:|---|---:|
| `S05_RM_DEPTH` | 32 | 384 | 4 | 152,003,173 | M02：32×512×2 | +1.04% |
| `S06_RM_TOKEN16` | 16 | 736 | 2 | 148,994,085 | M02：32×512×2 | -0.96% |

解释：

- `S05 vs M02`：近等参数下比较“更深更窄”与“更浅更宽”；
- `S06 vs M02`：近等参数下比较 16 Token 与 32 Token；S06 复用 v3 已冻结的 `5 common + 10 item + 1 creative` 语义分组，但仍用统一 Linear+Token-BN、PureFlat 和 Base MLP。

如果两个方向点都落在 M02 的等价区，可支持“在当前 149M–152M 附近，效果主要由参数量而非扩展轴决定”。如果 L4 或 T16 明显偏离，则只能声明“宽度 scaling 成立”，不能宣称通用的 T/D/L scaling law。

### 7.3 参数公式

在本设计的 `k=3`、Linear+Token-BN、PureFlat Base MLP 口径下：

```text
N_common(T,D) = 5,295,973 + (20,978 + 2,051T)D

N_RankMixer(T,D,L)
  = N_common(T,D)
  + L × (6TD² + 4TD + 4D)
```

其中：

- `5,295,973` 包含固定的 Input BN、SENet 和 MLP 后两层/输出等常数项；
- `20,978D` 是所有字段到 Token hidden 的投影权重；
- `2,051TD` 包含 Token 投影 bias、Token-BN 和 PureFlat 到第一层 2048 的权重；
- block 主项 `6LTD²` 是 Per-token FFN。

正式运行前必须用 TensorFlow 图中的 trainable variable 逐项复算，静态公式只作为防漂移断言。

---

## 8. Scaling Law 应如何拟合，才能避免“只是扫参”

### 8.1 主指标使用 LogLoss/NLL，AUC 为业务副指标

AUC 对小变化敏感但不具有天然可加的损失形式。Scaling Law 主拟合使用固定测试集的加权 BCE/NLL：

```text
L(N) = L_inf + a × N^(−alpha)
```

同时报告：

```text
AUC(N)、GAUC(N)、PR-AUC(N)、COPC(N)、ECE(N)
```

建议分别以以下横轴拟合：

1. `N_total`：全部 Dense trainable 参数，作为主口径；
2. `N_core`：仅 RankMixer block 参数，检查固定外壳是否掩盖趋势；
3. 静态 forward FLOPs 和实测训练 compute，作为效率口径。

### 8.2 最小外推验证

1. 只用 D128、D192、D256、D384 拟合 `L_inf, a, alpha`；约束 `a>0, alpha>0, L_inf<min(L)`。
2. D512 在拟合完成前保持盲态，是唯一预注册外推点。
3. 用 `search_id` 或 `user_id` block bootstrap 对测试样本重采样，重复拟合并形成参数和预测区间。
4. 若四点无法稳定识别三参数曲线，只增加一个 `D=96,H=32,L=2` 点；不要临时修改已有点或挑选更好看的区间。

### 8.3 宣称层级

只有满足对应条件，才使用相应措辞：

| 层级 | 必须满足的条件 | 允许表述 |
|---|---|---|
| Level 0 | 点不单调或大模型未收敛 | 未观察到可靠 scaling 趋势 |
| Level 1 | 五个宽度点总体单调，`alpha>0` | 当前单日、固定预算下存在宽度 scaling 趋势 |
| Level 2 | `alpha` 的 bootstrap 95% CI 不跨 0；D512 落在预注册预测区间；换 seed 后端点排序不变 | 当前数据与预算区间存在局部经验 scaling law |
| Level 3 | 第二个独立日期窗口仍得到接近指数和外推误差 | 有较稳健的跨窗口局部 scaling law 证据 |

不得根据一次单 seed、单日的 AUC 单调性直接写“验证了普适 Scaling Law”。

### 8.4 建议验收阈值

在运行前冻结以下默认阈值，若团队已有标准则整体替换：

- 五个宽度点的测试 NLL 中位数总体非增；
- `alpha` 在至少 95% bootstrap 拟合中大于 0；
- D512 的真实 NLL 落入预测 95% 区间；
- D512 的 excess-loss 外推相对误差不超过 20%；
- `S05/S06` 相对 M02 的 AUC 差处于 `±0.0001` 等价区，才支持扩展轴近似等价；
- 最大模型不能仍处于明显欠训练状态，否则结论是 data/optimization limited，而不是 scaling 失败。

### 8.5 不增加训练任务的学习曲线

每个 scaling 任务在 25%、50%、75%、100% 训练步保存 checkpoint，并在同一个固定小验证切片上计算 NLL/AUC。这样可观察：

- 大模型是否只是收敛更慢；
- 各尺度在相同 examples-seen 下是否排序稳定；
- D512 是否需要更多数据才能体现容量。

这些 checkpoint 曲线是 compute/learning curve，不能冒充独立的数据量 scaling。若确实需要联合数据 scaling，最省的补充是只对 D256 和 D512 各增加一次固定 hash 50% 数据训练，共 2 个额外任务。

---

## 9. 训练协议

### 9.1 数据与启动

建议沿用已经有完整配置的主窗口：

```text
训练：2026-08-14
测试：2026-08-15
Sparse 附加 checkpoint 日期：2026-08-13
Dense：全部随机冷启动
Sparse：全部来自同一 checkpoint
```

如果这些分区已被回刷或无法保证不可变，则选择新的 `D0 → D1`，但所有配置必须同批切换，不能混用历史结果。

固定要求：

- 相同数据目录、过滤、采样、label、文件数和 `drop_last_files`；
- 相同 `batch_size=2048`、epoch、训练步数、worker/PS 数和硬件；
- 相同优化器、LR、warmup/decay、L2、gradient clipping；
- 相同 Sparse 初始化和 Sparse 更新策略；
- `ignore_dense_checkpoint=true`；
- 每个任务使用事先确认为空的独立模型目录；
- 禁止 `auto_load_cp` 命中历史同名目录；
- 保存全量逐样本预测。

主比较建议先固定当前公共配置：

```text
optimizer=flood_adam
learning_rate=2e-5
schedule=gauss_decay(warmup=60000, decay=40000, min_rate=0.1)
batch_size=2048
batch_norm=true
mlp_act_type=gelu_2
epochs=1
```

不应根据正式 AUC 为每个模型单独挑 LR。正式任务前只允许做 500–1,000 step 的稳定性预检；若某个架构在公共 LR 下发散，可追加一个事先标记为 sensitivity/rescue 的半 LR 任务，但不能替换主表结果。

### 9.2 随机种子与共同初始化

主表单 seed 使用同一 `train_seed`，但仅设置全局 seed 还不够。不同 Mixer 分支创建的随机变量数量不同，可能使后创建的公共 MLP 获得不同初始化。

因此应：

1. 对 SENet、每个 Token projection 和 Base MLP 的 initializer 使用由 `train_seed + 固定 scope id` 派生的显式 seed；
2. 三个模型保持相同公共 variable scope 和创建顺序；
3. Mixer 私有参数使用独立的固定 seed namespace；
4. 数据 shuffle 也使用相同 seed；
5. manifest 记录所有 seed。

分布式 Sparse 更新仍可能引入非确定性，因此正式胜负不能只依赖一个 seed。

### 9.3 预检而非选模

所有分支先运行 500–1,000 step，仅检查：

- 图能够 build/train/test/export；
- Dense 参数量与预注册值一致；
- 输入和输出 shape 一致；
- loss、梯度、激活无 NaN/Inf；
- 所有 Mixer 私有变量都有非零梯度；
- 公共层变量集合完全相同；
- 实际 restore 清单只包含允许的 Sparse 变量；
- step time 和峰值内存未超预算。

预检 AUC 不得用于删点或改超参。

---

## 10. 分阶段执行和停跑规则

### Wave 0：工程闭环

完成统一模型文件、16 份机器可读配置、参数/FLOPs 计算器、静态测试和 smoke test。此阶段不产出算法结论。

### Wave 1：10 个单 seed 正式任务

同时运行 M00–M04、A01–A05。尽量同一集群批次提交，避免硬件和数据版本漂移。

运行后按以下 Gate 决策：

| 结果 | 动作 |
|---|---|
| M02 明显低于 M01，且置信区间完全低于 `−0.0001` | 不跑 RankMixer scaling；保留 M03/M04 的独立判断 |
| M02 至少非劣于 M01 | 进入 Wave 2 |
| M03/M04 均不优于 M01 | 停止 Mixer 家族扩展，不做 MoE/Global Token |
| 至少一个 Mixer 接近或超过 Base | 进入多 seed 确认 |
| 所有 Mixer 优于 M01 但仍明显低于 Base | 先完成归因；只允许追加一个“Base DCNM + Winner Core”的产品型实验 |

### Wave 2：6 个 Scaling 配置

运行 S01–S06；M02 已作为 D512 endpoint，无需重复。

### Wave 3：只确认 Top-2

在 selection split 上选出 M02/M03/M04 中的 Top-2，与 Base 一起补两个新 seed：

```text
3 个模型 × 2 个新增 seed = 6 个新增任务
```

若第三名与第二名的差值绝对值小于 `0.0001`，第三名也补两个 seed；否则不补。

### Wave 4：部署前多日链路

只对 Base 和 Top-2 的固定 seed-0 做至少 3 个连续训练日：第一天 Dense 冷启动，后续每天只加载本模型前一天 Dense checkpoint。该阶段用于检查收敛速度和日间稳定性，不再用于模块筛选。

---

## 11. 统计口径

### 11.1 测试集一次划分、两次使用

对测试日按稳定主键预先划分：

```text
hash(search_id) % 2 == 0：selection split
hash(search_id) % 2 == 1：confirmation split
```

selection split 用于：

- 判断 Gate；
- 选择 Top-2；
- 拟合 scaling 曲线。

confirmation split 只在模型和对比已经冻结后打开，用于最终报告。这样无需额外训练即可降低同一测试集反复选模的乐观偏差。

### 11.2 主指标和副指标

| 类型 | 指标 |
|---|---|
| 主业务指标 | ROC-AUC |
| Scaling 主指标 | weighted BCE / LogLoss / NLL |
| 排序补充 | GAUC、PR-AUC |
| 校准 | COPC、ECE、等频 Bucket Error |
| 稳定性 | seed 标准差、逐日均值/最差日 |
| 效率 | Dense 参数、active 参数、FLOPs、examples/s、step time、显存、导出大小、P99 |

### 11.3 Paired 检验

所有模型保存相同样本顺序的：

```text
example_id / search_id / user_id / label / prediction / slice tags
```

按 `user_id` 做 block bootstrap；若 user_id 不可用则按 `search_id`，至少 2,000 次重采样。报告每个预注册差值的点估计、95% CI 和 seed 分布。多个 RankMixer 消融使用 Holm 校正；不要只看单模型 AUC 的小数点差。

建议运行前冻结：

```text
δeq  = 0.0001  # AUC 工程等价边界
δwin = 0.0002  # 值得继续确认的业务增益
```

判定建议：

- **明确胜出**：点估计 `≥δwin`，且 paired CI 下界 `>0`；
- **非劣**：paired CI 下界 `>−δeq`；
- **等价**：paired CI 完整位于 `[-δeq,+δeq]`；
- **灰区**：其余情况，只在会改变路线选择时补 seed。

### 11.4 必做切片

至少报告：

- Query 频次：head / torso / tail；
- 用户活跃度与新老用户；
- 商品冷启动程度；
- 类目、价格段、促销/券场景；
- 召回源和候选位置；
- 正负样本率与预测分数分桶。

平均 AUC 正向但关键交易切片显著退化，不得直接晋级线上。

---

## 12. 结果解释矩阵

### 12.1 RankMixer 模块

| 观察 | 结论 | 下一步 |
|---|---|---|
| M02 > M01，A01 < M02 | 固定 Mixing 在本任务提供有效跨 Token 交互 | 保留 fixed mixing |
| A01 ≈ M02 | 当前收益主要来自 PFFN/容量，固定 Mixing 无可见增益 | 优先 TML/UniMixer 或更简单 PFFN |
| A02 明显低于 M02，且低于 scaling 曲线对其参数量的预期 | Token-specific 参数隔离有架构收益 | 保留 Per-token FFN |
| A02 的效果符合低参数量 scaling 预测 | 差距主要可由参数量解释 | 不宣称 token specificity 被独立验证 |
| A03/A04 下降或发散 | Residual/Norm 是当前训练稳定性的必要组件 | 不再删减 |
| A05 > M02 | 人工语义分组不适合当前数据或相关性过高 | 研究随机/可学习 Tokenization |
| A05 < M02 | 本地支持业务语义 Tokenization | 固化 32-token ABI |

### 12.2 三个 Mixer

| 观察 | 结论 | 下一步 |
|---|---|---|
| M03 > M02 | 对齐残差 + Reverting + pSwiGLU 的组合优于原始 block | 以 TML Core 为 Dense 主干 |
| M04 > M03，但真实延迟超预算 | UniMixer 表达更强但系统 ROI 不足 | 线上选 TML，UniMixer 保留研究线 |
| M04 > M03 且延迟可接受 | 可学习 Mixing 值得进入多日和线上验证 | 再做 UniMixer 专属温度/秩消融 |
| 三者近似等价 | 选择 step time/P99 最优且实现最简单者 | 通常优先 TML 或 RankMixer |
| 三者均低于 M01 | 当前 Token 交互模块无净收益 | 停止扩大 Mixer |
| 三者均高于 M01、但低于 Base | Token 化后缺失显式乘性交互可能仍是瓶颈 | 条件追加 DCNM + Winner，不回到 v8/v9 大矩阵 |

### 12.3 Scaling

| 观察 | 结论 |
|---|---|
| 宽度点单调、D512 可被外推、S05/S06 近等价 | 当前范围支持参数主导的局部 scaling law |
| 宽度单调，但 S05 明显低于 M02 | 宽度可扩，深度扩展失败；可能是残差/优化问题 |
| S06 明显偏离 M02 | Token 粒度是独立归纳偏置，不能只看参数量 |
| 大模型训练曲线仍快速下降 | 数据或训练步不足，不能据此否定 scaling |
| 曲线在 D256 后饱和 | 当前外壳/数据的有效容量接近饱和，继续增参 ROI 低 |

---

## 13. 建议新增的代码和配置，不建议继续复制大文件

### 13.1 新增统一模型

建议新增：

```text
src/models/rankmixer/cvr_bn_mixer_fair_v1.py
```

只保留一个训练生命周期和一份公共前后处理，通过图构建期字符串开关选择：

```text
mixer_type = none | rankmixer | tml_dense | unimixer_lite
token_grouping = semantic32 | hash32 | semantic16
rm_disable_mixing = true/false
rm_shared_ffn = true/false
rm_disable_skip = true/false
rm_disable_norm = true/false
token_num / hidden_dim / layer_num
```

分支选择必须发生在 Python 建图期，未选分支不能创建任何变量。

### 13.2 新增公共 Block 文件

若平台允许模块导入，建议新增：

```text
src/models/rankmixer/mixer_blocks.py
```

包含纯函数式的：

- `rankmixer_block`；
- `tml_dense_block`；
- `unimixer_lite_block`；
- 参数/FLOPs 元数据接口。

如果服务器发布要求单文件模型，则可在构建阶段把 helper 内联，但源码仓库仍应以一份公共实现为真源，避免 v11/v12/v13 再次漂移。

### 13.3 测试和机器可读实验清单

建议新增：

```text
src/models/rankmixer/tests/test_mixer_fair_v1_static.py
src/models/rankmixer/tools/count_mixer_params_flops.py
src/models/rankmixer/tools/fit_scaling_law.py
bash/mixer_fair_v1/manifest.json
bash/mixer_fair_v1/*.args.txt
introduce/mixer_fair_v1_results.csv
```

`manifest.json` 至少记录：

```text
experiment_id
model file SHA256 / git commit
expanded model_args
field/token ABI checksum
train/test/sparse dates
train seed / group seed / data shuffle seed
dense and sparse restore manifests
dense params / active params / FLOPs
model_dir / prediction_path
hardware and worker/PS topology
```

### 13.4 静态测试必须覆盖

1. 三个主模型的 SENet、Tokenizer、PureFlat、MLP AST 等价；
2. M02–M04 输入均为 `[B,32,512]`，输出均为 `[B,32,512]`；
3. 三模型任务头输入均为 `[B,16384]`；
4. 字段 385/835/14 恰好覆盖一次；
5. 预注册参数量精确匹配；
6. `A01–A05` 除目标开关外配置完全一致；
7. Scaling 点只改变 T/D/L 和由 shape 必然引起的参数；
8. 所有模型只输出 first-CVR，禁止隐式多任务 loss；
9. tau 的起始 global step 在冷启动和逐日恢复时口径正确；
10. Train/Test/Export 三种图均可构建。

---

## 14. 结果表模板

### 14.1 主结果

| ID | Seed | Dense Params | FLOPs | Train min | Step ms | Peak GB | AUC | GAUC | PR-AUC | NLL | COPC | ECE | 状态 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| M00 |  |  |  |  |  |  |  |  |  |  |  |  |  |
| M01 |  |  |  |  |  |  |  |  |  |  |  |  |  |
| M02 |  |  |  |  |  |  |  |  |  |  |  |  |  |
| M03 |  |  |  |  |  |  |  |  |  |  |  |  |  |
| M04 |  |  |  |  |  |  |  |  |  |  |  |  |  |

### 14.2 Paired 对比

| 对比 | ΔAUC | 95% CI | ΔNLL | 95% CI | Holm p | 实际成本变化 | 判定 |
|---|---:|---|---:|---|---:|---:|---|
| M02−M01 |  |  |  |  |  |  |  |
| M03−M02 |  |  |  |  |  |  |  |
| M04−M03 |  |  |  |  |  |  |  |
| A01−M02 |  |  |  |  |  |  |  |
| A02−M02 |  |  |  |  |  |  |  |
| A03−M02 |  |  |  |  |  |  |  |
| A04−M02 |  |  |  |  |  |  |  |
| A05−M02 |  |  |  |  |  |  |  |

### 14.3 Scaling 拟合

| 横轴 | `L_inf` | `a` | `alpha` | alpha 95% CI | Fit NRMSE | D512 预测 | D512 实际 | 外推误差 | 结论 |
|---|---:|---:|---:|---|---:|---:|---:|---:|---|
| Total Params |  |  |  |  |  |  |  |  |  |
| Core Params |  |  |  |  |  |  |  |  |  |
| Train Compute |  |  |  |  |  |  |  |  |  |

---

## 15. 关键风险和规避方式

### 15.1 单日冷启动偏向小模型

大模型可能需要更多数据才能收敛。通过 25/50/75/100% checkpoint 学习曲线判断；若 D512 明显未收敛，只能报告固定预算结果，不宣称 scaling 失败。

### 15.2 Per-token FFN 消融的参数混杂

Shared FFN 与 Per-token FFN 的 FLOPs相近但参数量差很多。使用 RankMixer scaling 曲线预测 52.8M 模型应有的表现，再看 A02 是高于还是低于该预测，避免把所有下降都归因给 token specificity。

### 15.3 UniMixer 的训练系统成本

UniMixer-Lite 的 forward FLOPs 仅约高 12%，但 Sinkhorn 和小矩阵操作可能拉低吞吐。必须报告实测 step time/P99，不按参数量代替真实成本。

### 15.4 Dense checkpoint 污染

当前配置同时出现 `auto_load_cp=true` 与 `ignore_dense_checkpoint=true`。正式任务必须记录实际 restored variable 清单，并在任务前断言输出目录为空；仅看参数文件不足以证明 Dense 冷启动。

### 15.5 结果归属和代码漂移

历史 v9 已出现“结果与当前代码是否对应”的文档冲突。每个任务必须保存 commit、模型文件 SHA256、args SHA256、字段 checksum、图参数量和 checkpoint 路径。

### 15.6 论文结果不可直接迁移

RankMixer、TML 和 UniMixer 的公开结果来自不同私有任务、目标、数据量和系统实现。它们只用于决定要消融哪些模块，不能作为本搜索 CVR 的先验胜负结论。

---

## 16. 最终推荐顺序

1. 不再补跑 v1–v10 的跨版本大跨度对比。
2. 先实现一个统一 `cvr_bn_mixer_fair_v1.py`，保证只有 Mixer Core 分支不同。
3. 第一波同时跑 M00–M04、A01–A05，共 10 个配置。
4. 若 M02 相对 M01 有效，再跑 S01–S06，共增加 6 个配置。
5. selection split 选 Top-2，只给 Base + Top-2 补到 3 seeds。
6. 只有 Top-1/Top-2 接近 Base 时，才进入三日连续训练和线上资源评估。
7. 若所有纯 Mixer 都低于 Base，但明显高于 NoMixer，只追加一个 `DCNM + Winner Core`；不要重新开启 v8/v9 式多分支大搜索。
8. 在 Dense Core 尚未成立前，不做 Global Token、MoE、辅助 loss、Rank、Basis、Temperature 等第二层超参消融。

这套设计用 10 个首轮配置即可回答最重要的问题，用 16 个唯一配置形成完整研究闭环，并把多 seed 资源集中到真正可能晋级的 2 个候选上。

---

## 17. 参考依据

- [RankMixer: Scaling Up Ranking Models in Industrial Recommenders](https://arxiv.org/abs/2507.15551)：原始 fixed Multi-head Token Mixing、Per-token FFN、Residual/LayerNorm 消融，以及 `T/D/L` 扩展讨论。
- [TokenMixer-Large: Scaling Up Large Ranking Models in Industrial Recommenders](https://arxiv.org/abs/2602.06563)：Mixing/Reverting、Pre-RMSNorm、双 Pertoken SwiGLU、跨层残差、辅助损失和 Sparse-Pertoken MoE。
- [UniMixer: A Unified Architecture for Scaling Laws in Recommendation Systems](https://arxiv.org/abs/2604.00590)：可学习局部/全局 Mixing、UniMixing-Lite、温度/Sinkhorn、SiameseNorm 与 scaling 拟合。
- [Mixer 家族文献综述](rankmixer_mixer_family_literature_review.md)：RankMixer、TML、UniMixer 及后续家族的统一梳理。
- [当前进度技术总结](current_progress_technical_summary_2026-08-28.md)：本仓库 v1–v10、Base、UniMixer 的结构、效果和工程状态审计。
