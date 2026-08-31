# RankMixer / TokenMixer-Large / UniMixer-Lite 严格对照实验设计

> 版本：2026-08-31  
> 适用工程：`/Users/goku/Documents/Codex/RSA_code_0816`  
> 实验目标：在同一电商搜索 CVR 骨架中，只替换 Mixer Block，比较 RankMixer、TokenMixer-Large 和 UniMixer-Lite。  
> 本设计优先级：高于此前“近似参数量配平”方案；本轮不要求三者参数量相等。

---

## 1. 最终建议

第一轮只做 4 个结构、每个结构 3 个配对随机种子，共 **12 次正式训练**：

| 实验 ID | 中间交互模块 | 作用 |
|---|---|---|
| `E0_NO_MIXER` | 不堆叠 Mixer Block | 测量共享 Tokenizer、Global Token 和任务头本身的能力 |
| `E1_RANKMIXER` | 2 层论文式 RankMixer Block | RankMixer 主对照 |
| `E2_TML` | 2 层 dense TokenMixer-Large Block | TokenMixer-Large 主对照 |
| `E3_UNIMIXER_LITE` | 2 层 UniMixer-Lite Block | UniMixer-Lite 主对照 |

四组必须共享以下完整路径：

```text
相同稀疏特征与 Embedding
  → 分桶 Input BN
  → 相同 Hierarchical SENet
  → 相同 31-Local + 1-Global Tokenizer，X0 ∈ R^(32×512)
  → {NoMixer | RankMixer | TokenMixer-Large | UniMixer-Lite}，仅此处变化
  → 相同 Readout = Concat[Global Token, Mean(Local Tokens)]，维度 1024
  → 相同 Base MLP：2048 → 2048 → 256
  → 相同线性输出层与 BCE
```

这一定义比“每个方案连同各自 Tokenizer、Pooling 一起替换”更严格，因为：

1. 三个模型看到完全相同的 32 个输入 Token；
2. 三个模型向任务头输出完全相同维度、完全相同语义的向量；
3. 第一层 MLP 的形状也完全相同；
4. 唯一的结构自变量是 Mixer Block 及其论文原生残差、归一化和训练机制；
5. 参数量仍允许自然不同，不做削宽、补层或虚构参数来强行配平。

本轮结论应表述为：

> 在统一的 `31 Local + 1 Global Token` 输入、统一 Readout 和统一任务头下，三类论文核心 Mixer Block 的效果、收敛性和效率比较。

不能表述为：

> 完整复现三篇论文在其私有数据、原始全模型和原始参数规模上的数值结果。

原因是三篇论文没有公开足以逐项重建工业模型的全部特征分组、隐藏扩展、训练日数和任务头超参；此外，本设计按要求给 RankMixer 和 UniMixer-Lite 都加入了共同的 Global Token。

---

## 2. 为什么把 Global Token 做成共同控制项

TokenMixer-Large 论文明确提出额外的 Global Token，用全量语义组输入生成全局表示。为了满足本实验要求，RankMixer 和 UniMixer-Lite 也使用同一种 Global Token。

Global Token 不应在三个分支里各写一套，否则其投影方式、初始化或位置差异会变成隐藏变量。正确做法是将它放入共享 Tokenizer：

```text
31 个 Local Token：由固定业务语义组分别做独立线性投影
1 个 Global Token：由全部 SENet 输出拼接后做独立线性投影
Token 顺序：Local[0:31] + Global[31]
```

Global Token 的固定定义：

\[
g_0 = W_g\,[e_{common}^{se};e_{item}^{se};e_{creative}^{se}] + b_g,
\qquad g_0\in\mathbb{R}^{512}.
\]

所有模型都把 Global Token 当成第 32 个普通 Token 送入各自 Block。这样可比较三种交互机制如何更新全局信息，而不是比较三种不同的全局信息生成器。

`E0_NO_MIXER` 很重要：由于 Global Token 自身已经看到了全部输入，它可能形成一个较强的直接通路。只有加入无 Mixer 对照，才能判断收益来自 Mixer，还是主要来自 Global Token 和宽 MLP。

---

## 3. 严格固定的公共部分

### 3.1 输入与特征

以下项目四组逐项相同：

- 相同的 `feature_version` 与 Feature ID 集合；
- 相同 Embedding 维度，当前工程固定为 17；
- 相同稀疏 Embedding checkpoint；
- 相同缺失值、截断、采样和样本权重逻辑；
- 相同 common / item / creative 字段顺序；
- 不允许某个分支额外使用 dense、sequence、coupon、DIN 或手工交叉输入；
- 只预测同一个 `fst_CVR` 主任务。

建议所有实验都设置：

```text
enable_last_cvr = false
enable_wide_cvr = false
enable_mlt_loss = false
enable_delay_train_mode = false
opt_goal = first_cvr
cvr_label_name = fst_cvr_label
```

### 3.2 Input BN 与 SENet

保持当前强 Base 中的顺序：

```text
common/item/creative 分桶拼接
  → 各桶 Input BN
  → Hierarchical SENet(common → item → creative)
```

固定配置：

```text
use_senet = true
use_senet_bn = true
senet_hidden_size = 128
```

Input BN 与 SENet 必须由同一个共享函数构图，不允许复制成三个略有不同的实现。四个分支的对应变量形状、初始化器、正则项和更新依赖必须一致。

### 3.3 共享语义 Tokenizer

直接采用当前 `rankmixer_v6` 的 31 个本地语义组：

| Bucket | Local Token 数 |
|---|---:|
| common | 10 |
| item | 20 |
| creative | 1 |
| 合计 | 31 |

再追加 1 个 Global Token，总数固定为：

```text
T = 32
D = 512
```

每个 Local Token 使用独立的线性投影：

\[
x_i=W_i\operatorname{Concat}(G_i)+b_i,\quad i=1,\ldots,31.
\]

为避免 Tokenizer 自身成为模型差异，本轮统一采用：

- `Linear` 投影；
- 不在投影后增加 Token BN；
- 不在投影后增加 LayerNorm / RMSNorm；
- 不使用 GELU、PReLU 或门控激活；
- 权重初始化统一为 `Normal(std=1/sqrt(fan_in))`；
- bias 初始化为 0。

注意：TML 论文把其语义投影写为 `MLP_i`，而 RankMixer / UniMixer 写为投影层。本轮使用共同的最小线性适配器，是为了只比较 Block；它是明确的实验控制，不是对三篇论文 Tokenizer 完全相同的宣称。

### 3.4 共享 Readout

三组都保留并显式使用最终 Global Token：

\[
z=\operatorname{Concat}\left[X_L[:,31,:],\ \operatorname{Mean}(X_L[:,:31,:])\right]
\in\mathbb{R}^{1024}.
\]

固定使用简单均值，不使用可学习 Pooling，原因是可学习池化本身会引入另一种交互机制。

禁止加入：

- Gated Pool；
- Global-conditioned attention pooling；
- Bucket Cross；
- Query–Item Cross；
- Flatten readout 分支；
- DCNM shortcut；
- Dual-view 或其他旁路。

### 3.5 共享任务头

Readout 后严格复用 Base 的任务头：

```text
1024
  → Linear(2048) → BN → GELU2
  → Linear(2048) → BN → GELU2
  → Linear(256)  → BN → GELU2
  → Linear(1) → Sigmoid
```

以下内容逐项相同：

- 三层宽度 `[2048, 2048, 256]`；
- BN 类型、momentum/decay、epsilon；
- `gelu_2` 实现；
- 权重初始化和 L2 正则；
- logit clip；
- 输出层 bias；
- BCE 定义。

由于三个分支的 Readout 都是 1024 维，任务头连第一层参数形状也完全一致。

---

## 4. 四个正式实验的精确定义

## 4.1 `E0_NO_MIXER`：无交互锚点

```text
X2 = X0
z = Concat[Global(X2), Mean(Local(X2))]
```

它不增加 Mixer 参数，不做两次伪 Identity Block。其作用不是替代现网 Base，而是回答：

> 在已有 Input BN、SENet、语义 Tokenizer、Global Token 和大 MLP 的情况下，Mixer 是否仍提供增益？

现有 `SENet → 2×DCNM → Base MLP` 可作为业务参考值另行展示，但不纳入三种 Mixer 的严格统计比较，因为它既没有相同 Tokenizer，也没有相同 Readout。

## 4.2 `E1_RANKMIXER`：论文式 RankMixer Block

固定配置：

```text
T = 32
H = 32
D = 512
L = 2
PFFN expansion k = 2
PFFN activation = GELU2
Norm = Post-LayerNorm
```

每层严格采用论文的两次 Add & Norm：

\[
S_l=\operatorname{LN}(\operatorname{TokenMix}(X_l)+X_l),
\]

\[
X_{l+1}=\operatorname{LN}(\operatorname{PFFN}_l(S_l)+S_l).
\]

实现约束：

- TokenMix 是无参数的 split–transpose–concat；
- `H=T=32`，保证输入输出均为 `[B,32,512]`；
- 每个 Token 拥有独立 PFFN 参数；
- 两层 Block 不共享参数；
- 不使用 MoE；
- Global Token 与其余 Token 一起被固定规则混合；
- 不加入额外 Final Norm；最后一层 Post-LN 输出直接进入共享 Readout。

相对原始 RankMixer 论文的差异只有两类：

1. 为严格对照统一成 `T=32,D=512`，而不是论文公开的 100M 点 `T=16,D=768,L=2`；
2. 按本实验要求加入共同 Global Token。

Block 拓扑、固定 Token Mixing、逐 Token FFN、Post-LN 和残差保持论文定义。

按大矩阵权重粗估，两个 Block 的 PFFN 参数约为：

\[
2\times L\times T\times D\times(kD)
=67.1\text{M}.
\]

该数值只用于资源规划，不用于参数配平。

## 4.3 `E2_TML`：Dense TokenMixer-Large Core

固定配置：

```text
T = 32
H = 32
D = 512
L = 2
pSwiGLU hidden M = 704
Norm = Pre-RMSNorm
RMS epsilon = 1e-6
down-projection init scale = 0.01
```

每层采用 Mixing–Reverting 和两套互不共享的 Per-token SwiGLU：

```text
M  = Mixing(X)                         # [B,T,D] → [B,H,D]
M1 = M + pSwiGLU_mixed(RMSNorm(M))
R  = Reverting(M1)                     # 回到原始 Token 语义位置
Y  = X + pSwiGLU_original(RMSNorm(R))
```

其中 `H=T=32` 是第一轮的共同形状设置，不是否定 TML 支持 `H≠T`。选择 `H=T` 有两个目的：

1. 减少同时改变 Head 数与 Block 拓扑带来的解释歧义；
2. 最大程度复用当前 `rankmixer_v6` 已验证的 Mixing/Reverting 张量路径。

实现约束：

- Mixing 与 Reverting 都无参数；
- 两个 pSwiGLU 的参数必须独立；
- 每个 Token 拥有独立 up/gate/down 参数；
- 使用 Swish gate；
- down 矩阵按论文使用 0.01 小尺度初始化；
- TML 内部线性核不使用 bias，遵循论文附录的 bias-free 说明；
- 不使用当前 v6 的 Global-conditioned Pool、Flatten readout 和增强融合；
- 第一轮使用 dense pSwiGLU，不启用 Sparse-Pertoken MoE。

选择 dense 版本不是删掉 TML 主体：论文提出的策略本身是先扩大 dense 模型，再做稀疏化；Sparse-Pertoken MoE 属于后续容量/效率实验。第一轮若直接引入 Router、Top-k、Shared Expert 和负载均衡，会把“Mixer 架构”与“稀疏专家策略”混为一个处理变量。

两个 Block 的两套 pSwiGLU 大矩阵权重粗估为：

\[
2\times L\times 3TDM
\approx 138.4\text{M}.
\]

这比 RankMixer 大是允许且预先声明的；结果必须同时报告参数、FLOPs 和吞吐，不能把单点 AUC 差异直接解释为更高的参数效率。

### 为什么第一轮不启用 Inter-Residual 与 AuxLoss

TML 论文说明跨层残差通常每 2 或 3 层设置一次，并且不建议加在最后一层。本轮所有方案都固定为 2 层，因此不存在一个既满足“间隔 2 层”又“不位于最后层”的合法位置。

所以 `E2_TML` 是完整的 **TML Core** 对照：Global Token、Mixing–Reverting、双 pSwiGLU、Pre-RMSNorm、长残差和 down-init 均保留；深层专用的 Inter-Residual 与 AuxLoss 留到第二阶段 `L=4/8` 深度实验中启用。这不是随意删项，而是避免为了一个只在深层有定义的机制破坏第一轮等深度对照。

## 4.4 `E3_UNIMIXER_LITE`：UniMixer-Lite + SiameseNorm

固定配置：

```text
T = 32
D = 512
flatten length F = 16384
UniMixer blocks M = 2
local block size B = 32
global block count G = F / B = 512
global low-rank r = 128
local basis count b = 8
pSwiGLU expansion = 2
Sinkhorn iterations = 10
tau_start = 1.0
tau_end = 0.05
RMS epsilon = 1e-8
```

UniMixing-Lite 使用：

\[
W_G=UV^\top,\qquad U,V\in\mathbb{R}^{512\times128},
\]

\[
W_B^{(i)}=\sum_{j=1}^{8}\alpha_{ij}Z_j.
\]

全局低秩矩阵和局部基矩阵都必须经过同样的约束链：

```text
Symmetry → divide by temperature → exp/log-stable transform
→ 10 次 Sinkhorn 行列交替归一化
```

SiameseNorm 双流严格写成：

\[
\hat Y_l=\operatorname{RMSNorm}(Y_l),\qquad
O_l=\operatorname{UniMixerLiteBlock}(X_l+\hat Y_l),
\]

\[
X_{l+1}=\operatorname{RMSNorm}(X_l+O_l),\qquad
Y_{l+1}=Y_l+O_l,
\]

\[
X_{out}=X_M+\operatorname{RMSNorm}(Y_M).
\]

`UniMixerLiteBlock` 内保持当前方案的论文式顺序：UniMixing-Lite、混合残差/RMSNorm、Per-token SwiGLU。每个 Block 的 Mixing 参数、pSwiGLU 参数和 Norm 参数均不共享。

与当前 `cvr_bn_unimixer_v1.py` 相比，正式对照必须修改三点：

1. 使用共同的 `31 Local + 1 Global`，不再是 `32 Local + 0 Global`；
2. 删除额外 Token BN，使用共同线性 Tokenizer；
3. 删除纯 Flatten Readout，改用共同的 `Global || LocalMean` Readout。

### 温度与 Warm-up

UniMixer 论文显示温度和 Warm-up 都会显著影响结果。为了不额外增加一次完整训练，本轮把 Warm-up 写进同一训练预算：

```text
前 20% update：tau = 1.0
后 80% update：tau 从 1.0 线性退火到 0.05
```

不得根据验证集临时修改 20% 比例、最低温度或 Sinkhorn 次数。若后续需要调参，应作为独立的 UniMixer-Lite 内部消融，不回写本轮主对照。

---

## 5. 三个 Mixer 的唯一差异矩阵

| 项目 | `E1_RANKMIXER` | `E2_TML` | `E3_UNIMIXER_LITE` |
|---|---|---|---|
| 输入 Token | 共同 31 Local + 1 Global | 相同 | 相同 |
| `T,D,L` | `32,512,2` | `32,512,2` | `32,512,2` |
| Mixing | 固定无参数重排 | 固定 Mixing + Reverting | 可学习低秩全局 + 基组合局部 Mixing |
| Channel MLP | Per-token GELU FFN | 两套 Per-token SwiGLU | Per-token SwiGLU |
| Norm | Post-LayerNorm | Pre-RMSNorm | SiameseNorm + RMSNorm |
| 主残差 | Mixed slot 与输入 slot 直接相加 | Revert 后回原语义 slot 再做长残差 | 双流残差 |
| 特有训练机制 | 无 | down matrix 0.01 小初始化 | 对称、Sinkhorn、温度 Warm-up/退火 |
| Readout | 共同 `Global || LocalMean` | 相同 | 相同 |
| 任务头 | 共同 `2048/2048/256` | 相同 | 相同 |

除本表中的 Block 原生差异，任何分支专属结构都视为实验污染。

---

## 6. 现有代码如何取舍

| 当前文件 | 结论 | 可复用部分 | 不能直接用于正式对照的原因 |
|---|---|---|---|
| `cvr_bn_rankmixer_v3.py` | 仅作 RankMixer Block 参考 | 固定 Token Mixing、Post-LN、Per-token FFN | 16 Token；默认 Gated Pool、Bucket Cross；不是共同 Readout |
| `cvr_bn_rankmixer_v6.py` | 作为共享 Tokenizer 与 TML Core 的主要来源 | 31 Local + Global、Mix/Revert、Pre-RMSNorm、pSwiGLU、down-init | 有增强 Pool、Flatten route；任务输出不是共同 Readout |
| `cvr_bn_unimixer_v1.py` | 作为 UniMixing-Lite/SiameseNorm 来源 | 低秩全局矩阵、局部 basis、Sinkhorn、温度、SiameseNorm | 当前是 32 Local、无 Global；额外 Token BN；纯 Flatten Readout |
| `cvr_bn_rankmixer_v4/v7/v8/v9/v10.py` | 第一轮舍弃 | 可留作历史参考 | 含 Query–Item Cross、DCN、Dual-view、shortcut 或其他复合改造 |
| `cvr_bn_rankmixer_v6_e2/e3.py` | 第一轮舍弃 | 可参考纯 Flat 分支实现 | 主要验证 RMSNorm/LN 与 Readout，不是三种论文 Block 的严格对照 |
| `src/models/unimixer/unimixer.py` | 只参考算子数学 | full/Lite Mixing 原型 | 生命周期、Token 定义和任务头未与本轮共同外壳统一 |

### 推荐新增代码结构

不要再复制三份完整的 2000 行模型。建议新增一个共享实验入口：

```text
src/models/rankmixer/
  cvr_bn_mixer_strict_v1.py       # 数据、Input BN、SENet、Tokenizer、Readout、Base MLP
  mixer_strict_blocks.py          # 三种纯 Block
```

通过唯一参数切换：

```text
mixer_type = none | rankmixer | tokenmixer_large | unimixer_lite
```

这样四个实验在代码层面天然共享：

- Feature lookup；
- Input BN；
- SENet；
- 31+1 Tokenizer；
- Readout；
- 任务头；
- loss、optimizer、metric 和 export。

建议新增四份仅改变 `mixer_type` 的启动配置：

```text
bash/set-mixer-strict-none-args.txt
bash/set-mixer-strict-rankmixer-args.txt
bash/set-mixer-strict-tml-args.txt
bash/set-mixer-strict-unimixer-lite-args.txt
```

配置生成后做结构化 diff；除 `mixer_type` 及该 Block 的专属参数外，其余键必须完全相同。

---

## 7. 实验次数与执行顺序

### 7.1 不计入结论的构图/短跑检查

每个配置先跑相同的短程 smoke test，例如 1,000～2,000 update。该阶段只判断实现是否正确，不比较 AUC，也不能据此筛掉表现差的方案。

必须通过：

- 输入、Token、Block 输出、Readout 形状正确；
- 四组共享变量清单与形状相同；
- 所有训练 loss 有限；
- 梯度不是全 0 或 NaN；
- UniMixer-Lite 的行和、列和误差处于预设阈值；
- Global Token 在三组中都有非零梯度；
- TML 的 `Revert(Mix(X))` 单元测试能够恢复原布局；
- RankMixer 固定重排与手工索引结果一致；
- train/eval/export 三种图均能构建。

### 7.2 正式训练

推荐随机种子：

```text
20260831
20260901
20260902
```

正式矩阵：

| 实验 | Seed 1 | Seed 2 | Seed 3 | 正式训练数 |
|---|---:|---:|---:|---:|
| `E0_NO_MIXER` | 1 | 1 | 1 | 3 |
| `E1_RANKMIXER` | 1 | 1 | 1 | 3 |
| `E2_TML` | 1 | 1 | 1 | 3 |
| `E3_UNIMIXER_LITE` | 1 | 1 | 1 | 3 |
| 合计 | 4 | 4 | 4 | **12** |

如果当前只能承担 4 次完整训练，可先用一个 Seed 跑四组作为探索批次，但只能输出“单次观察”，不能据此下最终架构结论。预算恢复后必须补齐相同的另外两个 Seed，而不是只补跑当前冠军。

### 7.3 为什么不先海量做单模块消融

第一轮的目标是先确认三种 Block 在统一外壳中的排序。若一开始同时运行：

- RankMixer 去残差、去 LN、共享 FFN；
- TML 去 Revert、改 PostNorm、加/去 AuxLoss；
- UniMixer-Lite 去温度、去对称、改 rank/basis；

实验会迅速扩张，而且可能在主方案尚未正确复现时消耗大量训练资源。正确顺序是先完成这 12 次主对照，再只对胜出或异常的方案做定向机制消融。

---

## 8. 训练条件必须锁死

### 8.1 数据预算

四组使用：

- 完全相同的训练日期和文件清单；
- 完全相同的样本数、epoch 数和 update 数；
- 完全相同的验证与测试日期；
- 相同 Seed 下完全相同的数据 shuffle 顺序；
- 相同 negative sampling 与样本权重；
- 主实验禁止按各模型验证集最优点提前停止。

更大的 Block 可能收敛更慢，因此同时保存并绘制 `AUC vs seen examples`。主结论仍以相同数据量末端 checkpoint 为准，best-checkpoint 只能作为次要结果。

### 8.2 初始化与 checkpoint

建议：

- 四组加载同一个稀疏 Embedding checkpoint；
- `ignore_dense_checkpoint=true`，不加载任何旧 Mixer、SENet 或任务头权重；
- 共享层在同一 Seed 下使用相同初始化；
- 分支独有层使用同一全局 Seed 派生的稳定子 Seed；
- 训练步数从本次实验的局部 step 0 计数；UniMixer 温度不得读取旧 checkpoint 的累计 global step。

如果业务原因必须加载 SENet checkpoint，则四组必须加载同一份 SENet 权重，并在报告中注明它是共同预训练组件。

### 8.3 优化器

以下内容不得按模型单独调优：

- dense optimizer；
- base learning rate；
- warm-up 与 decay；
- batch size；
- gradient clipping；
- L2；
- BN decay；
- mixed precision；
- loss scale。

建议直接锁定当前可稳定训练的共同配置，例如 `batch_size=2048`、`flood_adam`、`learning_rate=2e-5` 和相同的学习率曲线。UniMixer 的温度退火、TML 的 down-init 属于论文 Block 内部机制，不视为优化器特权。

若某一方案 NaN：

1. 先检查张量重排、Norm 轴、Sinkhorn 数值稳定性和初始化是否实现错误；
2. 若必须改变全局梯度裁剪或 LR，应四组一起重跑；
3. 不允许只给失败模型降低 LR 后继续与其余旧结果比较。

---

## 9. 指标与统计检验

### 9.1 主指标

主指标只设一个：

```text
固定测试集上的 fst_CVR AUC
```

主比较为：

1. `E2_TML − E1_RANKMIXER`；
2. `E3_UNIMIXER_LITE − E1_RANKMIXER`；
3. `E3_UNIMIXER_LITE − E2_TML`。

`E1/E2/E3 − E0_NO_MIXER` 是机制有效性的次级比较。

### 9.2 次级指标

效果指标：

- GAUC / UAUC；
- LogLoss；
- COPC 或预测均值 / 标签均值；
- 关键业务切片 AUC：新老用户、Query 频次、商品冷启动、类目等；
- 收敛曲线和末段 checkpoint 波动。

效率指标：

- Mixer 参数量、任务头参数量、总 dense 参数量；
- 训练 FLOPs / batch；
- samples/s；
- 单 step wall time；
- 峰值显存；
- 推理 p50/p95 延迟和吞吐。

参数量不作为准入条件，但必须报告，否则无法判断 AUC 增益是否由显著更大容量换来。

### 9.3 配对统计

同一 Seed 的四组使用相同样本顺序，并在相同测试样本上保存预测。分析时：

- 以用户 ID 或请求 ID 为 cluster 做 paired bootstrap；
- 每个 Seed 内计算配对 `ΔAUC`；
- 汇总 3 个 Seed 的均值、标准差与 95% CI；
- 三个主 pairwise 比较使用 Holm 校正，控制 family-wise error；
- 同时报告绝对 AUC 差和 `×10^-4` AUC bp。

预注册实用显著阈值：

```text
|ΔAUC| >= 0.0001
```

建议判定：

- **明确胜出**：校正后 95% CI 下界大于 0，且平均增益至少 0.0001；
- **统计并列**：CI 穿过 0，或虽显著但绝对增益小于 0.0001；
- **效果更好但代价显著更高**：不直接称为架构效率更强，改报效果/资源 Pareto；
- **不稳定**：多 Seed 符号不一致、NaN 或末段波动异常，先进入稳定性诊断。

---

## 10. 必须新增的静态和数值测试

### 10.1 共享外壳一致性

- 四种 `mixer_type` 的 Input BN、SENet、Tokenizer、Readout、MLP variable spec 完全一致；
- 用同一固定输入和同一共享权重，Mixer 前的 `X0` 逐元素相同；
- `z` 均为 `[batch,1024]`；
- MLP 三层均为 `2048/2048/256`；
- 禁止图中出现 `gated_pool`、`bucket_cross`、`dcnm_shortcut`、`flatten_readout`。

### 10.2 RankMixer

- `H=T=32`；
- `D % H == 0`；
- 固定小张量上的 split/transpose/concat 与预期排列一致；
- 每层恰好两次 Add & LayerNorm；
- 32 套 PFFN 参数互不共享；
- 两层 Block 参数互不共享。

### 10.3 TokenMixer-Large

- Mixing/Reverting 前后形状均为 `[B,32,512]`；
- 无 pSwiGLU 时 `Revert(Mix(X)) == X`；
- mixed 与 original 两套 pSwiGLU 不共享；
- Pre-RMSNorm 位置正确；
- 长残差加的是 Block 原始输入；
- down 权重初始化尺度符合 0.01 约束；
- 图中不存在 v6 的增强 Readout。

### 10.4 UniMixer-Lite

- `F=32×512=16384`；
- `F % B == 0`，`G=512`；
- `r=128 <= G`；
- `W_G` 由低秩矩阵生成，而不是完整自由参数；
- `W_B` 由 8 个 basis 组合，而不是每块完整自由参数；
- 约束后矩阵的行和、列和接近 1；
- 对称误差在容差内；
- tau 在训练 0%、20%、100% 处分别为约 `1.0、1.0、0.05`；
- Siamese 双流更新和最终融合与公式一致；
- 第 32 个 Token 确实是共同 Global Token。

---

## 11. 结果表模板

### 11.1 结构与资源

| ID | Block | Block 参数 | 总 dense 参数 | FLOPs/batch | samples/s | 峰值显存 | p95 latency |
|---|---|---:|---:|---:|---:|---:|---:|
| E0 | NoMixer |  |  |  |  |  |  |
| E1 | RankMixer |  |  |  |  |  |  |
| E2 | TML |  |  |  |  |  |  |
| E3 | UniMixer-Lite |  |  |  |  |  |  |

### 11.2 单 Seed 原始结果

| ID | Seed | AUC | GAUC | LogLoss | COPC | 最后 5 个点 AUC std | 是否 NaN |
|---|---:|---:|---:|---:|---:|---:|---|
| E0 | 20260831 |  |  |  |  |  |  |
| E1 | 20260831 |  |  |  |  |  |  |
| … | … |  |  |  |  |  |  |

### 11.3 配对主结论

| Contrast | Mean ΔAUC | 95% CI | Holm p | 实用阈值通过 | 结论 |
|---|---:|---:|---:|---|---|
| E2 − E1 |  |  |  |  |  |
| E3 − E1 |  |  |  |  |  |
| E3 − E2 |  |  |  |  |  |

### 11.4 相对 NoMixer

| Contrast | Mean ΔAUC | 95% CI | 说明 |
|---|---:|---:|---|
| E1 − E0 |  |  | RankMixer 固定交互增益 |
| E2 − E0 |  |  | TML Core 增益 |
| E3 − E0 |  |  | UniMixer-Lite 增益 |

---

## 12. 第一轮完成后的最小后续实验

第一轮不预先执行以下实验。只有出现相应结果时再触发：

### 情况 A：TML 明显优于 RankMixer

只新增 1 个桥接实验：

```text
E2A_TML_NO_REVERT
```

其余保持 TML，去掉 Reverting 并退化为 RankMixer 式错位残差，用于判断收益是否主要来自语义对齐残差。

### 情况 B：UniMixer-Lite 明显优于两者

只新增 2 个论文关键消融：

```text
E3A_UNILITE_FIXED_TAU       # tau 始终为 1.0
E3B_UNILITE_NO_SIAMESE      # 改为普通 PostNorm
```

先验证收益是否来自可学习 Mixing 本身，还是温度/SiameseNorm 的训练稳定性。

### 情况 C：三者接近

不要继续横向堆小改造，直接进入深度扩展：

```text
L/M = 2 → 4 → 8
```

此时：

- RankMixer 继续用论文式 Post-LN；
- TML 在 `L>=4` 启用 gap=2 的 Inter-Residual 和 AuxLoss，最后层不加 interval residual；
- UniMixer-Lite 保留 SiameseNorm；
- Tokenizer、Global Token、Readout、Base MLP 仍完全相同。

这组实验才用于回答三种方案的深度 scaling 能力，不能用第一轮单一深度的结果宣称 scaling law。

---

## 13. 结论解释边界

第一轮可以回答：

- 在相同 Token 输入和相同任务头下，哪种 Mixer Block AUC 更高；
- 三种 Block 是否相对 NoMixer 有增益；
- 哪种 Block 更稳定、更快或更省显存；
- Global Token 经过哪种交互块后提供更有效的最终表示。

第一轮不能回答：

- 哪篇论文的完整线上系统更强；
- 哪个模型拥有更好的参数 scaling law；
- 哪个模型在等参数或等 FLOPs 下更优；
- Sparse-Pertoken MoE 是否有效；
- TML 的深层辅助残差是否优于 UniMixer 的 SiameseNorm。

这些问题分别需要多尺度、等资源、MoE 或深度专项实验。

---

## 14. 开跑前冻结清单

- [ ] 四组都使用 31 Local + 1 Global；Global 固定为最后一个 Token。
- [ ] 四组 Mixer 输入均为 `[B,32,512]`。
- [ ] 四组 Readout 均为 `Global || Mean(Local)`，输出 1024 维。
- [ ] 四组任务头均为 `2048/2048/256`，实现与参数完全相同。
- [ ] RankMixer 不含 Gated Pool、Bucket Cross 和 Query–Item Cross。
- [ ] TML 不含 v6 的 Global-conditioned Pool 和 Flatten route。
- [ ] UniMixer-Lite 不含额外 Token BN 和纯 Flatten Readout。
- [ ] 第一轮所有 Block 深度均为 2。
- [ ] 正式比较使用相同 3 个 Seed、相同训练样本和相同测试样本。
- [ ] 不导入任何旧 dense checkpoint，或四组导入完全相同的共享层 checkpoint。
- [ ] 预先记录模型参数、FLOPs、吞吐和峰值显存。
- [ ] 主指标、主 contrasts、0.0001 实用阈值和统计方法在看结果前冻结。

---

## 15. 主要论文依据

1. [RankMixer: Scaling Up Ranking Models in Industrial Recommenders](https://arxiv.org/abs/2507.15551)：语义 Tokenization、固定 Multi-head Token Mixing、Per-token FFN、Post-LayerNorm、残差及公开的 `T/D/L` 扩展方向。
2. [TokenMixer-Large: Scaling Up Large Ranking Models in Industrial Recommenders](https://arxiv.org/abs/2602.06563)：Global Token、Mixing–Reverting、Per-token SwiGLU、Pre-RMSNorm、down-matrix 小初始化、Inter-Residual/AuxLoss 与 Sparse-Pertoken MoE。
3. [UniMixer: A Unified Architecture for Scaling Laws in Recommendation Systems](https://arxiv.org/abs/2604.00590)：UniMixing-Lite 的全局低秩分解、局部 basis 组合、对称/温度/Sinkhorn 约束、Per-token SwiGLU 与 SiameseNorm。

---

## 16. 一句话执行口径

> 用同一个 `Input BN → SENet → 31 Local + 1 Global Tokenizer → Global||LocalMean Readout → 2048/2048/256 MLP` 外壳，只把中间两层 Block 分别换成 RankMixer、dense TokenMixer-Large 和 UniMixer-Lite；先跑 NoMixer + 三方案的 4×3 Seed 主矩阵，再根据结果做最多 1～2 个定向消融。
