# 三桶 CVR 场景下的 RankMixer 诊断与下一代设计

## 1. 结论先行

当前对比结果为：

- `cvr_bn_senet_dcnm_fst.py`: fst_CVR ROC-AUC = **0.867060**
- `cvr_bn_rankmixer_v1.py`: fst_CVR ROC-AUC = **0.864326**
- 绝对差值 = **-0.002734**，即 **-0.2734 个百分点**；相对 base 约 **-0.3153%**。

这不是一个可以用随机波动解释的小差距。代码审计显示，v1 同时存在训练协议、tokenization 和参数预算三类问题，因此当前实验不能证明 RankMixer 架构本身弱于 base。

代码中的评估实现调用 `flood_auc` 和累计 `RocAucAccum`，因此本文将用户所称“预测准确率”按 **ROC-AUC** 解读，而不是阈值分类 accuracy。

最高优先级的原因依次是：

1. **checkpoint 恢复后的 dense 学习率可能只有 base 的 1/10**；
2. **所有 token 边界都切穿字段 embedding，并且两个 token 跨越三桶边界**；
3. **v1 核心稠密参数约 167.3M，而 base 约 90.2M，三天训练并非参数匹配实验**；
4. **v1 删除了 base 中有效的字段级 SENet 和显式乘性交叉归纳偏置**；
5. **v1 每层有三次 LayerNorm，而 RankMixer 论文公式只有两次 Add&Norm**。

因此推荐先验证“修复训练协议 + 语义 token + 参数匹配”的 dense RankMixer，再考虑 Sparse-MoE。

## 2. 代码证据

### 2.1 输入的真实规模

`data.cvr.cvr_fea_v10_base_cold` 只保留三桶，且每个字段 embedding 均为 17 维：

| 桶 | 字段数 | 展平维度 |
|---|---:|---:|
| common | 385 | 6,545 |
| item | 835 | 14,195 |
| creative | 14 | 238 |
| 合计 | 1,234 | 20,978 |

当前工作区的 `scripts/set-x.txt` 还存在一个实验追溯问题：文件中 `train_dates=2026-07-02:2026-07-07`，并非明确的三天区间，而且模型入口仍是 base；仓库中没有对应 RankMixer 运行脚本或训练日志。因此 0.867060/0.864326 是否严格使用相同三天、相同 step 和相同 checkpoint 策略，目前缺少本地证据，下一轮必须从平台日志固化这些字段。

### 2.2 v1 的等长切分破坏字段和桶语义

v1 将三桶直接拼成 20,978 维长向量，再切成 15 个 1,311 维和 1 个 1,313 维 token。

- `1,311 mod 17 = 2`，所以 15 个累计切点在字段内依次偏移 2、4、6……维；**没有任何切点位于完整字段边界**。
- token 4 包含 `1,301 common + 10 item` 维。
- token 15 包含 `1,075 item + 238 creative` 维。
- creative 仅 14 个字段，被塞进最后一个以 item 为主的 token，无法获得独立 PFFN 子空间。

RankMixer 论文明确要求用领域知识构建“semantically coherent clusters”。参数无关的 token mixing 只有在 token 本身代表稳定特征子空间时才有合理归纳偏置。

### 2.3 checkpoint 学习率恢复逻辑缺失

base 的 `train_init` 在恢复 checkpoint 后执行：

```python
session.run(learning_rate_utils.get_or_create_milestone_step_reset_op())
```

v1 的 `train_init` 只重新初始化数据 iterator，没有重置 milestone。

当前 `gauss_decay` 的实际逻辑为：

```text
step_rate = (global_step - milestone_step - 60000) / 40000
lr_factor = max(0.1, exp(-step_rate^2))
```

如果 checkpoint 中恢复的 `global_step >= 120,700`，v1 会直接使用最小因子 0.1，即把脚本配置的 `2e-5` 变成 `2e-6`；base 则会从 `2e-5` 重新开始。旧 checkpoint 很可能超过该阈值，这是当前最应先用日志确认的因素。

### 2.4 参数预算不匹配

论文给出的主要规模公式为：

```text
PFFN Params ≈ 2 * k * L * T * D^2
```

v1 默认 `T=16, D=768, L=2, k=4`：

| 组件 | 估算参数量 |
|---|---:|
| 16 个输入投影 | 16.12M |
| 两层 Per-token FFN | 151.12M |
| LN + 输出头 | 约 0.01M |
| v1 核心合计 | **约 167.25M** |

base 在当前三桶宽度、`dcnm_layer=500, cross_num=2, cvr_layers=[2048,2048,256]` 下，SENet + DCNM + MLP 核心约 **90.21M**。v1 是 base 的约 **1.85 倍**，却只训练相同时间；更大的冷启动 dense tower 可能尚未收敛。

v2 将 `k` 默认改为 2，并加入 SENet、池化和三桶交叉后约 **95.77M**，更接近 base 与 100M 档位。

### 2.5 结构与论文公式不一致

论文每个 block 为：

```text
S = LN(TokenMixing(X) + X)
X' = LN(PFFN(S) + S)
```

v1 在 PFFN 前又增加了一次 `LN(S)`，每层共三次 LN。额外归一化可能抹平推荐 embedding 中有意义的幅度，并改变论文已验证的优化路径。v2 恢复为恰好两次 Add&Norm。

### 2.6 base 的强归纳偏置被同时移除

base 不是普通 MLP，而是：

1. 字段级、样本自适应的分层 SENet；
2. 两层 DCNM 乘性交叉；
3. 三层大 MLP。

v1 一次性移除了 SENet 和 DCNM，仅依赖固定 token 置换与 PFFN。RankMixer 原论文能成立的前提是语义 token 正确；当前 token 已被破坏，因此同时移除两类强归纳偏置风险很高。

### 2.7 工程实现没有获得论文宣称的并行收益

v1 用 Python 循环创建 `L * T * 2 = 64` 个独立 FC 运算。论文的高 MFU 来自把 per-token FFN 融合为大批量矩阵乘法。v2 使用 `[T,D,kD]` 与 `[T,kD,D]` 的 batched matmul，参数仍按 token 独立，但执行拓扑更接近论文设计。

## 3. Semantic-Cross RankMixer v2

实现文件：`cvr_bn_rankmixer_v2.py`。

### 3.1 数据约束

模型只读取现有：

- common embeddings
- item embeddings
- creative embeddings

不引入 dense、sequence、gattr、DIN、外部统计特征或新标签。

### 3.2 语义且字段对齐的 tokenization

默认 `T=16`，按字段量比例且保证每桶至少一个 token，得到：

| 桶 | token 数 | 每个 token 的完整字段数 |
|---|---:|---|
| common | 5 | `[77,77,77,77,77]` |
| item | 10 | `[84,84,84,84,84,83,83,83,83,83]` |
| creative | 1 | `[14]` |

每个 token 只由同一桶中的完整字段组成，随后使用与 v1 一致的 `gelu_2` 非线性投影到 `D=768`。v2 的变化集中在按桶、按完整字段构造语义 token，而不是改变 token 投影激活。

### 3.3 字段重要性门控

复用 base 的层级 SENet 形式：

- common gate 由 common 字段生成；
- item gate 由 common + item 生成；
- creative gate 由 common + item + creative 生成。

门控为 `2 * sigmoid(.)`，初始量级靠近恒等映射，并能让 14 个 creative 字段在进入独立 token 前得到样本级重权。

### 3.4 RankMixer 主干

- parameter-free multi-head token mixing，`H=T=16`；
- 每个 token 独立参数的 PFFN；
- PFFN 用 batched matmul 实现；
- `L=2, D=768, k=2`；
- 每层严格采用两次 Add&Norm。

### 3.5 自适应池化

论文的固定 mean pooling 对所有 token 权重相同。v2 使用零初始化的单分数门控：训练开始时严格等价于 mean pooling，随后学习每个样本的 token 权重。它不是 token-token self-attention，不计算异构空间间的内积相似度。

### 3.6 三桶显式乘性交叉支路

从输入 token 分别聚合 common/item/creative 表示，构造：

```text
[C, I, A, C⊙I, C⊙A, I⊙A]
```

经过一个 6D→D 投影后，以初始约 0.119 的可学习门控残差融合到 RankMixer 输出。该支路补充 base DCNM 擅长的显式乘性交互，同时仍然只使用三桶输入。

## 4. 推荐训练配置

模型入口：

```text
models.rankmixer.cvr_bn_rankmixer_v2.MLPModel
```

建议模型参数：

```json
{
  "use_senet": true,
  "use_senet_bn": true,
  "rm_token_num": 16,
  "rm_head_num": 16,
  "rm_bucket_token_counts": [5, 10, 1],
  "rm_hidden_dim": 768,
  "rm_layer_num": 2,
  "rm_ffn_expand": 2,
  "rm_token_proj_act": "gelu_2",
  "rm_proj_ln": false,
  "rm_use_gated_pool": true,
  "rm_use_bucket_cross": true,
  "optimizer": "flood_adam",
  "learning_rate": 2e-5,
  "batch_size": 2048,
  "embedding_size": 17,
  "batch_norm": true,
  "use_riemann_bn": true
}
```

第一轮公平架构实验建议 base、v1、v2 都采用 `change_fea=cold`：只热启相同 sparse embedding，dense 全部冷启动。否则 base 能恢复完整 DCNM/MLP，而 RankMixer 的新变量随机初始化，比较不公平。

产品候选实验可以恢复 v2 与 base 同名的 `bn_input` 和 `senet` 变量，但必须单独标记为 warm-start 实验。

## 5. 最小消融矩阵

所有实验固定：同三天样本、相同日期顺序、相同 test_date、相同 600 个测试文件、相同 sparse checkpoint、相同有效全局 batch、相同 optimizer steps。

| 实验 | 目的 | 关键配置 |
|---|---|---|
| A0 base-cold | 公平参考 | base，dense 冷启 |
| A1 v1-original | 复现实验 | 原 v1 |
| A2 v1-lrfix | 验证 LR 根因 | 只补 milestone reset |
| A3 v2-core | 验证语义 token/参数匹配 | SENet=false, gated_pool=false, bucket_cross=false |
| A4 v2+SE | 验证字段门控 | use_senet=true |
| A5 v2+SE+pool | 验证动态池化 | rm_use_gated_pool=true |
| A6 v2-full | 验证乘性交叉 | rm_use_bucket_cross=true |

训练资源有限时，优先顺序为 `A2 → A3 → A4 → A6`。

`A2` 已提供独立入口，除学习率 milestone 外不改变 v1 架构：

```text
models.rankmixer.cvr_bn_rankmixer_v1_lrfix.MLPModel
```

## 6. 验证门槛

### 图构建与短跑

1. 在 Flood/Cayman 训练镜像中完成 train/test/export 三张图构建；
2. 先跑 100～500 step，确认无 NaN、梯度非空、BN/LN 变量复用正确；
3. 日志必须打印恢复后的 `global_step`、`milestone_step` 和实际 `learning_rate`；
4. 检查 16 个 token 的字段组为 `5/10/1`，总字段数恰为 `385/835/14`。

### 一天筛选

- 主指标：累计 ROC-AUC；
- 同时看 LogLoss、PR-AUC、COPC、正样本预测均值；
- 记录实际消费样本数、optimizer steps、每 step 时间与 peak memory；
- 若 v2 一天曲线仍在上升，不应仅按同 wall-clock 提前淘汰，应同时给出同 step 与同样本量结果。

### 三天决策

- 第一门槛：超过 v1 的 `0.864326`；
- 第二门槛：达到或超过 base 的 `0.867060`；
- COPC 建议处于 `[0.98, 1.02]`，否则需要校准或 bias 修正；
- 按 creative 特征活跃度/频次、item 长尾度和用户活跃度分桶观察 AUC，防止总 AUC 提升但长尾退化。

## 7. 后续可扩展方向

### 方向 A：先扩大 token 语义，而不是盲目加参数

若 v2-core 有效，可比较 `T=12/16/24`，每桶始终保留独立 token；不要再按裸维度跨桶切分。

### 方向 B：低秩 DCN-V2 支路

若三桶乘积支路有效但仍未达到 base，可在 16×D token 展平表示上加两层 rank-32/64 的低秩 DCN-V2，并与 RankMixer 并行融合。它能显式覆盖有界阶交互，成本远低于原始 20,978 维 full DCNM。

### 方向 C：实例级 mask

把当前静态 token 投影升级为 instance-guided token mask，但初始化为恒等，避免训练初期过度压制稀疏 creative 信号。

### 方向 D：同数据蒸馏

使用 base 对同一批三桶样本产生 teacher logit，训练目标采用 `BCE(label) + λ * BCE(teacher_prob)`。不增加输入数据，适合解决新 dense tower 冷启动；需要作为独立实验，不能与纯架构对比混在一起。

### 方向 E：最后再做 Sparse-MoE

只有 dense v2 已超过 base 后，才把 PFFN 替换成论文的 ReLU routing + dense-training/sparse-inference。当前阶段直接上 MoE 会把 tokenization 和优化问题与 expert balance 混在一起。

## 8. 文献依据

1. [RankMixer: Scaling Up Ranking Models in Industrial Recommenders](https://arxiv.org/abs/2507.15551) - 语义 token、参数无关 token mixing、per-token FFN、两次 Add&Norm 和规模公式。
2. [DCN V2: Improved Deep & Cross Network and Practical Lessons for Web-scale Learning to Rank Systems](https://arxiv.org/abs/2008.13535) - 低秩显式交叉和并行 deep/cross 结构。
3. [FiBiNET: Combining Feature Importance and Bilinear Feature Interaction for CTR Prediction](https://arxiv.org/abs/1905.09433) - 样本级 SENet 字段重要性与双线性交互。
4. [MaskNet: Introducing Feature-Wise Multiplication to CTR Ranking Models by Instance-Guided Mask](https://arxiv.org/abs/2102.07619) - 实例级 mask 与乘性交互。
5. [Hiformer: Heterogeneous Feature Interactions Learning with Transformers for Recommender Systems](https://arxiv.org/abs/2311.05884) - 异构特征空间不能简单按同质 token 处理。
6. [FinalMLP: An Enhanced Two-Stream MLP Model for CTR Prediction](https://arxiv.org/abs/2304.00902) - 隐式与显式交互双流、特征门控和流间融合。

## 9. 当前验证边界

本地已完成 Python 语法、路径白名单和静态参数/分组校验。当前桌面环境没有生产所需的 TensorFlow 1.x、Flood、Cayman 与 HDFS 数据，因此不能在本地声称 v2 已提升 AUC；最终结论必须由上述公平消融实验给出。
