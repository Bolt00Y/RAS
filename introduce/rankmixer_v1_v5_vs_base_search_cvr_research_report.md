# 搜索精排首次转化率模型：当前算法模块有效性与无效性分析报告

> 报告日期：2026-08-19  
> 研究对象：`fst_CVR` 搜索精排模型  
> 结论依据：当前仓库源码、启动参数及 `docs/background.md` 中记录的离线结果

## 0. 执行摘要

当前证据只回答“现有模块在当前搜索精排 CVR 场景中表现为何”，不延伸到后续架构或实验方案。

1. **Base 整体结构当前有效性最强。** 在相同首日口径下，Base AUC 为 **0.864538**，高于全部已完成测试的 RankMixer；v1 连续四天相对 Base 低 25–28 AUC bp。现有结果能够证明 `分层 SENet + 两层全维 DCN-M + 深 MLP` 这一整套结构有效，但没有 Base 内部消融，不能把领先幅度单独归因给其中任一模块。
2. **已完成测试的 RankMixer 中 v3 最好。** v3 AUC 为 **0.862991**，仍比 Base 低 **0.001547**，即 15.47 AUC bp（本文约定 `1 bp = 0.0001 AUC`）。
3. **业务语义分组是当前最清晰的正向模块证据。** v3 相对 v2 主要只改变 token 字段组织方式，参数量和主计算量不变，AUC 提升 **3.01 bp**。该结论限于当前单次冷启动结果，能够说明“本次运行有效”，不能证明跨 seed 稳定收益。
4. **当前形式的 QICross-Lite 是最清晰的负向模块证据。** v4 在 v3 上增加该分支后低 **2.82 bp**，且结果几乎退回 v2 水平。它说明这段具体实现未形成净收益，不等价于所有 Query–Item 交互都无效。
5. **v2 是组合式改善，单模块效果不可识别。** v2 比 v1 高 **6.57 bp**，但字段安全切分、SENet、`k=2`、Add&Norm、gated pooling、Bucket Cross 和学习率 reset 同时变化，只能判定整套修改在本次对照中有效。
6. **v1 的裸维度 tokenization 与字段结构不兼容。** 代码可确定它切断 17 维字段 embedding，并跨 common/item/creative 桶构造 token；v1 整体连续落后 Base，但当前结果不能定量分离字段切断、移除 SENet/DCN-M/深头、参数膨胀和学习率生命周期各自造成的损失。
7. **v5 目前只有“训练拟合有效”的组合证据。** 它约完成 `140000/270000=51.9%`，中间训练 AUC 超过 v3 并接近 Base；但 v5 同时改变 token 数、维度、分组、FFN、归一化、全局路径、读出和任务头，既无完整测试 AUC，也无法把训练提升归因给任何单个模块。

现有模块的证据状态可概括为：

| 证据状态 | 模块/结构 | 当前场景结论 |
|---|---|---|
| 单次直接正向 | v3 业务语义 token 分组 | 相对 v2 为 +3.01 bp |
| 单次直接负向 | v4 QICross-Lite | 相对 v3 为 -2.82 bp |
| 组合正向、不可拆分 | v2 整套修复；Base 整体主塔 | 整体有效，内部模块贡献未知 |
| 结构不兼容、损失不可拆分 | v1 裸维度切分 | 字段边界确定被破坏，整体效果持续落后 |
| 仅训练拟合正向 | v5 整体结构 | 中间训练 AUC 正向，测试有效性未知 |
| 未被单独验证 | SENet、Bucket Cross、gated pooling、Add&Norm、`k`、LR reset、固定 Mixer/PFFN，以及全部 v5 子模块 | 现有对照不足以判定独立 AUC 贡献 |

---

## 1. 证据范围与结论口径

### 1.1 主要证据源

| 类型 | 文件 | 用途 |
|---|---|---|
| 实验记录 | [`docs/background.md`](../docs/background.md) | 训练/测试日期协议与 AUC 结果 |
| Base 实现 | [`cvr_bn_senet_dcnm_fst.py`](../src/models/seq_model/cvr_bn_senet_dcnm_fst.py) | SENet、DCN-M、MLP 和损失实现 |
| RankMixer 实现 | [`cvr_bn_rankmixer_v1.py`](../src/models/rankmixer/cvr_bn_rankmixer_v1.py) 至 [`cvr_bn_rankmixer_v5.py`](../src/models/rankmixer/cvr_bn_rankmixer_v5.py) | 各版本前向结构和训练生命周期 |
| 启动参数 | [`bash`](../bash) 下的 `set-rankmixer-v*-args.txt` | 实际启用开关、维度、优化器及冷/热启动配置 |
| 复杂度核算 | [`model_parameters_flops_analysis.md`](model_parameters_flops_analysis.md) | Base、v1、v2、v3 的静态参数/FLOPs 复核 |
| 特征说明 | [`rankmixer_v2_三桶数据特征清单.txt`](../docs/数据特征清单/rankmixer_v2_三桶数据特征清单.txt) | 字段规模和业务语义 |

### 1.2 三类陈述严格分离

本报告使用三种证据等级：

- **代码事实**：可由当前源码和参数文件直接证明，例如 v1 的切分点会落在字段内部。
- **实验事实**：`background.md` 已记录的 AUC 或训练状态，例如 v3 首日测试 AUC 为 0.862991。
- **研究推论**：由代码与结果共同支持、但现有对照不能证实的解释，例如“v4 的零输出初始化可能造成新分支早期学习滞后”。

单次 AUC 差异不会被直接写成因果结论；训练 AUC 也不会替代测试 AUC。

### 1.3 共同实验协议

按背景记录，模型均使用相同三桶特征和相同日期协议：

- `2026-07-01` 约 5.5 亿样本用于首次训练；
- sparse embedding 热启动，dense 参数冷启动；
- `2026-07-02` 约 1.1 亿样本用于测试；
- 后续按天从前一天 checkpoint 热启动；
- 主指标为 `fst_CVR` ROC-AUC。

v2–v5 的展开参数进一步显示其共同配置包括：

- `batch_size=2048`；
- `optimizer=flood_adam`；
- dense learning rate `2e-5`；
- sparse learning rate `0.05`；
- `ignore_dense_checkpoint=True`、`ignore_sparse_checkpoint=False`；
- 单任务 BCE，未启用 Base 源码中可选的 wide、last、delay 和辅助多任务分支。

因此，当前 Base 与 RankMixer 的核心差异主要来自 dense 主塔，而不是输入字段或主标签不同。

---

## 2. 实验结果复盘

### 2.1 首日公平口径

下表均对应 `2026-07-01` 训练、`2026-07-02` 测试。

| 模型 | 测试 AUC | 相对 Base | 相对 v1 | 当前判断 |
|---|---:|---:|---:|---|
| Base | **0.864538** | — | +25.05 bp | 当前首日冠军 |
| RankMixer v1 | 0.862033 | -25.05 bp | — | 明显落后 |
| RankMixer v2 | 0.862690 | -18.48 bp | +6.57 bp | 修复后改善，但未追平 |
| RankMixer v3 | **0.862991** | **-15.47 bp** | **+9.58 bp** | 已完成 RankMixer 最优 |
| RankMixer v4 | 0.862709 | -18.29 bp | +6.76 bp | QICross 未形成净增益 |
| RankMixer v5 | 尚无完整测试结果 | 不可比较 | 不可比较 | 训练约 140k/270k 步 |

版本间最有信息量的差值为：

- v3 − v2 = **+0.000301**（+3.01 bp）；
- v4 − v3 = **−0.000282**（−2.82 bp）；
- v4 − v2 = **+0.000019**（+0.19 bp，几乎相同）。

### 2.2 Base 与 v1 的四日对齐结果

| 训练日序号 | Base AUC | v1 AUC | v1 − Base |
|---:|---:|---:|---:|
| 1 | 0.864538 | 0.862033 | -25.05 bp |
| 2 | 0.865633 | 0.862850 | -27.83 bp |
| 3 | 0.867060 | 0.864326 | -27.34 bp |
| 4 | 0.866114 | 0.863436 | -26.78 bp |
| **四日均值** | **0.865836** | **0.863161** | **-26.75 bp** |

v1 的对齐差距在四天内始终约为 25–28 bp，并没有随着逐日训练明显收敛到 Base。这比单日结果更支持“结构或优化协议存在持续性缺口”，而不只是首次 dense 冷启动尚未收敛。

### 2.3 v5 的当前状态应如何解读

v5 当前约训练 140,000 步，完整训练预计约 270,000 步：

```text
训练进度 ≈ 140000 / 270000 = 51.9%
```

中间训练 AUC 已超过 v3，并与 Base 的训练阶段 AUC 接近。这个现象能够支持和不能支持的结论边界如下：

- 能够支持：v5 **整套结构**在约 140,000 步时表现出比 v3 更强的训练集拟合；
- 不能支持：Global Token、双空间 SwiGLU、更多 token、Flatten 读出或深任务头中的任一个已独立有效；
- 不能支持：v1–v4 一定欠拟合，因为当前没有同结构容量对照；
- 不能支持：v5 的测试 AUC 已追平 Base，或 348M dense 容量没有扩大 train–test gap。

---

## 3. 统一输入、目标与复杂度

### 3.1 三桶输入

所有模型使用相同 1,234 个字段，每个字段 embedding 维度为 17：

| 桶 | 字段数 | 展平维度 | 字段占比 |
|---|---:|---:|---:|
| common | 385 | 6,545 | 31.20% |
| item | 835 | 14,195 | 67.67% |
| creative | 14 | 238 | 1.13% |
| **合计** | **1,234** | **20,978** | **100%** |

这使当前任务呈现三个与结构有效性直接相关的特征：

1. item 桶占绝对主体，粗粒度平均会承担更高的信息压缩风险；
2. query、用户状态和实时上下文主要在 common 桶，但它们需要与 item 发生条件交互；
3. creative 只占 1.13%，将其保留为单独 token 与其字段规模相符；当前结果没有专门衡量 creative 分支深度。

### 3.2 固定 dense 参数量

下表不含动态稀疏 embedding。若最终保留的 sparse key 总数为 `U`，所有模型还共同包含约 `17U` 个 embedding 参数。

| 模型 | 固定 dense 参数 | 相对 Base | 主要参数来源 |
|---|---:|---:|---|
| Base | 90,341,785（90.342M） | 1.00× | 两层 DCN-M + 三层 MLP |
| v1 | 167,293,157（167.293M） | 1.85× | `k=4` 的 16 路独立 PFFN |
| v2 | 95,809,126（95.809M） | 1.06× | `k=2` PFFN + SENet + Bucket Cross |
| v3 | 95,809,126（95.809M） | 1.06× | 与 v2 相同，仅分组变化 |
| v4 | 96,439,272（96.439M） | 1.07× | v3 + 0.630M QICross-Lite |
| v5 | **348,432,486（348.432M）** | **3.86×** | 双空间 Per-token SwiGLU 为主 |

v5 参数进一步拆分如下：

| v5 模块 | 参数量 | 占 v5 dense 参数 |
|---|---:|---:|
| 输入 BN + SENet | 0.564M | 0.16% |
| 31 个本地 token 投影与 RMSNorm | 21.545M | 6.18% |
| Global Token 两层编码 | 22.533M | 6.47% |
| 两层、双空间 Per-token SwiGLU | **277.266M** | **79.58%** |
| Global-conditioned pooling | 0.262M | 0.08% |
| Flatten readout | 16.254M | 4.66% |
| `[2048,2048,256]` 任务头 | 9.975M | 2.86% |
| 其他 RMSNorm | 0.033M | 0.01% |

因此，v5 的参数增长主要不是 Global Token 或深任务头，而是每层同时在 mixed space 和 original space 使用独立的 32 路 SwiGLU。其主矩阵乘前向计算约为 0.70 GFLOPs/样本量级，尚未计入 lookup、激活、归一化和训练反向。

---

## 4. Base：SENet + DCN-M + Deep MLP

### 4.1 有效前向流程

```text
三桶 sparse embedding
→ 每桶 BatchNorm
→ 分层 SENet 字段门控
→ 拼接为 20,978 维向量
→ 2 层 DCN-M（20,978→500→20,978）
→ MLP [2048, 2048, 256]，每层 BN + GELU
→ Linear(256→1)
→ sigmoid
```

当前启动参数关闭了源码中的 wide、last、delay 和多任务辅助塔，所以 Base 的有效训练目标也是单一 `fst_CVR` BCE。

### 4.2 分层 SENet

Base 不是对所有字段使用一个无条件 gate，而是按业务层级构造上下文：

- common gate 只看 common 字段；
- item gate 同时看 common + item；
- creative gate 同时看 common + item + creative。

每个 gate 使用 `2·sigmoid(·)`，初始化附近以 1 为中心，既能抑制也能增强字段。其价值在于同一个字段在不同用户、query、商品和请求上下文中可以拥有不同权重。

### 4.3 全维 DCN-M

每层交叉可概括为：

```text
h_l = W2_l(W1_l(x_l))              # 20,978 → 500 → 20,978
x_{l+1} = LN(x_0 ⊙ h_l + x_l)
```

关键点是乘法发生在完整 20,978 维空间，且每一层都与原始输入 `x_0` 相乘。它能保留字段位置并显式构造有界阶交互；相比只对三个桶均值做乘积，其交互粒度更细，但现有结果没有单独比较两种 Cross。

### 4.4 深任务头

DCN-M 后还有 `[2048,2048,256]` 的非线性 MLP。Base 因此同时具备：

- 样本级字段选择；
- 全维显式乘性交互；
- 大容量的最终任务映射。

这三类模块共同构成 Base 与 v1–v4 的主要结构差异，也与 Base 持续领先的现象一致；由于没有逐模块对照，它们只是整体领先的结构解释，不能被分别认定为独立增益来源。

---

## 5. RankMixer v1：裸维度切分的直接替换

### 5.1 前向结构

```text
三桶 embedding
→ 每桶 BatchNorm
→ [common; item; creative] 拼成 20,978 维
→ 按裸维度切成 16 段
→ 16 个独立非线性投影到 768 维
→ 2 层 RankMixer，T=H=16，D=768，k=4
→ Mean Pool
→ Linear(768→1)
```

v1 构造函数保留了 `use_senet` 参数，但主塔没有调用 SENet；Base 的 DCN-M 和三层 MLP 也被全部删除。

### 5.2 字段完整性问题是确定性事实

v1 的切分宽度为：

```text
20,978 = 1,311 × 15 + 1,313
1,311 mod 17 = 2
```

由于每个字段是 17 维，前 15 个切点没有一个落在字段边界上。结果包括：

- 大量字段 embedding 被拆到两个 token；
- 第 5 个 token 同时包含 common 尾部和 item 头部；
- 最后一个 token 同时包含 item 尾部和全部 creative；
- token 身份依赖偶然的维度位置，而不是稳定业务子空间。

参数无关 mixing 的归纳偏置建立在 token 代表稳定子空间的前提上；v1 在进入 Mixer 前已经破坏了这一前提。

### 5.3 Block 与参数预算

v1 每个 block 使用：

```text
Mix → Add → LN
→ LN → Per-token FFN(k=4)
→ Add → LN
```

即每层实际有三次 LayerNorm。16 个 token 各自拥有独立 PFFN 参数，`k=4` 使两层 PFFN 成为约 151M 参数的主体。v1 比 Base 大 85%，却使用同样的训练数据和近似训练预算，冷启动收敛压力更大。

### 5.4 学习率生命周期问题

原始 v1 的 `train_init()` 只重置数据 iterator，没有执行 milestone reset。Base 和 v2–v5 会在 chief worker 上执行该 reset。逐日从 checkpoint 恢复时，v1 可能沿用旧的衰减位置，导致实际 learning rate 低于配置值。

仓库提供了 [`cvr_bn_rankmixer_v1_lrfix.py`](../src/models/rankmixer/cvr_bn_rankmixer_v1_lrfix.py)，它只增加 milestone reset，不改变前向结构；但背景中没有该版本的独立 AUC，因此学习率问题的效果贡献尚未量化。

### 5.5 结果解读

v1 四日平均比 Base 低 26.75 bp，且差距没有明显缩小。由于字段切分、结构、参数量、输出头和学习率协议一次改变了多项因素，v1 结果不能用来否定 RankMixer 本身。

---

## 6. RankMixer v2：Field-safe Semantic-Cross RankMixer

### 6.1 相比 v1 的完整改动集合

v2 同时完成以下修改：

1. 只接受 common/item/creative，额外桶非空时立即报错；
2. 保留完整字段边界，按桶分配 `[5,10,1]` 个 token；
3. 恢复与 Base 同构的分层 SENet；
4. PFFN 从 Python 循环改为 token-major batched matmul；
5. `k=4 → 2`，把 dense 参数压回约 100M；
6. 每个 block 恢复为两次 Add&Norm；
7. Mean Pool 改为零初始化的 gated pooling；
8. 增加桶级显式乘积 residual；
9. 每次训练初始化时重置学习率 milestone。

### 6.2 Token 构造

v2 按 FeatureConfig 中的字段顺序连续均分：

| 桶 | token 数 | 每个 token 字段数 |
|---|---:|---|
| common | 5 | `[77,77,77,77,77]` |
| item | 10 | `[84×5, 83×5]` |
| creative | 1 | `[14]` |

相较 v1，它保证字段不被切断、桶不被混合；但同一个 token 仍可能机械地混合价格、文本、多模态、统计和召回等无关主题。

### 6.3 RankMixer 主干

每层为：

```text
h = LN(Mix(x) + x)
y = LN(PFFN_token(h) + h)
```

`Mix` 是无参数 reshape/transpose 置换。当 `H=T=16` 时，每个输出 token 会收集所有输入 token 的一个 head 子空间，再由该位置独立的 PFFN 做非线性变换。

### 6.4 Gated Pooling

v2 对每个隐藏 token 预测一个分数后 softmax 加权。分数层零初始化，因此训练第 0 步严格等价于平均池化，之后才允许样本自适应地选择 token。

### 6.5 Bucket Cross 与 Base DCN-M 不等价

v2 先对三桶 token 分别求均值 `c/i/a`，再构造：

```text
[c, i, a, c⊙i, c⊙a, i⊙a] → FC(6D→D) → LN → scalar gate
```

这个分支只在三个桶摘要上交互，无法恢复 Base 在 20,978 个位置上的字段级 cross。因此，“v2 已经有 cross 但仍低于 Base”不能证明 DCN-M 无效。

### 6.6 结果解读

v2 首日比 v1 高 6.57 bp，说明整体修复方向有效；但由于一次增加/修改了至少九项因素，不能把全部提升归因于 SENet、字段完整性或 Bucket Cross 中的任一个。

---

## 7. RankMixer v3：固定业务语义 token

### 7.1 v3 是当前最干净的结构对照

v3 保持 v2 的以下部分不变：

- `T=H=16、D=768、L=2、k=2`；
- 分层 SENet；
- gated pooling；
- Bucket Cross；
- 输出头、损失、优化器和学习率处理。

唯一核心变化是 token 的字段组织方式。

### 7.2 16 个语义组

| 桶 | 语义组 | 字段数 |
|---|---|---|
| common | 用户静态画像/设备 | 16 |
| common | 交易购买与消费价值 | 90 |
| common | 长期兴趣与行为历史 | 92 |
| common | Query 意图与召回上下文 | 85 |
| common | 实时会话与漏斗 | 102 |
| item | 商品身份/类目/静态质量 | 98 |
| item | 文本与 Query 相关性 | 71 |
| item | 多模态相似性 | 58 |
| item | 当前价格与促销供给 | 60 |
| item | 用户价格偏好 | 126 |
| item | 商品/类目/店铺全局统计 | 73 |
| item | 正向购买与收藏偏好 | 46 |
| item | 曝光、点击与停留互动 | 134 |
| item | 当前会话与位置上下文 | 33 |
| item | 召回、图关系与排序 | 136 |
| creative | 创意展示与促销表达 | 14 |

代码会校验全部 1,234 个字段恰好出现一次，禁止遗漏、未知字段、桶内重复和跨桶重复。每个 token scope 直接使用语义组名，checkpoint 和诊断也更可解释。

### 7.3 参数不变但输入归纳偏置改变

每个字段仍只进入一个投影，投影总权重始终为：

```text
20,978 × 768
```

因此 v2 与 v3 的参数量、主要 MatMul FLOPs 完全相同。v3 的 +3.01 bp 不能由“参数更多”解释。

### 7.4 当前证据支持与不支持什么

支持：

- 在当前单次对照中，业务语义分组优于桶内按字段顺序均分；
- 这 3.01 bp 差异不能由参数量或主 MatMul 计算量变化解释；
- 保持 query、价格、商品质量、历史偏好和实时会话等语义边界，与当前更高 AUC 同时出现。

不支持：

- 不能凭一个 seed 宣称语义分组稳定提升 3.01 bp；
- 不能宣称当前 16 组就是最优粒度；
- 最大 item 组包含 136 个字段，输入维度 2,312 压到 768，仍可能存在组内压缩和主题混杂。

---

## 8. RankMixer v4：QICross-Lite

### 8.1 新增结构

v4 在 v3 基础上读取三个输入语义 token：

- `common_query_intent_retrieval`；
- `item_text_relevance`；
- `item_static_identity_quality`。

Query 和两个 item token 分别从 768 维投影到 128 维。对每个 Query–Item 对构造：

```text
[q, i, q⊙i, q−i] → pair hidden
[q, i, q⊙i] → scalar gate
pair hidden → 768 维 residual
```

两个 residual 与 Bucket Cross 一起加入 pooled context，再共用一套 fusion LayerNorm。

### 8.2 初始化行为

v4 的 output projection 零初始化，gate 初始约为 `sigmoid(-2)=0.119`。因此：

- 第 0 步 QICross 输出严格为 0；
- 最初梯度主要更新 output projection；
- output 不再为 0 后，梯度才逐步传回 pair hidden 和低秩投影。

零输出初始化具有“初始不扰动原路径”的代码性质；当前实验却是 dense 全冷启动，不存在已经训练好的 v3 dense 主路径可供保护。它还会让上游低秩投影在最初阶段拿不到有效梯度，这是解释收敛滞后的一种机制，但当前日志不能证明它就是 AUC 下降原因。

### 8.3 参数与结果

QICross-Lite 只新增约 630,146 个参数，v4 总 dense 参数约 96.439M。它不是因模型过大而明显受罚。

但当前结果为：

```text
v4 − v3 = -2.82 bp
v4 − v2 = +0.19 bp
```

合理的待验证解释包括：

1. item text token 已包含大量人工 Query–Item 交叉字段，新分支信息重复；
2. Query token 本身包含 85 个查询、召回和上下文字段，低秩投影可能仍然过粗；
3. residual 只修改最终 context，没有直接更新目标 item token；
4. 零 output 初始化造成分支有效学习滞后；
5. 改善可能只发生在特定 query 切片，未反映到全局 AUC；
6. 2.82 bp 也可能包含随机初始化波动，因为启动配置未固定 seed。

因此，当前能下的结论只针对 **v4 这段 QICross-Lite 实现及本次 dense 冷启动运行**：它没有形成净增益。上述六点均属于可能机制，不能替代实验归因。

---

## 9. RankMixer v5：大容量双空间 Mixer

### 9.1 v5 与 v4 不是单变量关系

v5 同时改变 token 数、hidden dim、分组、归一化、FFN、池化、读出和任务头。即使最终 AUC 大幅提升，也不能直接说“是 SwiGLU 有效”或“是 Global Token 有效”。

### 9.2 31 个固定均衡本地 token

v5 使用：

- common：10 个 token，每组 38/39 个字段；
- item：20 个 token，每组 41/42 个字段；
- creative：1 个 token，14 个字段。

分组使用固定版本号、固定列表、组大小校验和 SHA-256 checksum，工程可复现性比运行时随机分组更好。

但这些组名为 `common_v5_00`、`item_v5_00` 等，不再表达业务语义。它取得的是“字段完整 + 宽度均衡 + 更多 token”，代价是 v3 的可解释语义聚类被弱化。

### 9.3 v5 显著缓解 token 投影压缩

v1–v4 的 token 总坐标为：

```text
16 × 768 = 12,288，仅为输入 20,978 维的 58.6%
```

v3 最大语义组为 136 个字段：

```text
136 × 17 = 2,312 → 768
```

v5 最大本地组为 42 个字段：

```text
42 × 17 = 714 → 1,024
```

31 个本地 token 共 31,744 个坐标，已经是输入宽度的 151.3%。这不等价于数学上的无损编码，但从维度关系看，它降低了单 token 投影的压缩压力。由于 token 数、维度和其他模块同时改变，现有训练曲线不能确认这一点贡献了多少 AUC。

### 9.4 Global Token

v5 将 SENet 后的完整三桶向量再编码为：

```text
20,978 → 1,024 → 1,024 → RMSNorm
```

Global Token 为模型提供一条不依赖本地分组质量的全局路径。即使某些均衡 token 语义混杂，最终读出仍能访问全局表示。

### 9.5 Mixing/Reverting + 双 SwiGLU

每个 v5 block 为：

```text
m = Mix(x)
m' = m + SwiGLU(RMSNorm(m))
r = Revert(m')
x' = x + SwiGLU(RMSNorm(r))
```

与 v1–v4 相比，v5 的主要变化是：

- 在 mixed space 先做一次 token-specific 非线性；
- 精确 Revert 回原 token 空间；
- 再做一次 token-specific 非线性；
- 使用 PreNorm 与长 identity residual；
- down projection 使用较小初始化尺度 `0.01`，降低冷启动时破坏主路径的风险。

每个 token 的 SwiGLU 参数独立，`T=32、D=1024、M=704、L=2` 使该部分达到 277.266M 参数，是 v5 成本的绝对主体。

### 9.6 三路读出

v5 最终上下文由三部分拼接：

1. 最终 Global Token：1,024 维；
2. Global-conditioned local pooling：1,024 维；
3. Flatten local-token readout：512 维，标量 gate 初始约 0.119。

最终 context 为 2,560 维，再进入 Base 风格 `[2048,2048,256]` 任务头。相比 v1–v4 的“768 维 pooled context → 1 个线性输出”，v5 显著增强了读出和任务映射能力。

### 9.7 当前训练信号为什么不能按模块归因

v5 相对 v3 同时发生至少五类变化：

- token 从 16 增加到 31，且 `D` 从 768 增加到 1024；
- 业务语义分组改为固定均衡分组；
- 增加 Global Token；
- GELU PFFN 改为双空间 SwiGLU，并切换为 RMSNorm/长残差；
- 单路池化线性输出改为三路读出和深任务头。

因此，约 140,000 步时出现的训练 AUC 改善只能记在“v5 整体结构”名下。277M SwiGLU、Global Token、Flatten readout、深任务头等每个子模块的测试有效性均为未证实。

---

## 10. 六种架构统一对比

| 维度 | Base | v1 | v2 | v3 | v4 | v5 |
|---|---|---|---|---|---|---|
| 字段完整性 | 完整 | **被切断** | 完整 | 完整 | 完整 | 完整 |
| token 分组 | 无 token | 裸维度等分 | 桶内顺序均分 | 业务语义分组 | 同 v3 | 固定均衡分组 |
| Token 数 | — | 16 | 16 | 16 | 16 | 31 local + 1 global |
| Hidden dim | — | 768 | 768 | 768 | 768 | 1024 |
| SENet | 开 | 未接入 | 开 | 开 | 开 | 开 |
| 显式交互 | 全维 2×DCN-M | 无 | 桶均值乘积 | 桶均值乘积 | 桶均值乘积 + QI | Global 条件选择，无 DCN-M |
| Mixer FFN | — | GELU, k=4 | GELU, k=2 | GELU, k=2 | GELU, k=2 | 双空间 SwiGLU, M=704 |
| 归一化 | BN + LN | Post-LN，3 次/层 | Post-LN，2 次/层 | 同 v2 | 同 v3 | Pre-RMSNorm + final RMSNorm |
| 池化/读出 | 保持全维后进 MLP | Mean | Gated | Gated | Gated + QI residual | Global-conditioned + Flatten |
| 任务头 | 2048→2048→256 | Linear | Linear | Linear | Linear | 2048→2048→256 |
| dense 参数 | 90.34M | 167.29M | 95.81M | 95.81M | 96.44M | 348.43M |
| 首日测试 AUC | **0.864538** | 0.862033 | 0.862690 | **0.862991** | 0.862709 | 未完成 |

代码中的结构变化顺序可以概括为：

```text
v1：直接替换 Base 主塔，但破坏字段边界
 ↓
v2：修复字段边界、恢复 SENet、控制参数并加粗粒度 Cross
 ↓
v3：将机械分组升级为业务语义分组
 ↓
v4：增加定向 Query→Item 低秩 residual，但当前无净增益

v5：大容量整体重构
    更多均衡 token + Global Token + 双空间 SwiGLU + 多路读出 + 深任务头
```

---

## 11. 当前算法模块有效性判定

### 11.1 判定标尺

为避免把结构解释写成因果结论，本节采用以下判定：

| 判定 | 含义 |
|---|---|
| **当前直接正向** | 相邻版本的核心结构差异较单一，且当前测试 AUC 上升；只对本次运行成立 |
| **当前直接负向** | 相邻版本增加具体模块后，当前测试 AUC 下降；只否定当前实现和训练条件 |
| **组合有效** | 整套结构相对对照更好，但同时变化多项，无法分配单模块贡献 |
| **结构性不适配** | 代码可直接证明模块破坏输入结构或违背其自身建模前提；AUC 损失仍不可定量拆分 |
| **仅训练拟合有效** | 只有训练阶段指标改善，没有完整独立测试结果 |
| **未证实** | 无单模块对照，现有结果既不能证明有效，也不能证明无效 |

### 11.2 模块级证据总表

| 算法模块 | 所在版本 | 可用对照或现象 | 当前判定 | 结论边界 |
|---|---|---|---|---|
| 三桶 BatchNorm | Base、v1–v5 | 所有模型共同使用，无关闭对照 | 未证实 | 无法从版本 AUC 中识别独立贡献 |
| Base 整体主塔：分层 SENet + 2×DCN-M + 深 MLP | Base | 首日高于最佳已完成 RankMixer 15.47 bp；连续四日高于 v1 25–28 bp | **组合有效** | 能证明整套结构适配当前任务，不能区分三个模块各自贡献 |
| 分层 SENet | Base、v2–v5 | v2 相比 v1 恢复 SENet，但同时还有八类变化 | 未证实 | 机制与样本级字段选择匹配，独立 AUC 收益未知 |
| 全维两层 DCN-M | Base | 仅 Base 使用，且 Base 同时具有不同 token/读出/任务头 | 未证实（组合相关） | Base 领先与其共现，但不能据此单独认定 DCN-M 有效 |
| `[2048,2048,256]` 深任务头 | Base、v5 | Base 有测试优势；v5 只有中途训练信号 | 未证实（组合相关） | 深头的独立测试贡献未知 |
| v1 裸维度 16-token 切分 | v1 | `1311 mod 17 = 2`，全部内部切点破坏字段边界，并出现跨桶 token；v1 整体持续落后 | **结构性不适配** | 可确认表示构造有缺陷，不能把全部 25–28 bp 差距归因于它 |
| 固定无参数 Mix | v1–v5 | 已完成 RankMixer 均使用，缺少同主塔的关闭对照 | 未证实 | “所有 RankMixer 低于 Base”不是对 Mix 的单模块反证 |
| v1 `k=4` 独立 PFFN | v1 | 参数约 151M，v1 落后；v2 同时把 `k` 改为 2 并修改多项结构 | 未证实 | 参数大与冷启动压力是代码层机制，AUC 影响不可分离 |
| 字段安全、桶内完整 tokenization | v2 | v2 比 v1 高 6.57 bp，但同时恢复 SENet、修改主干和读出等 | **组合有效的一部分** | 字段结构被确定性修复；其独立增益不能从 6.57 bp 中拆出 |
| 两次 Add&Norm | v2–v4 | 与 v1 的三次 LN 对照伴随多项变化 | 未证实 | 无独立 AUC 证据 |
| `k=2` batched PFFN | v2–v4 | 与字段切分、SENet、池化等同步变化 | 未证实 | 只能确认参数量下降，不能确认效果贡献 |
| Gated Pooling | v2–v4 | 替换 v1 Mean Pool 时同时发生多项变化 | 未证实 | 零初始化使起点等价于均值是代码事实，后续选择能力是否增益未知 |
| Bucket Cross | v2–v4 | 三桶摘要乘积与多项修复同步加入；v2–v4 仍低于 Base | 未证实 | 不能说它有效，也不能用未追平 Base 证明它无效；其粒度确定比全维 DCN-M 粗 |
| 学习率 milestone reset | Base、v2–v5 | v1 原版缺失，`v1_lrfix` 没有独立 AUC | 未证实 | 属于训练生命周期差异，效果量未知 |
| 固定业务语义 token 分组 | v3、v4 | v3 相对 v2 参数和主计算量不变，首日 +3.01 bp | **当前直接正向** | 当前场景单次运行有效；无固定 seed/重复运行，稳定性未知 |
| QICross-Lite | v4 | v4 相对 v3 仅增加该主要结构，首日 -2.82 bp | **当前直接负向** | 只说明当前低秩、末端 residual、零输出初始化实现无净收益，不否定 Query–Item 交互类别 |
| 31 个均衡本地 token + `D=1024` | v5 | 与全部 v5 重构同步；仅有中途训练 AUC | 未证实 | 维度压缩减弱是结构事实，独立测试收益未知 |
| Global Token | v5 | 无关闭对照 | 未证实 | 能提供全局旁路是表达能力事实，不等于 AUC 已受益 |
| Mix/Revert + 双空间 SwiGLU + RMSNorm | v5 | 无模块对照；占 v5 dense 参数约 79.58% | 未证实 | 计算和容量占比明确，效果与成本是否匹配未知 |
| Global-conditioned pooling + Flatten readout | v5 | 与深头等同步加入 | 未证实 | 保留局部 token 身份的机制明确，独立贡献未知 |
| v5 整体结构 | v5 | 约 140k 步训练 AUC 超过 v3、接近 Base 训练阶段 | **仅训练拟合有效** | 无完整训练和独立测试 AUC，不能判定泛化有效 |

### 11.3 当前能够认定有效的部分

#### 11.3.1 Base 整体结构：测试效果有效，内部归因不可分

Base 首日 AUC 为 0.864538，高于 v1–v4；在与 v1 对齐的四天里，Base 每天均领先 25–28 bp。这个结果不是单日偶然排序，而是当前唯一具有多日持续优势的结构证据。

代码上，Base 同时覆盖三种能力：SENet 做样本级字段重标定，DCN-M 在完整 20,978 维空间做显式乘性交互，深 MLP 做最终任务映射。三者与当前“字段极多、item 占比高、common 与 item 需要条件交互”的场景在机制上相容。测试结果证明的是这套能力组合整体有效，而不是证明其中任何一个模块单独贡献了全部或固定比例的 AUC。

#### 11.3.2 v2 整套修复：相对 v1 有效，模块贡献不可拆

v2 首日比 v1 高 6.57 bp，说明从 v1 到 v2 的整套变化产生正向净效果。代码同时发生至少九项改变，所以“字段安全切分有效”“SENet 有效”“Bucket Cross 有效”“gated pooling 有效”等单独说法都超出现有证据。能够确认的只有两点：v1 的字段破坏被修复；v2 组合结果优于 v1，但仍低 Base 18.48 bp。

#### 11.3.3 v3 业务语义分组：当前最明确的单模块正向证据

v2 与 v3 保持 `T=H=16、D=768、L=2、k=2`、SENet、Mixer、Bucket Cross、gated pooling 和输出头不变，dense 参数都为 95.809M；主要变化是字段由桶内顺序均分改为 16 个固定业务语义组。v3 比 v2 高 3.01 bp，因此在当前数据日期和本次冷启动中，语义分组有效。

这个判定不能扩展为“稳定提升 3.01 bp”或“16 组是最优划分”。仓库没有显式 seed，且只有一次结果，3.01 bp 仍可能混有初始化和分布采样波动。

### 11.4 当前能够认定无效或不适配的部分

#### 11.4.1 v1 裸维度切分：结构性不适配当前字段输入

当前输入的基本单位是 17 维字段 embedding，而 v1 将 20,978 维向量切为 `[1311]×15+[1313]`。由于 `1311 mod 17 = 2`，每个内部边界都会截断字段，第 5 个 token 跨 common/item，最后一个 token 跨 item/creative。独立 token 投影和 token-specific PFFN 随后会用不同参数处理同一字段的两段坐标。

这是不依赖 AUC 的确定性代码缺陷：token 不再代表完整、稳定的业务子空间，因而不满足当前 RankMixer 表示的基本前提。v1 连续四日落后 Base 与这一缺陷方向一致，但差距还混有移除 SENet/DCN-M/深头、`k=4` 参数膨胀和学习率 reset 缺失，不能把 26.75 bp 平均差全部记到切分模块。

#### 11.4.2 v4 QICross-Lite：当前实现没有测试净收益

v4 继承 v3 的语义 token、Mixer、Bucket Cross、池化和线性头，只增加约 0.630M 的 QICross-Lite，结果相对 v3 下降 2.82 bp，并与 v2 只差 +0.19 bp。因而在当前首日 dense 冷启动条件下，这个具体模块应判为直接负向。

代码揭示了几种与负向结果一致、但尚未被证明的机制：目标 item token 已含人工 Query–Item 相关字段，新增分支可能重复；85 字段 query token 压到 128 维可能过粗；residual 只在最终 context 接入；零初始化 output projection 使上游低秩层早期梯度滞后。这些解释不能上升为“显式 Query–Item 交互无效”。

#### 11.4.3 已完成 RankMixer 的整体替代效果：尚未达到 Base

v1、v2、v3、v4 的首日 AUC 均低于 Base，差距分别为 25.05、18.48、15.47、18.29 bp。因此，任何一个已完成版本作为完整主塔，在当前测试口径下都没有证明与 Base 等效。这个结论针对完整版本，不构成对固定 Mix、PFFN、SENet 或池化中某个单模块的否定。

### 11.5 当前不能判定有效性的部分

#### 11.5.1 Base 内部三个核心模块不能互相代替归因

Base 的 SENet、DCN-M 和深 MLP 从未在保持其他条件不变时分别关闭。Base 领先只能说明三者组合有效，不能确定领先主要来自字段重标定、全维交叉还是深任务映射。v2–v4 虽然也有 SENet，但它们同时改变了输入压缩、交互粒度和任务头，因此不能充当 SENet 的单变量反例。

#### 11.5.2 v2 的七类核心修改都没有独立效果量

字段安全切分、SENet、`k=2`、两次 Add&Norm、gated pooling、Bucket Cross 和 LR reset 同时出现。v2 的 +6.57 bp 是它们与其他工程变化的合计净值；其中可能同时包含正贡献、零贡献和负贡献，当前总差值无法分解。

#### 11.5.3 v1–v4 共同的固定 Mixer/PFFN 没有独立对照

已完成的四个 RankMixer 都依赖固定 reshape/transpose mixing 与 token-specific PFFN。它们都低于 Base，只能说明完整 RankMixer 主塔尚未追平，不能说明无参数 mixing 本身无效，因为输入 tokenization、显式交互和输出头均与 Base 不同。

#### 11.5.4 v5 的所有子模块都只有联合训练信号

v5 的中间训练 AUC 证明整个 348.432M dense 结构在约 140,000 步时拟合更强，但 31-token、`D=1024`、Global Token、双 SwiGLU、RMSNorm、三路读出和深头是同时启用的。训练尚未完成，也没有独立测试结果；所以这些子模块目前全部属于“测试有效性未证实”，v5 整体也只能属于“训练拟合有效”。

### 11.6 版本级最终判定

| 版本 | 当前场景判定 | 判定依据 |
|---|---|---|
| Base | **整体测试有效，已完成测试中当前最优** | 首日最高；对 v1 有连续四日稳定优势 |
| v1 | **完整结构无效；tokenization 结构性不适配** | 首日及四日均显著落后；代码确定切断字段并跨桶 |
| v2 | **相对 v1 组合有效，但未达到 Base** | +6.57 bp；同时变化过多，单模块不可归因 |
| v3 | **当前最佳 RankMixer；语义分组单次有效** | 相对 v2 +3.01 bp，参数和主要结构不变；仍低 Base 15.47 bp |
| v4 | **QICross-Lite 当前无效** | 相对 v3 -2.82 bp，且几乎退回 v2 水平 |
| v5 | **训练拟合有效；测试有效性未知** | 约 51.9% 进度的训练 AUC 正向，无完整测试 AUC |

---

## 12. 结论可信度与代码审计边界

当前仓库还存在若干会影响结论可信度的问题。

### 12.1 启动脚本与运行记录存在漂移

- `set-rankmixer-v2/v3/v4.txt` 当前日期文本与首日展开参数/背景记录不完全一致；
- `set-rankmixer-v3-args.txt` 的 checkpoint scheme 写成了 `hhdfs://`；
- v1 启动脚本中的模型入口名称与当前 `cvr_bn_rankmixer_v1.py` 不完全一致；
- 当前配置没有显式固定 TensorFlow 随机 seed。

这些问题不直接推翻 `background.md` 的已记录 AUC，但说明当前仓库不能独立还原每次服务器运行的完整状态。尤其 v3−v2 的 +3.01 bp 和 v4−v3 的 −2.82 bp 都来自单次运行，在没有固定 seed 和配对预测的情况下，应理解为“当前观测方向”，而不是已建立的稳定效果量。

### 12.2 若干配置字段在当前主路径中并未生效

从当前模型文件可见：

- v1–v4 的 `use_rankmixer` 不是可真正关闭主塔的开关；v5 才明确要求其为 true；
- `grad_clip_value` 被读取，但优化器路径没有显式执行 gradient clipping；
- `dropout` 被读取，但主塔没有调用 dropout；
- Base/v1–v4 部分层声明了 `weights_regularizer`，但 `self.loss` 没有显式合并 `REGULARIZATION_LOSSES`；v5 主要矩阵也未声明 L2 regularizer。

因此，梯度裁剪、Dropout 和 L2 不能被计入当前模型的有效模块，也不能用于解释版本 AUC 差异。`use_rankmixer` 在 v1–v4 中同样不能提供主塔开关对照。

### 12.3 AUC 记录的统计边界

模型图使用 `flood_auc(..., num_thresholds=2000)`。本报告将 `background.md` 中的数值视为既定实验记录，但仓库中没有逐样本预测、exact AUC、配对置信区间或重复 seed 结果。因此：

- Base 对 v1 的连续四日 25–28 bp 优势，是当前较强的版本级证据；
- v2 对 v1 的 6.57 bp 是多模块合成差值，统计显著性与模块归因是两个不同问题；
- v3 对 v2 的 +3.01 bp、v4 对 v3 的 −2.82 bp 可用于描述本次直接对照方向，但不能外推为稳定效果量；
- v4 对 v2 的 +0.19 bp 接近零，在现有记录下没有足够证据区分结构差异与运行波动。

### 12.4 各版本对照的归因强度

| 对照 | 同时变化规模 | 归因强度 | 当前可回答的问题 |
|---|---:|---|---|
| Base vs v1 | 很多 | 低 | Base 整套结构显著更好；不能定位单模块 |
| v2 vs v1 | 至少九项 | 低 | v2 整套修改净正向；不能拆出各模块贡献 |
| v3 vs v2 | 一个核心变化 | **当前最高** | 语义分组在本次运行中相对顺序均分正向 |
| v4 vs v3 | 一个主要新增模块 | **当前最高** | QICross-Lite 在本次运行中负向 |
| v5 vs v3/v4 | 多项且训练未完成 | 不具备测试归因条件 | 只能判断 v5 整体中途训练拟合更强 |

---

## 13. 最终结论

基于当前 AUC 现象和代码差异，模块有效性可以收敛为三组结论。

**已有正向证据：**

- Base 的 `分层 SENet + 全维 DCN-M + 深 MLP` 整体组合，在当前搜索精排 fst_CVR 场景下测试效果最强；内部模块贡献尚不可拆分。
- v2 的字段安全、SENet、主干修正、池化、Bucket Cross 和学习率生命周期修复作为整体，相对 v1 净提升 6.57 bp；任何单项都没有独立效果量。
- v3 的固定业务语义分组是当前唯一较清晰的 RankMixer 单模块正向证据，相对相同参数量的 v2 提升 3.01 bp；该结论限于单次运行。

**已有负向或不适配证据：**

- v1 的裸维度 tokenization 确定会切断 17 维字段并跨桶，结构上不适配当前输入；v1 完整结构四日平均比 Base 低 26.75 bp，但损失不能只归因于切分。
- v4 的 QICross-Lite 在当前 dense 冷启动实现中相对 v3 下降 2.82 bp，未产生测试净收益；这个结论不否定其他形式的 Query–Item 交互。
- v1–v4 任何一个已完成的完整 RankMixer 都未追平 Base，说明这些完整结构在当前测试口径下尚未证明等效替代能力。

**当前无法判定：**

- Base 内部 SENet、DCN-M、深任务头各自的独立贡献；
- v2 中 SENet、`k=2`、Add&Norm、gated pooling、Bucket Cross、LR reset 各自的独立贡献；
- 固定无参数 Mix 和 token-specific PFFN 本身的独立有效性；
- v5 的 31-token、`D=1024`、Global Token、双空间 SwiGLU、RMSNorm、三路读出及深任务头的独立有效性。

v5 当前唯一成立的判断是：其 **整套 348.432M dense 结构在约 51.9% 训练进度时表现出更强的训练拟合**。在没有完整独立测试 AUC 前，v5 及其任何子模块都不能被归入“测试有效”。
