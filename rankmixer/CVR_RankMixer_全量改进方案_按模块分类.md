# CVR RankMixer 全量改进方案：按模块分类的架构设计与实验指南

> 本文把当前工程中所有已经提出、能够独立消融的 RankMixer 改进点，按数据流所在模块重新整理。
> 每一类都先说明它属于 RankMixer 的哪个部分、在模型中承担什么职责，再说明为什么改、如何改、收益与风险。
>
> 主对比基线：<code>code/cvr_bn_rankmixer_v1.py</code>。<br>
> 工业参考实现：<code>code/commend_cvr.py</code>、<code>code/mlp_mixer_swiglu_fuse.py</code>。<br>
> 传统 CVR 参考：<code>code/cvr_fst_last_norpy.py</code>。<br>
> 结论边界：文中的“预期收益”是机制假设，不是 AUC 承诺；是否有效必须由同数据、同初始化、同预算的消融实验确认。

---

## 1. 先给结论：应该改什么，先后顺序是什么

当前 v1 的 AUC 约为 0.862，而原 CVR base 约为 0.865。这个差距不能直接归因于
“RankMixer block 不如 MLP/DCNM”，因为 v1 同时改变了输入、tokenizer、主干、读出、监督和初始化条件。

最重要的五个事实是：

1. v1 只使用 common、item、creative 三类稀疏 embedding，删除了 dense、DIN、gattr 等输入；
2. 20978 维长向量按标量等宽切成 16 段，所有内部边界都会切断 17 维字段；
3. 每段只经过一次 Dense + bias + GELU 投影，不是两层 MLP；
4. 两层 hybrid RankMixer 后只做 mean pooling，只训练 first CVR；
5. 约 167.3M dense 参数中的大部分是新增 PFFN 参数，热启动覆盖和收敛公平性都需要单独审计。

因此，正确顺序不是先堆更强的 mixer，而是：

~~~text
P0 正确性与公平性
  -> P1 输入恢复与稳定语义 token
  -> P1 双读出、Creative 侧塔、first/last 多任务
  -> P1 严格比较 block 拓扑与等参数 SwiGLU
  -> P2 显式交叉、深序列、蒸馏
  -> P3 可学习 mixing、表示扩秩、MoE 与系统优化
~~~

近期最推荐候选是：

~~~text
RM-v2-Parity
= 完整 base 输入
+ first/last 双任务
+ 完整字段语义 token
+ mean + low-rank flatten
+ Creative side tower
+ raw-logit BCE
~~~

它先回答一个最关键的问题：在输入、标签和训练条件公平时，RankMixer 到底能不能追回 base。

---

## 2. v1 的真实结构与问题位置

### 2.1 当前执行链

~~~mermaid
flowchart LR
    A["Sparse lookup"] --> U["Common<br/>6545"]
    A --> I["Item<br/>14195"]
    A --> C["Creative<br/>238"]
    U --> BN["分桶 BN"]
    I --> BN
    C --> BN
    BN --> V["拼接为 20978 维"]
    V --> S["标量等宽切分<br/>1311×15 + 1313"]
    S --> P["16 个独立<br/>Dense + bias + GELU"]
    P --> X["X₀: 16×768"]
    X --> R["2 个 hybrid<br/>RankMixer block"]
    R --> M["Mean pooling"]
    M --> H["Linear head"]
    H --> L["first CVR loss"]
~~~

源码依据：

- 三桶收集与 BN：<code>cvr_bn_rankmixer_v1.py:889-930</code>；
- 长向量拼接与等宽切分：<code>cvr_bn_rankmixer_v1.py:938-961</code>；
- 单层投影：<code>cvr_bn_rankmixer_v1.py:774-799</code>；
- token mixing、PFFN 与 hybrid block：<code>cvr_bn_rankmixer_v1.py:801-862</code>；
- mean pooling 与单头输出：<code>cvr_bn_rankmixer_v1.py:864-980</code>；
- first-only 概率式 loss：<code>cvr_bn_rankmixer_v1.py:409-420</code>。

### 2.2 标量切分为什么破坏字段

三桶宽度都来自 17 维字段：

| 域 | 总宽度 | 字段数 |
|---|---:|---:|
| Common | 6545 | 385 |
| Item | 14195 | 835 |
| Creative | 238 | 14 |
| 合计 | 20978 | 1234 |

当前每个普通分段宽 1311，而：

$$
1311 = 77\times17 + 2.
$$

所以第一个边界在某字段内部偏移 2 维，第二个偏移 4 维，以此类推。15 个内部边界没有一个与
17 维字段边界对齐。更具体地：

- 第 5 个 token 同时包含 Common 尾部和 Item 头部；
- 最后一个 token 同时包含 1075 维 Item 和全部 238 维 Creative；
- 同一个 token 位置长期承担的业务语义不稳定，削弱了 per-token 独立参数的意义。

### 2.3 “Dense + GELU 压成一个 token”到底是什么

对第 $t$ 个分段 $z_t\in\mathbb R^{d_t}$，源码执行：

$$
x_t=\mathrm{GELU}(z_tW_t+b_t),
\qquad
W_t\in\mathbb R^{d_t\times D},
\quad b_t\in\mathbb R^D.
$$

数据流是：

~~~text
[B, d_t]
  -> 一次矩阵乘法 W_t
  -> 加 bias
  -> GELU
  -> [B, D]
~~~

它是“一层带非线性的投影”，不是通常意义上的两层 MLP。它没有隐藏瓶颈、第二个 Dense 或输出投影。
<code>commend_cvr.py::embedding_to_tokens()</code> 虽然注释称为 MLP，本质上也只有一次
<code>tf.layers.dense</code>，然后 reshape。成熟实现的优势首先来自稳定的业务语义分组，而不是层数更多。

### 2.4 当前参数主要花在哪里

在 $T=16,D=768,k=4,L=2$ 时：

- tokenizer 权重约为 $20978\times768=16.11$M；
- 单个 GELU PFFN、单 token、单层约有 $2kD^2=4.72$M 权重；
- 16 token、2 层 PFFN 约有 151.0M 权重；
- 加上 bias、LN 和 head 后，dense 参数约 167.3M。

所以“增加模型参数”并不是当前第一优先级。更重要的是让这 151M PFFN 参数接收到稳定、完整、可训练的
token 表示。

---

## 3. RankMixer 可以修改的模块全景

### 3.1 原版 RankMixer 的模块边界

一个完整 RankMixer 可以拆成七个工程模块：

| 类别 | 在 RankMixer 中的位置 | 核心职责 | 本文方案编号 |
|---|---|---|---|
| P | 全链路前置条件 | 保证数据、shape、loss、训练和对照正确 | P1-P3 |
| A | Embedding 到 tokenizer 之前 | 选择输入、归一化、门控和小域特征处理 | A1-A4 |
| B | Tokenizer 与 token 表示 | 把异构字段变成稳定的 $T\times D$ token | B1-B6 |
| C | Token Mixing 与 residual | 在 token 间交换信息并定义残差坐标 | C1-C5 |
| D | Per-token FFN 与容量 | 每个 token 独立做通道非线性变换 | D1-D6 |
| E | 显式交叉与行为序列 | 补充固定 mixer 不擅长的乘性交叉和序列交互 | E1-E5 |
| F | Readout、融合与任务头 | 从 token 中读出全局/局部信息并构造监督 | F1-F6 |
| G | 初始化、优化与系统 | 决定大模型能否公平收敛和在线服务 | G1-G6 |

其中 A、E、F、G 是业务 CVR 在原版 RankMixer 外围的必要工程扩展；B、C、D 是 RankMixer 核心骨干。

### 3.2 总体改造图

~~~mermaid
flowchart TD
    X["完整 CVR 特征"] --> A["A 输入与预处理<br/>分域归一化 / Gate / 小域隔离"]
    A --> B["B Tokenizer<br/>字段安全分桶 / 表示增强"]
    B --> C["C Token Mixing<br/>固定 / Reverting / 可学习"]
    C --> D["D Per-token FFN<br/>GELU / SwiGLU / MoE"]
    D --> R["F Readout<br/>Mean / Group / Low-rank Flatten"]

    X --> E["E 显式交叉与序列<br/>DCNM / DIN / MixFormer"]
    E --> C
    E --> R

    R --> F["F 多任务 Heads<br/>first / last / 辅助任务"]
    F --> G["G 训练与系统<br/>Warm-start / KD / Kernel / Serving"]

    P["P 正确性与公平性"] -.约束.-> A
    P -.约束.-> B
    P -.约束.-> C
    P -.约束.-> D
    P -.约束.-> F
    P -.约束.-> G
~~~

### 3.3 与旧 M0-M14 方案的完整对应

| 旧编号 | 本文归类 | 是否保留 |
|---|---|---|
| M0 公平对齐 | P1-P3、G1、G4 | 完整保留并细分 |
| M1 稳定语义 token | B1-B3 | 完整保留并扩写投影选择 |
| M2 条件门控 | A3 | 完整保留 |
| M3 Creative/Coupon | A4、F2 | 完整保留 |
| M4 双读出 | F1-F2 | 完整保留 |
| M5 residual + SwiGLU | C2-C4、D1-D3 | 完整保留并严格拆开 |
| M6 DCNM | E1 | 完整保留 |
| M7 DIN/MixFormer | E2-E5 | 完整保留并补充 UI 解耦 |
| M8 多任务 | F3-F6 | 完整保留 |
| M9 热启/蒸馏/LR | G1-G3 | 完整保留 |
| M10 Soft-to-Hard | B4 | 完整保留 |
| M11 RankUp | B5-B6、F4 | 五个组件逐项保留 |
| M12 UniMixer-Lite | C4、D3 | 补充 SiameseNorm 与温度策略 |
| M13 RankElastor | C5、D3 | 完整保留 |
| M14 参数效率/MoE | D4-D6、G5 | 拆分 dense 压缩、两类 MoE 与系统优化 |

---

## 4. P 类：正确性与公平性前置模块

### 4.1 它属于 RankMixer 的哪个部分

P 类不改变某一层的数学表达式，而是约束整个 RankMixer 实验。它位于数据读取、建图、训练和评估全链路。
如果这一类没有完成，后续任何 AUC 变化都无法可靠归因。

### P1：输入、标签和样本口径完全对齐

**修改前。** base 使用 common、item、creative、dense、DIN、gattr，并训练 first/last；
v1 只使用三类稀疏输入和 first 标签。

**为什么改。** 0.865 与 0.862 不是同输入、同监督的 backbone 对照。

**如何改。**

1. 固定相同训练/测试日期、采样、过滤、label delay、正例率和数据量；
2. 先逐项恢复 dense、DIN、gattr，再恢复 last loss；
3. 每次只增加一个输入或一个监督，不同时更换 block；
4. 报告相同样本数、相同 FLOPs 和相同 wall-clock 三种口径。

**好处。** 能量化每个被删除组件造成的差距，并建立可解释的 RankMixer baseline。

**风险。** 一次恢复所有组件后即使 AUC 上升，也无法知道收益来自哪个模块。

### P2：shape、字段覆盖和 token contract 断言

**修改前。** v1 只断言 $D\%H=0$，没有显式断言 $T=H$、字段覆盖和配置长度一致。

**为什么改。** 当前 mixing 的最后一次 reshape 只有在 $T=H$ 时才保持论文中的 token/head 语义；
Python 的 <code>zip()</code> 还可能静默截断字段配置。

**如何改。**

~~~python
assert T == H
assert D % H == 0
assert len(tokens) == T
assert len(field_groups) == len(token_configs)
assert sorted(flatten(field_groups)) == sorted(all_fields)
assert len(flatten(field_groups)) == len(set(flatten(field_groups)))
~~~

同时给 token schema 增加版本号和哈希，schema 变化时禁止静默加载旧 tokenizer/PFFN checkpoint。

**好处。** 把静默结构错误变成建图时失败。

**风险。** 只检查张量 shape，不检查字段重复和遗漏，仍可能得到“能训练但语义错误”的图。

### P3：训练、导出、fused 与 checkpoint 一致性

**修改前。** <code>mlp_mixer_swiglu_fuse.py</code> 在 train 使用 fused 路径，在非 train 使用
optimized 路径；变量命名、初始化和 export 数值必须额外证明一致。

**为什么改。** forward 接近不代表输入梯度、权重梯度和 restore 一致。

**如何改。**

- 固定随机输入和相同权重，比较 fused/unfused forward；
- 比较对输入、gate/up/down kernel 的梯度；
- 保存 unfused checkpoint 后由 fused 图恢复，反向再测；
- 覆盖 train、test、export 三种 mode；
- 使用相对误差与绝对误差双阈值，不只用肉眼看日志。

**好处。** 避免训练有效、导出漂移，或 fused 路径未真正热启。

**风险。** zero-init down matrix 时，第一步 gate/up 梯度为零可能是结构预期，不能误判为永久断梯度。

---

## 5. A 类：输入选择、归一化与条件门控

### 5.1 它属于 RankMixer 的哪个部分

A 类位于 sparse/dense lookup 之后、tokenizer 之前。它决定 RankMixer 能看到哪些原始信息，以及进入 token
投影前各业务域的数值尺度和样本自适应权重。原版论文通常把输入视为已经准备好的长向量，但 CVR 工程不能忽略这一步。

### A1：恢复 dense 与 gattr

**修改前。** v1 的日志会打印 dense/seq/gattr 配置规模，但模型路径只收集 common/item/creative。

**为什么改。** 数值统计、价格、频次和全局属性可能是 CVR 的强信号；删除它们会让 RankMixer 在更弱输入上比较。

**如何改。**

- 完全复用 base 的 clip、减均值、除尺度和 <code>dense_scale</code>；
- gattr 只纳入配置中标记 <code>dnn_input=True</code> 的字段；
- 首轮走独立 context side branch，再比较转成 1-2 个 Context Token；
- 对空特征列表显式返回零宽或零向量，禁止空 <code>concat</code>。

**好处。** 恢复传统 MLP/DCNM 已验证的连续和全局信息，且 side branch 不破坏 token contract。

**风险。** dense 预处理与 base 不一致会制造分布差异；既走 side branch 又走 token 会重复计入。

### A2：分域归一化，而不是全局统一 BN

**修改前。** v1 已在拼接前对 common/item/creative 分别做 BN，这是正确方向，但没有 dense、DIN、gattr，
也没有字段/域分布监控。

**为什么改。** 用户、商品、创意、数值和序列表示的分布不同，先全拼接再统一归一化会让大域统计压制小域。

**如何改。**

1. 保留每域独立 BN；
2. dense 继续使用业务统计归一化，再决定是否额外 BN；
3. 序列聚合输出独立 LN/BN；
4. tokenizer 内优先比较无投影 LN 与投影后 LN，不能同时改；
5. 导出时验证 moving statistics 与训练图一致。

**好处。** 降低域间尺度冲突，特别保护 Creative、Coupon 和 DIN 小向量。

**风险。** 小流量域 BN 统计不稳；随意切换 BN/LN 会破坏 checkpoint 兼容。

### A3：identity-init User self gate 与 User-conditioned Item gate

**修改前。** v1 对所有样本使用相同静态投影；成熟实现先对 User 自门控，再用 User+Item 条件生成 Item gate。

**为什么改。** 同一商品字段对不同用户的价值不同，静态 tokenizer 不能在样本级抑制无关维度。

**如何改。**

$$
g_U=2\sigma(f_U(U)),\qquad
g_I=2\sigma(f_I([U,I])),
$$

$$
\widetilde U=U\odot g_U,\qquad
\widetilde I=I\odot g_I.
$$

最后一层 kernel 和 bias 使用零初始化，使初始 gate 恰好为 1。门控后再按稳定语义组切分。

**好处。** 从恒等模型开始训练，同时增加样本相关的字段选择能力。

**风险。**

- 普通 sigmoid 零初始化会得到 0.5，而不是恒等；
- gate 饱和后梯度变弱；
- gate 是动态幅度，不应同时让字段的 token 归属动态漂移。

**消融。** 无 gate → User gate → User+Item gate，并监控 gate 均值、分位数、接近 0/2 的比例与梯度。

### A4：Creative/Coupon 小域隔离与提前交互

**修改前。** 238 维 Creative 被放进最后一个 token，与 1075 维 Item 混合；v1 没有 Coupon 路径。

**为什么改。** 小而强的候选域经过长切片、PFFN 和 mean pooling 后容易被大域稀释。

**如何改。** 按成本从低到高比较：

1. Creative/Coupon → 分域 BN → 小型 side tower → final fusion；
2. Creative 生成 Item/token gate；
3. Creative 作为一个完整正式 token；
4. 每层后用轻量 FiLM 影响主干；
5. 只有 side tower 稳定增益后才比较同时 token 化。

**好处。** 保留候选局部强信号，在线成本可控，也方便独立回滚。

**风险。** 只在最后拼接可能交互不足；同时启用 side、token、FiLM 会无法归因。

### 5.2 A 类推荐数据流

~~~mermaid
flowchart LR
    U0["User fields"] --> UBN["User BN"]
    I0["Item fields"] --> IBN["Item BN"]
    C0["Creative/Coupon"] --> CBN["小域 BN"]
    D0["Dense/Gattr"] --> DN["业务归一化"]

    UBN --> UG["Identity User gate"]
    UG --> IG["User-conditioned<br/>Item gate"]
    IBN --> IG

    UG --> TOK["语义 Tokenizer"]
    IG --> TOK
    CBN --> SIDE["Side tower / FiLM"]
    DN --> CTX["Context side / Token"]
~~~

---

## 6. B 类：Tokenizer 与 token 表示模块

### 6.1 它属于 RankMixer 的哪个部分

Tokenizer 是 RankMixer 的入口。它把大量异构字段压成固定数量 $T$、统一宽度 $D$ 的 token。
后续 PFFN 的参数按 token 位置独立，因此 tokenizer 的首要目标不是“均匀切宽度”，而是让位置语义稳定。

### B1：完整字段、稳定业务语义硬分桶

**修改前。** v1 在 flatten 后按标量切 16 段，破坏字段边界并跨域。

**为什么改。** per-token 参数隔离只有在第 $t$ 个 token 长期表示相近语义时才有价值。

**如何改。**

- 只沿字段列表分组，17 维字段不可拆；
- 优先按用户画像、长期兴趣、实时意图、商品属性、价格、店铺、供给、创意等业务语义分桶；
- 每个字段恰好出现一次；
- 分桶配置独立成版本化文件，并将 schema hash 写入 checkpoint 元信息；
- 首个低风险版本可用 5 User + 10 Item + 1 Creative；
- 若 Creative 走侧塔，则可用 5 User + 11 Item；
- 成熟 32-token 结构可参考 11 User + 18 Item + 2 Sequence + 1 DIN。

**好处。** token 可解释，PFFN 能专门化，字段迭代和 checkpoint 行为更可控。

**风险。** 业务组过细导致单 token 输入太窄；过粗又恢复异构混合。

### B2：投影层选择——单层投影、两层 MLP 还是共享低秩

**修改前。** 每桶是一层 Dense + GELU。

**为什么改。** 一层投影可能不足以融合特别宽的语义桶，但直接改成两层 MLP会同时改变 tokenizer 容量，
使“分桶收益”和“增参收益”混杂。

**如何改。** 分三阶段：

| 版本 | 结构 | 用途 |
|---|---|---|
| B2-a | Dense$(d_t,D)$ + GELU | 首个字段安全基线，最容易归因 |
| B2-b | Dense$(d_t,r)$ + GELU + Dense$(r,D)$ | 超宽桶需要额外融合时使用 |
| B2-c | shared low-rank base + token adapter | 参数或服务压力较大时使用 |

首轮必须保持 B2-a，只改字段分桶。B2-b 的 hidden $r$ 单独扫参，并匹配总参数。B2-c 需保留少量
token-specific adapter，否则会丢失 RankMixer 的异质参数隔离。

**好处。** 可以在表达力、参数量和 token 专门化之间做可控选择。

**风险。** 把一层 Dense 口头称为 MLP 容易误判结构；两层投影过强可能让 mixer 只做很少工作。

### B3：Token budget 与 shape 合同

**修改前。** v1 固定 $T=H=16,D=768$；成熟实现固定 $T=H=32,D=512$。

**为什么改。** 添加 Context、Global、Sequence 或 Task Token 会改变 $T$，而 token mixing 对 $T/H/D$ 有硬约束。

**如何改。** 每次新增 token 必须二选一：

1. 在固定 $T$ 内重新分配现有语义组；
2. 同时调整 $T,H,D$，保持 $T=H$ 且 $D\%H=0$。

参数预算还应约束：

$$
P_{\mathrm{GELU\ PFFN}}\approx 2LTkD^2.
$$

所以从 16×768 改到 32×512 时，不能只比较 token 数，还要比较 $LTkD^2$、吞吐和显存。

**好处。** 防止 append token 后 reshape 仍“能跑”但语义错误。

**风险。** 只匹配参数量不匹配 kernel 效率；$T$ 增大还会降低每个 token 的字段宽度。

### B4：Soft-to-Hard 可学习分桶

**修改前。** B1 使用人工固定字段组。

**为什么改。** 字段完整不等于语义组合最优，可学习全局 assignment 可能发现跨域的有效组合。

**如何改。**

~~~mermaid
flowchart LR
    F["固定 Field identity"] --> A["全局 Soft assignment"]
    A --> R["负载均衡 + 熵正则"]
    R --> S["稳定性审计"]
    S --> H["Capacity-aware hardening"]
    H --> Z["冻结映射"]
    Z --> T["重新训练 / 微调"]
~~~

assignment 必须是全局模型参数，不是每样本动态改变 token 身份；同时约束每字段覆盖、token load 和熵。

**好处。** 有机会学习人工分桶未发现的组合，hard 化后仍保持高效固定 token。

**风险。** 字段塌缩到少数 token；soft 到 hard AUC 大跌；服务 schema 未版本化。

### B5：RankUp 式 Randomized Permutation Splitting 与 Multi-Embedding

**修改前。** v1 只有单份 embedding 和单一机械分片，token 之间可能高度相关。

**为什么改。** 增加 PFFN 参数不一定提高输入表示的有效秩；需要给主干更丰富、相关性更低的原料。

**如何改。**

- 以完整字段为单位，用固定 seed 生成 2-3 套 permutation；
- 每套映射在整个训练和服务生命周期保持不变；
- Multi-Embedding 只先用于关键 User/Item 字段，避免全量翻倍；
- 记录 token correlation、effective rank、梯度与 AUC。

**好处。** 增加多视角表示，可能减少 token 冗余。

**风险。** 每 step 重新随机会破坏 token 身份；对标量维随机会再次切字段；多 embedding 参数可能失控。

### B6：RankUp 式 Global、Cross Pre-trained 与 Task Token

**修改前。** v1 没有显式全局摘要、预训练交叉或任务专属 token。

**为什么改。**

- Global Token 给局部 token 一个全局条件；
- Cross Token 显式注入预训练 User/Item 匹配；
- Task Token 让不同任务从共享主干中读取不同表示。

**如何改。**

1. Global Token 优先由低秩全局分支生成，不使用完全自由常量代替业务上下文；
2. Cross Token 可用 $\mathrm{Proj}(e_u^{pre}\odot e_i^{pre})$；
3. Task Token 优先只用于 readout，稳定后再参与每层 mixing；
4. 三者分别实验，不一次 append；
5. 每次重新核算固定 token budget。

**好处。** 扩充全局、交叉和任务特异性表示。

**风险。** 无可靠预训练向量时 Cross Token 只是额外噪声；Task Token 参与主干可能污染共享表示。

---

## 7. C 类：Token Mixing 与 residual 拓扑

### 7.1 它属于 RankMixer 的哪个部分

Token Mixing 是 RankMixer 的跨 token 信息交换层。原版不是 self-attention：没有 Q/K/V、softmax 或点积，
而是把 $[B,T,H,D/H]$ 的 token/head 轴做固定置换。这个模块决定“谁能看到谁”，residual 则决定相加的坐标语义。

### C1：保留固定 mixing，但补齐严格 shape 语义

**修改前。** v1 实现的 reshape-transpose-reshape 在 $H=T=16$ 时成立，但只断言 $D\%H=0$。

**为什么改。** 如果配置把 $H$ 改成非 $T$，最终 shape 仍可能成立，但输出 token 轴不再对应论文语义。

**如何改。**

~~~python
def fixed_token_mix(x, T, H, D):
    assert T == H
    assert D % H == 0
    h = D // H
    x = reshape(x, [B, T, H, h])
    x = transpose(x, [0, 2, 1, 3])
    return reshape(x, [B, T, D])
~~~

**好处。** 保留零参数、低成本和强基线，同时消除配置静默错误。

**风险。** 固定置换对所有层、样本和任务相同，上限可能受限，但这应在 P/A/B 稳定后再验证。

### C2：从 v1 hybrid 恢复严格原版 RankMixer

**修改前。** v1 每层有 token-mix Post-LN、PFFN Pre-LN、PFFN Post-LN，共三次 LN：

$$
S=\mathrm{LN}(P(X)+X),
$$

$$
Y=\mathrm{LN}\left(S+\mathrm{PFFN}(\mathrm{LN}(S))\right).
$$

**为什么改。** 它既不是严格论文 block，也不是成熟公司 block；直接比较会混入额外 LN 的影响。

**如何改。** 增加独立 <code>rankmixer_strict_v1</code> scope：

$$
S=\mathrm{LN}(P(X)+X),\qquad
Y=\mathrm{LN}(S+\mathrm{PFFN}(S)).
$$

输入、tokenizer、PFFN、参数量和 readout 全部固定，只替换 LN 拓扑。

**好处。** 获得可归因的原版论文基线。

**风险。** 直接覆盖旧 scope 会错误复用形状相同但语义不同的 LN 参数。

### C3：公司 aligned residual

**修改前。** v1 把固定置换后的 $P(X)$ 与原坐标 $X$ 直接相加。

**为什么改。** shape 相同不代表 token 语义相同。成熟实现把 residual 放在同一个 mixed 坐标内：

$$
M=P(X),\qquad
Y=M+\mathrm{SwiGLU}(\mathrm{Norm}(M)).
$$

多层后再做 Final LN。

**如何改。** 每层先 mix，再在 mixed 表示上做 Pre-Norm PFFN residual；不额外加原始 $X$。

**好处。** 残差两支处于同一坐标，结构更简单，也更接近 <code>mlp_mixer_swiglu_fuse.py</code>。

**风险。** 奇数层输出可能处于 mixed 坐标；flatten readout 必须明确最后的坐标语义。

### C4：TokenMixer-Large 的 Mixing & Reverting

**修改前。** strict original 和 aligned block 都依赖固定置换后的坐标解释。

**为什么改。** 显式 inverse/revert 可以先跨 token 交互，再回到原 token 身份后做 residual。

**如何改。**

$$
M=P(X),\quad
\widehat M=M+\mathrm{PFFN}_{mix}(\mathrm{Norm}(M)),
$$

$$
R=P^{-1}(\widehat M),\quad
Y=X+R+\mathrm{PFFN}_{local}(\mathrm{Norm}(X+R)).
$$

**好处。** residual 坐标最清晰，适合更深网络。

**风险。** 一层包含两次 PFFN 时参数会翻倍；必须降低 hidden 宽度做等参数比较。

### C5：UniMixer-Lite 与 RankElastor 两条可学习 mixing 支线

**修改前。** C1-C4 的 $P$ 都是固定无参置换。

**为什么改。** 固定 mixing 可能限制更复杂的局部/全局交互；后续方法分别从“软置换”和“全坐标扩秩”改造。

**如何改。**

| 支线 | 核心结构 | 配套约束 |
|---|---|---|
| UniMixer-Lite | 共享局部基 + 低秩全局矩阵 | Sinkhorn、温度退火、SiameseNorm |
| RankElastor | Parameterized Full Mixing | 参数/FLOPs约束、effective-rank 监控 |

这两条都替换 fixed mixing，首轮互斥。必须用相同输入、tokenizer、PFFN、参数预算和训练预算比较。

**好处。** mixer 可以学习更合适的交换模式，或扩大坐标交互覆盖。

**风险。** 理论低秩不等于真实 kernel 高效；effective rank 上升也不等于 CVR AUC 上升。

### 7.2 四类 block 的简洁对比图

~~~mermaid
flowchart TB
    X["输入 X"] --> O["Strict Original<br/>P(X)+X → LN → PFFN residual"]
    X --> V["v1 Hybrid<br/>Original + 额外 PFFN Pre-LN"]
    X --> A["Company Aligned<br/>P(X) → PreNorm SwiGLU residual"]
    X --> R["Mixing & Reverting<br/>P → PFFN → P⁻¹ → add X"]

    O --> C["同输入、同参数、同 readout 比较"]
    V --> C
    A --> C
    R --> C
~~~

推荐顺序：v1 hybrid → strict original → company aligned → Mixing & Reverting → learned mixing。

---

## 8. D 类：Per-token FFN、归一化、容量与 MoE

### 8.1 它属于 RankMixer 的哪个部分

PFFN 是 RankMixer 的主要参数载体。每个 token 使用独立权重，在通道维上做非线性变换，不在 token 之间共享。
它负责保留异构特征的参数隔离，也是 v1 约 167.3M dense 参数的主要来源。

### D1：GELU PFFN 作为可归因基线

**修改前。** v1 使用每 token 独立的 $D\rightarrow4D\rightarrow D$ GELU FFN。

**为什么保留。** 在输入和 token 语义没有修正前换激活会混淆根因。GELU 版本还是所有新 block 的参数基准。

**如何改。** 首轮只修 tokenizer、输入或 readout时保持 GELU、$k=4$ 和初始化不变。

**好处。** 为后续 SwiGLU、低秩和 MoE 提供干净对照。

**风险。** 认为“旧激活一定弱”而跳过基线，会无法判断收益来自拓扑还是增参。

### D2：等参数 Pertoken SwiGLU

**修改前。** GELU PFFN 权重主项约为 $2D(4D)=8D^2$。

**为什么改。** SwiGLU 用 gate/value 乘法提供更强的通道选择，但相同 hidden 宽度会多 50% 参数：

$$
\mathrm{SwiGLU}(x)
=W_d\left(\mathrm{SiLU}(xW_g)\odot xW_u\right).
$$

**如何改。** 单次 SwiGLU 为匹配 $8D^2$，hidden 取：

$$
3Dm\approx8D^2
\quad\Rightarrow\quad
m\approx\frac{8D}{3}.
$$

当 $D=768$ 时可直接取 $m=2048$。如果一个 Mixing & Reverting block 使用两次 SwiGLU，则还要继续下调，
约取 $m=4D/3$ 才能接近单个 GELU PFFN 的权重预算。

**好处。** 在公平参数下验证门控非线性的真实收益。

**风险。** 直接沿用 expansion=4 会增参 50%，不能把全部增益归因于 SwiGLU。

### D3：Pre-Norm、Final Norm、small/zero-init 与深层辅助

**修改前。** v1 是混合 Pre/Post-LN；公司实现使用 Pre-LN SwiGLU、zero-init down 和 Final LN。

**为什么改。** 大而深的 PFFN 容易早期扰动主干分布；深层还可能出现梯度和有效秩振荡。

**如何改。**

1. 比较 Pre-RMSNorm 与 Pre-LN，固定其他条件；
2. down matrix 使用小初始化优先，zero-init 作为单独消融；
3. 堆深到 6 层以上时增加 inter-residual；
4. 在中间层 readout 加小权重 auxiliary loss；
5. UniMixer 支线再单独比较 SiameseNorm；
6. 所有版本保留 Final Norm 的开关消融。

**好处。** 改善深层训练稳定性，让新增容量更快进入有效学习。

**风险。** zero-init 首步 gate/up 无梯度；auxiliary loss 太大可能限制深层表示；多种 norm 一次更换无法归因。

### D4：参数压缩——降宽、低秩、共享基座加 token adapter

**修改前。** v1 的 per-token 独立 FFN 参数极大。

**为什么改。** 如果训练窗、显存或服务延迟无法承载，理论容量没有实际价值。

**如何改。**

- 先扫 $D$、$k$、$L$，保持效果/成本 Pareto；
- 对 up/down 做低秩分解；
- 同一业务组共享 base FFN，每 token 保留低秩 adapter；
- readout flatten 必须先低秩压缩；
- 用大模型 teacher 蒸馏到较小 Student。

**好处。** 降低显存、checkpoint、训练吞吐和在线延迟压力。

**风险。** 全共享 FFN 会破坏 token 异质性；只算理论 FLOPs 可能忽略真实 kernel。

### D5：两类 Sparse MoE

**修改前。** v1 是 dense PFFN。

**为什么改。** 希望训练更大容量而保持稀疏计算，但不同 MoE 方法的训练/推理语义并不相同。

**如何改。**

| 方法 | 属于哪一部分 | 核心做法 | 使用边界 |
|---|---|---|---|
| 原版 RankMixer MoE | PFFN expert 路由 | ReLU routing、DTSI 等 | 先验证路由与专家平衡 |
| TokenMixer-Large Sparse-Pertoken MoE | 每个 token 自己的专家组 | first enlarge then sparse、gate scaling | 需要 grouped sparse kernel |

两类都必须报告 expert load、路由熵、激活专家数、通信和真实延迟。

**好处。** 增大条件容量，同时控制激活计算。

**风险。** router 塌缩；通信抵消稀疏收益；框架若仍执行所有专家，就只有复杂度没有加速。

### D6：fused SwiGLU 与 grouped kernel

**修改前。** Python 循环、per-token matmul 和 train/export 双路径可能限制吞吐。

**为什么改。** RankMixer 的理论并行性只有在底层能把 token 独立 GEMM 合并时才会转化为系统收益。

**如何改。**

- 在数值 parity 通过后启用 fused SwiGLU；
- 将多个 token 的小 GEMM 组织成 grouped GEMM；
- 对 MoE 使用真正的 sparse grouped kernel；
- 记录端到端 samples/s，而不是只测单算子 microbenchmark。

**好处。** 降低 kernel launch 和 Python 图膨胀成本。

**风险。** 变量 scope 或 bias 初始化不一致会破坏热启；单算子加速可能被数据和通信瓶颈抵消。

---

## 9. E 类：显式交叉与行为序列模块

### 9.1 它属于 RankMixer 的哪个部分

E 类是 RankMixer 骨干的补充信息路径。固定 token mixing 擅长结构化交换，但不一定能在有限训练窗内替代
DCNM 的显式乘性交叉，也不会自动完成候选相关的行为序列注意力。

### E1：小型 DCNM late branch 或 Global Token

**修改前。** base 有两层 DCNM；v1 完全删除。

**为什么改。** v1 可能丢失已经验证的显式高阶交叉。

**如何改。** 比较两个互斥接法：

~~~text
E1-a: all inputs -> low-rank DCNM -> 256 -> late fusion
E1-b: all inputs -> low-rank DCNM -> Global Token -> RankMixer
~~~

先做 E1-a，因为不改变 token 数和 mixer contract；如果稳定有益，再试 E1-b。

**好处。** 快速补回全局乘性交叉，并能量化 DCNM 的真实边际贡献。

**风险。** 两条大塔重复计算；Global Token append 后忘记调整 $T/H/D$。

### E2：原样恢复 candidate-aware DIN

**修改前。** v1 lookup 了 sequence columns，但没有执行 base 的 sequence attention 并送入主路径。

**为什么改。** CVR 对用户意图、购买阶段和候选匹配高度依赖历史行为。

**如何改。**

- 复用 base 的 padding、mask 和候选 query；
- 先将 DIN 输出作为 late side；
- 再把 DIN 压成一个正式 token；
- 保持序列窗口与 label delay 时间安全。

**好处。** 恢复已知强信号，复杂度低于深序列 backbone。

**风险。** mask 错位、未来泄漏、候选相关用户塔破坏请求级缓存。

### E3：短期、长期 Sequence Token

**修改前。** 一个 DIN 向量可能过度压缩多种行为时间尺度。

**为什么改。** 短期浏览、长期购买和候选感知 DIN 表示的语义不同。

**如何改。** 在固定 budget 内分配 2 个 sequence token + 1 个 DIN token，或压缩其他 Item 组腾出位置。

**好处。** 保留不同时间尺度，并继续使用普通 RankMixer 主干。

**风险。** 直接 append 使 token contract 失效；序列 token 仍是一次性汇总，无法逐层读取原始历史。

### E4：MixFormer 式逐层深融合

**修改前。** E2/E3 只在输入阶段把序列压成固定向量。

**为什么改。** 深融合允许高阶 query 在每层读取原始行为 K/V，而不是只依赖一次 DIN 汇总。

**如何改。** 每层执行：

~~~text
Dense tokens
  -> Query Mixer
  -> Cross Attention(query, raw sequence K/V, mask)
  -> Output Fusion
  -> next block
~~~

只有 E2/E3 已证明序列是主要增益来源，才升级 E4。

**好处。** 稠密和序列表示共同演化，序列容量可随层数扩展。

**风险。** 显存、训练吞吐和服务延迟显著增加；没有原始 K/V 时不能把普通 DIN 冒充 MixFormer。

### E5：User-Item 解耦与请求级复用

**修改前。** candidate-aware DIN 或双向深融合可能让 User 表示依赖每个 Item，破坏请求级复用。

**为什么改。** 排序线上一个请求通常有多个候选，User 侧重复计算会放大延迟。

**如何改。**

- 把纯 User token/sequence K/V 做请求级预计算；
- 使用单向 mask，限制 Item 信息回流到可缓存 User 表示；
- 在候选侧执行小型 cross/query fusion；
- 报告单请求多候选下的真实 P95/P99。

**好处。** 在保留深融合能力时恢复服务可用性。

**风险。** 过度解耦会降低 User-Item 交互；离线单样本 latency 无法反映请求级收益。

### 9.2 序列升级决策图

~~~mermaid
flowchart LR
    S0["S0 无序列<br/>当前 v1"] --> S1["S1 Base DIN<br/>Late side"]
    S1 --> S2["S2 DIN Token"]
    S2 --> S3["S3 Short/Long<br/>Sequence Tokens"]
    S3 --> Q{"序列是否稳定贡献<br/>且预算允许？"}
    Q -->|是| S4["S4 MixFormer<br/>逐层 Cross Attention"]
    Q -->|否| K["保留 S1-S3"]
    S4 --> UI["User-Item 解耦<br/>请求级复用"]
~~~

---

## 10. F 类：Readout、融合与任务监督

### 10.1 它属于 RankMixer 的哪个部分

F 类位于最后一个 RankMixer block 之后。Readout 决定哪些 token 信息能到达预测头；多任务层决定共享表示
接受什么监督。即使主干很强，单一 mean pooling 和 first-only loss 仍可能形成瓶颈。

### F1：Mean + low-rank Flatten 双读出

**修改前。** v1 对 16 个 token 等权平均。

**为什么改。** mean 保留全局统计，但丢失 token 身份、小域峰值和位置特异信息；全量 flatten 又会参数爆炸。

**如何改。**

$$
r_{mean}=\frac1T\sum_t h_t,
\qquad
r_{flat}=\mathrm{Proj}_{low-rank}(\mathrm{vec}(H)).
$$

然后拼接 $[r_{mean},r_{flat}]$。若使用奇数层 aligned mixing，先确认 flatten 的坐标含义。

**好处。** 同时保留全局鲁棒性和位置身份，成本远低于大 flatten MLP。

**风险。** low-rank 宽度太大仍会形成高成本 head；wide 分支增益不能误归 RankMixer block。

### F2：Grouped、weighted 与 attention pooling

**修改前。** 所有 token 权重相同。

**为什么改。** User、Item、Sequence、Global 和 Task token 的统计意义不同。

**如何改。** 按复杂度依次比较：

1. User/Item/Sequence 分组 mean；
2. 每组可学习静态权重；
3. 样本级 gated pooling；
4. 最后才尝试 attention pooling。

**好处。** 保留域级结构，让小域不被大域平均。

**风险。** attention readout 可能重复主干交互且增加不必要复杂度。

### F3：Creative/Coupon side fusion

**修改前。** Creative 被混进 token，Coupon 缺失。

**为什么改。** A4 负责小域输入处理，F3 负责它在输出侧如何与主干结合。

**如何改。**

~~~text
RankMixer dual readout
    + Creative tower
    + Coupon tower
    + optional Context/DCNM branch
    -> 256 shared fusion
    -> task heads
~~~

**好处。** 保留局部强信号，同时使主干和小域路径可独立消融。

**风险。** 插入层配置无效或拼接顺序变化会破坏热启。

### F4：恢复 first + last，多任务头与 Task Token

**修改前。** v1 只解析和优化 first；base 使用 first/last，成熟实现还有 no-refund 等任务。

**为什么改。** 相关任务可提升样本效率并正则共享表示，但任务冲突需要显式处理。

**如何改。** 第一阶段只恢复：

$$
\mathcal L
=\mathcal L_{first}
+\lambda_{last}\mathcal L_{last},
\qquad
\lambda_{last}\in\{0.1,0.2,0.5\}.
$$

稳定后再增加 task-specific readout 或 Task Token；新增任务必须验证标签窗口、缺失 mask 和可观测性。

**好处。** 恢复 base 的辅助监督，并允许不同任务读取不同 token 信息。

**风险。** 辅助权重过大造成负迁移；不同窗口标签混用造成泄漏。

### F5：Pairwise、consistency 与校准联合目标

**修改前。** 只优化 pointwise BCE，而最终关注 AUC、COPC 和校准。

**为什么改。** BCE 与 AUC 的排序目标不完全一致，多任务标签之间还可能存在业务包含关系。

**如何改。**

- 在 query 内构造正负对，加入 0.02-0.1 小权重 pairwise softplus；
- 对 first/last/no-refund 的真实包含关系加 consistency 或 monotonic loss；
- 始终保留 BCE 主损失；
- 同时报 AUC、PR-AUC、LogLoss、COPC、ECE。

**好处。** 可能改善排序，同时保持概率可用性。

**风险。** 纯 pairwise 会损害校准；跨 query 采样会引入偏差；错误的标签包含假设会伤害模型。

### F6：ESMM 的条件性使用

**修改前。** 当前数据看起来是 CVR 训练样本，不足以自动证明拥有全曝光链路。

**为什么改。** 只有完整的曝光→点击→转化数据，ESMM 才能建模样本选择偏差。

**如何改。** 先审计是否存在全曝光样本、点击标签和转化标签，再构造 CTR/CVR/CTCVR 的一致概率关系。

**好处。** 数据条件满足时可缓解 clicked-only CVR 的选择偏差。

**风险。** 没有全曝光数据却套 ESMM，数学前提不成立。

### 10.2 输出与监督流程图

~~~mermaid
flowchart LR
    H["RankMixer H<br/>B×T×D"] --> M["Mean / Group Pool"]
    H --> F["Low-rank Flatten"]
    M --> Z["Shared fusion"]
    F --> Z
    C["Creative/Coupon side"] --> Z
    D["DCNM/Context side"] --> Z

    Z --> A["first head"]
    Z --> B["last head"]
    Z --> T["可选 Task readout"]
    A --> L["Raw-logit BCE<br/>+ 小权重辅助目标"]
    B --> L
    T --> L
~~~

---

## 11. G 类：初始化、优化、蒸馏与系统模块

### 11.1 它属于 RankMixer 的哪个部分

G 类横跨参数初始化、优化器、checkpoint、训练调度、算子和服务。RankMixer 的 PFFN 参数远大于普通 MLP，
如果仍用极小学习率冷启或低效小 GEMM，结构优势可能完全无法显现。

### G1：公平冷启或等成熟度热启

**修改前。** base 可能复用成熟 DCNM/MLP checkpoint，而 RankMixer 新 scope 大量随机初始化。

**为什么改。** 固定训练 10 天会偏向热启覆盖更广的模型。

**如何改。**

- 对照一：base 与 RM 都冷启；
- 对照二：base 使用原热启，RM 尽可能加载共享 embedding/side tower，并用 teacher/KD 补主干；
- 输出每个 scope loaded、missing、random-init 的参数量；
- 比较相同样本、相同 FLOPs、相同 wall-clock 的收敛曲线。

**好处。** 把“结构能力”和“初始化成熟度”分开。

**风险。** 只看 restore 成功日志，不核对变量数量和 shape。

### G2：分组学习率

**修改前。** 新 tokenizer/PFFN 和成熟 sparse embedding 可能使用同一学习率。

**为什么改。** 新增 100M 级 dense 参数需要更快学习，而成熟 embedding 需要保护。

**如何改。**

| 参数组 | 建议相对 LR |
|---|---:|
| 新 tokenizer / RankMixer / head | 1.0 |
| 热启 side tower | 0.2-0.5 |
| 热启 sparse embedding | 0.1 或原 sparse optimizer |

**好处。** 缩短新主干追平时间，降低旧表征被破坏的风险。

**风险。** scope 匹配错误会让参数落入错误优化器；必须打印每组变量清单和参数量。

### G3：base teacher 蒸馏与 Student

**修改前。** RankMixer 需要从随机大主干重新学习 base 已有知识。

**为什么改。** 蒸馏可以把成熟 base 的函数先验迁移给 RM，特别适合训练窗固定的大模型。

**如何改。**

$$
\mathcal L
=(1-\alpha)\mathcal L_{label}
+\alpha T^2\mathrm{KL}
\left(
\sigma(z_{teacher}/T),
\sigma(z_{student}/T)
\right).
$$

从 $T\in\{1,2\}$、首日 $\alpha=0.5$ 逐渐降至 0.1 开始。teacher logits 优先离线写入，避免双模型前向。
RM 稳定后还可反向蒸馏到较小 Student。

**好处。** 加快追平、平滑优化，并形成可服务的小模型路径。

**风险。** teacher/student 样本、标签或校准口径不同；蒸馏权重过大限制学生上限。

### G4：raw-logit loss、真正的梯度裁剪与配置闭环

**修改前。**

- v1 先 sigmoid，再调用 <code>tf.losses.log_loss</code>；
- logits 在 sigmoid 前 clip；
- <code>grad_clip_value</code>、dropout、<code>use_rankmixer</code> 等配置未必真正影响图；
- 优化器计算梯度后直接 apply，没有执行显式裁剪。

**为什么改。** 数值稳定性和极端样本梯度会影响大模型早期收敛；“有配置但图中无效”会造成错误实验记录。

**如何改。**

- 训练使用 <code>sigmoid_cross_entropy_with_logits</code> 直接读取 raw logits；
- 预测单独 sigmoid，不在 loss 前 clip logits；
- 使用 global-norm clipping 并记录裁剪前后 norm；
- 每个配置开关必须有图结构或变量数量差异测试；
- dropout 只在 train 激活，test/export 确认关闭。

**好处。** 提高数值稳定性，让实验配置与实际模型一致。

**风险。** 切换 loss 同时改 label 权重会无法归因；clip 阈值过小会长期限制学习。

### G5：系统优化——grouped GEMM、Token Parallel、FP8 与缓存

**修改前。** 理论可并行的 per-token 计算不一定被当前 TF1 图和 PS 系统高效执行。

**为什么改。** 最终能否上线由真实吞吐、显存、通信和 P99 决定。

**如何改。**

1. grouped GEMM/fused SwiGLU；
2. 大模型阶段评估 Token Parallel；
3. 硬件与数值验证充分后评估 FP8 推理；
4. User-Item 解耦恢复请求级缓存；
5. checkpoint 分片与恢复耗时纳入成本；
6. MoE 只有在真实 sparse kernel 可用时开启。

**好处。** 把模型结构优势转化为端到端成本收益。

**风险。** 论文系统数字不能直接外推到当前 TF1/PS 环境。

### G6：可观测性与验收指标

**修改前。** 只看总体 AUC 会掩盖校准、切片和系统退化。

**为什么改。** CVR 改造可能在总体平均上接近，却伤害冷启动、长尾或某个标签窗口。

**如何改。** 每个实验固定报告：

- 效果：AUC/GAUC、PR-AUC、LogLoss、COPC、ECE；
- 切片：冷启动、长尾、行为长度、User/Item/Creative/Coupon；
- 收敛：按样本、FLOPs、wall-clock 的曲线；
- 表示：token correlation、effective rank、gate/router 分布；
- 成本：dense params、FLOPs、samples/s、peak memory、checkpoint；
- 服务：P50/P95/P99、单请求多候选成本。

**好处。** 能判断“为什么有效”和“是否可上线”。

**风险。** 只用单 seed、单时间窗或最后一个 checkpoint 结论不稳。

---

## 12. 模块组合、冲突与依赖

### 12.1 自然可叠加的主线

~~~text
P 正确性
 -> A 完整输入与 identity gate
 -> B 字段安全 tokenizer
 -> C 选定一种 block
 -> D 选定一种 PFFN
 -> E DIN / DCNM side
 -> F 双读出与 first/last
 -> G 热启、蒸馏和系统优化
~~~

这些位于不同层次，但仍要逐项打开。

### 12.2 互斥或高度重叠

| 选择组 | 为什么首轮不能一起开 |
|---|---|
| 人工硬桶 vs Soft-to-Hard vs 随机 permutation | 都改变字段到 token 的映射 |
| strict original vs aligned vs Mixing & Reverting | 都替换 block residual 拓扑 |
| fixed mixing vs UniMixer-Lite vs RankElastor | 都替换 mixing |
| GELU vs SwiGLU vs MoE | 都改变 PFFN 容量与非线性 |
| DCNM Global Token vs DCNM late branch | 都验证显式全局交叉 |
| mean+flat vs attention pooling vs Task readout | 都改变输出信息通路 |
| Creative token vs side tower vs FiLM | 都改变小域进入主干的方式 |

### 12.3 有前置条件的方案

| 方案 | 必须先满足 |
|---|---|
| ESMM | 全曝光、点击、转化链路完整 |
| Cross Pre-trained Token | 有可靠且时间安全的预训练 User/Item 向量 |
| MixFormer | 保留原始序列 K/V 和正确 mask |
| User-Item 解耦 | 线上确有请求级多候选复用 |
| Sparse MoE | 有真实 sparse/grouped kernel 与负载监控 |
| FP8 | 硬件支持且离线/在线数值 parity 通过 |
| Learned mixing | P/A/B 与 fixed mixing baseline 已稳定 |

---

## 13. 四套可落地架构方案

### 13.1 方案 A：RM-v2-Parity——先追回 base

**架构目标。** 只修复比较不公平和明显信息瓶颈，不引入高风险研究模块。

**结构。**

~~~text
完整 base 输入
 -> 分域归一化
 -> 完整字段 16-token hard tokenizer
 -> strict original RankMixer 或 v1 block 单变量对照
 -> mean + low-rank flatten
 -> Creative side
 -> first + last heads
 -> raw-logit BCE
~~~

**为什么这样设计。** 它能分离输入、tokenization、readout 和辅助监督对 0.003 AUC 差距的贡献。

**预期好处。** 风险最低、可解释性最高，最适合作为新的工程 baseline。

**不要同时加入。** SwiGLU、可学习 mixing、Soft-to-Hard、MoE。

### 13.2 方案 B：RM-v3-Aligned——近期上限候选

**架构目标。** 在方案 A 已稳定后，引入成熟公司结构中最有依据的动态选择和 block 改造。

**结构。**

~~~text
方案 A
 + identity User gate
 + User-conditioned Item gate
 + 32 个稳定语义 token
 + company-aligned PreNorm Pertoken SwiGLU
 + DIN token
 + Creative/Coupon side
 + small DCNM late branch
 + base teacher distillation
~~~

**为什么这样设计。** 输入、token、残差和非线性分别解决 v1 的主要结构缺陷，又保留可回滚 side path。

**预期好处。** 是最有希望稳定超过 base 的近期候选。

**关键约束。** SwiGLU 必须等参数；32 token 必须重新核算 $T/H/D$；fused parity 先通过。

### 13.3 方案 C：RM-v4-Sequence——序列主导场景

**架构目标。** 当 E2/E3 证明行为序列贡献最大时，让稠密 token 与原始序列逐层融合。

**结构。**

~~~text
方案 B 的稳定 tokenizer/readout
 -> Query Mixer
 -> Cross Attention to raw behavior sequence
 -> Output Fusion
 -> 多层堆叠
 -> User-Item 单向解耦
 -> first/last/task heads
~~~

**为什么这样设计。** 避免把全部历史压缩成一个 DIN 向量，并保留请求级复用。

**预期好处。** 在长行为序列和候选匹配强相关的流量上提升上限。

**关键约束。** 必须测多候选 P99；没有原始 K/V 时不实施。

### 13.4 方案 D：RM-v5-Research——表示、mixing 与稀疏扩展

**架构目标。** 在 B/C 已有稳定 checkpoint 后探索中长期 scaling。

独立复制四条支线：

- D1：Soft-to-Hard；
- D2：RankUp 表示扩展；
- D3：UniMixer-Lite 或 RankElastor，二选一；
- D4：Student 或 Sparse-Pertoken MoE。

**为什么这样设计。** 四条支线改变的核心假设不同，拆开才能判断上限来自表示、mixing 还是容量。

**预期好处。** 探索有效秩、深度和条件容量的进一步增长。

**关键约束。** 每条支线只替换一个核心模块，且必须报告系统成本。

### 13.5 四套架构的演进关系

~~~mermaid
flowchart LR
    A["A Parity<br/>公平输入与监督"] --> B["B Aligned<br/>Gate + 语义 Token + SwiGLU"]
    B --> C["C Sequence<br/>逐层读取行为序列"]
    B --> D1["D1 表示支线<br/>Soft-to-Hard / RankUp"]
    B --> D2["D2 Mixer 支线<br/>UniMixer / RankElastor"]
    B --> D3["D3 容量支线<br/>Student / Sparse MoE"]
~~~

---

## 14. 统一伪代码

### 14.1 模块化 forward

~~~python
def forward(features, mode, cfg):
    # P/A: 完整输入与分域预处理
    parts = build_exact_base_inputs(
        features,
        include_dense=cfg.include_dense,
        include_gattr=cfg.include_gattr,
        include_sequence=cfg.include_sequence,
    )
    parts = domain_normalize(parts, mode)

    user = parts.user * identity_user_gate(parts.user, cfg)
    item = parts.item * identity_item_gate(
        concat([parts.user, parts.item]), cfg
    )

    # B: 字段安全 tokenizer
    token_groups = versioned_field_groups(cfg.token_schema)
    assert_field_coverage(token_groups, parts.fields)
    tokens = project_groups_once(
        user, item, token_groups,
        projection=cfg.token_projection,
    )

    # E: 序列与全局交叉
    if cfg.sequence_level >= 2:
        tokens = insert_with_fixed_budget(
            tokens,
            build_sequence_tokens(parts.sequence, parts.item, cfg),
        )
    assert_token_contract(tokens, cfg.T, cfg.H, cfg.D)

    # C/D: 只选择一种 mixer 和一种 PFFN
    h = rankmixer_stack(
        tokens,
        topology=cfg.block_topology,
        mixing=cfg.mixing_type,
        pffn=cfg.pffn_type,
        parameter_budget=cfg.parameter_budget,
    )

    # F: 双读出与侧路融合
    rm_repr = concat([
        grouped_or_mean_pool(h),
        low_rank_flatten(h),
    ])
    side_repr = concat_nonempty([
        creative_coupon_side(parts, cfg),
        dcnm_or_context_side(parts, cfg),
        din_side(parts, cfg),
    ])
    shared = fusion_head(concat([rm_repr, side_repr]))

    logits = build_task_heads(shared, cfg.tasks)
    return logits
~~~

### 14.2 loss 与优化

~~~python
logits = forward(features, mode="train", cfg=cfg)

loss = raw_logit_bce(logits["first"], labels["first"])
loss += cfg.last_weight * raw_logit_bce(
    logits["last"], labels["last"]
)
loss += optional_pairwise_or_consistency(logits, labels, cfg)
loss += optional_teacher_distillation(logits, teacher_logits, cfg)

groups = group_variables_by_scope(
    new_rankmixer=1.0,
    warm_side=0.2_to_0.5,
    warm_sparse=0.1,
)
grads = compute_gradients(loss, groups)
grads = clip_by_global_norm(grads, cfg.clip_norm)
train_op = apply_grouped_learning_rates(grads, groups)
~~~

---

## 15. 推荐实验阶梯

| 阶段 | 唯一主要变化 | 回答的问题 |
|---|---|---|
| E0 | 原样复现 base 与 v1，多 seed | 0.003 是否稳定 |
| E1 | v1 恢复 dense | dense 边际贡献 |
| E2 | E1 恢复 DIN | 序列边际贡献 |
| E3 | E2 恢复 gattr | 全局属性贡献 |
| E4 | E3 恢复 last loss | 辅助监督贡献 |
| E5 | E4 改字段安全 hard tokenizer | 标量切分损失 |
| E6 | E5 + Creative side | 小域隔离收益 |
| E7 | E6 + identity gates | 样本动态选择收益 |
| E8 | E7 + low-rank flatten | token 身份读出收益 |
| E9 | 固定 E8 比较四种 block | residual 拓扑收益 |
| E10 | 最佳 block 上比较 GELU/等参 SwiGLU | 激活与门控收益 |
| E11 | E10 + DCNM late | 显式交叉边际 |
| E12 | E11 + distillation | 收敛迁移收益 |
| E13 | E12 + Task readout/辅助目标 | 多任务收益 |
| E14+ | Soft-to-Hard、RankUp、learned mixing、MoE | 中长期上限 |

训练资源建议：

~~~text
1k-step 图与数值验证
 -> 固定 1-2 日小窗
 -> 5 日中窗
 -> 最佳 1-2 个跑完整 10 日
 -> 另一不重叠时间窗复测
~~~

淘汰时不能只看首日 AUC。大主干冷启较慢，应同时观察同样本、同 FLOPs 和同 wall-clock 曲线。

### 15.1 实验决策流程

~~~mermaid
flowchart TD
    A["E0 可重复？"] -->|否| A0["修数据、随机性和评估"]
    A -->|是| B["E1-E4 输入与监督追回差距？"]
    B -->|是| C["进入字段安全 tokenizer"]
    B -->|否| D["复查热启、loss、梯度和样本口径"]
    C --> E["E5-E8 稳定增益？"]
    E -->|否| F["停在 Parity 并定位表示瓶颈"]
    E -->|是| G["等参数比较 block/PFFN"]
    G --> H["加入 DCNM/DIN/KD"]
    H --> I{"效果、校准、成本均通过？"}
    I -->|是| J["跨时间窗与在线灰度"]
    I -->|否| K["回滚最近单一模块"]
~~~

---

## 16. 每个实验的最小验收清单

### 16.1 正确性

- [ ] 每个字段恰好进入一个预期组；
- [ ] token schema 有版本和 hash；
- [ ] $T=H$、$D\%H=0$、最终 token 数等断言通过；
- [ ] train/test/export shape 和数值口径一致；
- [ ] fused/unfused forward、gradient、restore parity 通过；
- [ ] raw-logit loss 与预测 sigmoid 分离；
- [ ] global-norm gradient clipping 真正执行；
- [ ] loaded/missing/random-init 参数量可审计。

### 16.2 效果与归因

- [ ] 一次只改变一个核心模块；
- [ ] first AUC/GAUC、PR-AUC、LogLoss、COPC、ECE 全部报告；
- [ ] 多 seed 或 paired bootstrap 方向一致；
- [ ] 至少两个不重叠时间窗复测；
- [ ] 冷启动、长尾、行为长度和小域切片无系统性退化；
- [ ] 参数、FLOPs、样本和 wall-clock 口径同时给出。

### 16.3 成本与服务

- [ ] trainable dense params、FLOPs、samples/s、peak memory；
- [ ] checkpoint 大小、保存与恢复时长；
- [ ] serving P50/P95/P99；
- [ ] 多候选请求级复用成本；
- [ ] gate/router 负载与稀疏 kernel 实际激活量；
- [ ] 新模块有明确关闭开关和 checkpoint 回滚路径。

---

## 17. 源码修改落点

| 模块 | 主要源码位置 | 建议新 scope |
|---|---|---|
| P 数据/任务 | <code>cvr_bn_rankmixer_v1.py:380-462</code>；参考 <code>cvr_fst_last_norpy.py:481-538</code> | <code>loss_rawlogit_v1</code> |
| A 输入/BN/Gate | <code>cvr_bn_rankmixer_v1.py:889-930</code>；参考 <code>commend_cvr.py:2243-2323</code> | <code>input_adapter_v2</code> |
| B Tokenizer | <code>cvr_bn_rankmixer_v1.py:774-799,938-962</code>；参考 <code>commend_cvr.py:2386-2500,5190-5213</code> | <code>tokenizer_field_safe_v1</code> |
| C Mixing/Block | <code>cvr_bn_rankmixer_v1.py:801-862</code> | <code>rankmixer_strict_v1</code>、<code>rankmixer_aligned_v1</code> |
| D SwiGLU/fused | <code>mlp_mixer_swiglu_fuse.py:90-304,339-386</code> | <code>pffn_swiglu_matched_v1</code> |
| E DCNM | <code>cvr_fst_last_norpy.py:772-817</code> | <code>cross_branch_v1</code> |
| E DIN/Sequence | <code>cvr_fst_last_norpy.py:917-962,1691+</code>；<code>commend_cvr.py:2180-2241,2337-2500</code> | <code>sequence_adapter_v1</code> |
| F Readout/Side | <code>cvr_bn_rankmixer_v1.py:864-980</code>；参考 <code>commend_cvr.py:2507-2594</code> | <code>readout_dual_v1</code> |
| F 多任务 | <code>cvr_fst_last_norpy.py:481-538,1132-1173</code> | <code>multitask_head_v1</code> |
| G Warm/优化 | <code>cvr_bn_rankmixer_v1.py:57-70,432-462,760-765</code> | <code>optimizer_groups_v1</code> |

注意：<code>cvr_bn_rankmixer_v1.py:960</code> 的注释写“无 bias”，但实际
<code>_rm_tokenize_bucket()</code> 在 791 行创建了零初始化 bias。文档和后续重构应以实际图为准并修正注释。

---

## 18. 延伸文档

- 快速选型版：[CVR_RankMixer_v1_改进方案汇总与选型指南.md](CVR_RankMixer_v1_改进方案汇总与选型指南.md)
- AUC 差距源码诊断：[CVR_RankMixer_v1_AUC差距诊断与改进路线.md](CVR_RankMixer_v1_AUC差距诊断与改进路线.md)
- 四套输入与监督方案：[CVR_RankMixer_四套改进方案设计.md](CVR_RankMixer_四套改进方案设计.md)
- SwiGLU 源码严格对比：[CVR_RankMixer_SwiGLU_源码分析与原版对比.md](CVR_RankMixer_SwiGLU_源码分析与原版对比.md)
- RankMixer 演进方法调研：[RankMixer及其演进方法详细调研.md](RankMixer及其演进方法详细调研.md)
- 原版论文详解：[RankMixer_论文详解.md](RankMixer_论文详解.md)
- TokenMixer-Large：[../tokenmixer/TokenMixer-Large_论文详解.md](../tokenmixer/TokenMixer-Large_论文详解.md)
- MixFormer：[../mixformer/MixFormer_论文详解.md](../mixformer/MixFormer_论文详解.md)
- RankUp：[../RankUp/RankUp_论文详解.md](../RankUp/RankUp_论文详解.md)
- UniMixer：[../UniMixer/UniMixer_论文详解.md](../UniMixer/UniMixer_论文详解.md)
- RankElastor：[../RankElastor/RankElastor_论文详解.md](../RankElastor/RankElastor_论文详解.md)

---

## 19. 最终建议

1. 先完成 P 类审计，否则所有结构结论都不可靠；
2. 第一条主线只做 A1/A2、B1、F1/F3/F4 和 G4，构建 RM-v2-Parity；
3. Parity 稳定后再做 A3、C3、D2、E1/E2 和 G3，构建 RM-v3-Aligned；
4. 序列已经证明是主增益时才做 MixFormer；
5. Soft-to-Hard、RankUp、UniMixer、RankElastor 和 MoE 必须拆成独立支线；
6. 每一次修改都同时回答三个问题：效果是否更好、原因是否可归因、成本是否可服务。

一句话总结：

> 先把输入和 token 语义做对，再把 residual 与 PFFN 做强，最后才讨论可学习 mixing 和稀疏扩展。
