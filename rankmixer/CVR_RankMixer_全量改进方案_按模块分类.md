# CVR RankMixer 全量改进方案：基于真实 SENet+DCNM Base 的严格模块对照

> 真实 Base：<code>code/cvr_bn_senet_dcnm.py</code>。<br>
> RankMixer v1：<code>code/cvr_bn_rankmixer_v1.py</code>。<br>
> Base 运行证据：<code>set-x.args.resolved.txt</code>、<code>0721args.txt</code>。<br>
> RankMixer 运行证据：<code>code/set-xcal.txt</code>。<br>
> 工业 RankMixer 参考：<code>code/commend_cvr.py</code>、<code>code/mlp_mixer_swiglu_fuse.py</code>。<br>
> 重要边界：<code>code/cvr_fst_last_norpy.py</code> 只是另一次模型训练尝试，不是本文 Base，不能用于证明 Base 的输入、任务或网络结构。

本文重新建立完整证据链：先以真实运行参数判断“实际打开了什么”，再用真实 Base 源码解释数据如何流动，
最后才提出 RankMixer 改进。文中的“预期收益”是机制假设，不是 AUC 承诺。

---

## 1. 修正后的核心结论

### 1.1 原结论中必须撤销的三项

此前把 <code>cvr_fst_last_norpy.py</code> 当作 Base，导致以下结论错误：

| 原结论 | 真实 Base 证据 | 修正 |
|---|---|---|
| RankMixer 删除了 Base 的 dense、DIN、gattr | Base 源码 920 行明确写这些域为 0；938-953 行只收集三桶 | 撤销；两边实际输入相同 |
| RankMixer 删除了 Base 的 last 辅助任务 | Base resolved 参数设置 <code>enable_last_cvr=false</code> | 撤销；两边都只训练 first |
| 恢复 DIN/last 是追回 0.003 的 P0 步骤 | 它们不在真实 Base 运行图中 | 降为独立研究扩展 |

这三个方向仍可以研究，但不能再解释“Base 0.865、RankMixer v1 0.862”的现有差距。

### 1.2 真实差异是什么

在用户说明“Base 与 RankMixer 运行方式基本相同，只替换程序名”的前提下，运行参数进一步证明两边共享：

- 相同 feature version：<code>data.cvr.cvr_fea_v10_base_cold</code>；
- common、item、creative 三桶稀疏 embedding，embedding size 17；
- 三桶分别做输入 BN；
- <code>flood_adam</code>、学习率 $2\times10^{-5}$、batch size 2048；
- first CVR 单任务；
- wide、last、multi-task、delay 全部关闭；
- 概率式 BCE、logit clip，以及没有真正执行的 gradient clipping。

真正不同的是：

| 数据流位置 | 真实 Base | RankMixer v1 |
|---|---|---|
| 字段动态选择 | 层级 SENet，运行时开启 | 没有；参数为 false，代码也未接入 |
| 早期显式交叉 | 2 层 DCNM，bottleneck 500 | 没有 |
| 主干 | MLP 20978→2048→2048→256 | Tokenizer + 2 层 RankMixer |
| Tokenizer | 不做 token 化 | 标量等宽切 16 段，切断字段 |
| 跨 token 交互 | 不适用 | 固定无参置换 |
| 通道变换 | 共享大 MLP | 每 token 独立 $768\to3072\to768$ |
| Readout | 256 维 MLP 表示接 head | 16 个 token 直接 mean 后接 head |
| 近似 dense 参数 | 约 90.3M | 约 167.3M |

因此，修正后的根因优先级是：

1. **P0：v1 没有复用 Base 已开启的层级 SENet。**
2. **P0：v1 tokenizer 在字段内部切分，破坏 per-token 参数专门化。**
3. **P0/P1：Base 在压缩前有两层全局乘性交叉，v1 没有同等机制。**
4. **P1：v1 只做 mean readout，丢失 token 位置身份。**
5. **P1：v1 block 是额外 Pre-LN 的 hybrid，并非严格原版或成熟公司版本。**
6. **P1：v1 参数约为 Base 的 1.85 倍，却可能有更多随机初始化参数，训练预算不公平。**

### 1.3 最推荐的新实验顺序

~~~text
E0 真实 Base 与当前 v1 严格复现
 -> E1 v1 前增加“完全相同的 Base SENet”
 -> E2 只修复完整字段语义 tokenizer
 -> E3 只增加 mean + low-rank flatten
 -> E4 固定输入比较 strict / aligned / reverting block
 -> E5 将 rm_ffn_expand 从 4 改为 2，构造约 91.7M 等参数 RM
 -> E6 保留 Base SENet+DCNM，只把 Base MLP 替换为 RankMixer
 -> E7 在最佳公平版本上比较等参数 SwiGLU
 -> E8 再做蒸馏、序列、多任务、可学习 mixing、MoE
~~~

这个顺序先回答“差距来自缺少 Base 模块，还是 RankMixer 本身”，再讨论超过 Base。

---

## 2. 证据口径：代码默认值不等于真实运行图

### 2.1 证据优先级

本文使用以下优先级：

1. resolved/实际启动参数；
2. 真实 Base 与 RankMixer 的 <code>model_fn()</code> 调用链；
3. 配置默认值和未打开分支；
4. 其他训练尝试与论文，只用于设计新方案。

这是必要的，因为 Base 类中的默认值与真实运行不同：

| 配置 | 类默认值 | Base resolved 运行值 | 实际影响 |
|---|---:|---:|---|
| <code>use_senet</code> | false | true | SENet 实际启用 |
| <code>use_senet_bn</code> | false | true | 三个 gate hidden 都做 BN |
| <code>enable_last_cvr</code> | true | false | last 分塔与 loss 不在运行图 |
| <code>enable_wide_cvr</code> | true | false | wide 分塔与 loss 不在运行图 |
| <code>enable_mlt_loss</code> | true | false | 多目标辅助塔不在运行图 |
| <code>cross_num</code> | 2 | 未覆盖，仍为 2 | 两层 DCNM |

仅看类代码存在某分支，不能说 Base 实际使用了它。

### 2.2 两个运行模板的已证实共同条件

Base resolved 参数位于 <code>set-x.args.resolved.txt:27-76</code>；RankMixer 参数位于
<code>set-xcal.txt:248-298</code>。二者共同条件包括：

~~~text
optimizer       = flood_adam
learning_rate   = 2e-5
batch_size      = 2048
embedding_size  = 17
batch_norm      = true
feature_version = data.cvr.cvr_fea_v10_base_cold
opt_goal        = first_cvr
wide / mlt / last / delay = false
~~~

唯一不能仅从静态文件确认的是具体 AUC 实验对应的完整 checkpoint 加载结果。两边都提供
<code>checkpoint_import_dir</code>，但 Base 的 SENet/DCNM/MLP scope 与历史模型同名，RankMixer 主干 scope 是新增的。
到底加载了多少参数必须以服务器 restore 日志为准。

### 2.3 非 Base 文件应如何使用

<code>cvr_fst_last_norpy.py</code> 可以提供 dense、DIN、gattr、last 等实现参考，但只能放在“研究扩展”：

~~~text
真实 Base 事实证明：禁止引用
研究方案伪代码：可以参考
效果归因：必须重新实验
~~~

---

## 3. 真实 Base 的完整结构

### 3.1 数据流

~~~mermaid
flowchart LR
    A["Sparse lookup"] --> U["Common<br/>6545 = 385×17"]
    A --> I["Item<br/>14195 = 835×17"]
    A --> C["Creative<br/>238 = 14×17"]

    U --> BN["三桶独立 BN"]
    I --> BN
    C --> BN

    BN --> SE["层级 SENet<br/>Common self<br/>Item conditioned on U+I<br/>Creative conditioned on U+I+C"]
    SE --> X["拼接 20978 维"]
    X --> DCN["2× DCNM<br/>20978→500→20978<br/>乘法交叉 + residual + LN"]
    DCN --> MLP["MLP<br/>20978→2048→2048→256<br/>BN + GELU"]
    MLP --> H["Linear 256→1"]
    H --> L["first CVR<br/>概率式 BCE"]
~~~

源码依据：

- Base 配置：<code>cvr_bn_senet_dcnm.py:150-208</code>；
- 三桶收集：<code>cvr_bn_senet_dcnm.py:915-953</code>；
- 三桶 BN 与 SENet 调用：<code>cvr_bn_senet_dcnm.py:955-980</code>；
- 层级 SENet：<code>cvr_bn_senet_dcnm.py:830-905</code>；
- DCNM：<code>cvr_bn_senet_dcnm.py:783-828,982-983</code>；
- 三层 MLP：<code>cvr_bn_senet_dcnm.py:988-1016</code>；
- first head：<code>cvr_bn_senet_dcnm.py:1097-1108</code>；
- first loss：<code>cvr_bn_senet_dcnm.py:527-548</code>；
- resolved 关闭 last/wide/mlt：<code>set-x.args.resolved.txt:70-75</code>。

### 3.2 Base SENet 的精确数学结构

先把每个域恢复为字段矩阵：

$$
U\in\mathbb R^{385\times17},\quad
I\in\mathbb R^{835\times17},\quad
C\in\mathbb R^{14\times17}.
$$

沿 17 维 embedding 做均值，得到字段级 squeeze：

$$
s_U=\mathrm{Mean}_{emb}(U),\quad
s_I=\mathrm{Mean}_{emb}(I),\quad
s_C=\mathrm{Mean}_{emb}(C).
$$

三个 gate 具有层级条件关系：

$$
g_U=2\sigma\left(W_{U2}\tanh(\mathrm{BN}(W_{U1}s_U))\right),
$$

$$
g_I=2\sigma\left(W_{I2}\tanh(\mathrm{BN}(W_{I1}[s_U,s_I]))\right),
$$

$$
g_C=2\sigma\left(W_{C2}\tanh(\mathrm{BN}(W_{C1}[s_U,s_I,s_C]))\right).
$$

最后每个 gate 标量乘到对应字段的 17 个 embedding 维：

$$
\widetilde U_f=g_{U,f}U_f,\quad
\widetilde I_f=g_{I,f}I_f,\quad
\widetilde C_f=g_{C,f}C_f.
$$

这不是普通独立 SENet：

- Common 只看 Common；
- Item 同时看 Common 与 Item；
- Creative 同时看三域；
- gate 范围是 $(0,2)$；
- Base 使用 Glorot 初始化，不是严格 identity-init，但初始 logits 近零时 gate 大致靠近 1。

### 3.3 Base DCNM 的精确结构

令 $x_0$ 为 SENet 后 20978 维拼接向量，第 $l$ 层：

$$
u_l=W_{l,2}\phi(W_{l,1}x_l),
\qquad
W_{l,1}:\ 20978\to500,
\quad
W_{l,2}:\ 500\to20978,
$$

$$
x_{l+1}
=\mathrm{LN}\left(x_0\odot u_l+x_l\right).
$$

实际运行中 <code>use_cross_act</code> 没有打开，所以 $\phi$ 是 identity。DCNM 的重要作用不是单纯增加两层
Dense，而是在第一次强压缩前，让每个原始坐标获得全局条件并与 $x_0$ 做乘法。

### 3.4 Base 的参数量

按实际三桶宽度、SENet hidden 128、DCNM bottleneck 500、两层交叉、MLP
$[2048,2048,256]$ 估算：

| 部分 | 权重主项 |
|---|---:|
| 层级 SENet | 0.521M |
| 两层 DCNM | 41.956M |
| 三层 MLP | 47.682M |
| Head | 0.0003M |
| 加 bias、BN、LN 后近似 trainable | **90.3M** |

该数值是源码结构估算，最终应以训练图变量清单为准。

---

## 4. RankMixer v1 的真实结构

### 4.1 数据流

~~~mermaid
flowchart LR
    A["与 Base 相同三桶"] --> BN["与 Base 相同的三桶 BN"]
    BN --> V["直接拼接 20978 维<br/>没有 SENet"]
    V --> S["标量等宽切分<br/>1311×15 + 1313"]
    S --> P["16 个独立<br/>Dense + bias + GELU"]
    P --> X["X₀: 16×768"]
    X --> R["2 个 hybrid RankMixer block<br/>固定 mixing + per-token FFN"]
    R --> M["Mean pooling<br/>16×768→768"]
    M --> H["Linear 768→1"]
    H --> L["与 Base 相同<br/>first CVR 概率式 BCE"]
~~~

### 4.2 Tokenizer 为什么是直接缺陷

总宽度为 1234 个 17 维字段：

$$
20978=1234\times17.
$$

当前普通分段宽：

$$
1311=77\times17+2.
$$

由于 17 是质数且 $\gcd(2,17)=1$，前 15 个内部边界都不会落在字段边界上：

- 第 5 个 token 跨 Common/Item 域；
- 最后一个 token 含 1075 维 Item 与全部 238 维 Creative；
- 每个 per-token FFN 收到的是被截断和跨域混合的字段片段。

### 4.3 一段 Dense+GELU 不是两层 MLP

第 $t$ 段执行：

$$
x_t=\mathrm{GELU}(z_tW_t+b_t).
$$

只有一次仿射维度变化。它是带非线性的单层投影，不含 hidden bottleneck 或第二个输出投影。

### 4.4 v1 block 与参数量

v1 block：

$$
S=\mathrm{LN}(P(X)+X),
$$

$$
Y=\mathrm{LN}\left(S+\mathrm{PFFN}(\mathrm{LN}(S))\right).
$$

每层三次 LN。$T=16,D=768,k=4,L=2$ 时：

| 部分 | 近似 trainable |
|---|---:|
| 三桶输入 BN | 0.042M |
| Tokenizer | 16.123M |
| 两层 PFFN | 151.118M |
| LN + Head | 0.010M |
| 合计 | **167.3M** |

它约为真实 Base 的 1.85 倍。参数更多但 AUC 更低，说明当前瓶颈不是简单的“容量不够”。

---

## 5. 严格差异归因

### 5.1 已直接证明的差异

| 优先级 | 差异 | 为什么可能影响 0.003 |
|---:|---|---|
| P0 | Base 开 SENet，v1 不开 | v1 缺少已验证的样本级字段选择 |
| P0 | v1 切断字段 | per-token 专属参数无法绑定稳定语义 |
| P0/P1 | Base 有两层 DCNM | Base 在压缩前已有全局乘性交叉 |
| P1 | mean-only readout | token 身份与小域强信号被平均 |
| P1 | hybrid block | 额外 LN 与残差坐标未经干净比较 |
| P1 | 参数 90.3M vs 167.3M | 大量新参数可能在相同训练窗内收敛不足 |

### 5.2 已证明不是差异的项目

| 项目 | Base | v1 | 结论 |
|---|---|---|---|
| 输入域 | Common/Item/Creative | Common/Item/Creative | 相同 |
| Dense/DIN/Gattr | 不使用 | 不使用 | 不能解释差距 |
| 输入 BN | 三桶独立 | 三桶独立 | 基本相同 |
| 主任务 | first only | first only | 相同 |
| last/wide/mlt | 关闭 | 关闭 | 不能解释差距 |
| loss 写法 | sigmoid 后 log_loss | sigmoid 后 log_loss | 都可改，但非差异 |
| logit clip | 有 | 有 | 非差异 |
| grad clipping | 配置存在、未执行 | 配置存在、未执行 | 非差异 |

### 5.3 仍需服务器日志确认

- Base SENet/DCNM/MLP 实际 restore 数量；
- RankMixer tokenizer/PFFN/head 的随机初始化数量；
- 两个 AUC 对应的精确日期、样本数和 checkpoint；
- 两边实际训练 step、处理样本和 wall-clock；
- 运行框架是否对未显式列出的变量做额外 restore。

这些不确定项不能通过源码静态分析伪造结论。

---

## 6. 可修改模块全景

| 类别 | 位于 RankMixer 哪一部分 | 目标 | 方案 |
|---|---|---|---|
| P | 实验与建图前置 | 保证比较只改变一个核心因素 | P1-P3 |
| A | Embedding 到 Tokenizer 之前 | 对齐 Base SENet、归一化和小域选择 | A1-A4 |
| B | Tokenizer / Token 表示 | 构造字段完整、位置稳定的 token | B1-B6 |
| C | Token Mixing / Residual | 定义跨 token 交换和残差坐标 | C1-C5 |
| D | Per-token FFN | 控制非线性、深度、容量和稀疏化 | D1-D6 |
| E | 显式交叉 / 序列扩展 | 对齐 DCNM，研究额外交互与序列 | E1-E5 |
| F | Readout / Heads | 保留 token 身份并管理任务监督 | F1-F6 |
| G | 参数、初始化与系统 | 公平训练并满足服务成本 | G1-G6 |

~~~mermaid
flowchart TD
    X["相同三桶 Embedding"] --> A["A Base-aligned SENet<br/>与输入预处理"]
    A --> B["B Field-safe Tokenizer"]
    B --> C["C Token Mixing"]
    C --> D["D Per-token FFN"]
    D --> F["F Readout / Heads"]

    A --> E["E Base DCNM / 新交互分支"]
    E --> B
    E --> F

    P["P 公平性与正确性"] -.约束.-> A
    P -.约束.-> B
    P -.约束.-> C
    P -.约束.-> D
    G["G 参数与系统"] -.约束.-> C
    G -.约束.-> D
    G -.约束.-> F
~~~

---

## 7. P 类：公平性与正确性前置

### 7.1 它属于 RankMixer 的哪个部分

P 类不改变某个 block，而是约束数据、配置、变量和评估。真实 Base 已表明“代码里存在”不等于“运行时启用”，
所以必须把 resolved 参数和图变量作为一等证据。

### P1：真实运行配置锁定

**修改前。** 仅根据类默认值和其他训练文件推测 Base。

**为什么改。** 这会把关闭的 last/DIN 等功能错误归因给 Base。

**如何改。**

- 将 Base/RM 的完整 resolved args 保存到每个实验产物；
- 对 feature version、日期、采样、batch、优化器、标签和开关逐项 diff；
- 训练开始时输出 active graph summary；
- 每次实验保存 git commit、配置 hash 和 token schema hash。

**好处。** 防止再次把“可选代码”误当“真实运行图”。

**风险。** 只保存启动脚本模板，不保存变量展开后的真实值。

### P2：字段、shape 和 token contract 断言

**修改前。** 只检查 $D\%H=0$，不检查字段覆盖与 $T=H$。

**为什么改。** shape 能成立不代表 token/head 语义正确。

**如何改。**

~~~python
assert T == H
assert D % H == 0
assert len(tokens) == T
assert every_field_appears_exactly_once(groups, feature_config)
assert no_group_splits_embedding_dimension(groups, embedding_size=17)
assert schema_hash == checkpoint_schema_hash
~~~

**好处。** 将静默错误变成建图失败。

**风险。** 只检查字段数量而不检查字段 ID，会漏掉“一处重复、一处遗漏”。

### P3：fused、export 与 restore parity

**修改前。** SwiGLU train 使用 fused 路径，test/export 可能使用另一实现。

**为什么改。** forward 相近不代表梯度和 checkpoint 命名一致。

**如何改。** 比较 forward、input gradient、全部 kernel gradient、save/restore 和 export；覆盖 train/test/export。

**好处。** 避免离线训练有效但导出漂移。

**风险。** zero-init down 时第一步 gate/up 梯度为零是预期现象，不能误判。

---

## 8. A 类：Base-aligned SENet 与输入预处理

### 8.1 它属于 RankMixer 的哪个部分

A 类位于三桶 BN 之后、tokenizer 之前。真实 Base 在这里执行字段级动态选择，而 v1 直接拼接。
这是当前最明确、最低风险的结构差异。

### A1：原样复制真实 Base SENet

**修改前。** v1 完全跳过 SENet。

**为什么改。** Base 运行时明确开启 <code>use_senet=true</code> 和 <code>use_senet_bn=true</code>。

**如何改。**

1. 直接复用 Base <code>senet_layer()</code>；
2. Common gate 只看 Common；
3. Item gate 看 Common+Item；
4. Creative gate 看全部三域；
5. hidden=128、tanh、BN、$2\sigma$ 与 Base 完全一致；
6. SENet 后才进入当前 v1 tokenizer；
7. 首轮保留原 Base scope，验证 checkpoint restore。

**好处。** 这是最干净的“补回真实 Base 差异”实验。

**风险。** 同时修 tokenizer 会无法区分 SENet 与字段完整性的贡献。

### A2：identity-init SENet 改进

**修改前。** Base 用 Glorot 初始化，gate 初始通常靠近 1，但不是严格恒等。

**为什么改。** 大模型接入 gate 时，严格从原输入开始可降低早期扰动。

**如何改。** 在 A1 复现成功后，单独把 gate 最后一层零初始化，保持 $g=2\sigma(0)=1$。

**好处。** 更稳定的冷启和更清晰的增量学习。

**风险。** 这不再是 Base parity；必须使用新 scope，不能冒充 A1。

### A3：保持三桶 BN，不引入虚假输入差异

**修改前。** Base 与 v1 都对三桶独立 BN。

**为什么改。** 该部分不是差距来源，但重构 tokenizer 时容易不小心改变 BN scope 或统计。

**如何改。**

- 首轮完全复用 <code>bn_input_common/item/creative</code>；
- 检查 train/export moving statistics；
- SENet 内的三个 hidden BN 与输入 BN 分开；
- 不在同一次实验增加 tokenizer LN。

**好处。** 保持唯一变量原则和 checkpoint 兼容。

**风险。** 误把 SENet 内 BN 与输入 BN 合并。

### A4：Creative/Coupon 小域隔离

**修改前。** Base 用全域条件 gate 保护 Creative；v1 把 Creative 混入最后一个 Item token。

**为什么改。** 14 个 Creative 字段虽小，但候选相关性强。

**如何改。** 顺序比较：

1. A1 的精确 Creative gate；
2. Creative 单独完整 token；
3. Creative side tower；
4. Creative 生成 Item/token FiLM；
5. Coupon 仅在真实特征存在时单独加入。

**好处。** 防止小域被机械切分与 mean 稀释。

**风险。** 同时走 token、side 和 FiLM 会重复计入并失去归因。

### 8.2 Base-aligned 输入流程

~~~mermaid
flowchart LR
    U["Common BN"] --> GU["Base Common gate"]
    I["Item BN"] --> GI["Base Item gate<br/>conditioned on U+I"]
    C["Creative BN"] --> GC["Base Creative gate<br/>conditioned on U+I+C"]
    GU --> T["Field-safe Tokenizer"]
    GI --> T
    GC --> T
~~~

---

## 9. B 类：Tokenizer 与 token 表示

### 9.1 它属于 RankMixer 的哪个部分

Tokenizer 把 1234 个字段压成 $T$ 个统一宽度 token。PFFN 按 token 位置拥有独立参数，所以字段完整性和
位置稳定性是 RankMixer 成立的前提。

### B1：完整字段语义硬分桶

**修改前。** 20978 个标量等宽切 16 段。

**为什么改。** 15 个内部边界全部切断 17 维字段，且跨域。

**如何改。**

- 以 FeatureConfig 的完整字段列表为原子；
- 先分 Common/Item/Creative，再按业务语义细分；
- 每字段恰好出现一次；
- 低风险 16-token 起点：5 User + 10 Item + 1 Creative；
- 分桶配置版本化并写入 checkpoint；
- 先保留一层 Dense+GELU，避免同时增加 tokenizer 深度。

**好处。** per-token FFN 获得稳定职责。

**风险。** 人工语义组未经实际字段列表审计；不能只按字段数均分后宣称语义分桶。

### B2：投影层选择

**修改前。** 每个切片只有一层 Dense+GELU。

**为什么改。** 超宽业务组可能需要更多融合，但增层会混入容量变化。

**如何改。**

| 版本 | 结构 | 何时使用 |
|---|---|---|
| B2-a | Dense$(d_t,D)$ + GELU | 首个字段安全基线 |
| B2-b | Dense$(d_t,r)$ + GELU + Dense$(r,D)$ | B2-a 明确欠拟合后 |
| B2-c | shared low-rank base + token adapter | 参数或服务压力明显时 |

**好处。** 分离“分桶正确”与“投影更强”。

**风险。** 把单层 Dense 叫 MLP，导致结构和参数描述错误。

### B3：压缩时机与信息宽度

**修改前。** v1 在任何全局可学习交叉前，把 20978 维压成 $16\times768=12288$ 维。

**为什么改。** Base 先在 20978 维做 SENet+DCNM，再压到 2048；v1 过早丢弃约 41.4% 的总宽度。

**如何改。** 比较：

1. SENet → Tokenizer；
2. SENet → Base DCNM → Tokenizer；
3. 提高 $T\times D$，但匹配总参数；
4. 加低秩 global summary token，而不是盲目增大所有 token。

**好处。** 判断差距来自“压缩太早”还是 mixer 不够强。

**风险。** 只增加 $T/D$ 会进一步增大 PFFN 参数。

### B4：Token budget 合同

**修改前。** v1 固定 $T=H=16,D=768$。

**为什么改。** Global/Task/Sequence token 会改变 $T$。

**如何改。** 固定预算内重分配，或同时调整 $T,H,D$，始终保持 $T=H$ 和 $D\%H=0$。

**好处。** 防止 append 后 reshape 仍能跑但语义错误。

**风险。** 只匹配参数，不测 grouped GEMM 的真实效率。

### B5：Soft-to-Hard 可学习分桶

**修改前。** B1 依赖人工字段组。

**为什么改。** 完整字段不代表人工组合最优。

**如何改。**

~~~mermaid
flowchart LR
    F["固定 Field identity"] --> S["全局 Soft assignment"]
    S --> R["覆盖 + 负载 + 熵约束"]
    R --> A["稳定性审计"]
    A --> H["Capacity-aware hardening"]
    H --> Z["冻结 schema 后重训"]
~~~

**好处。** 自动发现跨域组合，hard 化后保持高效。

**风险。** 每样本动态 assignment 会破坏 token 身份；所有字段可能塌缩到少数 token。

### B6：RankUp 表示扩展

**修改前。** 单份 embedding、单一 token 视角。

**为什么改。** 大 PFFN 不保证输入表示有效秩增加。

**如何改。** 独立比较固定字段 permutation、关键字段 Multi-Embedding、预训练 User×Item Cross Token、
Global Token 和 Task Token。

**好处。** 给大主干更丰富、相关性更低的表示原料。

**风险。** 每 step 重新随机、标量级随机或无可靠预训练向量都会破坏假设。

---

## 10. C 类：Token Mixing 与 residual

### 10.1 它属于 RankMixer 的哪个部分

Token Mixing 负责跨 token 交换。原版不是 attention，而是 reshape-transpose-reshape 的固定置换。
residual 决定相加分支是否处于同一语义坐标。

### C1：严格固定 mixing

**修改前。** 只断言 $D\%H=0$。

**为什么改。** $H\ne T$ 时最终 shape 仍可能成立，但不再是论文语义。

**如何改。**

~~~python
assert T == H
assert D % H == 0
x = reshape(x, [B, T, H, D // H])
x = transpose(x, [0, 2, 1, 3])
x = reshape(x, [B, T, D])
~~~

**好处。** 保留零参数、低成本强基线。

**风险。** 固定 mixing 上限有限，但必须在 A/B 修正后再判断。

### C2：strict original block

**修改前。** v1 在 PFFN 前额外加 LN。

**为什么改。** 当前结构既非严格论文，也非成熟公司实现。

**如何改。**

$$
S=\mathrm{LN}(P(X)+X),\qquad
Y=\mathrm{LN}(S+\mathrm{PFFN}(S)).
$$

**好处。** 获得可归因论文基线。

**风险。** 复用旧 LN scope 会混淆 checkpoint。

### C3：company-aligned residual

**修改前。** 直接相加 $P(X)$ 与 $X$。

**为什么改。** shape 相同不代表 token 坐标相同。

**如何改。**

$$
M=P(X),\qquad
Y=M+\mathrm{PFFN}(\mathrm{Norm}(M)).
$$

堆叠后使用 Final Norm。

**好处。** residual 两支处于同一 mixed 坐标。

**风险。** 奇数层输出坐标必须在 flatten readout 前定义清楚。

### C4：Mixing & Reverting

**修改前。** 原版和 aligned 依赖置换坐标。

**为什么改。** 显式逆置换可以恢复原 token 身份再 residual。

**如何改。**

$$
M=P(X)\to\mathrm{PFFN}\to P^{-1}(M)\to +X\to\mathrm{LocalPFFN}.
$$

**好处。** 残差语义最清晰，适合更深网络。

**风险。** 两次 PFFN 会翻倍参数，必须缩 hidden。

### C5：UniMixer-Lite 与 RankElastor

**修改前。** 固定无参置换。

**为什么改。** 稳定基线后可探索可学习局部/全局 mixing。

**如何改。**

- UniMixer-Lite：共享局部基、低秩全局矩阵、Sinkhorn、温度退火；
- RankElastor：Parameterized Full Mixing，并监控 effective rank。

**好处。** 学习更合适的交换模式或扩大坐标覆盖。

**风险。** 两者首轮互斥；理论低秩不等于真实服务高效。

### 10.2 Block 对照

~~~mermaid
flowchart TB
    X["相同 SENet + Tokenizer 输入"] --> V["v1 Hybrid"]
    X --> O["Strict Original"]
    X --> A["Company Aligned"]
    X --> R["Mixing & Reverting"]
    V --> C["固定参数、readout、数据比较"]
    O --> C
    A --> C
    R --> C
~~~

---

## 11. D 类：Per-token FFN 与容量

### 11.1 它属于 RankMixer 的哪个部分

PFFN 是 v1 的主要参数来源，每个 token 使用独立通道权重。它应在 token 身份稳定后再扩容。

### D1：GELU 等参数基线

**修改前。** $D\to4D\to D$，两层共约 151.1M PFFN 参数。

**为什么改。** 当前 RM 已显著大于 Base，容量不是首因。

**如何改。** 保留 GELU，先比较 $k=4$ 与参数匹配的 $k=2$。

**好处。** 分离容量与结构质量。

**风险。** 只比较最终 AUC，不比较收敛速度和 FLOPs。

### D2：等参数 SwiGLU

**修改前。** GELU 权重主项 $2D(4D)=8D^2$。

**为什么改。** SwiGLU 有 gate/value 乘法，但同 hidden 会增参 50%。

**如何改。**

$$
3Dm\approx8D^2
\Rightarrow m\approx\frac{8D}{3}.
$$

$D=768$ 时单次 SwiGLU hidden 取约 2048。两次 SwiGLU 的 Reverting block 还要继续下调。

**好处。** 公平验证非线性门控收益。

**风险。** 直接使用 expansion=4 会把增参收益误归 SwiGLU。

### D3：Pre-Norm、Final Norm 与 small/zero-init

**修改前。** v1 是混合 Pre/Post-LN。

**为什么改。** 大 PFFN 早期可能强烈扰动表示。

**如何改。** 独立比较 Pre-LN/Pre-RMSNorm、Final Norm、small-init down、zero-init down。

**好处。** 提升深层训练稳定性。

**风险。** 一次更换多个 norm 和初始化无法归因。

### D4：Inter-residual 与 auxiliary loss

**修改前。** 两层 v1 不需要深层训练辅助，但扩展到更深后可能不稳。

**为什么改。** 深层 RankMixer 可能出现梯度与有效秩振荡。

**如何改。** 仅在 6 层以上实验 inter-residual 和中间层小权重辅助头。

**好处。** 改善深层可训练性。

**风险。** 辅助权重过大限制深层表示。

### D5：低秩、共享基座和 Student

**修改前。** 全部 token、全部 block 的 FFN 完全独立。

**为什么改。** 参数和服务压力可能超过收益。

**如何改。** 低秩 up/down、业务组共享 base FFN + token adapter、Teacher→Student 蒸馏。

**好处。** 保留异质性同时降低成本。

**风险。** 全共享会破坏 RankMixer 核心参数隔离。

### D6：Sparse MoE

**修改前。** dense PFFN。

**为什么改。** 稳定 dense baseline 后可扩大条件容量。

**如何改。** 分开比较原版 RankMixer ReLU/DTSI 路由与 TokenMixer-Large Sparse-Pertoken MoE。

**好处。** 在稀疏激活下增加专家容量。

**风险。** router 塌缩、通信成本和伪稀疏执行。

---

## 12. E 类：真实 Base DCNM 与新交互扩展

### 12.1 它属于 RankMixer 的哪个部分

E 类负责补充显式全局交叉。这里必须区分“对齐真实 Base 的 DCNM”和“Base 没有的 DIN/序列研究”。

### E1：Base SENet+DCNM 后只替换 MLP

**修改前。** v1 同时删除 SENet、DCNM、MLP，再加入 tokenizer、RankMixer、mean。

**为什么改。** 当前实验一次改了四个结构，无法回答 MLP 与 RankMixer 谁更好。

**如何改。**

~~~text
Base:  BN -> SENet -> DCNM -> MLP -> first head
Swap:  BN -> SENet -> DCNM -> Field-safe Tokenizer -> RM -> Readout -> first head
~~~

SENet、DCNM、输入、loss、优化器全部复用 Base，只替换 DCNM 之后的 MLP。

**好处。** 这是最严格的 backbone 对照。

**风险。** DCNM 输出虽然保持 20978 坐标，但字段组必须严格沿原始坐标边界切。

### E2：DCNM late branch 或 Global Token

**修改前。** E1 把 DCNM 串在 RM 前，成本较高。

**为什么改。** 需要找到显式交叉与 RM 的更高效组合。

**如何改。**

1. 低秩 DCNM → 256 → late fusion；
2. 低秩 DCNM → Global Token；
3. 两者互斥比较；
4. 先做 late branch，不改变 token contract。

**好处。** 保留全局乘性交叉并方便回滚。

**风险。** 两条大塔重复计算；Global Token 会占用 token budget。

### E3：DIN/Sequence 是新能力，不是 Base 恢复

**修改前。** 真实 Base 与 v1 都没有 DIN。

**为什么改。** 行为序列仍可能提高 CVR，但它不能用于解释当前 0.003。

**如何改。** 在核心差距实验完成后，依次比较 DIN late side、DIN Token、短期/长期 Sequence Token。

**好处。** 拓展用户意图建模。

**风险。** 把新输入收益错误归因给 RankMixer；mask、未来泄漏与请求级复用问题。

### E4：MixFormer 深序列

**修改前。** E3 只做一次序列汇总。

**为什么改。** 逐层 Cross Attention 可让高阶 query 多次读取原始历史。

**如何改。** Query Mixer → Cross Attention → Output Fusion，只有 E3 稳定增益后开启。

**好处。** 稠密和序列共同演化。

**风险。** 显存、训练吞吐和在线延迟显著增加。

### E5：User-Item 解耦

**修改前。** candidate-aware 深融合会破坏 User 请求级复用。

**为什么改。** 一个请求多候选时重复计算成本高。

**如何改。** User token/KV 预计算、单向 mask、候选侧轻量融合。

**好处。** 恢复服务可用性。

**风险。** 过度解耦损害 User-Item 交互。

### 12.2 严格 Backbone Swap 图

~~~mermaid
flowchart TD
    X["相同三桶 + BN"] --> S["相同 Base SENet"]
    S --> D["相同 2×DCNM"]
    D --> B["Base 分支<br/>2048→2048→256 MLP"]
    D --> R["Swap 分支<br/>Field-safe Tokenizer→RankMixer"]
    B --> HB["相同 first head/loss"]
    R --> HR["相同 first head/loss"]
~~~

---

## 13. F 类：Readout 与任务

### 13.1 它属于 RankMixer 的哪个部分

F 类位于最后一个 block 之后。真实 Base 的任务对齐是 first-only；恢复 last 不是 parity。

### F1：Mean + low-rank Flatten

**修改前。** v1 只做 mean。

**为什么改。** mean 丢失 token 身份，而 Base MLP 在第一层前保留全部 20978 坐标。

**如何改。**

$$
r=\left[
\frac1T\sum_t h_t,\quad
\mathrm{Proj}_{low-rank}(\mathrm{vec}(H))
\right].
$$

**好处。** 同时保留全局统计和位置身份。

**风险。** flatten 投影过宽造成 head 参数爆炸。

### F2：Grouped / weighted pooling

**修改前。** User、Item、Creative 等权。

**为什么改。** 不同域统计意义不同。

**如何改。** 依次比较分组 mean、静态权重、样本 gate、最后才是 attention pooling。

**好处。** 保护小域和局部峰值。

**风险。** attention readout 重复主干复杂度。

### F3：Creative side fusion

**修改前。** Creative 只在错误的最后 token 内。

**为什么改。** Base 的 Creative gate 使用全域条件，v1 缺少这种保护。

**如何改。** RM readout + Creative side + 可选 DCNM side → 256 fusion → head。

**好处。** 保留候选强信号，路径可独立消融。

**风险。** 与 Creative token 同时启用会重复。

### F4：first-only 是 parity，多任务是研究扩展

**修改前。** 旧文档把 first+last 当作 Base 对齐。

**为什么改。** resolved Base 与 RM 都设置 last/wide/mlt=false。

**如何改。**

- E0-E7 全部只训练 first；
- 只有最佳公平 RM 稳定后，单独增加 last；
- 再比较 multi-task/Task Token；
- 标签窗口与缺失 mask 必须重新审计。

**好处。** 不再把任务增益误当 backbone 增益。

**风险。** 代码默认 <code>enable_last_cvr=true</code>，若忘记显式配置会破坏 parity。

### F5：Pairwise 与 consistency

**修改前。** pointwise BCE。

**为什么改。** AUC 与 BCE 目标不完全一致。

**如何改。** 保留 BCE 主损失，加入 0.02-0.1 query 内 pairwise；多标签存在真实包含关系时再加 consistency。

**好处。** 可能改善排序并保留校准。

**风险。** 纯 pairwise 损害概率校准。

### F6：ESMM

**修改前。** 当前证据不能证明有完整曝光链路。

**为什么改。** ESMM 只有在曝光→点击→转化定义完整时成立。

**如何改。** 先审计全曝光样本与标签，再构造 CTR/CVR/CTCVR。

**好处。** 条件满足时缓解选择偏差。

**风险。** 数据前提不满足时数学关系无效。

---

## 14. G 类：参数、公平训练与系统

### 14.1 它属于 RankMixer 的哪个部分

G 类决定 167M 新主干能否在与 90M Base 相同的时间内公平收敛。

### G1：构造参数匹配 RM

**修改前。** v1 约 167.3M，Base 约 90.3M。

**为什么改。** 参数更多既增加容量，也增加收敛和服务成本。

**如何改。** 保持 $T=H=16,D=768,L=2$，仅将 <code>rm_ffn_expand</code> 从 4 改为 2：

| 配置 | 近似 trainable |
|---|---:|
| Base | 90.3M |
| v1，$k=4$ | 167.3M |
| RM-Matched，$k=2$ | **91.7M** |

**好处。** 几乎不改变 tokenizer/mixing shape，就得到约 1.6% 参数差的公平基线。

**风险。** 参数匹配不等于 FLOPs、内存访问和 kernel 效率匹配。

### G2：checkpoint 加载审计

**修改前。** 两边都提供历史 checkpoint，但 scope 匹配不同。

**为什么改。** Base 可能加载成熟 SENet/DCNM/MLP，而 RM 主干只能随机初始化。

**如何改。**

- 输出每个 scope 的 loaded/missing/random-init 参数；
- 比较都冷启与各自等成熟度热启；
- 记录 checkpoint 日期和训练累计样本；
- 不根据配置字符串猜 restore 结果。

**好处。** 分离结构能力与初始化成熟度。

**风险。** “restore 成功”不代表所有核心变量都加载。

### G3：分组学习率与蒸馏

**修改前。** 新 RM 与成熟 embedding 可能用相同 dense LR。

**为什么改。** 新主干需要更快学习，旧 embedding 需要保护。

**如何改。**

| 参数组 | 相对 LR |
|---|---:|
| 新 tokenizer/RM/readout | 1.0 |
| 热启 SENet/DCNM | 0.2-0.5 |
| 热启 sparse embedding | 0.1 或原 sparse optimizer |

再用真实 Base logits 做 teacher，temperature 1-2，蒸馏权重从 0.5 退火到 0.1。

**好处。** 缩短追平时间。

**风险。** teacher 与 student 样本或标签定义不一致。

### G4：共同训练缺陷应对两边同时修

**修改前。** Base 与 RM 都使用 sigmoid 后 log_loss、clip logits、配置了但未执行的 grad clipping。

**为什么改。** 这些会影响数值稳定性，但不是当前差异来源。

**如何改。** 新建 Base-clean 与 RM-clean：

- raw logits 进入 <code>sigmoid_cross_entropy_with_logits</code>；
- 预测单独 sigmoid；
- loss 前不 clip logits；
- 真正执行 global-norm clipping；
- dropout/use_rankmixer 等开关增加图结构测试。

**好处。** 改善两边共同的训练质量，同时保持公平。

**风险。** 只修 RM 会把公共修复误认为 RankMixer 增益。

### G5：fused/grouped kernel 与服务

**修改前。** per-token 小 GEMM 可能产生大量 kernel launch。

**为什么改。** 理论并行性要靠 fused/grouped GEMM 才能实现。

**如何改。** 数值 parity 后启用 fused SwiGLU、grouped GEMM；更大模型再评估 Token Parallel、FP8 和 MoE kernel。

**好处。** 降低端到端延迟与显存。

**风险。** 单算子 microbenchmark 不能代表完整训练与服务。

### G6：完整可观测性

**修改前。** 只看总体 AUC。

**为什么改。** 需要区分效果、校准、收敛和成本。

**如何改。** 固定报告：

- AUC/GAUC、PR-AUC、LogLoss、COPC、ECE；
- 多 seed 或 paired bootstrap；
- User/Item/Creative、冷启动、长尾切片；
- loaded/random-init 参数；
- params、FLOPs、samples/s、peak memory；
- P50/P95/P99 与多候选请求成本。

**好处。** 能解释“为什么有效”和“是否能上线”。

**风险。** 单时间窗和单 seed 的结论不稳。

---

## 15. 方案组合与冲突

### 15.1 自然主线

~~~text
P 真实运行锁定
 -> A1 精确 Base SENet
 -> B1 字段安全 tokenizer
 -> F1 双读出
 -> G1 参数匹配
 -> C2/C3 block 比较
 -> E1 严格 Backbone Swap
 -> D2 等参数 SwiGLU
 -> G2/G3 热启与蒸馏
~~~

### 15.2 互斥或高度重叠

| 选择组 | 原因 |
|---|---|
| Base SENet vs identity-init SENet | 后者是改进，不是 parity |
| 人工硬桶 vs Soft-to-Hard vs permutation | 都改变字段到 token 的映射 |
| strict vs aligned vs reverting | 都改变 residual |
| fixed vs UniMixer vs RankElastor | 都替换 mixing |
| GELU vs SwiGLU vs MoE | 都改变 PFFN |
| DCNM pre-RM vs late branch vs Global Token | 都改变显式交叉路径 |
| mean+flat vs attention readout | 都改变输出信息通路 |

### 15.3 不能用于解释当前差距的扩展

- Dense/DIN/Gattr；
- last/wide/multi-task；
- MixFormer；
- ESMM；
- RankUp、UniMixer、RankElastor；
- Sparse MoE。

这些可以提高未来模型，但必须在真实 Base 差距归因完成后独立研究。

---

## 16. 四套重构方案

### 16.1 方案 A：RM-GateParity

**架构。**

~~~text
三桶 -> 相同 BN -> 完全相同 Base SENet
-> 保留当前 v1 tokenizer/block/mean
-> first head
~~~

**回答。** 0.003 中有多少来自缺失 SENet？

**好处。** 修改最小、证据最直接。

**限制。** tokenizer 仍切断字段。

### 16.2 方案 B：RM-FieldSafe

**架构。**

~~~text
三桶 -> Base SENet
-> 完整字段语义 tokenizer
-> strict RankMixer
-> mean + low-rank flatten
-> first head
~~~

**回答。** 在不依赖 DCNM 时，正确构造的纯 RankMixer 能否追回 Base？

**好处。** 保留 RankMixer 的纯模型定位。

**限制。** 与 Base 相比仍缺少早期显式乘性交叉。

### 16.3 方案 C：RM-BackboneSwap

**架构。**

~~~text
三桶 -> 相同 BN -> 相同 Base SENet -> 相同 2×DCNM
-> 完整字段 tokenizer -> 参数匹配 RankMixer
-> 双读出 -> first head
~~~

**回答。** 当所有上游模块一致时，RankMixer 是否优于 Base MLP？

**好处。** 最严格的 backbone 因果对照。

**限制。** 成本包含 DCNM，未体现纯 RM 的极简结构。

### 16.4 方案 D：RM-AlignedHybrid

**架构。**

~~~text
方案 B
+ identity-init SENet 消融
+ company-aligned 等参数 SwiGLU
+ 小型 DCNM late branch
+ Base teacher distillation
~~~

**回答。** 在公平基础上，成熟 RankMixer 设计能否稳定超过 Base？

**好处。** 是近期上限候选。

**限制。** 必须由 A/B/C 逐步演进，不能一次全开。

### 16.5 演进关系

~~~mermaid
flowchart LR
    B0["真实 Base<br/>SENet+DCNM+MLP"] --> A["A GateParity"]
    A --> B["B FieldSafe Pure RM"]
    A --> C["C BackboneSwap"]
    B --> D["D AlignedHybrid"]
    C --> D
    D --> R1["序列 / 多任务"]
    D --> R2["Learned Mixing"]
    D --> R3["Student / MoE"]
~~~

---

## 17. 统一伪代码

### 17.1 严格实验容器

~~~python
def build_shared_frontend(features, mode, cfg):
    common, item, creative = lookup_same_three_domains(features)
    common = base_input_bn(common, mode, scope="bn_input_common")
    item = base_input_bn(item, mode, scope="bn_input_item")
    creative = base_input_bn(creative, mode, scope="bn_input_creative")

    if cfg.senet == "base_exact":
        common, item, creative = base_hierarchical_senet(
            common, item, creative,
            hidden=128,
            use_senet_bn=True,
        )
    elif cfg.senet == "identity_variant":
        common, item, creative = identity_init_hierarchical_senet(
            common, item, creative
        )
    return common, item, creative


def base_forward(parts, cfg):
    x = concat(parts)
    x = exact_base_dcnm(x, layers=2, bottleneck=500)
    x = exact_base_mlp(x, layers=[2048, 2048, 256])
    return linear_head(x)


def rankmixer_forward(parts, cfg):
    x = concat(parts)
    if cfg.keep_base_dcnm:
        x = exact_base_dcnm(x, layers=2, bottleneck=500)

    groups = load_versioned_complete_field_groups(cfg.schema)
    assert_field_coverage(groups)
    tokens = project_complete_groups(x, groups, D=cfg.D)
    assert_token_contract(tokens, cfg.T, cfg.H, cfg.D)

    h = rankmixer_stack(
        tokens,
        topology=cfg.topology,
        pffn=cfg.pffn,
        expansion=cfg.k,
    )
    r = concat([group_or_mean_pool(h), low_rank_flatten(h)])
    r = optional_fuse(r, dcnm_late_branch(parts, cfg))
    return linear_head(r)
~~~

### 17.2 公平 loss 与优化

~~~python
# Parity 阶段只允许 first 标签
base_logit = base_forward(shared_parts, cfg_base)
rm_logit = rankmixer_forward(shared_parts, cfg_rm)

base_loss = raw_logit_bce(base_logit, label_first)
rm_loss = raw_logit_bce(rm_logit, label_first)

assert_same_data_and_label_pipeline()
assert_resolved_config_diff_is_allowlisted()
report_loaded_and_random_initialized_parameters()
report_params_flops_samples_and_wall_clock()
~~~

---

## 18. 推荐实验阶梯

| 实验 | 唯一主要变化 | 目的 |
|---|---|---|
| E0 | 原样 Base 与 v1，多 seed | 固定 0.865/0.862 与方差 |
| E1 | v1 + exact Base SENet | SENet 缺失贡献 |
| E2 | E1 + field-safe tokenizer | 标量切分损失 |
| E3 | E2 + low-rank flatten | mean readout 瓶颈 |
| E4 | 固定 E3 比较 v1/strict/aligned | block 拓扑贡献 |
| E5 | 最佳结构将 $k=4\to2$ | 90M 等参数公平性 |
| E6 | Base 保持不变；DCNM 后 MLP→RM | 严格 backbone swap |
| E7 | 最佳 GELU→等参 SwiGLU | 门控非线性贡献 |
| E8 | DCNM pre/late/global 三选一 | 显式交叉组合 |
| E9 | Base teacher distillation | 收敛迁移 |
| E10+ | DIN、多任务、RankUp、learned mixing、MoE | 新能力研究 |

资源阶梯：

~~~text
图构建与 1k-step 数值检查
 -> 固定 1-2 日小窗
 -> 5 日中窗
 -> 最佳 1-2 个完整训练窗
 -> 不重叠时间窗复测
 -> 在线灰度
~~~

### 18.1 决策流程

~~~mermaid
flowchart TD
    A["E0 是否复现？"] -->|否| X["修数据、配置、restore"]
    A -->|是| B["E1 SENet 是否追回？"]
    B --> C["E2/E3 Tokenizer+Readout"]
    C --> D{"是否接近 Base？"}
    D -->|否| E["E6 严格 Backbone Swap"]
    D -->|是| F["E4/E5 Block+参数匹配"]
    E --> F
    F --> G{"公平版本是否超过 Base？"}
    G -->|是| H["SwiGLU / KD / 新能力"]
    G -->|否| I["定位 DCNM、压缩时机和收敛"]
~~~

---

## 19. 验收清单

### 19.1 Base 事实

- [ ] 唯一 Base 文件是 <code>cvr_bn_senet_dcnm.py</code>；
- [ ] resolved Base 参数归档；
- [ ] use_senet/use_senet_bn 为 true；
- [ ] last/wide/mlt/delay 为 false；
- [ ] 输入只有 common/item/creative；
- [ ] <code>cvr_fst_last_norpy.py</code> 未被用于 Base 证明。

### 19.2 公平性

- [ ] feature version、日期、采样、batch、优化器一致；
- [ ] first-only 标签一致；
- [ ] Base 与 RM 的公共训练清理同时启用；
- [ ] loaded/missing/random-init 参数按 scope 报告；
- [ ] 参数匹配和原 167M 两套结果都报告；
- [ ] 一次只改变一个核心模块。

### 19.3 RankMixer 正确性

- [ ] 每字段恰好进入一个组；
- [ ] 无 17 维字段被切断；
- [ ] token schema hash 与 checkpoint 一致；
- [ ] $T=H$、$D\%H=0$；
- [ ] fused/unfused forward、gradient、restore parity；
- [ ] readout 坐标语义明确。

### 19.4 效果与成本

- [ ] AUC/GAUC、PR-AUC、LogLoss、COPC、ECE；
- [ ] 多 seed/paired bootstrap；
- [ ] 冷启动、长尾、Creative 等切片；
- [ ] params、FLOPs、samples/s、peak memory；
- [ ] checkpoint 大小与恢复耗时；
- [ ] P50/P95/P99。

---

## 20. 源码定位

| 事实/模块 | 真实源码位置 |
|---|---|
| Base resolved 参数 | <code>set-x.args.resolved.txt:27-76</code> |
| RM resolved 模板 | <code>code/set-xcal.txt:248-298</code> |
| Base 配置默认值 | <code>cvr_bn_senet_dcnm.py:150-228</code> |
| Base labels/loss | <code>cvr_bn_senet_dcnm.py:461-513,527-652</code> |
| Base SENet | <code>cvr_bn_senet_dcnm.py:830-905</code> |
| Base 三桶路径 | <code>cvr_bn_senet_dcnm.py:907-980</code> |
| Base DCNM | <code>cvr_bn_senet_dcnm.py:783-828,982-983</code> |
| Base MLP/head | <code>cvr_bn_senet_dcnm.py:988-1016,1097-1108</code> |
| RM 三桶/BN | <code>cvr_bn_rankmixer_v1.py:889-930</code> |
| RM tokenizer | <code>cvr_bn_rankmixer_v1.py:774-799,938-962</code> |
| RM block | <code>cvr_bn_rankmixer_v1.py:801-862</code> |
| RM mean/head | <code>cvr_bn_rankmixer_v1.py:864-980</code> |
| RM first loss | <code>cvr_bn_rankmixer_v1.py:409-420</code> |
| 工业 gate/token 参考 | <code>commend_cvr.py:2243-2500</code> |
| SwiGLU/fused 参考 | <code>mlp_mixer_swiglu_fuse.py:90-304,339-386</code> |

<code>cvr_bn_rankmixer_v1.py:960</code> 的注释写“无 bias”，但
<code>_rm_tokenize_bucket()</code> 在 791 行实际创建了零初始化 bias，应以图为准修正文档和注释。

---

## 21. 延伸文档

- 快速选型：[CVR_RankMixer_v1_改进方案汇总与选型指南.md](CVR_RankMixer_v1_改进方案汇总与选型指南.md)
- Corrected AUC 诊断：[CVR_RankMixer_v1_AUC差距诊断与改进路线.md](CVR_RankMixer_v1_AUC差距诊断与改进路线.md)
- 输入/监督方案设计：[CVR_RankMixer_四套改进方案设计.md](CVR_RankMixer_四套改进方案设计.md)
- SwiGLU 源码对比：[CVR_RankMixer_SwiGLU_源码分析与原版对比.md](CVR_RankMixer_SwiGLU_源码分析与原版对比.md)
- RankMixer 演进调研：[RankMixer及其演进方法详细调研.md](RankMixer及其演进方法详细调研.md)
- TokenMixer-Large：[../tokenmixer/TokenMixer-Large_论文详解.md](../tokenmixer/TokenMixer-Large_论文详解.md)
- MixFormer：[../mixformer/MixFormer_论文详解.md](../mixformer/MixFormer_论文详解.md)
- RankUp：[../RankUp/RankUp_论文详解.md](../RankUp/RankUp_论文详解.md)
- UniMixer：[../UniMixer/UniMixer_论文详解.md](../UniMixer/UniMixer_论文详解.md)
- RankElastor：[../RankElastor/RankElastor_论文详解.md](../RankElastor/RankElastor_论文详解.md)

---

## 22. 最终建议

1. 不再用 dense、DIN、gattr、last 解释当前 0.003；
2. 第一优先级是 exact Base SENet、字段安全 tokenizer 和双读出；
3. 用 $k=2$ 构造约 91.7M 参数匹配 RM；
4. 用“保留 SENet+DCNM，只替换 MLP”的方案做最严格 backbone 对照；
5. Base 与 RM 共同存在的 loss/clip/grad 问题必须对两边同时修；
6. 序列、多任务、RankUp、learned mixing 和 MoE 放到追回真实 Base 之后；
7. 所有结论必须同时报告效果、收敛、restore 和服务成本。

一句话总结：

> 真实 Base 的优势不是更多输入或更多任务，而是压缩前的层级字段门控与全局乘性交叉；RankMixer 应先对齐这两点，再证明自身主干价值。
