# CVR RankMixer v1 改进方案汇总与选型指南：真实 Base 修正版

> 真实 Base：<code>code/cvr_bn_senet_dcnm.py</code>。<br>
> 当前 RankMixer：<code>code/cvr_bn_rankmixer_v1.py</code>。<br>
> 运行证据：Base 使用 <code>set-x.args.resolved.txt</code> / <code>0721args.txt</code>；RankMixer 使用 <code>code/set-xcal.txt</code>。<br>
> 实验背景：Base AUC 约 0.865，RankMixer v1 AUC 约 0.862。<br>
> 详细证据与全部方法卡：[全量改进方案](CVR_RankMixer_全量改进方案_按模块分类.md)；专项归因：[0.003 AUC 差距诊断](CVR_RankMixer_v1_AUC差距诊断与改进路线.md)。

> 边界修正：<code>code/cvr_fst_last_norpy.py</code> 不是 Base，只是另一种训练尝试。它可以提供 DIN、dense、gattr、last 等研究实现，但不能用于解释本次 Base 与 RankMixer 的 AUC 差距。

---

## 1. 一页结论

真实运行图表明，Base 与 RankMixer v1 实际上具有相同的输入域和主任务：

- 都只使用 Common/User、Item、Creative 三桶稀疏 embedding；
- 都先对三桶分别做 BatchNorm；
- 都使用 first CVR 单任务；
- wide、multi-task、last CVR、delay train 在两次实际运行中都关闭；
- optimizer、学习率、batch size、embedding size 和特征版本基本一致。

因此，以下旧结论全部撤销：

| 旧判断 | 真实结论 |
|---|---|
| v1 删除了 Base 的 dense、DIN、gattr | 错；真实 Base 运行图也没有使用它们 |
| v1 删除了 Base 的 last 辅助任务 | 错；真实 Base resolved 参数关闭 last |
| 先恢复更多输入和多任务才能公平对比 | 错；这会引入 Base 没有的新变量 |
| 0.003 主要来自任务/输入缺失 | 无证据；差异集中在门控、交叉、tokenizer、主干与读出 |

当前最可信的差距来源是：

1. Base 实际开启了层级 SENet，v1 没有；
2. v1 用等宽标量分段切断全部内部 17 维字段边界；
3. Base 在 20978 维原空间做两层 DCNM 乘性交叉，v1 直接压成 16×768；
4. v1 最终只做 mean pooling，主动抹掉 token 身份；
5. v1 的 hybrid RankMixer block 不是严格论文结构，也不是成熟 SwiGLU 结构；
6. v1 约 167.3M 参数，Base 约 90.3M；新增大主干的初始化和训练预算并不公平。

首选实验顺序不是“一次堆满所有增强”，而是：

~~~text
E0 复现 Base 与 v1，多 seed
 -> E1 v1 + 完全相同的 Base SENet
 -> E2 E1 + 字段安全 tokenizer
 -> E3 E2 + mean/分组池化 + 低秩 flatten 双读出
 -> E4 E3 + 严格 RankMixer block
 -> E5 E4 + 参数匹配（优先 k: 4 -> 2）
 -> E6 Base 的 SENet+DCNM 不变，只用 RankMixer 替换 Base MLP
 -> E7 再比较 aligned SwiGLU、蒸馏和研究扩展
~~~

---

## 2. 真实 Base 与 v1 的严格结构对照

### 2.1 实际运行配置

| 项目 | 真实 Base | RankMixer v1 | 差异 |
|---|---:|---:|---|
| 输入域 | Common + Item + Creative | Common + Item + Creative | 无 |
| Embedding size | 17 | 17 | 无 |
| Batch size | 2048 | 2048 | 无 |
| Learning rate | 2e-5 | 2e-5 | 无 |
| Optimizer | flood_adam | flood_adam | 无 |
| Feature version | cvr_fea_v10_base_cold | cvr_fea_v10_base_cold | 无 |
| 分域 BN | 开 | 开 | 无 |
| 层级 SENet | 开，且 SENet BN 开 | 关 | 核心差异 |
| 显式交叉 | 2×DCNM，bottleneck 500 | 无 | 核心差异 |
| 压缩主干 | 2048→2048→256 MLP | 16×768，2×RankMixer | 核心差异 |
| Readout | MLP 保留坐标后压缩 | 16 token mean | 核心差异 |
| first CVR | 开 | 开 | 无 |
| last / wide / MLT / delay | 全关 | 全关 | 无 |

Base 的类默认值并不等于实际训练值。必须优先信任 resolved args，再沿已启用分支读源码。

### 2.2 两条端到端数据流

~~~mermaid
flowchart LR
    X["同一三桶稀疏 Embedding"] --> B1["三桶独立 BN"]
    B1 --> S["层级 SENet<br/>Common→Item→Creative"]
    S --> D["2×DCNM<br/>20978↔500"]
    D --> M["MLP<br/>20978→2048→2048→256"]
    M --> HB["first head"]

    X --> B2["三桶独立 BN"]
    B2 --> C["concat 20978"]
    C --> P["等宽标量分段<br/>1311×15+1313"]
    P --> T["16×独立 Dense+GELU<br/>→16×768"]
    T --> R["2×hybrid RankMixer"]
    R --> A["mean pooling"]
    A --> HR["first head"]
~~~

### 2.3 真实 Base 的层级 SENet

三桶宽度分别是：

$$
d_U=385\times17=6545,
\quad d_I=835\times17=14195,
\quad d_C=14\times17=238.
$$

Base 先将每域恢复成字段矩阵，再沿 embedding 维求均值：

$$
s_U=\operatorname{Mean}(U,\text{axis}=17),
\quad s_I=\operatorname{Mean}(I,\text{axis}=17),
\quad s_C=\operatorname{Mean}(C,\text{axis}=17).
$$

随后使用三级条件 gate：

$$
g_U=2\sigma(W_{U2}\tanh(BN(W_{U1}s_U))),
$$

$$
g_I=2\sigma(W_{I2}\tanh(BN(W_{I1}[s_U;s_I]))),
$$

$$
g_C=2\sigma(W_{C2}\tanh(BN(W_{C1}[s_U;s_I;s_C]))).
$$

最后将每个字段的标量 gate 广播回 17 维 embedding。Common 自条件、Item 由 Common+Item 条件化、Creative 由三域共同条件化。这是 v1 完全缺失的样本级字段选择能力。

### 2.4 真实 Base 的 DCNM

对拼接向量 $x_0\in\mathbb R^{20978}$，每层近似为：

$$
z_l=W_{down,l}x_l+b_{down,l}\in\mathbb R^{500},
$$

$$
u_l=W_{up,l}z_l+b_{up,l}\in\mathbb R^{20978},
$$

$$
x_{l+1}=LN(x_l+x_0\odot u_l).
$$

实际运行使用两层，且 cross activation 关闭。它不是普通 MLP：乘法项让每一坐标在压缩前就能被全局条件调制。

---

## 3. v1 Tokenizer 的精确拆解

### 3.1 不是“两层 MLP”，而是单层投影

第 $t$ 个分段执行：

$$
h_t=\operatorname{GELU}(x_tW_t+b_t),
\quad W_t\in\mathbb R^{d_t\times768}.
$$

其维度变化只有一次：

~~~text
[B, d_t]
 -> Dense(W_t, b_t)
 -> [B, 768]
 -> GELU
 -> [B, 768]
~~~

因此它是“带 bias 和 GELU 的单层全连接投影”，不是具有 hidden layer 的两层 MLP。每段拥有独立矩阵，16 段输出堆叠成 16 个 token。

### 3.2 “每桶”准确说法是什么

当前 v1 中的“桶”不是业务字段桶，而是 flatten 后的等宽标量段：

~~~text
20978 = 1311×15 + 1313
~~~

由于 $1311\bmod17=2$，15 个内部边界全部落在 17 维字段内部。它还会产生两个跨域 token：

- Common 结束位置 6545 落在第 5 段内部；
- 最后一段包含约 1075 维 Item 尾部和全部 238 维 Creative。

所以“每桶 Dense+GELU 压成一个 token”的严格含义是：

> 每个等宽标量分段各用一套独立 Dense+bias 做一次维度投影，再过 GELU，得到一个 768 维 token；它不是语义桶，也不是两层 MLP。

### 3.3 为什么会有不可逆的信息瓶颈

Tokenizer 总输出宽度为：

$$
16\times768=12288.
$$

输入宽度为 20978，所以第一次进入 Mixer 前已经减少：

$$
20978-12288=8690,
\quad \frac{8690}{20978}\approx41.4\%.
$$

这不意味着一定损失 41.4% 的有效信息，但说明 Base 在原空间完成 SENet+DCNM 后才压缩，而 v1 先完成一个强且不可逆的随机初始化压缩。

---

## 4. 改进模块地图

| 编号 | 模块 | 修改位置 | 回答的问题 | 优先级 |
|---|---|---|---|---:|
| M0 | 运行与评估对齐 | 实验入口 | 0.003 是否稳定、配置是否同源 | P0 |
| M1 | 精确 Base SENet | BN 后 | 缺少动态字段选择贡献多少 | P0 |
| M2 | 字段安全 tokenizer | SENet 后 | 等宽切字段是否是主因 | P0 |
| M3 | 双读出 | Mixer 后 | mean 是否抹掉关键信号 | P1 |
| M4 | 严格 block 对照 | Mixer 内 | hybrid 拓扑是否拖累 | P1 |
| M5 | 参数匹配 | Mixer 内 | 大参数冷启动是否不公平 | P1 |
| M6 | 严格 backbone swap | DCNM 后 | RM 能否替代 Base MLP | P0/P1 |
| M7 | aligned SwiGLU | Mixer 内 | 成熟块设计能否增益 | P2 |
| M8 | 热启/蒸馏 | 优化层 | 新主干收敛是否不足 | P1/P2 |
| M9 | DIN/序列 | 新输入分支 | 新能力能否超过 Base | P2 研究 |
| M10 | 多任务/Task Token | 监督层 | 新监督能否继续提升 | P2 研究 |
| M11 | Soft-to-Hard/MoE | Tokenizer/FFN | 长期结构搜索 | P3 研究 |

P0 表示归因和正确性前置，P1 表示近期候选，P2/P3 是追回真实 Base 后再研究。

---

## 5. M0：先建立可信的公平基线

修改前：只比较一次 Base 0.865 与 v1 0.862，容易把 seed、checkpoint restore 和训练预算混入架构结论。

如何改：

- 固定同一份 feature version、日期、采样、label delay 和过滤；
- 归档两侧完整 resolved args，而非只保存手写启动脚本；
- 至少 3 个 seed，报告均值、标准差和最差值；
- 保存每个 scope 的 loaded、missing、random-init 参数数量；
- 对齐 step 数、seen examples、wall-clock 和早停规则；
- 同时报 AUC、GAUC、logloss、校准、分桶 AUC 和推理延迟；
- Base 与 RM 共同的 loss/clip/grad 清理必须双边同时启用。

收益：让后续每一个结构改动只回答一个问题。

风险：若没有服务器 restore 日志，仅凭静态代码不能断言 checkpoint 实际加载比例。

---

## 6. M1：原样复制 Base SENet

修改前：v1 在三桶 BN 后直接 concat。

如何改：第一轮直接复用 Base 的 <code>senet_layer()</code>，保持 hidden=128、tanh、SENet BN、$2\sigma$、三级条件关系和 scope 命名完全一致。

~~~python
u = domain_bn(common)
i = domain_bn(item)
c = domain_bn(creative)

u, i, c = exact_base_senet(u, i, c)
x = concat([u, i, c])
tokens = current_v1_tokenizer(x)
~~~

收益：这是最干净的 Base 差异消融，也可能复用历史 gate 权重。

风险：identity-init SENet、RMSNorm 或换激活都属于后续改进，不能与“精确对齐 Base”混在同一实验。

---

## 7. M2：字段安全 Tokenizer

修改前：20978 个 scalar 被切成 16 个近似等宽段，内部边界全部切字段，并跨域混合。

如何改：

1. 先 reshape 成 $[B,F,17]$；
2. 仅沿字段轴分组，字段 17 维不可拆；
3. 先保证 U/I/C 域不跨组；
4. 用版本化配置记录每个 field id 到 token id 的映射；
5. 每个语义组仍只做一次独立 Dense+GELU，避免同时增加网络深度；
6. assert 每个字段恰好出现一次。

一个 16-token 的起始方案：

~~~text
Common:   5 个完整字段组
Item:    10 个完整字段组
Creative: 1 个完整字段组
总计:    16 tokens
~~~

~~~python
def field_safe_tokenize(u_fields, i_fields, c_fields, groups):
    all_ids = []
    tokens = []
    for group in groups:
        assert group.domain in {"U", "I", "C"}
        fields = gather_domain_fields(group.domain, group.field_ids)
        flat = reshape(fields, [batch_size, -1])
        tokens.append(gelu(dense(flat, 768, scope=group.name)))
        all_ids.extend(group.global_field_ids)
    assert sorted(all_ids) == list(range(1234))
    assert len(set(all_ids)) == 1234
    return stack(tokens, axis=1)
~~~

收益：token 位置拥有稳定业务职责，Per-token FFN 才能形成可靠专门化。

风险：业务字段表变更必须升级 schema 版本；新旧 tokenizer checkpoint 不应静默混用。

---

## 8. M3：Mean + 身份保留读出

修改前：

$$
r_{mean}=\frac1T\sum_{t=1}^Th_t.
$$

它对 token 置换近似不敏感，Creative 等小域信息会被 1/16 稀释。

如何改：并行保留全局统计和 token 身份。

~~~python
mean_repr = reduce_mean(h, axis=1)            # [B, D]
group_repr = concat([
    reduce_mean(h[:, user_ids, :], axis=1),
    reduce_mean(h[:, item_ids, :], axis=1),
    h[:, creative_id, :],
], axis=-1)
flat_low_rank = dense(reshape(h, [B, T * D]), 256)
repr = concat([mean_repr, group_repr, flat_low_rank], axis=-1)
~~~

收益：mean 提供稳健全局统计，group/flatten 分支保留域和位置身份。

风险：直接将 16×768 flatten 接大 MLP 会再次造成参数爆炸；第一轮只用低秩 128/256 维投影。

---

## 9. M4：严格比较三种 RankMixer block

当前 v1 是 hybrid：mixing residual 后 Post-LN，PFFN 前额外 Pre-LN，再在 residual 后 Post-LN。它既不是论文原版，也不是成熟 aligned SwiGLU。

需要固定 tokenizer、D、L、k、readout，只替换 block：

### 9.1 v1 hybrid

~~~text
s = LN(P(x) + x)
y = LN(PFFN(LN(s)) + s)
~~~

### 9.2 严格论文式对照

~~~text
s = LN(P(x) + x)
y = LN(PFFN(s) + s)
~~~

### 9.3 Mixing & Reverting

~~~text
m = P(x)
m = PFFN(m)
x = x + P^-1(m)
y = x + LocalFFN(Norm(x))
~~~

收益：把“固定 mixing 本身”与“残差坐标、归一化拓扑”分开归因。

风险：$P(x)$ 与 $x$ shape 相同不代表 token 语义相同；任何 permutation 改动都需要可逆性和奇偶层测试。

---

## 10. M5：参数匹配，而不是继续放大

按主要可训练权重估算：

| 模型 | 主要参数量 |
|---|---:|
| 真实 Base SENet | 约 0.52M |
| 真实 Base 2×DCNM | 约 41.96M |
| 真实 Base 2048→2048→256 MLP | 约 47.68M |
| 真实 Base 合计 | 约 90.3M |
| 当前 RM v1，k=4 | 约 167.3M |
| RM v1，k=2 | 约 91.7M |

最简单的公平版本保持 $T=H=16,D=768,L=2$，只把 PFFN expansion 从 $k=4$ 改成 $k=2$。

收益：参数约为 Base 的 1.016 倍，可回答“结构优劣”而不是“容量和冷启动差异”。

风险：参数相等不等于 FLOPs、显存、kernel 效率或收敛速度相等；这些仍需单独报告。

---

## 11. M6：最严格的 Backbone Swap

这是整个路线中最有解释力的实验：Base 的输入、BN、SENet、DCNM、head、loss 和训练配置全部不变，只替换 DCNM 后面的 MLP。

~~~mermaid
flowchart LR
    X["三桶输入"] --> BN["相同 BN"]
    BN --> S["相同 Base SENet"]
    S --> D["相同 2×DCNM"]
    D --> A["Base 分支<br/>2048→2048→256 MLP"]
    D --> B["实验分支<br/>字段安全 RM + 双读出"]
    A --> H1["相同 first head/loss"]
    B --> H2["相同 first head/loss"]
~~~

伪代码：

~~~python
x = exact_base_input_and_bn(features)
x = exact_base_senet(x)
x = exact_base_dcnm(x)

if backbone == "base_mlp":
    repr = base_mlp(x)
elif backbone == "rankmixer":
    tokens = field_safe_tokenize_after_dcnm(x)
    h = parameter_matched_rankmixer(tokens)
    repr = dual_readout(h)

logit = identical_first_head(repr)
loss = identical_first_loss(logit, label)
~~~

收益：上游显式交叉完全相同后，差异才能主要归因于 MLP 与 RankMixer 的表示能力。

风险：DCNM 输出仍是 20978 维，后续字段分组必须保留与原字段坐标一致的切片协议。

---

## 12. M7：Aligned SwiGLU 作为后续增强

SwiGLU：

$$
\operatorname{SwiGLU}(x)=
W_d\left(\operatorname{SiLU}(W_gx+b_g)\odot(W_ux+b_u)\right)+b_d.
$$

它与普通两矩阵 GELU FFN 的联系是都做升维、非线性、降维；区别是 SwiGLU 多一条 gate 投影并进行逐元素乘法。要公平对比，不能直接保留相同 hidden size。

建议：

- 普通 GELU hidden 为 $4D$ 时，单次 SwiGLU hidden 先取约 $8D/3$；
- 如果一个 block 含两次 SwiGLU，再下调 hidden 以匹配总参数；
- FFN 输入使用 Pre-Norm；
- residual 必须处于相同 token 坐标；
- down projection small-init 或 zero-init 要单独消融；
- fused 与 unfused 实现需要前向、输入梯度、三组权重梯度 parity 测试。

收益：更强的通道级门控和成熟工业 kernel 潜力。

风险：同时换 activation、norm、residual、初始化和 fused kernel 会完全失去归因。

---

## 13. M8：热启动与蒸馏

静态代码只能确认两侧都提供 checkpoint import/auto-load，不能证明实际加载数量。Base 的 SENet/DCNM/MLP scope 更可能匹配历史模型，RM 主干是新增 scope。

实施顺序：

1. 从 restore 日志统计每个 scope 的 loaded/missing 参数；
2. tokenizer projection 可用分桶内旧 embedding 统计做初始化；
3. 对新 RM 主干使用较大学习率，对复用 embedding/gate 使用较小学习率；
4. 先做结构对齐，再用真实 Base logits 做 teacher；
5. temperature 1–2，蒸馏权重从 0.5 退火到 0.1；
6. 同时保留 hard-label BCE，不能只拟合 teacher。

收益：减少 167M/92M 新主干在固定训练窗内的收敛劣势。

风险：KD 能追回 teacher，不证明 RankMixer 结构本身优于 Base；必须保留无 KD 对照。

---

## 14. 三种主推方案

### 14.1 方案 A：RM-GateParity

目标：用最小改动验证 Base SENet 的贡献。

~~~mermaid
flowchart LR
    X["U/I/C"] --> BN["分域 BN"]
    BN --> S["精确 Base SENet"]
    S --> V["现有等宽 tokenizer"]
    V --> R["现有 v1 RankMixer"]
    R --> M["mean + first head"]
~~~

伪代码：

~~~python
u, i, c = domain_bn(u, i, c)
u, i, c = exact_base_senet(u, i, c)
h = current_rankmixer(current_scalar_tokenizer(concat([u, i, c])))
logit = first_head(reduce_mean(h, axis=1))
~~~

优点：改动最小、归因最强、可能直接复用 gate checkpoint。<br>
缺点：字段切分、早期压缩和 mean readout 问题仍在。

### 14.2 方案 B：RM-FieldSafe

目标：构建不依赖 DCNM 的纯 RankMixer 公平版。

~~~mermaid
flowchart LR
    X["U/I/C"] --> BN["分域 BN"]
    BN --> S["精确 Base SENet"]
    S --> F["16 个完整字段语义组"]
    F --> R["严格 block<br/>k=2 参数匹配"]
    R --> Q["mean + grouped + low-rank flatten"]
    Q --> H["first head"]
~~~

伪代码：

~~~python
u, i, c = exact_base_frontend(features)
tokens = field_safe_tokenize(u, i, c, groups_5_10_1)
h = strict_rankmixer(tokens, layers=2, dim=768, expansion=2)
repr = dual_readout(h)
logit = first_head(repr)
~~~

优点：修复三个最明显的 RM 表示问题，参数接近 Base。<br>
缺点：仍缺少 Base 压缩前的显式乘性交叉。

### 14.3 方案 C：RM-BackboneSwap

目标：在完全相同 Base 前端和 DCNM 上，只比较后端主干。

~~~mermaid
flowchart LR
    X["U/I/C"] --> B["Base BN+SENet+2×DCNM"]
    B --> T["字段安全 tokenizer"]
    T --> R["参数匹配 RankMixer"]
    R --> D["双读出"]
    D --> H["相同 first head/loss"]
~~~

优点：最严格、最能证明 RankMixer 是否优于 Base MLP。<br>
缺点：算力较高，不能回答“纯 RM 能否替代 DCNM”。

---

## 15. 推荐实验矩阵

| 实验 | 唯一主要改动 | 关键问题 |
|---|---|---|
| E0-B | 原样真实 Base，3+ seeds | Base 方差是多少 |
| E0-R | 原样 v1，3+ seeds | 0.003 是否稳定 |
| E1 | v1 + exact SENet | Base gate 贡献多少 |
| E2 | E1 + 字段安全 tokenizer | 切字段代价多少 |
| E3 | E2 + 双读出 | mean 信息损失多少 |
| E4 | E3 + strict block | hybrid 拓扑影响多少 |
| E5 | E4，k=4→2 | 参数匹配后是否更稳 |
| E6 | Base DCNM 后 MLP→RM | RM 能否替代 Base MLP |
| E7 | E6 + aligned SwiGLU | 成熟 FFN 是否增益 |
| E8 | E6/E7 + warm/KD | 收敛迁移贡献多少 |
| E9 | 加 DIN/sequence | 新能力是否超过 Base |
| E10 | 加 multi-task | 新监督是否继续提升 |

每个实验至少报告：

- 多 seed AUC/GAUC 均值与标准差；
- logloss、ECE/校准曲线；
- 新老用户、热门/长尾 Item、Creative 分桶指标；
- 参数量、FLOPs、峰值显存、step time、线上 p95；
- loaded/random-init 参数量和训练前 10% 的收敛曲线。

---

## 16. 研究扩展与 Base 归因必须分开

以下方向可以提高未来模型，但真实 Base 本次运行没有它们：

- DIN 与多路序列；
- dense/gattr 侧路；
- last CVR、多任务、Task Token；
- ESMM、pairwise/RankUp loss；
- Soft-to-Hard 分桶；
- UniMixer、RankElastor、Sparse MoE；
- Creative FiLM、Global Token 和更大 token budget。

它们应在 E0–E6 完成后独立立项，不能把增益描述为“恢复 Base”。

---

## 17. 上线与回滚检查单

### 17.1 数据与结构

- [ ] 唯一 Base 文件记录为 <code>cvr_bn_senet_dcnm.py</code>；
- [ ] resolved args 与 commit SHA 一同归档；
- [ ] 每个 17 维字段只进入一个 tokenizer group；
- [ ] U/I/C 域不在单个 token 内混杂；
- [ ] $T=H$、$D\bmod H=0$ 有显式断言；
- [ ] token schema 变更时升级 checkpoint 版本；
- [ ] first-only 是 parity，其他任务都有单独实验编号。

### 17.2 训练与数值

- [ ] 使用 raw logits 的稳定 BCE，或 Base/RM 同时保持旧 loss；
- [ ] gradient clipping 确实接到 train op；
- [ ] loaded/missing/random-init scope 有日志；
- [ ] 参数匹配与 FLOPs 匹配分别报告；
- [ ] fused/unfused 前向与梯度误差通过阈值；
- [ ] 多 seed 收敛曲线无异常分叉。

### 17.3 服务

- [ ] 导出图与训练图使用相同 tokenizer/schema；
- [ ] p50/p95/p99 延迟和峰值内存合格；
- [ ] 灰度期间可按配置切回 Base；
- [ ] 异常校准、长尾分桶和 Creative 分桶有监控；
- [ ] checkpoint 与特征 schema 双重版本化。

---

## 18. 源码证据定位

| 事实 | 源码/运行证据 |
|---|---|
| Base resolved 参数 | <code>set-x.args.resolved.txt:27-76</code>、<code>0721args.txt</code> |
| RM resolved 参数 | <code>code/set-xcal.txt:248-298</code> |
| Base 只取三桶 | <code>cvr_bn_senet_dcnm.py:915-953</code> |
| Base 三桶 BN 与 SENet | <code>cvr_bn_senet_dcnm.py:955-980</code> |
| Base 层级 SENet | <code>cvr_bn_senet_dcnm.py:830-905</code> |
| Base 两层 DCNM | <code>cvr_bn_senet_dcnm.py:783-828,982-983</code> |
| Base MLP 与 head | <code>cvr_bn_senet_dcnm.py:988-1016,1097-1108</code> |
| Base first loss | <code>cvr_bn_senet_dcnm.py:527-548</code> |
| RM 三桶与 BN | <code>cvr_bn_rankmixer_v1.py:889-930</code> |
| RM 投影函数与等宽分段 | <code>cvr_bn_rankmixer_v1.py:774-799,938-962</code> |
| RM block 与 mean readout | <code>cvr_bn_rankmixer_v1.py:801-866,964-980</code> |

---

## 19. 最终选型建议

若目标是最快定位 0.003：先做 E1 exact SENet，再做 E2 字段安全 tokenizer 和 E3 双读出。

若目标是证明 RankMixer 主干价值：做 E6 Backbone Swap；它比把大量新输入、任务和结构一次性加入更有解释力。

若目标是构建可上线版本：以 RM-FieldSafe 为低成本候选，以 RM-BackboneSwap 为高置信候选，二者都使用参数匹配、版本化 tokenizer、严格 restore 日志和多 seed 验证。

一句话总结：

> 真实 Base 的优势不是更多输入或更多任务，而是压缩前的层级字段门控与全局乘性交叉；RankMixer 应先补齐这些可验证差异、修复字段切分和读出，再证明自身主干是否更强。
