# CVR RankMixer 输入重构与 AUC 优化：四套可落地方案

> 关联源码：commend_cvr.py、mlp_mixer_swiglu_fuse.py<br>
> 基线分析：[CVR_RankMixer_SwiGLU_源码分析与原版对比.md](CVR_RankMixer_SwiGLU_源码分析与原版对比.md)<br>
> 结合 0.865 base 与 0.862 RankMixer v1 实验后的专项诊断：[CVR_RankMixer_v1_AUC差距诊断与改进路线.md](CVR_RankMixer_v1_AUC差距诊断与改进路线.md)<br>
> 方案日期：2026-08-14<br>
> 文档性质：架构设计与实验规范，不表示四套方案已经在生产代码中实现。

---

## 目录

1. [设计范围与必须先修正的问题](#1-设计范围与必须先修正的问题)
2. [统一的严格 RankMixer 主干](#2-统一的严格-rankmixer-主干)
3. [方案一：三域完整字段硬分桶](#3-方案一三域完整字段硬分桶)
4. [方案二：条件门控、Creative 侧塔与双读出](#4-方案二条件门控creative-侧塔与双读出)
5. [方案三：Soft-to-Hard 可学习语义分桶](#5-方案三soft-to-hard-可学习语义分桶)
6. [方案四：CVR 多任务与全空间建模](#6-方案四cvr-多任务与全空间建模)
7. [四套方案的严格比较](#7-四套方案的严格比较)
8. [与当前源码的落点映射](#8-与当前源码的落点映射)
9. [实验路线与验收标准](#9-实验路线与验收标准)
10. [最终建议](#10-最终建议)

---

## 1. 设计范围与必须先修正的问题

### 1.1 本文采用的紧凑输入接口

本文延续前序讨论中的紧凑建模接口：

$$
U\in\mathbb{R}^{B\times385\times17}
$$

$$
I\in\mathbb{R}^{B\times835\times17}
$$

$$
C\in\mathbb{R}^{B\times14\times17}
$$

其中：

- $B$ 是 batch size；
- $385$、$835$、$14$ 分别是 User、Item、Creative 字段数；
- 每个字段当前按 17 维表示；
- 一个字段的 17 维必须作为不可拆分的最小语义单元。

三个域 flatten 后的宽度分别为：

$$
d_U=385\times17=6545
$$

$$
d_I=835\times17=14195
$$

$$
d_C=14\times17=238
$$

总宽度是：

$$
d_{all}=6545+14195+238=20978
$$

这里必须特别指出：

> 总宽度是 20978，不是 20798。

此外，源码主 Teacher 当前使用的是 11 个 User 桶、18 个 Item 桶、2 个 Sequence token 和 1 个 DIN token，默认形成 $32\times512$ 的 Mixer 输入。本文的 $16\times768$ 是一条新的紧凑实验塔，不能假定与现有 checkpoint 直接兼容。将方案落到现有大塔时，应保留本文的“按完整字段分桶、token 语义稳定、主干残差正确”等原则，再按实际 11/18 桶配置实例化。

### 1.2 待改造的等宽标量切分为什么有结构性问题

如果先把三个域全部 flatten 并拼成：

$$
X_{flat}\in\mathbb{R}^{B\times20978}
$$

再近似等宽切成 16 段，段长只能是：

$$
20978=2\times1312+14\times1311
$$

但是：

$$
1312\bmod17=3
$$

$$
1311\bmod17=2
$$

这意味着切分边界落在字段内部，而不是字段之间。结果至少有三类问题：

1. 同一个字段的 17 维被拆到相邻 token；
2. 某个 token 会同时含有 User 尾部和 Item 头部；
3. 最后一个 token 会同时含有 Item 尾部和 Creative。

这种切分方式看起来保证了 token 宽度接近，但破坏了 RankMixer 最关键的前提：

> 每个 token 应当长期对应稳定的业务语义，Per-token FFN 才能学到稳定的专属参数。

### 1.3 必须建立的四个输入不变量

无论选择后面的哪套方案，都应先加入以下约束：

~~~python
assert user.shape == [B, 385, 17]
assert item.shape == [B, 835, 17]
assert creative.shape == [B, 14, 17]

# 只能沿字段轴分桶，不能沿 flatten 后的标量轴等宽切分
assert every_bucket_contains_complete_fields()
~~~

如果第 17 维并不是与前 16 维同质的 embedding，而是 mask、value、weight 或统计量，应先显式拆成：

~~~python
embedding = x[..., :16]
side_value = x[..., 16:]
~~~

再决定：

- side value 是否与 embedding 一起进入投影；
- 是否只用于 gate；
- 是否只用于 mask；
- 是否应该独立归一化。

在确认这件事之前，不能默认 17 维完全同质。

### 1.4 统一设计目标

四套方案共同优化以下目标：

| 目标 | 含义 |
|---|---|
| 语义完整性 | 不切断单字段 embedding，不跨 User/Item/Creative 边界 |
| RankMixer 对齐 | token 数、head 数、维度整除关系明确，保留原版关键残差 |
| AUC 提升空间 | 增强候选条件化、token 身份保留和辅助监督 |
| 可解释消融 | 每次只引入一个主要变量，能判断 AUC 变化来源 |
| 工程可控 | 参数、显存、延迟、checkpoint 和 fused kernel 风险可量化 |

---

## 2. 统一的严格 RankMixer 主干

### 2.1 推荐的公共配置

前三套结构方案统一使用：

| 参数 | 值 |
|---|---:|
| Token 数 $T$ | 16 |
| Token Mixing head 数 $H$ | 16 |
| Token 宽度 $D$ | 768 |
| 单 head 子空间 | $D/H=48$ |
| RankMixer block 数 $L$ | 2 |
| GELU PFFN hidden | 3072 |
| 参数匹配 SwiGLU hidden | 2048 |

必须满足：

$$
T=H
$$

$$
D\bmod H=0
$$

这里把 $L$ 设为 2，有两个目的：

1. 与原版 RankMixer 的双块主干更容易严格对齐；
2. Token Mixing 是一个满足 $P^2=I$ 的置换，偶数层不会把最终表示永久留在交换后的坐标系。

### 2.2 Multi-head Token Mixing

输入：

$$
X\in\mathbb{R}^{B\times T\times D}
$$

先把每个 token 的通道切成 $H=T$ 个子空间：

$$
X\rightarrow
\mathbb{R}^{B\times T\times H\times(D/H)}
$$

交换 token 轴和 head 轴：

$$
\mathbb{R}^{B\times T\times H\times48}
\rightarrow
\mathbb{R}^{B\times H\times T\times48}
$$

最后重新拼成：

$$
P(X)\in\mathbb{R}^{B\times T\times D}
$$

伪代码：

~~~python
def token_mix(x):
    # x: [B, T, D], T == H == 16, D == 768
    assert D % T == 0
    h = reshape(x, [B, T, T, D // T])   # [B, 16, 16, 48]
    h = transpose(h, [0, 2, 1, 3])      # exchange token and head
    return reshape(h, [B, T, D])        # [B, 16, 768]
~~~

这个操作没有可训练参数。它不是 attention，也不计算样本相关权重；它只是把所有 token 的对应子空间进行确定性交换。

### 2.3 原版 RankMixer block

严格原版拓扑应写成：

$$
S=\operatorname{LN}_{mix}(P(X)+X)
$$

$$
Y=\operatorname{LN}_{ffn}(\operatorname{PFFN}(S)+S)
$$

PFFN 的参数对 token 隔离：

$$
\operatorname{PFFN}_t(s_t)
=W_{2,t}\operatorname{GELU}(W_{1,t}s_t+b_{1,t})+b_{2,t}
$$

其中不同 token 的 $W_{1,t}$、$W_{2,t}$ 不共享。

完整伪代码：

~~~python
def per_token_ffn(s, hidden=3072):
    outputs = []
    for t in range(16):
        h = gelu(s[:, t, :] @ W1[t] + b1[t])
        y = h @ W2[t] + b2[t]
        outputs.append(y)
    return stack(outputs, axis=1)

def rankmixer_block(x):
    mixed = token_mix(x)
    s = layer_norm(mixed + x, scope="mix_post_ln")
    f = per_token_ffn(s)
    y = layer_norm(f + s, scope="ffn_post_ln")
    return y

def rankmixer_backbone(tokens):
    x = tokens
    for layer in range(2):
        x = rankmixer_block(x)
    return x
~~~

### 2.4 与当前 mlp_mixer_swiglu 模块的边界

当前源码中的 mlp_mixer_swiglu 主要执行：

$$
M=P(X)
$$

$$
Y=M+\operatorname{SwiGLU}(\operatorname{LN}(M))
$$

最后再执行一次 Final LayerNorm。

它保留了：

- 无参数 Token Mixing；
- Per-token 参数隔离；
- SwiGLU 非线性；
- FFN residual。

但它没有原版的：

$$
\operatorname{LN}(P(X)+X)
$$

也就是删除了 mixing residual，并把原版双 Post-LN 改成了 FFN Pre-LN 加 Final LN。

因此本文的方案一至方案三默认先使用“严格原版 block”做干净基线；当前 SwiGLU 变体作为独立消融项比较，不能只把它当成激活函数替换。

### 2.5 GELU 与 SwiGLU 的公平参数匹配

原版 GELU PFFN 若 expansion ratio 为 4：

$$
h_{GELU}=4D=3072
$$

忽略 bias 后，两矩阵参数量为：

$$
2Dh_{GELU}=8D^2
$$

SwiGLU 有 Gate、Up、Down 三个矩阵。若希望参数量近似相同：

$$
3Dh_{SwiGLU}=8D^2
$$

所以：

$$
h_{SwiGLU}=\frac{8D}{3}=2048
$$

若直接令 SwiGLU hidden 也等于 3072，则 PFFN 矩阵参数比 GELU expansion 4 多约 50%。这种实验同时改变了激活函数和模型容量，不能作为公平对照。

### 2.6 输出必须使用 raw logits

训练阶段：

~~~python
logit = dense(head_hidden, 1, activation=None)
loss = sigmoid_cross_entropy_with_logits(
    labels=label,
    logits=logit,
)
~~~

预测阶段：

~~~python
probability = sigmoid(logit)
~~~

不能在 BCEWithLogits 之前执行 sigmoid，也不应在训练图里对 logit 或 probability 做 clip。clip 会制造零梯度区间和大量相同分数，直接伤害 AUC 排序能力。

---

## 3. 方案一：三域完整字段硬分桶

### 3.1 方案定位

这是最低风险、必须先做的正确性基线。

它只解决三件事：

1. 不再按标量维度等宽切分；
2. 每个 token 只包含同一业务域的完整字段；
3. 主干恢复为严格原版 RankMixer。

不引入动态 gate、可学习分桶和辅助任务，便于判断“修正 token 语义”本身带来的收益。

### 3.2 分桶规则

User 的 385 个字段均分成 5 桶：

$$
385=5\times77
$$

Item 的 835 个字段分成 10 桶：

$$
835=5\times84+5\times83
$$

Creative 的 14 个字段组成 1 桶。

最终：

$$
T=5+10+1=16
$$

| Token 范围 | 域 | 每 token 字段数 | 投影前宽度 | 投影后 |
|---|---|---:|---:|---:|
| 0-4 | User | 77 | $77\times17=1309$ | 768 |
| 5-9 | Item-A | 84 | $84\times17=1428$ | 768 |
| 10-14 | Item-B | 83 | $83\times17=1411$ | 768 |
| 15 | Creative | 14 | $14\times17=238$ | 768 |

### 3.3 架构图

~~~mermaid
flowchart LR
    U["User B×385×17"] --> US["按字段轴切成 5×77"]
    I["Item B×835×17"] --> IS["按字段轴切成 5×84 + 5×83"]
    C["Creative B×14×17"] --> CS["完整 14 字段"]

    US --> UP["5 个独立 Dense+GELU<br/>1309 → 768"]
    IS --> IP["10 个独立 Dense+GELU<br/>1428/1411 → 768"]
    CS --> CP["1 个独立 Dense+GELU<br/>238 → 768"]

    UP --> TOK["Concat: B×16×768"]
    IP --> TOK
    CP --> TOK

    TOK --> R1["Strict RankMixer Block 1"]
    R1 --> R2["Strict RankMixer Block 2"]
    R2 --> POOL["Mean Pool: B×768"]
    POOL --> HEAD["MLP Head → raw logit"]
~~~

### 3.4 “Dense+GELU 压成一个 token”到底是什么

对第 $k$ 个桶：

$$
X_k\in\mathbb{R}^{B\times n_k\times17}
$$

先只在桶内 flatten：

$$
\tilde X_k\in\mathbb{R}^{B\times(17n_k)}
$$

然后：

$$
z_k=\tilde X_kW_k+b_k
$$

$$
t_k=\operatorname{GELU}(z_k)
$$

其中：

$$
W_k\in\mathbb{R}^{(17n_k)\times768}
$$

结论非常明确：

> 这是一个独立的线性投影矩阵，加一次逐元素 GELU；它不是带隐藏层的两层 MLP。

虽然源码函数名是 embedding_to_tokens，注释里称其为 MLP，但实际核心是一次 tf.layers.dense。对于 token_num=1：

~~~python
def bucket_to_one_token(bucket, token_dim=768):
    # bucket: [B, field_count, 17]
    x = reshape(bucket, [B, field_count * 17])
    z = dense(x, units=token_dim, activation=None)
    token = gelu(z)
    return token[:, None, :]  # [B, 1, 768]
~~~

只有改成下面这样，才是常见意义上的两层 MLP：

~~~python
h = gelu(x @ W1 + b1)
token = h @ W2 + b2
~~~

方案一不建议一开始增加这个隐藏层，因为单投影已经能完成：

- 输入维度对齐；
- 桶内全部字段的线性混合；
- GELU 非线性截断；
- 不同语义桶的参数隔离。

先保持 tokenizer 简单，才能把 AUC 变化归因于字段分桶和 RankMixer 主干，而不是额外 MLP 容量。

### 3.5 完整伪代码

~~~python
def split_fields(x, sizes):
    assert sum(sizes) == x.shape[1]
    return split(x, sizes, axis=1)

def independent_bucket_token(bucket, scope):
    with variable_scope(scope):
        x = reshape(bucket, [B, -1])
        z = dense(x, 768, activation=None, name="projection")
        return gelu(z)[:, None, :]

def build_scheme_1(user, item, creative):
    user_buckets = split_fields(user, [77, 77, 77, 77, 77])
    item_buckets = split_fields(
        item,
        [84, 84, 84, 84, 84, 83, 83, 83, 83, 83],
    )
    creative_buckets = [creative]

    tokens = []
    for k, bucket in enumerate(user_buckets):
        tokens.append(independent_bucket_token(bucket, "user_token_%d" % k))
    for k, bucket in enumerate(item_buckets):
        tokens.append(independent_bucket_token(bucket, "item_token_%d" % k))
    tokens.append(independent_bucket_token(creative, "creative_token"))

    x = concat(tokens, axis=1)
    assert x.shape[1:] == [16, 768]

    for layer in range(2):
        x = strict_rankmixer_block(x, scope="rankmixer_%d" % layer)

    pooled = reduce_mean(x, axis=1)
    hidden = gelu(dense(pooled, 512))
    hidden = gelu(dense(hidden, 256))
    logit = dense(hidden, 1, activation=None)
    return logit
~~~

### 3.6 参数量

16 个独立 tokenizer 投影的参数量为：

$$
5(1309\times768+768)
$$

$$
+5(1428\times768+768)
$$

$$
+5(1411\times768+768)
$$

$$
+1(238\times768+768)
=16{,}123{,}392
$$

重要的是：

> 只要每个输入标量最终仍只连接到一个 768 维 token，重新按完整字段划分不会显著增加 tokenizer 总参数；它主要改变连接结构和 token 语义。

两层、16 token、$D=768$、GELU expansion 4 的 Per-token FFN 参数约为：

$$
2\times16\times
\left(2\times4\times768^2+5\times768\right)
=151{,}117{,}824
$$

不含 LayerNorm 和预测头的 tokenizer 加 PFFN 小计约 1.672 亿参数。由此也能看出，主要容量在 Per-token FFN，而不在 tokenizer。

### 3.7 优点、风险与适用条件

优点：

- 修复字段被切断和跨域污染；
- token 身份固定，最符合 Per-token FFN 假设；
- 不需要额外标签和外部特征；
- 参数变化小，便于热启动之外的公平重训实验。

风险：

- 分桶仍由人工按顺序决定，未必是最佳语义组合；
- Creative 只有一个 token，可能容量不足；
- Mean Pool 会丢失 token 身份；
- 严格原版主干与当前 SwiGLU checkpoint 不兼容。

适用条件：

- 字段顺序稳定；
- 385、835、14 的字段清单可冻结和版本化；
- 可以从头训练或接受新 scope 冷启动。

---

## 4. 方案二：条件门控、Creative 侧塔与双读出

### 4.1 方案定位

这是本文最推荐的工业落地方案。

它在方案一“完整字段硬分桶”的基础上增加：

1. User 自条件 gate；
2. User 条件化 Item gate；
3. Creative 独立侧塔；
4. Mean Pool 与低秩 Flatten 双读出。

核心动机是：

- User 决定哪些 Item 字段对当前样本更重要；
- Creative 不必被压进唯一一个 Mixer token；
- 主任务既获得稳定的全局平均表示，也保留部分 token 坐标信息。

### 4.2 为什么改成 5 个 User token 和 11 个 Item token

Creative 移出 Mixer 后，Mixer 内仍保持 16 个 token：

$$
T_U=5
$$

$$
T_I=11
$$

$$
T_U+T_I=16
$$

User：

$$
385=5\times77
$$

Item：

$$
835=10\times76+1\times75
$$

| 域 | 桶配置 | 投影前宽度 | token 数 |
|---|---|---:|---:|
| User | $5\times77$ 字段 | 1309 | 5 |
| Item | $10\times76+1\times75$ 字段 | 1292/1275 | 11 |
| Creative | 不进 Mixer | 238 | 侧塔 128 |

### 4.3 字段级条件 gate

直接定义为：

$$
g_U=2\sigma(G_U(\operatorname{vec}(U)))
$$

$$
g_I=2\sigma(G_I([\operatorname{vec}(U),\operatorname{vec}(I)]))
$$

$$
g_C=2\sigma(G_C(\operatorname{vec}(C)))
$$

形状为：

$$
g_U\in\mathbb{R}^{B\times385}
$$

$$
g_I\in\mathbb{R}^{B\times835}
$$

$$
g_C\in\mathbb{R}^{B\times14}
$$

gate 作用在完整字段上：

$$
\hat U_{b,f,:}=g^U_{b,f}U_{b,f,:}
$$

$$
\hat I_{b,f,:}=g^I_{b,f}I_{b,f,:}
$$

$$
\hat C_{b,f,:}=g^C_{b,f}C_{b,f,:}
$$

最后一层 gate 网络使用全零初始化：

~~~python
last_kernel = zeros()
last_bias = zeros()
gate = 2.0 * sigmoid(context @ last_kernel + last_bias)
~~~

初始化时：

$$
g=2\sigma(0)=1
$$

因此新模型一开始近似恒等映射，不会随机压死某批字段。

完整 flatten gate 最直接，但参数可能较大。生产实现可把 context 换成字段 squeeze 统计：

~~~python
field_mean = reduce_mean(x, axis=-1)
field_norm = sqrt(reduce_sum(square(x), axis=-1) + eps)
context = concat([field_mean, field_norm], axis=-1)
~~~

这不会改变“User 自条件、User 条件化 Item”的架构含义，只是降低 gate 网络成本。

### 4.4 双读出

Mixer 输出：

$$
X_{out}\in\mathbb{R}^{B\times16\times768}
$$

第一路是 Mean Pool：

$$
h_{mean}=\operatorname{Mean}(X_{out},axis=1)
\in\mathbb{R}^{B\times768}
$$

第二路保留全部 token 坐标，再做低秩压缩：

$$
h_{flat}
=\phi(\operatorname{vec}(X_{out})W_{flat}+b_{flat})
\in\mathbb{R}^{B\times256}
$$

其中：

$$
W_{flat}\in\mathbb{R}^{12288\times256}
$$

Creative 侧塔：

$$
h_C=\operatorname{MLP}(\operatorname{vec}(\hat C))
\in\mathbb{R}^{B\times128}
$$

融合：

$$
h=[h_{mean},h_{flat},h_C]
\in\mathbb{R}^{B\times1152}
$$

预测头：

$$
1152\rightarrow512\rightarrow256\rightarrow1
$$

### 4.5 架构图

~~~mermaid
flowchart TD
    U["User B×385×17"] --> UN["User Norm"]
    I["Item B×835×17"] --> IN["Item Norm"]
    C["Creative B×14×17"] --> CN["Creative Norm"]

    UN --> UG["User self gate<br/>2·sigmoid"]
    UN --> IG["User context"]
    IN --> IG["User-conditioned Item gate<br/>2·sigmoid"]
    CN --> CG["Creative self gate<br/>2·sigmoid"]

    UG --> UB["5×77-field buckets"]
    IG --> IB["10×76 + 1×75-field buckets"]
    UB --> UT["5 independent projections<br/>→ 5×768"]
    IB --> IT["11 independent projections<br/>→ 11×768"]

    UT --> TOK["B×16×768"]
    IT --> TOK
    TOK --> RM["2× Strict RankMixer"]

    RM --> MEAN["Mean readout<br/>B×768"]
    RM --> FLAT["Flatten 12288 → 256"]
    CG --> CT["Creative side tower<br/>238 → 128"]

    MEAN --> F["Concat B×1152"]
    FLAT --> F
    CT --> F
    F --> H["512 → 256"]
    H --> LOGIT["first_cvr raw logit"]
~~~

### 4.6 完整伪代码

~~~python
def identity_gate(context, output_fields, scope):
    with variable_scope(scope):
        h = gelu(dense(context, 256))
        gate_logit = dense(
            h,
            output_fields,
            activation=None,
            kernel_initializer=zeros(),
            bias_initializer=zeros(),
        )
        return 2.0 * sigmoid(gate_logit)

def apply_field_gate(x, gate):
    return x * gate[:, :, None]

def build_scheme_2(user, item, creative):
    user_n = layer_norm(user, axis=-1, scope="user_norm")
    item_n = layer_norm(item, axis=-1, scope="item_norm")
    creative_n = layer_norm(creative, axis=-1, scope="creative_norm")

    user_context = reshape(user_n, [B, -1])
    item_context = concat(
        [reshape(user_n, [B, -1]), reshape(item_n, [B, -1])],
        axis=1,
    )
    creative_context = reshape(creative_n, [B, -1])

    g_user = identity_gate(user_context, 385, "user_gate")
    g_item = identity_gate(item_context, 835, "item_gate")
    g_creative = identity_gate(creative_context, 14, "creative_gate")

    user_g = apply_field_gate(user_n, g_user)
    item_g = apply_field_gate(item_n, g_item)
    creative_g = apply_field_gate(creative_n, g_creative)

    user_buckets = split(user_g, [77] * 5, axis=1)
    item_buckets = split(item_g, [76] * 10 + [75], axis=1)

    tokens = []
    for k, bucket in enumerate(user_buckets):
        tokens.append(independent_bucket_token(bucket, "user_token_%d" % k))
    for k, bucket in enumerate(item_buckets):
        tokens.append(independent_bucket_token(bucket, "item_token_%d" % k))

    x = concat(tokens, axis=1)
    assert x.shape[1:] == [16, 768]

    for layer in range(2):
        x = strict_rankmixer_block(x, scope="rankmixer_%d" % layer)

    h_mean = reduce_mean(x, axis=1)                    # [B, 768]
    h_flat = gelu(dense(reshape(x, [B, 12288]), 256)) # [B, 256]
    h_creative = gelu(
        dense(reshape(creative_g, [B, 238]), 128)
    )                                                  # [B, 128]

    h = concat([h_mean, h_flat, h_creative], axis=1)  # [B, 1152]
    h = gelu(dense(h, 512))
    h = gelu(dense(h, 256))
    logit = dense(h, 1, activation=None)
    return logit
~~~

### 4.7 增量参数与收益来源

低秩 Flatten 投影参数：

$$
12288\times256+256=3{,}145{,}984
$$

Creative 单层侧塔参数：

$$
238\times128+128=30{,}592
$$

融合头的主要矩阵约为：

$$
1152\times512+512\times256
\approx0.72M
$$

相对于 1.5 亿级 Per-token FFN，这些增量较小。

潜在收益来自三个不同方向：

1. gate：增强 User 对 Item 特征的样本级条件选择；
2. Creative 侧塔：防止 14 个 Creative 字段在单 token 中过早压缩；
3. low-rank Flatten：弥补 Mean Pool 丢失 token 身份的问题。

### 4.8 建议拆成三个独立消融

不要一次把全部增强同时打开。按以下顺序：

| 实验 | 相对方案一的唯一变化 |
|---|---|
| E2 | Creative 从 Mixer 移到侧塔，其他不变 |
| E3 | 增加 identity-init User/Item/Creative gate |
| E4 | 增加 low-rank Flatten readout |

这样可以知道 AUC 提升究竟来自早期字段选择、Creative 处理还是输出读取。

### 4.9 风险

- Item gate 同时读取完整 User 和 Item 时，gate MLP 输入很宽；
- gate 容易饱和，应监控均值、P1/P50/P99 和梯度；
- low-rank Flatten 会绑定固定 token 顺序；
- Creative 晚融合不能参与 Mixer 内部的 User-Item 交互；
- 必须保证 gate 最后一层严格零初始化，否则冷启动分布会漂移。

---

## 5. 方案三：Soft-to-Hard 可学习语义分桶

### 5.1 方案定位

方案一和方案二仍假设“按字段列表顺序近似均分”是合理的。若字段语义资料不完整，或者人工分桶难以维护，可以让模型先学习字段到 token 的静态归属，再把结果硬化成稳定分桶。

关键限制是：

> 学习的是全局静态字段归属，不是每个样本动态改变字段属于哪个 token。

Per-token FFN 的参数身份与 token 绑定。如果同一个 token 在不同样本里代表完全不同的字段集合，独立 PFFN 很难形成稳定专长。

### 5.2 字段编码

对每个 17 维字段使用共享编码器：

$$
e_f=\operatorname{GELU}(x_fW_e+b_e+q_f)
$$

其中：

- $W_e\in\mathbb{R}^{17\times128}$ 是域内共享字段编码器；
- $q_f\in\mathbb{R}^{128}$ 是稳定的 field-ID embedding；
- $e_f\in\mathbb{R}^{128}$。

User 和 Item 可以使用不同的共享编码器，避免两个域的统计分布互相污染。

### 5.3 可学习归属矩阵

User：

$$
L_U\in\mathbb{R}^{385\times5}
$$

Item：

$$
L_I\in\mathbb{R}^{835\times10}
$$

温度为 $\tau$ 时：

$$
A^U_{f,k}
=\operatorname{softmax}_k(L^U_{f,:}/\tau)
$$

$$
A^I_{f,k}
=\operatorname{softmax}_k(L^I_{f,:}/\tau)
$$

每个字段对所有 token 的权重和为 1：

$$
\sum_k A_{f,k}=1
$$

### 5.4 Soft token 聚合

User 第 $k$ 个 soft token：

$$
\bar e^U_k=
\frac{\sum_f A^U_{f,k}e^U_f}
{\sum_f A^U_{f,k}+\epsilon}
$$

再用 token-specific 投影：

$$
t^U_k=\bar e^U_kW^U_k+b^U_k
\in\mathbb{R}^{768}
$$

Item 同理得到 10 个 token。Creative 固定作为第 16 个 token：

$$
t_C=\operatorname{GELU}
(\operatorname{vec}(C)W_C+b_C)
$$

最终仍为：

$$
5+10+1=16
$$

### 5.5 防止所有字段塌缩到同一 token

负载均衡损失：

$$
L_{balance}
=\sum_{k=1}^{K}
\left(
\frac{1}{F}\sum_{f=1}^{F}A_{f,k}
-\frac{1}{K}
\right)^2
$$

归属熵：

$$
L_{entropy}
=-\frac{1}{F}\sum_f\sum_k
A_{f,k}\log(A_{f,k}+\epsilon)
$$

训练目标：

$$
L=L_{task}
+\lambda_{bal}L_{balance}
+\lambda_{ent}L_{entropy}
$$

早期使用较高温度，让多个 token 都能收到梯度；后期降低温度并逐步增加熵惩罚，让每个字段趋向单一归属。

示意调度：

~~~python
tau = linear_or_cosine_decay(
    start=2.0,
    end=0.2,
    training_progress=progress,
)
lambda_entropy = warmup(0.0, target_entropy_weight)
~~~

具体数值必须通过离线实验确定，不能把示意值当作通用最优值。

### 5.6 两阶段 Soft-to-Hard

阶段 A：学习 soft assignment。

阶段 B：硬化并重建 tokenizer。

$$
k_f^*=\arg\max_k A_{f,k}
$$

把所有 $k_f^*=k$ 的完整 17 维字段组成第 $k$ 个硬桶：

~~~python
hard_bucket[k] = [
    field_f
    for field_f in all_fields
    if argmax(assignment[f]) == k
]
~~~

然后不再做加权平均，而是恢复完整字段 concat：

$$
t_k=
\operatorname{GELU}
(\operatorname{vec}(\operatorname{HardBucket}_k)W_k+b_k)
$$

这样做的好处是：

- 学习阶段自动发现字段组合；
- 部署阶段 token 语义固定；
- 不需要在线计算 soft assignment；
- 保留每个字段完整的 17 维信息；
- Per-token FFN 获得长期稳定的职责。

### 5.7 架构图

~~~mermaid
flowchart TD
    U["User 385 fields"] --> UE["Shared field encoder 17→128<br/>+ Field-ID embedding"]
    I["Item 835 fields"] --> IE["Shared field encoder 17→128<br/>+ Field-ID embedding"]

    AU["Static logits Aᵤ: 385×5"] --> US["Temperature Softmax"]
    AI["Static logits Aᵢ: 835×10"] --> IS["Temperature Softmax"]

    UE --> UA["Weighted aggregation<br/>5 soft groups"]
    US --> UA
    IE --> IA["Weighted aggregation<br/>10 soft groups"]
    IS --> IA

    UA --> UP["5 token-specific 128→768"]
    IA --> IP["10 token-specific 128→768"]
    C["Creative 14×17"] --> CP["Fixed projection → 1×768"]

    UP --> TOK["B×16×768"]
    IP --> TOK
    CP --> TOK
    TOK --> RM["2× Strict RankMixer"]
    RM --> HEAD["Readout and first_cvr logit"]

    US --> REG["Balance + Entropy losses"]
    IS --> REG
    REG --> HARD["Stage B: argmax hardening"]
    HARD --> HB["Export stable full-field buckets"]
~~~

### 5.8 完整伪代码

~~~python
def encode_fields(x, field_ids, scope):
    # x: [B, F, 17]
    with variable_scope(scope):
        content = einsum("bfd,dh->bfh", x, W_encoder) + b_encoder
        identity = embedding_lookup(field_id_table, field_ids)
        return gelu(content + identity)  # [B, F, 128]

def soft_group(encoded, assignment_logits, token_count, tau, scope):
    # assignment is static across samples
    assignment = softmax(assignment_logits / tau, axis=1)  # [F, K]
    numerator = einsum("fk,bfh->bkh", assignment, encoded)
    denominator = reduce_sum(assignment, axis=0)[None, :, None] + 1e-8
    grouped = numerator / denominator

    tokens = []
    for k in range(token_count):
        tokens.append(
            dense(grouped[:, k, :], 768, scope="%s_token_%d" % (scope, k))
        )
    return stack(tokens, axis=1), assignment

def grouping_regularization(assignment):
    field_count, token_count = assignment.shape
    load = reduce_mean(assignment, axis=0)
    balance = reduce_sum(square(load - 1.0 / token_count))
    entropy = -reduce_mean(
        reduce_sum(assignment * log(assignment + 1e-8), axis=1)
    )
    return balance, entropy

def build_scheme_3(user, item, creative, tau):
    user_e = encode_fields(user, user_field_ids, "user_encoder")
    item_e = encode_fields(item, item_field_ids, "item_encoder")

    user_t, a_user = soft_group(
        user_e, user_assignment_logits, 5, tau, "user_group"
    )
    item_t, a_item = soft_group(
        item_e, item_assignment_logits, 10, tau, "item_group"
    )
    creative_t = independent_bucket_token(creative, "creative_token")

    x = concat([user_t, item_t, creative_t], axis=1)
    assert x.shape[1:] == [16, 768]

    for layer in range(2):
        x = strict_rankmixer_block(x, scope="rankmixer_%d" % layer)

    logit = prediction_head(reduce_mean(x, axis=1))

    bal_u, ent_u = grouping_regularization(a_user)
    bal_i, ent_i = grouping_regularization(a_item)
    group_loss = (
        lambda_balance * (bal_u + bal_i)
        + lambda_entropy * (ent_u + ent_i)
    )
    return logit, group_loss, a_user, a_item
~~~

### 5.9 硬化前必须检查的内容

| 检查项 | 目的 |
|---|---|
| 每 token 字段数量 | 防止空桶或极端大桶 |
| Assignment 最大概率分布 | 判断归属是否足够确定 |
| 字段归属跨随机种子稳定性 | 排除偶然分组 |
| 每桶业务字段清单 | 让业务专家检查明显冲突 |
| Soft 与 Hard AUC 差距 | 判断硬化信息损失 |
| Hard tokenizer 参数和延迟 | 确认部署成本 |

### 5.10 风险

- Assignment logits 可能只学到字段频率，而非业务语义；
- balance 与 entropy 权重冲突时会造成训练震荡；
- 某些字段天然应该服务多个 token，硬化会损失多归属能力；
- field-ID 必须稳定，字段增删要有版本迁移方案；
- 相比方案一和方案二，训练与上线链路更复杂。

因此方案三应作为独立研究分支，不应阻塞方案一和方案二落地。

---

## 6. 方案四：CVR 多任务与全空间建模

### 6.1 方案定位

前三套方案主要改变特征组织和 backbone。方案四与它们正交，主要改变监督信号和预测头。

它适用于当前只优化 first_cvr，但训练数据中还存在以下可靠标签的情况：

- last_cvr；
- first/last no-refund CVR；
- click；
- favorite/cart；
- 其他与转化链路一致且时间窗无泄漏的行为。

源码当前核心头名称包括 fst_nrfnd、lst_nrfnd、fst、lst，辅助头包括相似点击和收藏。外部标签配置决定它们的精确业务含义，因此实施前必须先审计 label mapping。尤其要注意源码中的 fst_auc 实际读取 fst_nrfnd_pred，不能只根据监控名称推断主标签。

### 6.2 共享骨干与任务专属塔

以方案二或方案三的共享表示 $h_{shared}$ 为输入，每个任务使用独立小塔：

$$
h_t=\phi(W_{t,2}\phi(W_{t,1}h_{shared}))
$$

$$
z_t=w_t^\mathsf{T}h_t+b_t
$$

不要让所有任务只共享最后一个线性头，因为：

- 不同任务的标签噪声和样本空间不同；
- first/last、refund/no-refund 的决策边界不完全一致；
- 任务专属塔能缓解负迁移。

### 6.3 Masked 多任务 BCE

对任务 $t$：

$$
m_{i,t}=\mathbb{1}[y_{i,t}\ge0]
$$

$$
L_t=
\frac{
\sum_i m_{i,t}
\operatorname{BCEWithLogits}(z_{i,t},y_{i,t})
}{
\max(1,\sum_i m_{i,t})
}
$$

总损失：

$$
L_{MTL}
=L_{first\_cvr}
+\sum_{t\ne first\_cvr}\alpha_tL_t
$$

推荐原则：

- 主任务权重固定为 1；
- 辅助任务从 0.05 至 0.2 的小权重开始；
- 根据共享层梯度 cosine 和主任务验证 AUC 调整；
- 不应因为任务数量多就把所有权重设成 1。

这里的数值是起始搜索区间，不是固定最优值。

### 6.4 可选 ESMM

如果当前 CVR 模型只在点击样本上训练，却需要在全曝光空间打分，并且能够获得完整曝光与点击标签，可采用：

$$
pCTR=\sigma(z_{ctr})
$$

$$
pCVR=\sigma(z_{cvr})
$$

$$
pCTCVR=pCTR\times pCVR
$$

同时监督：

- 全曝光空间的 click；
- 全曝光空间的 click-and-conversion。

ESMM 的作用是缓解 sample selection bias 和数据稀疏问题，但使用条件非常严格：

> 没有全曝光样本和正确的 click→conversion 链路时，不应为了“模型更复杂”强行使用 ESMM。

### 6.5 可选 Pairwise AUC loss

在同一 batch 或同一 request/group 内采样正负对：

$$
L_{pair}
=\operatorname{mean}
\operatorname{softplus}
(-(z^+-z^-))
$$

总损失：

$$
L
=L_{MTL}
+\lambda_{pair}L_{pair}
$$

Pairwise loss直接优化正样本分数高于负样本的排序关系，但要注意：

- 最好在同一用户、请求或可比曝光组内采样；
- 全局随机负样本可能只学到用户或场景偏差；
- BCE 仍负责概率校准，pairwise 只作为小权重辅助项；
- 需要同时检查 AUC 和 ECE/校准曲线。

### 6.6 架构图

~~~mermaid
flowchart TD
    IN["Scheme 2 or Scheme 3 inputs"] --> BB["Shared tokenizer + RankMixer"]
    BB --> SH["Shared representation"]

    SH --> M["Main tower<br/>first_cvr"]
    SH --> L["Aux tower<br/>last_cvr"]
    SH --> N["Aux tower<br/>no-refund CVR"]
    SH --> C["Aux tower<br/>click"]
    SH --> A["Aux tower<br/>favorite/cart"]

    M --> BCE["Masked BCE<br/>weight 1.0"]
    L --> AUX["Aux BCE<br/>small weights"]
    N --> AUX
    C --> AUX
    A --> AUX

    M --> PAIR["Within-group pairwise AUC loss"]
    C --> ESMM["Optional ESMM"]
    M --> ESMM

    BCE --> TOTAL["Total loss"]
    AUX --> TOTAL
    PAIR --> TOTAL
    ESMM --> TOTAL
~~~

### 6.7 完整伪代码

~~~python
def task_tower(shared, name):
    with variable_scope(name):
        h = gelu(dense(shared, 256))
        h = gelu(dense(h, 128))
        return dense(h, 1, activation=None)

def masked_bce(logit, label):
    mask = cast(label >= 0, float32)
    safe_label = maximum(label, 0)
    per_example = sigmoid_cross_entropy_with_logits(
        labels=safe_label,
        logits=logit,
    )
    return reduce_sum(mask * per_example) / maximum(reduce_sum(mask), 1.0)

def pairwise_auc_loss(main_logit, main_label, group_id):
    positive, negative = sample_comparable_pairs(
        main_logit,
        main_label,
        group_id,
    )
    return reduce_mean(softplus(-(positive - negative)))

def build_scheme_4(shared, labels, group_id):
    logits = {
        "first_cvr": task_tower(shared, "first_cvr_tower"),
        "last_cvr": task_tower(shared, "last_cvr_tower"),
        "first_nrfnd": task_tower(shared, "first_nrfnd_tower"),
        "last_nrfnd": task_tower(shared, "last_nrfnd_tower"),
        "click": task_tower(shared, "click_tower"),
        "favorite": task_tower(shared, "favorite_tower"),
    }

    main_loss = masked_bce(logits["first_cvr"], labels["first_cvr"])

    auxiliary_loss = 0.0
    for task, weight in auxiliary_weights.items():
        auxiliary_loss += weight * masked_bce(logits[task], labels[task])

    rank_loss = pairwise_auc_loss(
        logits["first_cvr"],
        labels["first_cvr"],
        group_id,
    )

    total_loss = main_loss + auxiliary_loss + lambda_pair * rank_loss

    if use_esmm:
        p_ctr = sigmoid(logits["click"])
        p_cvr = sigmoid(logits["first_cvr"])
        p_ctcvr = p_ctr * p_cvr
        total_loss += esmm_loss(p_ctr, p_ctcvr, labels)

    return logits, total_loss
~~~

### 6.8 防止负迁移

必须监控共享 backbone 上各任务梯度：

$$
\cos(g_{main},g_t)
=\frac{g_{main}\cdot g_t}
{\|g_{main}\|\|g_t\|}
$$

如果某辅助任务长期：

- 与主任务梯度显著冲突；
- 单独指标提升但主任务 AUC 下降；
- 存在不同标签窗口或延迟回填；

应依次尝试：

1. 降低辅助权重；
2. 延后辅助任务接入层；
3. 增大任务专属塔；
4. 对辅助梯度 stop-gradient 到部分 backbone；
5. 删除该辅助任务。

### 6.9 标签泄漏审计

在多任务实验前逐项确认：

- 标签观测窗是否晚于线上打分时刻；
- refund/no-refund 是否在相同成熟窗口内；
- first/last 的业务包含关系是否真实成立；
- 缺失标签是否用 -1 mask，而不是错误当作 0；
- 训练、验证、线上三个口径是否一致；
- AUC 指标读取的 prediction 是否与名称一致。

多任务能放大有效监督，也能放大标签错误。标签审计优先级高于调权。

---

## 7. 四套方案的严格比较

### 7.1 总表

| 方案 | 核心变化 | Mixer token | 额外数据 | 工程风险 | 推荐定位 |
|---|---|---:|---|---|---|
| 方案一 | 完整字段硬分桶 + 严格原版 block | 5U+10I+1C | 不需要 | 低 | 必做正确性基线 |
| 方案二 | 条件 gate + Creative 侧塔 + 双读出 | 5U+11I | 不需要 | 中 | 最推荐工业方案 |
| 方案三 | 学习静态字段归属，再硬化 | 5U+10I+1C | 稳定 field ID | 中高 | 独立研究分支 |
| 方案四 | 多任务、ESMM、Pairwise 排序 | 继承方案二/三 | 需要可靠辅助标签；ESMM 需全曝光 | 中高 | 监督层增强 |

### 7.2 与原版 RankMixer 的联系

| 维度 | 原版 RankMixer | 方案一 | 方案二 | 方案三 | 方案四 |
|---|---|---|---|---|---|
| 语义 token | 人工稳定分组 | 是 | 是 | 先学习后固定 | 继承 backbone |
| Token Mixing | 无参数交换 | 严格保留 | 严格保留 | 严格保留 | 不改变 |
| Mixing residual | 有 | 有 | 有 | 有 | 不改变 |
| Per-token FFN | 独立参数 | 独立 | 独立 | 独立 | 不改变 |
| Readout | 论文任务头 | Mean | Mean+low-rank Flat | 可选 | 多任务专属塔 |
| 额外交互 | 无 | 无 | User→Item gate | Assignment 学习 | 标签级共享 |

### 7.3 与当前源码 SwiGLU 变体的关系

| 项目 | 当前 mlp_mixer_swiglu | 本文干净基线 |
|---|---|---|
| Token Mixing | 保留 | 保留 |
| Mixing residual | 删除 | 恢复 |
| FFN | Per-token SwiGLU | 先用 Per-token GELU |
| LN | FFN Pre-LN + Final LN | 两个 residual 后 Post-LN |
| Down projection | Zero init | 常规初始化 |
| 默认层数 | 3 | 2 |
| 默认 $T,D$ | 32, 512 | 16, 768 |
| 目的 | 工业大容量变体 | 可归因的严格对照基线 |

推荐先得到方案一的严格原版结果，再做以下四格实验：

| Block | GELU 参数 | SwiGLU 参数匹配 | SwiGLU expansion 4 |
|---|---:|---:|---:|
| 原版双 residual | A | B | C |
| 当前 Pre-LN、无 mixing residual | D | E | F |

这样可以分离：

- block 拓扑影响；
- 激活函数影响；
- 参数量影响。

### 7.4 方案之间不是互斥关系

推荐组合关系：

~~~mermaid
flowchart LR
    S1["方案一<br/>正确分桶基线"] --> S2["方案二<br/>工业增强"]
    S1 --> S3["方案三<br/>学习分桶研究"]
    S2 --> S4["方案四<br/>多任务与排序"]
    S3 --> S4
~~~

最现实的主线是：

$$
\text{方案一}\rightarrow\text{方案二}\rightarrow\text{方案四}
$$

方案三作为并行研究线，得到稳定硬分桶后，可反向替换方案二中的人工分桶。

---

## 8. 与当前源码的落点映射

### 8.1 源码修改位置

| 位置 | 当前职责 | 方案落点 |
|---|---|---|
| commend_cvr.py 的字段列表与原始桶 | 11 User、18 Item 语义分组 | 冻结字段清单；禁止 flatten 后跨字段切分 |
| embedding_to_tokens | 一次 Dense+activation，再 reshape | 方案一/二硬桶 tokenizer；明确不是两层 MLP |
| User/Item SENet | User 自 gate、User 条件化 Item | 方案二可复用其条件门控思想 |
| mlp_mixer_swiglu_fuse.py 的 mix_up | 无参数 token/head 交换 | 三套 backbone 共同复用 |
| mlp_mixer_swiglu | 当前 Pre-LN SwiGLU 变体 | 新增 strict_rankmixer 实验实现，不直接覆盖旧 scope |
| Mean 与 Flatten readout | 主头 Mean、辅助头 Flatten | 方案二让主 CVR 同时读取低秩 Flatten |
| _build_losses | 多头 masked BCE | 方案四增加小权重辅助任务和可选 pairwise |

### 8.2 不建议直接覆盖当前函数

建议新增清晰、可并存的 scope：

~~~text
tokenizer_hard_v1/
rankmixer_strict_v1/
readout_dual_v1/
multitask_head_v1/
~~~

而不是直接改变：

~~~text
mlp_mixer/
pwff_fc1_*/
pwff_fc2_*/
pwff_fc3_*/
~~~

原因是当前 fused SwiGLU 对变量名和 checkpoint 有显式依赖。直接复用旧 scope 可能产生：

- shape 不匹配；
- 错误热启；
- fused 与 unfused 权重语义不一致；
- 难以回滚；
- 实验结果无法区分新旧 block。

### 8.3 必须新增的 shape 断言

~~~python
assert len(user_buckets) == expected_user_tokens
assert len(item_buckets) == expected_item_tokens
assert all_fields_used_once(user_buckets, user_field_ids)
assert all_fields_used_once(item_buckets, item_field_ids)

assert token_tensor.shape[1] == mixup_token_num
assert token_tensor.shape[2] == mixup_token_dim
assert mixup_token_dim % mixup_token_num == 0
~~~

对于当前 Teacher 大塔还要补：

~~~python
assert len(user_senet_parts) == len(USER_TOKEN_CONFIG)
assert len(item_senet_parts_without_hs) == len(ITEM_TOKEN_CONFIG)
~~~

源码当前使用 zip 遍历配置和 tensor；两侧长度不一致时 zip 会静默截断，所以仅依赖最终 reshape 不能及时暴露字段丢失。

### 8.4 配置化字段桶

硬桶不应散落在 Python 控制流中。建议版本化保存：

~~~yaml
schema_version: compact_rankmixer_v1
field_width: 17
user:
  token_0: [u_field_000, ..., u_field_076]
  token_1: [u_field_077, ..., u_field_153]
item:
  token_0: [i_field_000, ..., i_field_083]
creative:
  side_tower: [c_field_000, ..., c_field_013]
~~~

启动时验证：

- 无重复字段；
- 无遗漏字段；
- 顺序与 Feature 配置一致；
- schema version 与 checkpoint 一致。

---

## 9. 实验路线与验收标准

### 9.1 推荐实验顺序

| 编号 | 实验 | 回答的问题 |
|---|---|---|
| E0 | 修正总维度、字段轴切分、shape assert、raw-logit loss | 基线是否存在结构或训练错误 |
| E1 | 方案一 | 完整字段语义 token + 严格原版 RankMixer 是否有效 |
| E2 | 方案二仅 Creative 侧塔 | Creative 晚融合是否优于单 token |
| E3 | E2 + identity gate | User 条件化字段选择是否有效 |
| E4 | E3 + low-rank Flatten | 主任务保留 token 身份是否有效 |
| E5 | E4 + 小权重多任务 | 辅助标签是否改善共享表示 |
| E6 | E5 + pairwise AUC loss | 排序辅助目标是否进一步改善 AUC |
| E7 | 方案三 Soft assignment | 自动分桶是否优于人工均分 |
| E8 | 方案三 Hard 化 | 部署友好的固定桶能否保留收益 |

若具备全曝光数据，再单独增加 ESMM 实验；不要把 ESMM 混进 E5 的第一轮多任务实验。

### 9.2 每个实验保持不变的条件

为了公平归因，以下条件应固定：

- 数据时间窗与样本过滤；
- embedding 初始化和学习率；
- batch size；
- optimizer 与 schedule；
- 主任务 loss 权重；
- 随机种子集合；
- early stopping 规则；
- 评估切片；
- 线上候选集合。

如果模型参数量变化明显，至少同时报告：

1. 固定训练 step 的结果；
2. 固定训练 FLOPs 或 wall-clock budget 的结果。

### 9.3 离线指标

主指标：

- first_cvr AUC/GAUC；
- first_cvr LogLoss；
- 正负样本排序对正确率。

校准指标：

- ECE；
- reliability curve；
- 分桶预测均值与真实转化率。

稳定性指标：

- 多随机种子均值与方差；
- paired bootstrap 置信区间；
- 训练 loss 和 gradient norm；
- 各 token 激活 norm；
- gate 分位数；
- assignment 负载与熵。

业务切片：

- 新老用户；
- 高低活跃用户；
- 头部/长尾 Item；
- 不同 Creative 类型；
- 不同流量场景；
- 不同曝光位置。

工程指标：

- 参数量；
- 训练显存；
- tokens/s；
- 在线 P50/P95/P99 latency；
- fused/unfused 数值差；
- checkpoint 体积与加载时间。

### 9.4 验收原则

本文不预设“某方案必然提升多少 AUC”。一个方案进入下一阶段至少应满足：

1. 主任务 AUC 的提升跨随机种子稳定；
2. LogLoss 与校准没有不可接受退化；
3. 关键业务切片没有系统性下降；
4. 参数和延迟增量符合线上预算；
5. shape、字段覆盖、fused parity 等正确性测试全部通过；
6. 标签映射与预测头名称经过人工核对。

### 9.5 回滚策略

每个增强都使用独立配置开关：

~~~python
use_complete_field_buckets = True
use_strict_rankmixer = True
use_conditional_gate = False
use_creative_side_tower = False
use_low_rank_flatten = False
use_learned_grouping = False
use_multitask = False
use_pairwise_auc = False
use_esmm = False
~~~

并保留旧模型完整建图路径。这样线上出现：

- 延迟回归；
- checkpoint 恢复失败；
- 指标漂移；
- gate 或 assignment 异常；

可以只关闭对应增量，不需要回滚整份模型代码。

---

## 10. 最终建议

### 10.1 推荐主线

首选落地顺序：

1. 先完成方案一，建立没有字段切断、没有跨域 token 污染、block 拓扑严格的可信基线；
2. 再按 E2、E3、E4 分步构建方案二；
3. 标签审计完成后，在方案二上增加方案四；
4. 方案三独立训练，只有在 Soft 与 Hard 两阶段都稳定优于人工桶时，才替换主线分桶。

### 10.2 最推荐的最终候选

在不依赖额外全曝光数据的条件下，最推荐的候选架构是：

~~~text
完整字段输入
  → 域内归一化
  → identity-init User / User-conditioned Item gate
  → 5 User + 11 Item 稳定硬桶
  → 16×768 tokens
  → 2 层严格 RankMixer
  → Mean 768 + Low-rank Flatten 256
  → Creative side tower 128
  → 1152→512→256
  → first_cvr raw logit
~~~

它在四个方面取得相对平衡：

- 不破坏字段完整性；
- 保留 RankMixer 的核心结构；
- 增加样本相关的 User→Item 条件化；
- 以较小参数增量补回 token 身份信息。

### 10.3 三条不可妥协的结论

1. 不再对 20978 个标量做近似等宽 token 切分；只能沿完整字段轴分桶。
2. Dense+GELU tokenizer 是一次投影加激活，不是两层 MLP；若增加隐藏层，必须作为独立容量实验。
3. 当前 SwiGLU 模块与原版 RankMixer 不只是激活不同；mixing residual、LayerNorm 位置、初始化和参数量都必须分别对照。

---

## 参考资料

- 本仓库原论文：[RankMixer.pdf](RankMixer.pdf)
- 本仓库源码分析：[CVR_RankMixer_SwiGLU_源码分析与原版对比.md](CVR_RankMixer_SwiGLU_源码分析与原版对比.md)
- [RankMixer: Scaling Up Ranking Models in Industrial Recommenders](https://arxiv.org/abs/2507.15551)
- [FiBiNET: Combining Feature Importance and Bilinear Feature Interaction for Click-Through Rate Prediction](https://arxiv.org/abs/1905.09433)
- [Entire Space Multi-Task Model: An Effective Approach for Estimating Post-Click Conversion Rate](https://arxiv.org/abs/1804.07931)
- [Deep Interest Network for Click-Through Rate Prediction](https://arxiv.org/abs/1706.06978)
- [DCN V2: Improved Deep & Cross Network and Practical Lessons for Web-scale Learning to Rank Systems](https://arxiv.org/abs/2008.13535)
