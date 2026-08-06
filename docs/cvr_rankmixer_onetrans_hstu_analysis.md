# 超大规模 CVR 预估：从 RankMixer 迁移到 OneTrans 与 HSTU

> 分析对象：推荐 / 广告 / 搜索排序中的 post-click CVR 预估  
> 数据规模：约 5.5 亿条训练样本 / 天  
> 当前稀疏特征 embedding 维度：17  
> 当前 RankMixer：16 tokens，`d=768`，2 个 block，Mean Pooling，二分类损失

## 1. 结论先行

1. **当前 RankMixer 的等长切分需要先修正。** 总维度 `20,978` 平均分成 16 份后，前 15 份长度为 1,311，最后一份为 1,313。由于 `1,311 % 17 = 2`，这种切分会从中间切断单个字段的 17 维 embedding；第 5 个 segment 还同时包含 user 与 item，第 16 个 segment 同时包含 item 与 creative。RankMixer 的 token 本应代表相对一致、完整的特征语义，因此这会削弱 per-token FFN 的意义。
2. **OneTrans 是对当前 RankMixer 最自然的下一步升级。** 它可以复用现有静态特征 tokenizer，将其作为 NS-tokens，再引入用户行为序列 S-tokens，在同一个 causal Transformer 中联合完成序列建模和高阶特征交互。
3. **HSTU 不应被理解为“把 16 个静态 token 换成另一种 attention”。** HSTU 的核心是用户行为时序、target-aware mask、位置与时间偏置、pointwise aggregated attention，以及按用户 / 请求摊销历史计算。若没有原始行为序列，只对 16 个静态 token 使用 HSTU，不能算有效复现。
4. **落地顺序建议：** 先修复 RankMixer tokenizer；再做 OneTrans-S；随后做官方 DLRM-HSTU 风格的“历史用户塔 + 独立 item/creative 塔”；最后再尝试完整 Generative Recommender 训练。
5. **不要直接把 OneTrans/HSTU 的隐藏维度设成 768。** OneTrans 的 NS-token QKV 和 FFN 是 token-specific 的，`d=768` 会带来非常大的稠密参数量。首版建议 `d=256, 6 layers, 4 heads`；HSTU 首版同样建议从 `d=256, 4~6 layers` 开始。
6. **CVR 样本空间必须先定义清楚。** 若仅在点击样本上训练，预测的是 `P(conversion | click, x)`；若使用全曝光数据，优先考虑 ESMM 风格的 CTR + CTCVR 联合训练，缓解 clicked-only 训练的样本选择偏差与正例稀疏问题。

---

## 2. 当前 RankMixer 方案审计

### 2.1 输入规模

| 特征域 | 字段数 | 单字段 embedding | 展开维度 |
|---|---:|---:|---:|
| user | 385 | 17 | 6,545 |
| item | 835 | 17 | 14,195 |
| creative | 14 | 17 | 238 |
| **合计** | **1,234** | 17 | **20,978** |

当前流程为：

```text
user       [B,  6545]
item       [B, 14195]
creative   [B,   238]
concat  -> [B, 20978]
equal split 16 segments
16 independent projections -> [B, 16, 768]
2 x RankMixer block
mean pooling -> [B, 768]
CVR head
```

### 2.2 等长切分存在三个结构性问题

`20,978 = 15 × 1,311 + 1,313`。

#### 问题 A：切断单个字段的 embedding

每个字段是一个完整的 17 维向量，但 `1,311 = 77 × 17 + 2`。因此绝大多数 segment 边界都会把某个字段的 embedding 切成两部分。例如一个字段的前 2 维位于 token 1，剩余 15 维位于 token 2。这样会使 token 的语义不稳定，也破坏字段内部维度的联合投影。

#### 问题 B：跨特征域混合

- segment 5：约 `1,301` 维 user + `10` 维 item；
- segment 16：`1,075` 维 item + 全部 `238` 维 creative。

因此 creative 没有独立 token，边界 token 的 per-token FFN 同时服务于完全不同的数据分布。

#### 问题 C：token 编号依赖字段排列

一旦上游增删字段或改变拼接顺序，后续所有 segment 的含义都会漂移，模型 warm-start、增量发布和可解释性都会变差。

### 2.3 不增加参数量的修复方案

首个强基线建议仍保持 16 tokens，但只在完整字段边界上切分：

| 域 | token 数 | 每个 token 包含的字段数 | 输入维度 |
|---|---:|---:|---:|
| user | 5 | 每个 77 个字段 | 每个 1,309 |
| item | 10 | 5 个 token 各 84 字段，5 个各 83 字段 | 1,428 / 1,411 |
| creative | 1 | 14 个字段 | 238 |
| **合计** | **16** |  |  |

由于独立投影的总输入维度仍然是 20,978，`sum(input_dim_i × 768)` 基本不变，但不再切断字段，也不再跨域。

更好的版本应根据业务语义分组，而非仅按数量平均。例如：

- user：身份 / 长期兴趣、短期统计、价格偏好、地域设备、搜索意图；
- item：ID 与类目、店铺品牌、价格促销、质量统计、供给状态、内容属性、交叉统计等；
- creative：模板、素材类型、文案 / 图片属性、历史效果。

### 2.4 当前稠密参数量的量级

若 per-token FFN 扩张比为 `r=4`，仅两层 PFFN 的权重约为：

```text
2 × r × num_layers × num_tokens × d²
= 2 × 4 × 2 × 16 × 768²
≈ 151.0M parameters
```

16 个输入 projector 约为：

```text
20,978 × 768 ≈ 16.1M parameters
```

因此当前方案即使没有 sparse MoE，稠密部分也已经是百 M 级。后续比较 OneTrans/HSTU 时，应同时报告：

- 不含 sparse embedding 的 dense params；
- 单样本 / 单请求 FLOPs；
- 训练吞吐；
- 推理 p95 / p99；
- 显存与 CPU/DRAM embedding 压力。

---

## 3. 建模前必须补齐的数据结构

OneTrans 与 HSTU 的主要增益都依赖用户行为序列。推荐把当前逐行样本升级为请求级或会话级结构：

```text
user_static_embeddings        [B, 385, 17]
history_event_embeddings      Jagged[sum(L), F_history, 17]
history_action_type           Jagged[sum(L)]
history_timestamp/time_gap    Jagged[sum(L)]
query_and_request_context      [B, F_query, 17]
candidate_item_embeddings     [B, C, 835, 17]
candidate_creative_embeddings [B, C, 14, 17]
click_label                    [B, C]
conversion_label               [B, C]
```

其中：

- `B` 是请求 / 用户序列数，而不是独立候选行数；
- `C` 是同一次请求中的候选数；
- `L` 是每个用户在该请求时间点之前可见的历史长度；
- history 必须严格截断到当前 impression time，禁止未来行为泄漏；
- query、场景、流量来源、广告位、设备等搜索上下文应单独建模，不建议埋在 user/item 的大向量里。

### 3.1 行级训练格式的成本问题

若同一请求有 `C` 个候选，逐候选样本会把相同用户历史复制 `C` 次。OneTrans 的 KV cache 和 HSTU 的多 target / generative 训练都要求尽量按请求连续组织候选，才能把历史侧计算从 `O(C)` 摊销到接近 `O(1)`。

### 3.2 历史 event 不要复制全部 835 个 item 字段

不建议在每个历史位置都放入完整的 835 个 item 字段。可选择：

- item ID、category、brand/shop、price bucket、关键质量字段；
- action type：曝光、点击、加购、收藏、购买等；
- query / domain / scene；
- creative type；
- timestamp、time gap、position；
- 或离线生成的 compact content embedding。

当前候选的完整 835 + 14 个字段仍可进入 target/item tower；历史位置只携带稳定且高价值的紧凑特征。

---

## 4. CVR 标签与损失函数

### 4.1 clicked-only CVR

若训练集只包含已点击样本：

```text
pCVR = sigmoid(z_cvr)
L_cvr = BCEWithLogits(z_cvr, conversion_label), only where click=1
```

这对应：

```text
P(conversion = 1 | click = 1, user, query, item, creative, history)
```

二分类 `CrossEntropyLoss` 也能工作，但单 logit 的 `BCEWithLogitsLoss` 更直接；不要先 sigmoid 再传入 `BCEWithLogitsLoss`。

### 4.2 全曝光空间：建议 ESMM 风格联合训练

若每天 5.5 亿条是 impression 样本，推荐至少增加 CTR 与 CTCVR：

```text
pCTR   = sigmoid(z_ctr)
pCVR   = sigmoid(z_cvr)
pCTCVR = pCTR * pCVR

L = BCE(pCTR, click)
  + lambda_ctcvr * BCE(pCTCVR, click * conversion)
```

优点：

- CVR tower 能从全曝光空间获得监督；
- 减轻 clicked-only 样本选择偏差；
- CTCVR 约束保证概率链条更一致；
- 最终排序可按 `pCTR × pCVR × value` 或业务效用打分。

可进一步加入加购、收藏、支付成功、GMV 回归等辅助任务，但首版不要让辅助任务数量过多。

### 4.3 延迟转化与校准

CVR 常见的主要数据问题不是模型结构，而是标签延迟：

- 训练样本必须达到成熟窗口后再判负；
- 历史 action 只能使用请求时间点之前已发生的状态；
- 负采样后要保留 sampling weight，或在推理后做先验修正；
- class weight / focal loss 可能提升召回，但会改变概率校准，需单独评估 ECE、可靠性曲线和分桶 CVR。

---

## 5. OneTrans 建模方案

## 5.1 为什么 OneTrans 最适合作为第一步

OneTrans 将两类 token 放进同一个 backbone：

- **S-tokens**：用户多行为序列，所有位置共享一套 Q/K/V 与 FFN；
- **NS-tokens**：user、query/context、item、creative 等非序列特征，每个 token 使用独立 Q/K/V 与 FFN。

token 顺序为：

```text
[S-tokens ordered by time] [NS-tokens]
```

使用 causal mask 后，所有 NS-token 都能读取完整历史，而用户历史侧可以缓存。它比“先序列编码，再拼接 RankMixer”更充分地联合建模 sequence-feature interaction。

### 5.2 推荐的 S-token 构造

对点击、加购、购买、搜索等多种行为，可分别建立 event projector：

```text
event_click  -> MLP_click  -> d
event_cart   -> MLP_cart   -> d
event_order  -> MLP_order  -> d
event_search -> MLP_search -> d
```

然后按时间戳从旧到新合并，并添加：

- behavior type embedding；
- relative time / time-gap embedding；
- position embedding；
- 必要时的 `[SEP]` token。

有可靠时间戳时优先 timestamp-aware merge，而不是把所有订单、加购、点击各自整段拼接。

### 5.3 推荐的 NS-token 分配

首版保持 `L_NS=16`，建议：

```text
5 user tokens
8 item tokens
1 creative token
1 query/request-context token
1 learned [PRED] token
-----------------------------
16 NS tokens
```

如果没有独立 query 特征，可使用：

```text
5 user + 9 item + 1 creative + 1 [PRED]
```

固定顺序建议：

```text
[S history] -> [USER] -> [QUERY/CONTEXT] -> [ITEM] -> [CREATIVE] -> [PRED]
```

因为是 causal attention，顺序会决定信息流。最后的 `[PRED]` 能看到此前所有 token，适合作为任务 head 的输入。也可以让最后一个 candidate token 兼任 prediction carrier，但显式 `[PRED]` 更容易保持不同 tokenizer 版本间的接口稳定。

### 5.4 NS tokenizer：先 Group-wise，再做低秩 Auto-Split

论文提供两种 tokenizer：

1. Group-wise：每组特征独立 MLP；
2. Auto-Split：所有 NS 特征先进入一个 MLP，再 reshape 为多个 token。

对当前 `20,978` 维输入，直接做一次 `20,978 -> 16 × d` 的全连接非常大：

| d | 仅第一层权重 |
|---:|---:|
| 256 | `20,978 × 4,096 ≈ 85.9M` |
| 384 | `20,978 × 6,144 ≈ 128.9M` |
| 768 | `20,978 × 12,288 ≈ 257.8M` |

因此建议：

**V1：边界安全的 Group-wise tokenizer**

```text
user_flat     -> 5 group projectors -> [B, 5, d]
item_flat     -> 8 group projectors -> [B, 8, d]
creative_flat -> 1 projector        -> [B, 1, d]
query_flat    -> 1 projector        -> [B, 1, d]
learned pred                         -> [B, 1, d]
```

**V2：低秩 Auto-Split**

```text
x_ns [B, 20978]
 -> Linear(20978, 1024 or 2048)
 -> SiLU/RMSNorm
 -> Linear(1024 or 2048, 15*d)
 -> reshape [B, 15, d]
 -> append [PRED]
```

也可先按 user/item/creative/query 分别压缩，再做一次联合 Auto-Split。这样能保留 Auto-Split 的跨域组合能力，同时避免 2.6 亿级 tokenizer 权重。

### 5.5 OneTrans block

每层采用 pre-RMSNorm：

```text
x = x + MixedCausalMHA(RMSNorm(x))
x = x + MixedFFN(RMSNorm(x))
```

参数规则：

```text
S positions : shared QKV + shared FFN
NS position i: token-specific QKV_i + token-specific FFN_i
```

不要把所有 token 共享同一个 FFN，否则会丢失 OneTrans 针对异构 NS-token 的关键设计；也不要继续使用 RankMixer 的 token mixing 来替代 causal MHA。

### 5.6 Pyramid schedule

首版序列长度可从 128 或 256 开始。例如 `L_S=256, L_NS=16, layers=6`：

```text
retained S-query count:
256 -> 192 -> 128 -> 96 -> 64 -> 32 -> 0
```

每层保留序列尾部，最上层最终只剩 NS-token。具体数量可按硬件向 32 的倍数对齐。Pyramid 只减少 query/output 数量；当前层的 K/V 仍覆盖完整输入，使长历史信息逐层汇聚到尾部 NS-token。

### 5.7 输出 head

不建议对 S + NS 做全局 mean pooling，因为大量历史 token 会稀释 candidate 信息。建议：

```text
h = final_state_of_[PRED]            # [B, d]
# 或 concat / attentive pooling of final NS states
z_ctr, z_cvr = multitask_head(h)
```

若同一请求有 `C` 个候选，则输出为 `[B, C, d]` 或在 candidate 维度展开后恢复。

### 5.8 首版超参数

| 项目 | 建议 V1 | 后续扩展 |
|---|---:|---:|
| hidden dim | 256 | 384 |
| layers | 6 | 8 |
| heads | 4 | 6 / 8 |
| NS tokens | 16 | 16 / 24 |
| S length | 128 或 256 | 512、1024+ |
| FFN ratio | 2~4 | 按参数预算调整 |
| norm | Pre-RMSNorm | 不变 |
| precision | BF16 | BF16 + recompute |
| attention | causal + FlashAttention | pyramid + KV cache |

OneTrans 论文的小模型也是 `6 layers, d=256, 4 heads`，大模型为 `8 layers, d=384`。这比把现有 `d=768` 原样搬过去更合理。

### 5.9 为什么不建议 OneTrans 直接使用 d=768

粗略忽略 attention output projection，只计算每个 NS-token 的 QKV 与 `r=4` FFN：

```text
per layer ≈ (3 + 2r) × L_NS × d²
```

当 `L_NS=16`：

- `d=256`：约 11.5M / layer；
- `d=768`：约 103.8M / layer。

6 层 `d=768` 仅这部分就超过 620M，还不含 S-side 参数、tokenizer 和 head。因此应该先用更深、更窄的配置，并优先扩展序列长度。

### 5.10 请求级 KV cache

训练与服务数据应把同一请求的候选连续组织：

```text
Stage 1: user S-tokens 只计算一次并缓存各层 K/V
Stage 2: 每个候选只计算其 NS-tokens，并读取共享 S-side cache
```

跨请求缓存时，只有用户新增行为需要计算增量 K/V。若仍将 5.5 亿条样本完全随机打散成独立 `(user, item)` 行，这个优势基本无法利用。

### 5.11 没有行为序列时的降级版本

可以令 `L_S=0`，只对 16 个 NS-token 使用 mixed causal attention，并用 `[PRED]` 输出。但该模型应命名为：

```text
OneTrans-NS ablation / OneTrans-style feature mixer
```

它可以测试 token-specific QKV 的价值，但不代表完整 OneTrans，也无法获得 pyramid、历史 KV cache、跨行为序列交互等主要收益。

---

## 6. HSTU 建模方案

## 6.1 HSTU 的正确定位

HSTU 将推荐问题表示为按时间排列的内容与动作序列。对 ranking，可写成：

```text
content_0, action_0, content_1, action_1, ..., target_content
```

在 target content 位置预测下一动作，例如点击、加购、转化。核心 block 为：

```text
U, V, Q, K = Split(SiLU(Linear(x)))
A V        = SiLU(Q K^T + relative_position_time_bias) V
y          = Linear(Norm(A V) * U)
```

关键区别：

- attention 权重使用逐点 SiLU，而不是对整条序列做 softmax；
- 包含相对位置与相对时间 bias；
- `Norm(AV) * U` 同时承担交互与门控；
- 使用 target-aware causal mask；
- 依赖 jagged/ragged fused kernel 才能在长序列上高效。

若实现仍是标准 softmax attention + Transformer FFN，就不能称为 HSTU 复现。

## 6.2 两种可落地路线

### 路线 A：完整 Generative Recommender

构建交替内容 / 动作序列，并在许多历史 content 位置都计算 action loss：

```text
[Phi_0, a_0, Phi_1, a_1, ..., Phi_T, a_T, Phi_candidate]
```

在 `Phi_candidate` 的输出位置预测当前候选的 click / conversion。优点是一次 forward 能产生多个历史位置的监督，充分复用每天 5.5 亿条数据；缺点是数据管线、训练采样和服务形式变化最大。

### 路线 B：DLRM-HSTU 混合架构，推荐先做

官方开源实现提供了更接近工业排序模型的结构：

1. HSTU 历史塔生成 candidate-specific user embedding；
2. 当前 item 特征通过独立 item MLP/tower；
3. user 与 item embedding 做逐元素乘法；
4. MLP 输出多任务 logits。

对当前特征可设计为：

```text
user static 385 fields
    -> 4~8 contextual prefix tokens

history events
    -> compact event embeddings
    -> HSTU stack

candidate target token
    <- selected item + creative + query features
    -> target-aware HSTU output h_user_target [B, C, d]

full candidate item 835 + creative 14 + query/context
    -> item/creative tower -> h_item [B, C, d]

interaction = h_user_target * h_item
    -> multitask MLP -> ctr/cvr/ctcvr logits
```

这种方案不会在每个历史位置复制 835 个 item 字段，同时保留完整当前候选特征。

## 6.3 HSTU 输入 token 设计

### Contextual prefix

将 385 个 user 字段压缩为 4~8 个 context tokens，例如：

```text
[user_profile]
[user_long_term_interest]
[user_recent_stats]
[user_price_preference]
[user_device_geo]
```

它们位于历史序列之前，并对后续所有 history / target 位置可见。不要把 385 个字段各自当作一个时序 token。

### History event token

每个事件建议包含：

```text
item/content id embedding
+ category / brand / shop
+ compact price and quality features
+ action embedding
+ behavior/source embedding
+ query/domain embedding
+ relative time embedding
```

可选择“一次交互一个融合 token”的工程版本，也可使用论文更忠实的 content/action 交替 token。

### Target token

当前候选 target token 只使用当前请求时可见的信息：

```text
selected item features + creative features + query/request context
```

目标 click/conversion action 不能出现在输入中。

## 6.4 多候选 target-aware mask

同一请求可追加 `C` 个 target positions。mask 必须保证：

- 每个 target 能读取 context 与全部历史；
- target 不能读取未来行为；
- 不同 candidate target 默认互相不可见，避免候选间标签 / 顺序泄漏；
- 只有明确做 listwise ranking 时，才设计候选间交互。

输出只取 target positions，不做整个序列 mean pooling。

## 6.5 Item / creative tower

当前 item + creative 展开维度为：

```text
14,195 + 238 = 14,433
```

可以使用：

1. 小型分域 MLP；
2. 修复后的 10 item + 1 creative RankMixer tokenizer，再用 1~2 个轻量 block；
3. 低秩 Auto-Split 后得到一个 `d` 维 item embedding。

首版建议简单 MLP / low-rank tower，先隔离 HSTU 用户历史塔本身的增益。之后再把 item tower 升级为 RankMixer。

## 6.6 首版 HSTU 超参数

| 项目 | 建议 V1 | 后续扩展 |
|---|---:|---:|
| model dim | 256 | 384 / 512 |
| layers | 4 或 6 | 8 / 12+ |
| heads | 4 | 4 / 8 |
| qk dim / head | 32~64 | 64~128 |
| value/hidden dim / head | 64 | 64~128 |
| max history | 128 或 256 | 512、1024、4096+ |
| context tokens | 4~8 | 按语义增加 |
| precision | BF16 | BF16 + recompute |
| input layout | jagged / packed | fused HSTU kernel |
| output | target positions | 多 target / generative |

不要直接照搬开源代码中的某个默认 12 层配置；应先匹配当前 RankMixer 的 dense params、训练 FLOPs和线上时延。

## 6.7 长序列训练的工程要求

- 使用 jagged tensor / offsets，而不是为所有用户 padding 到最大长度；
- 按 sequence length 或 `sum(L_i²)` 做动态 batch；
- 可以保持 global batch 规模，但单卡 micro-batch 往往需要下降，并使用 gradient accumulation；
- 采用 length bucketing、activation recomputation、BF16；
- 序列很长后再使用 stochastic length / 多时间尺度采样；
- 训练样本按用户 / session 产生多个 target，摊销历史 encoder 成本。

HSTU 在长度 1K~8K 时的价值更明显。若实际有效行为序列只有几十个事件，OneTrans 通常更容易先获得收益。

## 6.8 没有行为序列时不建议使用 HSTU

将 16 个静态 feature tokens 输入 HSTU，会失去：

- 内容 / 动作的序列转导定义；
- 时间偏置；
- 长度稀疏性；
- generative multi-position supervision；
- 历史计算摊销。

这种实验最多应称为 `HSTU-like static mixer`，优先级低于修复后的 RankMixer 与 OneTrans-NS。

---

## 7. OneTrans 与 HSTU 的选择

| 条件 | 推荐 |
|---|---|
| 只有静态 user/item/creative | 修复 RankMixer；可做 OneTrans-NS ablation |
| 已有 128~512 长度行为序列，静态特征非常丰富 | **OneTrans 优先** |
| 同请求候选较多，需要复用用户历史 | OneTrans KV cache 或 DLRM-HSTU |
| 计划使用 1K~8K+ 长历史 | **HSTU 更值得长期投入** |
| 希望最小改造现有 RankMixer | OneTrans：复用现有 tokenizer 作为 NS tokenizer |
| 愿意重构为 user/session 多 target 训练 | 完整 HSTU Generative Recommender |
| 当前线上时延严格、序列数据尚未准备好 | 先优化 RankMixer，不要强上 HSTU |

**对当前项目的推荐路线：OneTrans -> DLRM-HSTU -> 完整 HSTU GR。**

---

## 8. 建议实验矩阵

所有模型必须使用相同 sparse embeddings、标签成熟窗口、时间切分和采样权重；同时做参数量 / FLOPs 匹配。

| ID | 模型 | 目的 |
|---|---|---|
| RM0 | 当前等长切分 RankMixer | 原始基线 |
| RM1 | 5 user + 10 item + 1 creative | 验证字段完整性与域边界 |
| RM2 | 语义分组 + 原 mean pooling | 验证 tokenizer |
| RM3 | RM2 + `[PRED]`/attention pooling | 验证输出聚合方式 |
| RM4 | 更窄更深，如 d=384/512、4~6 层 | 在相近 FLOPs 下测试 depth |
| OT0 | OneTrans-NS，16 NS，无序列 | 验证 mixed QKV/FFN |
| OT1 | d=256,L=6,H=4,S=128 | OneTrans 首版 |
| OT2 | OT1，S=256/512 | 优先验证 length scaling |
| OT3 | d=384,L=8,S=512 | 扩大模型 |
| OT4 | Group-wise vs 低秩 Auto-Split | tokenizer ablation |
| H0 | HSTU user tower d=256,L=4,S=128 | 工程首版 |
| H1 | HSTU d=256,L=6,S=256/512 | 深度与长度 |
| H2 | H1 + 多 target / 多任务历史监督 | generative 收益 |
| H3 | 1K+ history + ragged fused kernel | 长序列能力 |

### 8.1 评估指标

离线至少包括：

- CVR AUC、UAUC/GAUC；
- CTCVR AUC；
- LogLoss；
- PR-AUC（正例稀少时更敏感）；
- calibration / ECE；
- query/request 级 NDCG 或业务 value-weighted metric；
- 新用户、长尾 item、长序列、短序列、不同流量来源分桶；
- dense params、FLOPs、samples/s、p95/p99 latency、显存。

数据切分应严格按时间；所有特征使用 impression-time snapshot。不要随机切分同一用户的未来行为到训练集、过去行为到测试集。

---

## 9. 伪代码与张量形状

### 9.1 OneTrans

```python
# Static NS features
u_ns = user_tokenizer(user_emb.flatten(1))          # [B, 5, d]
i_ns = item_tokenizer(item_emb.flatten(1))          # [B, 8, d]
c_ns = creative_tokenizer(creative_emb.flatten(1))  # [B, 1, d]
q_ns = query_tokenizer(query_emb.flatten(1))         # [B, 1, d]
pred = learned_pred.expand(B, 1, d)                  # [B, 1, d]

# Multi-behavior history; actual implementation should be jagged/packed
s = event_tokenizer(history_features, action, timestamp)  # [B, Ls, d]

# Causal order: S first, prediction carrier last
x = torch.cat([s, u_ns, q_ns, i_ns, c_ns, pred], dim=1)

for block, keep_tail in pyramid_blocks:
    x = block(x, token_type="mixed", causal=True)
    x = x[:, -keep_tail:, :]

h = x[:, -1, :]                                      # [PRED], [B, d]
z_ctr, z_cvr = heads(h)
```

### 9.2 DLRM-HSTU 风格

```python
# 385 user fields -> contextual prefix tokens
ctx = user_context_tokenizer(user_emb)                # [B, Nctx, d]

# Compact historical event representation
hist = content_mlp(history_content)
hist = hist + action_mlp(history_action)
hist = hist + time_encoder(history_timestamp)

# Selected candidate features become target tokens
candidate_target = target_tokenizer(
    selected_item_features,
    creative_emb,
    query_context,
)                                                     # [B, C, d]

# Jagged concat + target-aware mask
x, offsets, num_targets = pack(ctx, hist, candidate_target)
y = hstu(x, offsets, timestamps, num_targets)
h_user_target = gather_target_positions(y)            # [B, C, d]

# Full current candidate feature tower
h_item = item_tower(item_emb, creative_emb, query_context)  # [B, C, d]

h = h_user_target * h_item
z_ctr, z_cvr = multitask_head(h)
```

---

## 10. 训练与系统建议

### 10.1 Batch 不应继续只用固定样本数描述

RankMixer 的计算基本与固定 16 tokens 成正比，但序列模型计算高度依赖 `L`。因此需要同时定义：

- requests per batch；
- candidates per request；
- total S tokens；
- `sum(L_i²)` attention budget；
- global batch 与 micro-batch。

`B=2048` 可以保留为 global batch 目标，但不能预设 OneTrans/HSTU 也能以相同单卡 micro-batch 运行。

### 10.2 Optimizer

可沿用大规模推荐常见的双优化器：

- sparse embedding：Adagrad / row-wise Adagrad；
- dense backbone：AdamW、RMSProp 或现有生产优化器；
- dense/sparse 分别做 grad norm 监控；
- BF16；
- global gradient clipping；
- warmup + 稳定的 streaming learning-rate schedule。

不建议机械照搬论文学习率，应按 global batch、正例率、embedding 更新频率重新调参。

### 10.3 5.5 亿日样本下的优先级

数据量已经足够大，首要瓶颈通常变成：

1. 特征与标签时间一致性；
2. 请求级去重和历史复用；
3. 稀疏 embedding 通信 / 存储；
4. 序列长度与 attention kernel；
5. 参数扩展后的线上时延。

因此优先扩大“有效行为历史长度”和监督密度，而不是首先把隐藏维度从 256 扩到 768。

---

## 11. 最终实施计划

### Phase 0：一周内可完成的强基线

- 将 RankMixer 改为完整字段边界的 `5 user + 10 item + 1 creative`；
- 增加 query/context 独立分组；
- 明确 clicked-only CVR 或 ESMM 全空间目标；
- 建立时间切分、标签成熟和 calibration 指标。

### Phase 1：OneTrans-S

- 准备 128/256 长度多行为历史；
- `d=256, layers=6, heads=4, L_NS=16`；
- 先 Group-wise / 低秩 Auto-Split；
- pre-RMSNorm、mixed causal QKV/FFN、pyramid；
- 使用最后 `[PRED]` token；
- 按请求组织候选，后续加入 KV cache。

### Phase 2：DLRM-HSTU

- 4~8 个 user context prefix tokens；
- compact history event tokens；
- target-aware HSTU user tower；
- 完整 item+creative 独立 tower；
- element-wise interaction + CTR/CVR/CTCVR heads；
- jagged packed kernel。

### Phase 3：完整 Generative HSTU

- content/action 交替序列；
- 一条 user/session 序列产生多个 target loss；
- 1K+ 历史；
- stochastic length；
- 多候选推理摊销与 cache。

---

## 12. 参考资料

- [RankMixer: Scaling Up Ranking Models in Industrial Recommenders](https://arxiv.org/abs/2507.15551)
- [OneTrans: Unified Feature Interaction and Sequence Modeling with One Transformer in Industrial Recommender](https://arxiv.org/abs/2510.26104)
- [Actions Speak Louder than Words: Trillion-Parameter Sequential Transducers for Generative Recommendations (HSTU)](https://arxiv.org/abs/2402.17152)
- [Meta Generative Recommenders / HSTU official implementation](https://github.com/meta-recsys/generative-recommenders)
- [Entire Space Multi-Task Model for Post-Click CVR](https://arxiv.org/abs/1804.07931)
