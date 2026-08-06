# 01｜尽量贴近原论文的 HSTU-CVR 基线

## 1. 设计目标

本方案优先保留 HSTU 原论文的核心建模假设，而不是追求第一版就达到最低线上延迟。它用于回答：

> 在推广搜索数据中，把用户历史重写为内容—动作序列，并使用 HSTU 的 pointwise aggregated attention 和目标位置监督，是否能相对静态 RankMixer/MLP 获得稳定 CVR 增益？

忠实性来自以下五点：

1. 用户历史按时间组织；
2. 内容 token 与动作 token 交错；
3. 当前候选作为待预测内容位置追加在序列尾部；
4. 使用 HSTU 原生的 SiLU attention、相对位置/时间偏置和 gated output；
5. 在候选位置预测动作，不做全序列 mean pooling。

## 2. 输入序列

### 2.1 上下文前缀

将 385 个 user 域字段按语义压缩为 `C=8` 个上下文 token：

```text
c_profile
c_long_interest
c_consumption
c_device_geo
c_query_intent
c_request_scene
c_time_context
c_recent_statistics
```

实际组名必须由字段注册表决定。若一个组不存在，不能用无关字段硬凑；若 query/request 字段很多，可以增加到 10–12 个 token。

输出：

```text
X_context ∈ R[B, C, d]
```

### 2.2 历史内容 token

历史第 `i` 个曝光事件包含当时可见的 item、creative、query/context 子集：

```text
item_i:     selected semantic groups from 835 fields
creative_i: selected groups from 14 fields
request_i:  historical query/scene groups
```

编码为：

```text
φ_i = EventEncoder(item_i, creative_i, request_i) ∈ R^d
```

推荐 EventEncoder：

```text
field 17 -> 32
semantic gated pooling
concat selected group tokens
MLP: [G*32 -> 2d -> d]
RMSNorm
```

不得把完整 20,978 维向量重复放到每个时间步。

### 2.3 历史动作 token

对每个历史内容事件构造动作集合，例如：

```text
impression
click
long_click / dwell bucket
add_cart
favorite
order
pay / conversion
negative feedback
```

动作可以是 bitmask，也可以是多热集合。首版建议：

```text
a_i = action_type_embedding
    + dwell_bucket_embedding
    + conversion_type_embedding
    + time_gap_embedding
```

再投影到 `R^d`。当前待预测候选没有已知动作，不能把标签 action 写入输入。

### 2.4 当前候选 token

对同一请求的 `M` 个候选分别编码：

```text
τ_j = CandidateEncoder(item_j, creative_j, query_j, request_context) ∈ R^d
```

候选编码可与历史 EventEncoder 部分共享，但建议保留独立的 target projection 或 target-type embedding，以区分“已观察历史事件”和“待预测候选”。

### 2.5 忠实交错序列

完整序列为：

```text
X = [c_1, ..., c_C,
     φ_1, a_1,
     φ_2, a_2,
     ...,
     φ_T, a_T,
     τ_1, ..., τ_M]
```

张量长度：

```text
L = C + 2T + M
```

例如 `C=8, T=128, M=16`：

```text
L = 8 + 256 + 16 = 280
X ∈ R[B_user, 280, 256]
```

注意 `B_user` 是用户/请求序列批次，不一定等于原先曝光批次 2048。真正控制训练吞吐的是批次内 candidate target 总数和 token 总数。

## 3. Candidate-isolated causal mask

若一次追加多个候选，必须防止候选之间相互读取。对候选 `τ_j`：

- 可以读取全部上下文和历史；
- 可以读取自己；
- 不能读取其他候选 token；
- 历史位置不能读取当前候选。

用集合表示：

```text
AllowedKeys(τ_j) = context ∪ history ∪ {τ_j}
```

严格单候选版本 `M=1` 最接近原始逐目标预测，适合先验证正确性；多候选隔离版本用于提升训练和推理效率。

## 4. HSTU Layer

设输入为 `Z ∈ R[L,d]`。每层执行：

```text
X = Norm(Z)
U, V, Q, K = Split(SiLU(f1(X)))
A = SiLU(QK^T + B_pos + B_time) ⊙ Mask
H = A V
Y = f2(Norm(H) ⊙ U)
Z_next = Z + Dropout(Y)
```

其中：

- 不使用 softmax；
- `B_pos` 是相对位置 bias；
- `B_time` 是相对时间 bucket bias；
- `Mask` 是因果、candidate-isolated 且可包含线上延迟约束的 mask；
- `Norm(H) ⊙ U` 是 HSTU 的门控输出；
- 序列采用 jagged/packed 表示，不为短历史补齐到统一最大长度。

### 4.1 推荐维度

首版：

```text
d_model = 256
num_heads = 4
attention_dim/head = 64
linear_dim/head = 64
num_layers = 6
```

根据官方 STU 参数形式，每层主要 dense 权重约为：

```text
UVQK projection:
256 × (2×64 + 2×64) × 4 = 262,144

output projection:
(3×64×4) × 256 = 196,608

合计约 458,752 / layer
6 层约 2.75M
```

该估算不包含 embedding tables、tokenizer、bias、norm 和任务头。

生产扩容配置可参考官方 DLRM-HSTU 的 512 维形态：

```text
d_model = 512
num_heads = 4
attention_dim/head = 128
linear_dim/head = 128
num_layers = 5
```

主要 dense 权重约 9.18M，仍远小于高基数 embedding tables。

## 5. 候选表示与任务头

### 5.1 不做均值池化

经过 `N` 层 HSTU 后，直接取每个候选位置输出：

```text
h_seq_j = HSTU(X)[position(τ_j)] ∈ R^d
```

全序列 mean pooling 会把不同时间、动作与目标候选混在一起，丢失 target-aware 语义，不符合原论文的 ranking formulation。

### 5.2 候选独立塔

参考官方 DLRM-HSTU，可以再用一个轻量 item/creative tower 得到：

```text
h_item_j = ItemTower(current item_j, creative_j) ∈ R^d
```

然后构造：

```text
h_interact_j = h_seq_j ⊙ h_item_j
h_final_j = [h_seq_j; h_item_j; h_interact_j]
```

任务头：

```text
MLP(3d -> 512 -> num_tasks)
activation: SiLU / SwishLayerNorm
```

候选独立塔不是为了重复编码，而是让 HSTU 输出专注于 candidate-conditioned user/history representation，同时保留当前 item/creative 的稳定表征。

## 6. CVR 训练目标

### 6.1 最小忠实版本：动作多任务预测

HSTU 原论文在内容位置预测后续动作，因此基线至少输出：

```text
z_click
z_conversion
```

若 conversion 是点击后条件目标：

```text
L_click = BCEWithLogits(z_click, y_click)
L_cvr_clicked = y_click × BCEWithLogits(z_cvr, y_conv)
L = L_click + λ_cvr × L_cvr_clicked
```

此版本结构上忠实，但 CVR 仍受到 clicked-only 条件训练的稀疏性影响。

### 6.2 推荐基线：ESMM 约束

为了在全曝光空间训练：

```text
p_ctr = sigmoid(z_ctr)
p_cvr = sigmoid(z_cvr)
p_ctcvr = p_ctr × p_cvr
```

```text
L_ctr = BCE(p_ctr, y_click)
L_ctcvr = BCE(p_ctcvr, y_click × y_conv)
L = L_ctr + λ_ctcvr L_ctcvr
```

可选增加点击样本上的校准辅助项：

```text
L_clicked_cvr = y_click × BCE(p_cvr, y_conv)
L_total = L_ctr + λ1 L_ctcvr + λ2 L_clicked_cvr
```

建议首轮：

```text
λ1 = 1.0
λ2 = 0.1–0.3
```

`λ2` 必须通过 clicked-space CVR 和 entire-space CTCVR 的联合验证确定，不能只看单一 AUC。

### 6.3 历史位置辅助监督

原论文 generative training 的关键是对序列中多个位置提供动作监督。可对历史 content token `φ_i` 的输出预测其已知动作 `a_i`：

```text
L_history = average_i MultiTaskBCE(head(h_i), historical_actions_i)
```

最终：

```text
L_total = L_candidate + λ_history L_history
```

建议 `λ_history=0.1–0.2` 起步。历史 supervision 只能使用该位置之后真实发生的动作，不得把未来 item 内容泄漏给前面的 query。

## 7. 时间编码

仅用位置 index 不足以表示推广搜行为中的不规则时间间隔。对每个 token 构造：

```text
absolute_time_features:
  hour_of_day
  day_of_week

relative_time_features:
  log-bucket(rank_time - event_time)
  session_boundary
  days_since_last_click
  days_since_last_conversion
```

相对时间 bucket 示例：

```text
0–1min, 1–5min, 5–30min, 30min–2h,
2–12h, 12–24h, 1–3d, 3–7d, 7–30d, >30d
```

时间 bucket 应进入 attention bias，而不是只拼接到 token embedding。候选之间共享同一 rank time。

## 8. 首轮配置与形状

### 8.1 Smoke Test

```yaml
history_events: 64
context_tokens: 8
candidates_per_request: 4
event_layout: interleaved_item_action
d_model: 256
num_layers: 4
num_heads: 4
attention_dim_per_head: 64
linear_dim_per_head: 64
precision: bf16
```

长度：

```text
L = 8 + 2×64 + 4 = 140
```

目标：验证 mask、时间排序、label alignment、单候选与多候选输出一致性。

### 8.2 Paper-Faithful Baseline

```yaml
history_events: 128
context_tokens: 8
candidates_per_request: 8-16
event_layout: interleaved_item_action
d_model: 256
num_layers: 6
num_heads: 4
attention_dim_per_head: 64
linear_dim_per_head: 64
input_dropout: 0.20
output_dropout: 0.10
```

### 8.3 Production-Oriented Baseline

正确性和收益成立后，使用 action-aware merged event：

```text
x_i = item_event_i + action_embedding_i
```

当前候选 action embedding 固定为零。配置：

```yaml
history_events: 256-512
context_tokens: 8
candidates_per_request: 16
event_layout: merged_item_action
d_model: 512
num_layers: 5
num_heads: 4
attention_dim_per_head: 128
linear_dim_per_head: 128
```

`C=8,T=256,M=16` 时：

```text
strict interleaved L = 536
merged event L       = 280
```

在 dense attention 近似下，attention 元素数比例约为 `(536/280)^2 ≈ 3.66`。因此忠实交错版用于论文复现与消融，合并版更适合生产扩容。

## 9. 训练样本组织

### 9.1 以用户/请求为 batch 单元

不再把 2048 理解为 2048 条独立曝光，而是定义：

```text
targets_per_global_batch ≈ 2048
```

例如：

```text
128 user/request sequences × 16 candidates = 2048 targets
```

不同序列长度通过 packed jagged tensor 表示。每个 rank 按 token budget 组 batch，避免某个长历史拖慢全部 worker。

### 9.2 同一序列的多训练切点

对一个用户的长时间线，可以抽取多个 target request：

```text
history before t1 -> request at t1
history before t2 -> request at t2
...
```

必须保证每个切点的 feature snapshot 与线上一致。可在数据层缓存 event embedding references，避免重复存储。

## 10. 推理路径

### 10.1 用户历史缓存

对历史前缀执行 HSTU prefill，缓存每层 K/V：

```text
cache_key = user_id + history_version + feature_version
```

当前请求到达时只追加：

```text
query/request context delta
candidate target tokens
```

若上下文前缀随请求变化，稳定用户上下文可缓存，query/request token 作为 delta 追加；需要验证 mask 与训练结构一致。

### 10.2 候选分块

在线候选数较大时：

- 历史 K/V 只计算一次；
- 每 16/32 个候选一块；
- 候选间使用隔离 mask；
- 合并各块分数后排序。

## 11. 必做正确性测试

1. **无未来性测试**：修改候选之后的事件，不应改变该候选预测；
2. **候选隔离测试**：修改候选 B 的字段，不应改变候选 A 分数；
3. **单候选一致性**：单独打分与多候选 batch 中该候选分数应在数值误差内一致；
4. **padding/jagged 一致性**：dense reference 与 jagged kernel 输出对齐；
5. **action leakage 测试**：当前候选 action 输入必须为零；
6. **时间回放测试**：以历史日期回放时只能使用当时可见特征；
7. **label maturation 测试**：未成熟转化不得当作普通负例；
8. **排序稳定性测试**：候选输入顺序改变不应改变各候选自身分数。

## 12. 基线是否成功的判断

不能只比较模型总参数量。应固定：

- 同一训练日期与标签归因；
- 同一 candidate set；
- 相近训练 FLOPs 或相近 GPU 时间；
- 同一 embedding 表和特征版本；
- 同一数据采样。

至少对比：

```text
Static MLP
现有 RankMixer
Static-HSTU（无历史，结构控制组）
HSTU strict interleaved
HSTU merged event
HSTU + history auxiliary loss
```

只有当“有时间历史的 HSTU”稳定优于 Static-HSTU，才能把增益归因于 HSTU 的序列建模，而非只是增加参数或更换非线性层。
