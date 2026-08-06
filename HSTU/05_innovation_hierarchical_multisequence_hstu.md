# 05｜创新方案四：Hierarchical Multi-Sequence HSTU

简称 **HM-HSTU**。

## 1. 动机

把所有用户行为放入单一时间线存在三个问题：

1. **上下文污染**：大量与当前 query、候选或购买意图无关的行为占据注意力；
2. **层次缺失**：session 内短期意图与跨月长期偏好具有不同时间尺度；
3. **计算分配不合理**：高价值点击/下单行为与普通曝光使用相同序列容量。

当历史从 128 扩展到 1,000、10,000 甚至更长时，单序列 dense HSTU 的二次复杂度会成为瓶颈。HM-HSTU 将用户历史拆成多个具有明确语义的通道，并在通道内使用共享或部分共享 HSTU，再进行候选条件化融合。

本方案优先从确定性、多通道结构开始；只有在确定性通道证明有效后，才增加学习路由。

## 2. 第一阶段：确定性四通道

建议构造以下四条序列：

### 2.1 Recent Session Sequence

```text
最近一个或若干 session 内的曝光/点击事件
长度 32–128
```

作用：捕获当前短期意图、刚发生的点击和 query reformulation。

### 2.2 Query/Category Matched Long-Term Sequence

从长期历史中选择与当前 query/category 相关的事件：

```text
query category match
item category/brand match
learned low-dimensional relevance top-K
```

长度 64–256。

### 2.3 Advertiser/Brand/Shop Affinity Sequence

保留与当前候选广告主、品牌、店铺或相邻商业实体有关的历史：

```text
same advertiser/shop/brand
same business category
similar price band
```

长度 32–128。

### 2.4 High-Intent Action Sequence

仅保留高价值动作：

```text
long click
add cart
favorite
order
pay/conversion
```

长度 32–128。该序列稀疏但信噪比高。

## 3. 通道编码

### 3.1 参数共享策略

首版推荐共享 HSTU 主权重，仅增加 channel embedding：

```text
x_i^(k) = event_encoder(event_i) + channel_embedding_k
h_k = SharedHSTU(x^(k), candidate)
```

优点：

- 参数量可控；
- 各通道共享基础行为语义；
- 易于公平比较单序列与多序列。

如果通道梯度冲突显著，再采用：

```text
shared first N layers
channel-specific final 1–2 layers
```

不建议一开始为四个通道复制完整 HSTU。

### 3.2 每通道候选目标

每条通道后追加同一候选 target token：

```text
[channel_context, events_k, target_j]
```

输出：

```text
h_{k,j} ∈ R^d
```

候选集合训练时，每个通道仍使用 candidate-isolated mask。

## 4. 候选条件化通道融合

对候选 `j`：

```text
s_{k,j} = w^T tanh(W_h h_{k,j} + W_q q + W_c candidate_j)
α_{k,j} = softmax_k(s_{k,j})
h_multi_j = Σ_k α_{k,j} h_{k,j}
```

加入主时间线或 recent 通道的 residual：

```text
h_out_j = h_recent,j + W_o h_multi_j
```

这样即使某些通道为空，模型仍有稳定路径。

### 4.1 空通道处理

通道为空时：

- 使用 `empty_channel_embedding_k`；
- fusion mask 将其 gate 设为无效；
- 记录空通道比例；
- 不用全零向量直接参与 LayerNorm。

## 5. Session Hierarchy

单纯按事件平铺无法显式表达 session 边界。构造两级结构：

```text
Level 1: intra-session HSTU
Level 2: inter-session HSTU over session summaries
```

### 5.1 Session 内编码

对每个 session：

```text
S_m = HSTU_session(events in session m)
s_m = last/attention summary of S_m
```

不要 mean pooling。可使用 session end token 或最后一个可见事件表示。

### 5.2 Session 间编码

```text
H_session = HSTU_long([s_1, s_2, ..., s_K, target])
```

当前 session 的原始 recent events 可同时保留一条 bypass：

```text
h_final = Gate(h_current_session, h_long_sessions)
```

### 5.3 时间编码

- session 内：分钟级相对时间；
- session 间：小时/天级相对时间；
- session summary 包含 session query/category/action statistics；
- 所有特征必须是当时可见值。

## 6. 第二阶段：学习式多序列构造

确定性通道稳定后，可以让 router 学习事件属于哪些序列。

### 6.1 Soft Router

对历史事件 `e_i` 和当前请求 `q`：

```text
r_i = MLP([e_i; q; e_i⊙q; time_features])
p_i = softmax(r_i) ∈ R^K
```

每条通道输入：

```text
x_i^(k) = p_i[k] × e_i
```

Soft router 计算量大且每个事件进入所有通道，适合离线验证，不适合最终高效实现。

### 6.2 Top-2 Router

```text
Top2(p_i) -> route event to at most 2 channels
```

加入：

```text
L_balance = K × Σ_k mean(p_i[k]) × mean(1[event routed to k])
L_entropy = controlled router entropy regularization
```

路由必须遵守每通道 token budget，超出时按 relevance、recency 或 reservoir policy 截断。

### 6.3 防止语义崩塌

路由学习存在全部事件进入同一通道的风险。建议：

- 由确定性通道规则初始化 router；
- 前若干训练步冻结 HSTU，只训练 router adapter；
- 使用小权重负载均衡；
- 保留 recent 通道为硬路由，不允许 router 删除最新事件；
- 监控每通道事件类型、时间跨度和 action 分布。

## 7. 超长历史的 Semi-Local Attention

当单通道长度超过 1,000，可将 full causal attention 替换为半局部结构：

```text
local window K_local: 捕获相邻行为和 session 内模式
global window K_global: 保留若干全局/高价值历史位置
```

近似复杂度：

```text
O((K_local + K_global) × L)
```

而不是 `O(L²)`。

### 7.1 推荐窗口

首轮：

```yaml
K_local: 128-256
K_global: 64-128
```

全局位置选择可以是：

- 最新若干事件；
- 高意图动作；
- query/category 匹配事件；
- session summaries。

需要确保候选 target 可以读取全部被选中的 global positions。

## 8. Attention Truncation

另一种深度扩容方式：

```text
前 N1 层：处理完整长序列
选择高价值子序列 L' < L
后 N2 层：只处理子序列与候选
```

选择器可以依据：

```text
recency
query relevance
action value
first-stage attention score
```

首版建议确定性选择，避免可学习 top-k 的梯度与稳定性问题。

示例：

```yaml
full_history_length: 1024
first_stage_layers: 3
truncated_length: 256
second_stage_layers: 3
```

## 9. Stochastic Length 与负载均衡

训练时长历史导致不同 GPU 计算量差异巨大。可随机截断部分用户历史，但必须：

- 短历史尽量保持完整；
- 长历史按可控分布采样长度；
- 各 rank 的 `Σ length^γ` 尽量接近；
- 验证/推理使用完整允许长度。

定义 rank load：

```text
load_r = Σ_{u in rank r} length_u^γ
γ ∈ (1,2]
```

组 batch 时以 load 而非用户数平衡，减少同步训练 straggler。

## 10. 与当前字段规模的结合

HM-HSTU 不会在每个通道重复使用全部 1,234 字段。

推荐：

```text
history event token:
  选定 item/creative ID与核心语义组 + action + time

current candidate:
  完整语义组 tokenizer

static interaction branch:
  可与 FI-HSTU 组合，但必须在单独实验后再组合
```

长期通道可进一步使用更小 event dimension：

```text
long-term events: d_event=128 -> project to d_model
recent events:    d_event=256/512
```

## 11. 缓存与服务

### 11.1 分通道缓存

```text
cache_recent(user)
cache_query_category(user, coarse_category)
cache_affinity(user, advertiser/brand)
cache_high_intent(user)
```

不能为每个 query 建任意细粒度缓存。query-matched 长期序列应基于有限 coarse category、ANN 检索或两阶段选择。

### 11.2 请求时路径

```text
1. 获取稳定通道缓存；
2. 根据当前 query/candidate 选择相关长期事件；
3. 增量追加最近未缓存事件；
4. 共享候选 target，执行各通道 HSTU；
5. 候选条件化融合并出多任务分数。
```

若在线预算有限，可先只对 Top-N 粗排候选使用 HM-HSTU。

## 12. 实验阶段

### Stage H0：单序列控制组

```text
all history, T=256, shared HSTU
```

### Stage H1：确定性双通道

```text
recent + high-intent
```

### Stage H2：确定性四通道

```text
recent + query/category + affinity + high-intent
```

### Stage H3：Session hierarchy

```text
current session events + historical session summaries
```

### Stage H4：Top-2 learned router

只在 H2/H3 已证明多通道有效后进行。

### Stage H5：Semi-local attention / attention truncation

只在历史长度扩展到 1,000+ 且 dense attention 成为瓶颈后进行。

## 13. 消融矩阵

| ID | 通道 | 路由 | Session hierarchy | Attention | 历史长度 |
|---|---|---|---|---|---:|
| H0 | 单序列 | 无 | 否 | full | 256 |
| H1 | recent + high-intent | 硬规则 | 否 | full | 256 total |
| H2 | 四通道 | 硬规则 | 否 | full | 512 total |
| H3 | 四通道 | 硬规则 | 是 | full | 512–1024 |
| H4 | 四通道 | Top-2 | 是 | full | 512–1024 |
| H5 | 四通道 | Top-2/硬规则 | 是 | semi-local | 1K–10K |

必须额外比较：

```text
同总 token budget：单序列 vs 多序列
同总 FLOPs：full attention vs semi-local
共享 HSTU vs channel-specific layer
候选无关 fusion vs candidate-conditioned fusion
```

## 14. 重点切片

HM-HSTU 的增益应集中在：

- 长历史用户；
- 多兴趣、多类目用户；
- session 意图快速变化；
- 当前 query 与长期主兴趣不一致；
- 有少量高价值历史转化；
- 同品牌/店铺复购；
- 大量低价值曝光造成噪声的用户。

对短历史用户，如果 HM-HSTU 明显变差，需检查空通道处理和 fusion residual。

## 15. 风险

1. 确定性通道规则引入人为偏见；
2. 多通道重复事件导致算力反而增加；
3. learned router 塌缩或不稳定；
4. 同一事件跨通道引起过度计数；
5. 多缓存键导致线上系统复杂度显著上升；
6. 长历史增加训练收益但无法满足线上 p99；
7. 选择器使用当前候选未来反馈产生泄漏。

控制措施：

- 统计事件 duplication ratio；
- 固定总 token budget；
- 保留 recent residual；
- 先硬路由后学习路由；
- 路由输入只使用曝光时可见特征；
- 训练与服务共享同一序列选择器；
- 为每个阶段提供单序列回退开关。

## 16. 可证伪假设

> 在相同 token/FLOP 预算下，将用户历史按短期、query 相关、商业实体亲和和高意图动作拆分，并做候选条件化融合，应比单一平铺序列更有效；增益应主要来自长历史、多兴趣和上下文污染严重的用户。

若多序列只在增加总 token/FLOP 后提升，固定预算后不再提升，则不能说明结构本身有效。若 learned router 不优于确定性规则，则应保留更简单、可解释、易服务的硬通道版本。

## 17. 建议优先级

HM-HSTU 是高潜力但高工程成本方案。推荐在以下条件同时满足后启动：

1. 论文式 HSTU 相对静态模型已有稳定增益；
2. history length ablation 显示 256 以上仍有收益；
3. attention 或历史噪声成为明确瓶颈；
4. 在线缓存与序列选择基础设施已具备；
5. QC-HSTU 和 ES-HSTU 的收益已单独验证。

否则先做全空间多任务和 query-conditioned 候选建模，通常风险更低、验证周期更清晰。
