# 00｜问题定义与数据契约

## 1. 任务定义

推广搜索中的排序单元通常是一条曝光：

```text
(user, query/request, item, creative, context, timestamp, click, conversion)
```

当前已知输入为：

```text
user:     [B, 385, 17]
item:     [B, 835, 17]
creative: [B,  14, 17]
```

其中 `B` 是当前训练批次中的曝光数。总字段数为 1,234，总展平维度为 20,978。

### 1.1 CVR 的三个相关概率

令：

- `y_click ∈ {0,1}`；
- `y_conv ∈ {0,1}`，且有效业务链路通常要求 conversion 之前存在 click；
- `x` 为曝光时可见特征。

定义：

```text
pCTR(x)   = P(click=1 | impression, x)
pCVR(x)   = P(conversion=1 | click=1, impression, x)
pCTCVR(x) = P(click=1, conversion=1 | impression, x)
```

满足：

```text
pCTCVR(x) = pCTR(x) × pCVR(x)
```

如果训练集只保留点击样本，单一 BCE 头学习的是条件 CVR；如果线上对全部曝光候选推理，则必须关注样本选择偏差。本文默认推荐全曝光多任务训练，同时保留点击样本上的条件 CVR 评估。

## 2. 为什么需要重组数据，而不是直接更换网络层

HSTU 原始范式以用户历史（User Interaction History, UIH）为输入。当前每条曝光数据本身已经包含用户、商品和创意特征，因此不要求引入新的外部数据，但必须有以下主键：

```text
user_id
request_id / impression_id
query_id 或 query 文本/类目表示（若推广搜存在）
event_time
item/ad_id
creative_id
click_time（可空）
conversion_time（可空）
conversion_type / attribution rule
```

按 `user_id` 和 `event_time` 排序后，每一条历史曝光都可以被编码为一个 content/event token；其后续点击、加购、下单或支付可编码为 action。训练样本从“独立曝光行”重组为：

```text
UserSequenceSample {
    contextual_features,
    historical_events[0:T],
    current_request_candidates[0:M],
    labels[0:M]
}
```

这样才能获得 HSTU 的三项主要收益：

1. 利用长短期用户行为；
2. 当前候选作为 target token 读取历史；
3. 一个用户序列承载多个候选或多个时间位置的监督，摊薄历史编码成本。

## 3. 建议的数据表结构

### 3.1 曝光事实表

每行一个真实曝光，至少包含：

```text
user_id
request_id
rank_time
query/request context
item_id
creative_id
user_sparse_ids[385]
item_sparse_ids[835]
creative_sparse_ids[14]
click_label
conversion_label
click_time
conversion_time
propensity / logging policy（若可获得）
```

推荐存储 sparse IDs，而非 20,978 维 dense embedding。embedding table 由训练系统统一管理和分片。

### 3.2 用户历史索引

对每个目标曝光时间 `t`，历史只能包含 `event_time < t - online_delay_buffer` 的事件。索引可以是：

```text
(user_id, day/hour shard) -> ordered list of event references
```

每个 event reference 指向曝光事实表中的 item/creative/action/context。训练时按需 lookup 并构造 jagged sequence，避免复制整段历史。

### 3.3 请求候选集合

同一 `request_id` 下的候选应尽量同时进入一个样本：

```text
history shared once
candidate_1 ... candidate_M appended as target positions
```

训练初期可取真实曝光候选中的 `M=8–16`；若一个请求的候选更多，可按真实曝光、困难负例和随机负例组成小集合。线上可用候选分块或 KV cache 处理更大的候选集合。

## 4. 字段语义分组

禁止按 20,978 维连续位置平均切片。建议建立显式注册表：

```yaml
feature_name:
  domain: user | request | item | creative | action
  semantic_group: profile | query | category | price | statistics | ...
  availability: request_time | historical_only | post_event
  cacheability: user_cache | item_cache | request_dynamic
  cardinality: ...
  embedding_dim: 17
  leakage_risk: low | medium | high
```

### 4.1 user 385 字段

至少拆分为：

- 长期用户画像；
- 长期兴趣与消费能力；
- 近期统计特征；
- 设备、地域、网络；
- query/request/场景/流量入口；
- 时间相关上下文；
- 可能由未来窗口聚合得到的高风险特征。

推广搜索中必须找到 query/request 相关字段。如果这些信息并不存在，模型无法区分同一用户在不同检索意图下的候选相关性。

### 4.2 item 835 字段

拆分为：

- **固有、可缓存属性**：ID、类目、品牌、店铺、商品内容；
- **动态但曝光时可见属性**：价格、库存、活动、实时统计；
- **query-item 交叉或召回阶段特征**；
- **曝光后特征**：点击、停留、支付结果等，禁止输入当前目标位置。

用于历史 event 的字段必须取历史时刻快照，不能用当前时刻回填值，否则会产生时间穿越。

### 4.3 creative 14 字段

创意字段数量少，建议保留独立语义：

- creative ID / template；
- 图片、标题、文案类型；
- 落地页或素材形态；
- 创意版本、审核或展示形态。

不建议把 creative 238 维直接并入一个巨大的 item 展平向量后失去身份。应增加 `creative_type_embedding`，并在当前候选交互塔中单独保留 creative token。

## 5. Tokenizer 数据契约

### 5.1 字段级预投影

每个 17 维字段先进入字段共享或语义组共享投影：

```text
e_f ∈ R^17
z_f = RMSNorm(SiLU(W_group e_f + b_group)) ∈ R^32 or R^64
z_f += field_id_embedding + domain_embedding
```

如果为每个字段分别设置 `17 -> 32` 权重，权重规模约为：

```text
1,234 × 17 × 32 = 671,296
```

远低于直接把 20,978 维投影到 512 维所需的约 1,074 万个权重，并且保留字段边界。

### 5.2 语义组压缩

对同一语义组内字段使用 gated pooling：

```text
α_f = softmax(w_g^T tanh(W_g z_f))
group_token = Σ_f α_f z_f
token = Linear(group_token) -> d_model
```

建议：

```text
user/request context:  6–10 tokens
historical content:    每个事件 1 content token
historical action:     每个事件 1 action token（忠实方案）
current candidate:     每个候选 1 target token
```

### 5.3 历史事件字段裁剪

不应为每个历史事件重放全部 835+14 字段。先通过消融筛选：

- ID/类目/品牌/店铺/创意；
- 当时的 query/场景；
- 价格和重要状态；
- 历史动作与时间差。

完整 1,234 字段只用于当前候选的静态交互分支；历史 HSTU token 采用压缩字段集合。否则 I/O、embedding lookup 和历史存储将主导训练成本。

## 6. 时间与标签泄漏检查

以下字段必须逐一审计：

1. 未来 1/7/30 天转化率、GMV、点击统计；
2. 使用目标曝光之后事件计算的用户统计；
3. 当前请求最终排名、曝光位置或竞价结果，但线上打分时尚不可知；
4. 由转化标签衍生的订单状态、支付状态；
5. 数据回填后的商品状态；
6. click/conversion action 被写入当前候选 token；
7. 同一次 session 内线上尚未来得及进入特征平台的事件。

推荐统一使用：

```text
feature_snapshot_time <= rank_time - feature_pipeline_delay
history_event_time     <= rank_time - online_history_delay
label_time             > rank_time
```

对于 session 内近实时行为，可设置 delay-aware mask：训练时只允许读取线上确实可见的历史位置。

## 7. 转化延迟与成熟负样本

当天未观察到转化不一定是真负例。设业务归因窗口为 `W`，则：

- 正例：窗口内已转化；
- 成熟负例：`observation_time >= rank_time + W` 且未转化；
- 未成熟样本：暂不进入标准 CVR BCE，或进入带生存/删失建模的 delay head。

推荐至少维护：

```text
conversion_label_matured
conversion_delay_bucket
sample_observation_age
```

否则 5.5 亿条新鲜样本会系统性低估 CVR，且新广告、新创意受到更强偏置。

## 8. 时间切分

必须按时间划分：

```text
train: 较早日期
validation: 后续连续日期
test: 再后续连续日期
```

不允许随机打散后切分，因为同一用户、商品、创意及其未来统计会跨集合泄漏。建议额外评估：

- 新用户；
- 新 item/ad；
- 新 creative；
- 新 query 或低频 query；
- 历史极短和历史极长用户。

## 9. 只有当前行、无法构造历史时的退化方案

若数据链路确实没有 user_id/event_time，无法连接历史，可构造“Static-HSTU”用于结构试验：

```text
8 user/request semantic tokens
16 item semantic tokens
1 creative token
1 prediction token
```

输入为 `[B, 26, d]`，使用 2–4 个 HSTU block，预测 token 只能读取前面的字段 token。需要明确：

- 该模型没有用户行为序列；
- token 顺序是人工语义顺序，不是时间；
- generative multi-position supervision 不成立；
- 它更接近带 HSTU 单元的字段交互网络，而不是原论文方案。

因此该退化方案只作为工程 smoke test，不应成为最终研究结论。

## 10. 数据验收清单

在开始大规模训练前，必须通过以下检查：

- [ ] 每个目标请求可恢复至少一个历史事件；
- [ ] 历史严格早于目标请求且满足线上延迟；
- [ ] query/request 字段已从 user 域中识别并单独建模；
- [ ] item/creative 的未来统计与标签字段已排除；
- [ ] 当前候选 action embedding 被置零或完全缺省；
- [ ] 同请求候选之间不存在标签互看；
- [ ] 转化负例已成熟，或显式建模删失；
- [ ] 训练/验证/测试严格按时间切分；
- [ ] 用户历史构造、tokenizer 和线上特征处理完全一致；
- [ ] 能输出各长度 bucket、各场景和冷启动切片的数据统计。
