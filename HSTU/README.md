# HSTU 在推广搜索 CVR 预测中的建模研究

本目录面向推广搜索（Sponsored Search / Ads Ranking）中的 CVR 预测，给出一套尽量贴近 HSTU 原论文范式的可复现方案，以及四套具有明确问题动机、可验证增量和工程落地路径的创新方案。

## 1. 已知数据条件

每天最多可使用约 **5.5 亿条曝光日志**。每条日志经过 sparse embedding 后包含：

| 特征域 | 字段数 | 单字段 embedding | 展平维度 |
|---|---:|---:|---:|
| user | 385 | 17 | 6,545 |
| item | 835 | 17 | 14,195 |
| creative | 14 | 17 | 238 |
| 合计 | 1,234 | 17 | 20,978 |

最终目标为 CVR。这里的 CVR 必须先区分两种统计口径：

1. `pCVR = P(conversion | click, impression)`：点击后转化率；
2. `pCTCVR = P(click & conversion | impression)`：全曝光空间的点击且转化概率。

生产排序通常同时需要二者，不能把“在点击样本上训练的 CVR”直接等同于全曝光空间目标。

## 2. 核心结论

### 2.1 HSTU 不是静态字段 Mixer

HSTU 原论文的核心贡献不是把 1,234 个字段排成一个长度为 1,234 的序列，而是：

- 按用户和时间组织历史内容与动作；
- 将推荐问题改写为 sequential transduction；
- 在历史序列后追加当前候选；
- 在候选位置读取 candidate-aware 表示并预测动作；
- 用同一条长序列监督多个历史位置，从而摊薄用户历史编码成本。

因此，只拥有当前曝光行的 user/item/creative embedding 时，可以实现“使用 HSTU block 的静态模型”，但不能称为忠实复现 HSTU。最重要的数据工程工作是：使用同一批曝光/点击/转化日志，按 `user_id + event_time` 聚合为用户历史，而不是增加新的外部特征。

### 2.2 不应把 20,978 维历史向量原样存储或送入每个时间步

5.5 亿条样本若把 20,978 维向量全部物化：

- BF16 约为 `550M × 20,978 × 2 bytes ≈ 23.1 TB/day`；
- FP32 约为 `46.2 TB/day`。

建议保留 sparse IDs 并由分布式 embedding lookup 在线构造，或者离线缓存 256/512 维事件向量：

- 256 维 BF16：约 281.6 GB/day；
- 512 维 BF16：约 563.2 GB/day。

### 2.3 字段必须按语义分组，而不是平均切片

推荐先将 1,234 个字段归入约 20–40 个语义组，例如：

- 用户长期画像、短期兴趣、设备、地域；
- query/request、场景、时段、流量入口；
- item ID/类目/品牌/店铺/价格/统计特征；
- creative 模板、素材类型、文案/图片侧特征；
- 只在历史位置可见的行为与反馈。

每个字段仍为 17 维，先做共享或分组的 `17 -> 32/64` 投影，再在组内 gated pooling，最后投影到 HSTU 的 `d_model`。必须避免切断某个 17 维字段 embedding，也必须排除标签泄漏和未来统计量。

## 3. 推荐实施路线

### 阶段 A：数据契约与无泄漏基线

1. 确认曝光、点击、下单/支付、归因时间；
2. 明确哪些 user 字段实际属于 query/request context；
3. 将 item/creative 字段划分为可缓存的固有属性、请求相关属性和曝光后才产生的属性；
4. 构造严格时间切分与成熟负样本；
5. 保留现有 RankMixer/MLP 作为强静态基线。

### 阶段 B：论文式 HSTU 基线

推荐首版张量：

```text
context prefix:             [B, C=8, d]
historical content/action:  [B, 2T, d]   # 忠实交错版本
current candidates:         [B, M, d]
full sequence:              [B, C+2T+M, d]
candidate outputs:          [B, M, d]
```

首轮配置：

```yaml
history_events: 128
d_model: 256
num_layers: 6
num_heads: 4
attention_dim_per_head: 64
linear_dim_per_head: 64
context_tokens: 8
candidates_per_request: 8-16
input_dropout: 0.20
output_dropout: 0.10
precision: bf16
```

完成正确性验证后，再切换到 item/action 合并事件、`T=256/512`、`d=512` 的生产基线。

### 阶段 C：按优先级验证创新方案

| 优先级 | 方案 | 主要解决的问题 | 预计改造复杂度 |
|---:|---|---|---|
| P0 | Entire-Space Causal Multi-Task HSTU | CVR 样本选择偏差、正例稀疏、延迟反馈 | 中 |
| P0 | Query-Conditioned Candidate-Set HSTU | 推广搜意图、候选集合训练、位置/slot 上下文 | 中 |
| P1 | Field-Interaction Augmented HSTU | 长序列压缩后丢失当前 user-item-creative 显式交叉 | 低至中 |
| P2 | Hierarchical Multi-Sequence HSTU | 超长历史、上下文污染、多兴趣通道竞争 | 高 |

不建议一次性叠加四个方案。应先复现基线，然后每次只增加一个变量，最后再组合通过单模块消融验证过的组件。

## 4. 文档索引

- [`00_problem_and_data_contract.md`](00_problem_and_data_contract.md)：问题定义、数据组织、泄漏检查、静态退化方案；
- [`01_paper_faithful_hstu_cvr.md`](01_paper_faithful_hstu_cvr.md)：尽量贴近原论文的 HSTU-CVR 基线；
- [`02_innovation_query_conditioned_candidate_hstu.md`](02_innovation_query_conditioned_candidate_hstu.md)：查询条件化与多候选隔离；
- [`03_innovation_entire_space_multitask_hstu.md`](03_innovation_entire_space_multitask_hstu.md)：全空间因果多任务 CVR；
- [`04_innovation_field_interaction_hstu.md`](04_innovation_field_interaction_hstu.md)：HSTU 与显式字段交互双塔融合；
- [`05_innovation_hierarchical_multisequence_hstu.md`](05_innovation_hierarchical_multisequence_hstu.md)：分层、多序列与长历史稀疏化；
- [`06_training_system_and_experiment_plan.md`](06_training_system_and_experiment_plan.md)：5.5 亿样本/日下的训练系统、实验矩阵与上线指标；
- [`configs/hstu_cvr_baseline.yaml`](configs/hstu_cvr_baseline.yaml)：建议起始配置；
- [`references.md`](references.md)：论文与官方实现资料。

## 5. 最终推荐架构

在当前约束下，建议最终目标不是“纯 HSTU 替换 RankMixer”，而是以下推广搜专用结构：

```text
                           ┌──────────────────────────┐
user history + actions ──> │ target-aware HSTU tower │ ──> h_seq(candidate)
query/request prefix ─────>│                          │
current candidate token ──>│                          │
                           └──────────────────────────┘

current user/item/creative semantic groups
          └──────────────> compact DCNv2 / gated interaction tower ──> h_cross

h = h_seq + sigmoid(Wg[h_seq; h_cross]) ⊙ Ws(h_cross)

h ──> CTR head
  ├─> conditional CVR head
  ├─> CTCVR = pCTR × pCVR
  └─> optional add-cart/order/payment/delay heads
```

该结构保留 HSTU 对用户历史和候选条件化的优势，同时保留推广搜索排序中极其重要的当前 query-user-item-creative 显式交互，并用全曝光多任务目标解决 CVR 的统计偏差问题。

## 6. 重要前提

若 385 个 user 字段中完全没有 query、request、场景或检索意图信息，那么模型最多是“广告推荐 CVR 模型”，不能充分建模推广搜索。此时第一优先级不是加深 HSTU，而是完成字段归属审计并接入检索请求上下文。
