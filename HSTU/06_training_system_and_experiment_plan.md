# 06｜训练系统、实验矩阵与上线计划

## 1. 规模换算

每天最多 5.5 亿条曝光样本。若以 2,048 个 candidate targets 为一个全局 step：

```text
steps/day = 550,000,000 / 2,048 ≈ 268,555
```

若 24 小时完成一遍数据，需要：

```text
optimizer steps/s ≈ 3.11
candidate targets/s ≈ 6,366
```

这只是端到端最低吞吐目标，不包括数据延迟、checkpoint、评估和容错。实际系统应预留 20%–40% 余量。

HSTU 训练的 batch 不应只以“用户序列数”定义，而应同时约束：

```text
targets_per_batch
tokens_per_batch
estimated_attention_load = Σ length^γ
```

其中 dense attention 可取 `γ≈2`，使用 stochastic length 或 sparse attention 后可取 `1<γ<2`。

## 2. 数据与 embedding 系统

### 2.1 不物化全量 dense embedding

20,978 维 dense embedding 的每日体积：

```text
BF16: 550M × 20,978 × 2 bytes ≈ 23.1 TB
FP32: 550M × 20,978 × 4 bytes ≈ 46.2 TB
```

推荐优先级：

1. 保存 sparse IDs，训练时由分片 embedding table lookup；
2. 对历史 event 缓存 256/512 维压缩 token；
3. 当前候选仍读取完整 sparse fields，历史读取精选字段；
4. 禁止为每个训练切点复制整段历史 dense tensor。

### 2.2 Embedding sharding

高基数字段通常占绝大多数参数。建议：

- table-wise sharding：不同字段表分配到不同 rank；
- row-wise sharding：超大 item/user/ad 表按行切分；
- column-wise 仅在单表维度很大时考虑；
- embedding optimizer state 使用分片/低精度；
- 热点 ID 做 cache，但训练与线上 hash/version 必须一致。

可以使用 TorchRec/FBGEMM 类似能力，也可接入现有参数服务器。关键不是框架名称，而是保证 sparse lookup、jagged sequence 和 dense HSTU 的通信拓扑不会相互阻塞。

### 2.3 用户级数据分片

尽量让同一用户的历史和多个 target request 落在同一数据 worker：

```text
shard = hash(user_id) % num_data_shards
```

这样可以：

- 顺序构造历史；
- 复用 event references；
- 降低跨 worker join；
- 避免同用户时间线乱序。

## 3. Packed Jagged Batch

### 3.1 动态 batch

每条用户序列长度不同，不能统一 padding 到最大长度。数据加载器输出：

```text
values:       [total_tokens, d]
lengths:      [B_user]
offsets:      [B_user + 1]
num_targets:  [B_user]
timestamps:   [total_tokens]
```

### 3.2 Token-budget batching

示例约束：

```yaml
max_candidate_targets_per_global_batch: 2048
max_tokens_per_gpu: hardware_profiled_value
max_sequences_per_gpu: safety_limit
```

按长度 bucket 组 batch：

```text
0–64
65–128
129–256
257–512
513–1024
>1024
```

bucket 内再根据 candidate 数和 attention load 装箱。

### 3.3 示例分布式 batch

仅作为容量规划示例：

```text
64 GPUs
每 GPU 32 candidate targets
全局 targets = 2048
```

如果每个请求 16 个候选，则每 GPU 约 2 条请求序列；如果每请求 8 个候选，则约 4 条。实际 microbatch 由序列长度决定，不能固定照搬。

## 4. 精度与内存

### 4.1 第一阶段

```text
embedding output: BF16
HSTU dense compute: BF16
attention accumulation / sensitive norm: framework推荐精度
optimizer master weights: FP32 或稳定低精度实现
```

### 4.2 激活重计算

优先重计算：

```text
normalized X
U/V/Q/K projection
post-attention intermediate
```

而不是对整个模型做粗粒度 checkpoint。需要 profile 计算/显存比后决定。

### 4.3 FP8/INT4

FP8 HSTU GEMM 与 INT4 embedding 通信属于后期系统优化。只有在 BF16 模型质量、数值和线上收益稳定后再引入，并必须做：

- 相同 checkpoint 的离线数值对齐；
- AUC/LogLoss/ECE 对齐；
- 长尾 ID、极端 logit 和长序列切片；
- 线上 shadow traffic。

## 5. Optimizer 与训练日程

建议起始设置：

```yaml
optimizer: AdamW
learning_rate_dense: 1.0e-3
learning_rate_embedding: 1.0e-3 or existing proven value
weight_decay_dense: 1.0e-5 to 1.0e-4
weight_decay_embedding: 0 or table-specific
grad_clip_global_norm: 1.0
warmup_steps: 2,000-10,000
schedule: constant_then_cosine_or_production_schedule
precision: bf16
```

官方公开小数据配置中的 `lr=1e-3` 可作为量级参考，但 5.5 亿样本、分布式 embedding 和多任务 loss 需要重新调参。

### 5.1 学习率分组

```text
embedding tables
field tokenizer
HSTU backbone
new innovation modules
task heads
```

当从 HSTU baseline 增加新模块时：

```text
LR_new_module = base LR
LR_backbone = 0.2–0.5 × base LR
```

稳定后再统一微调。

### 5.2 样本顺序

按时间流式训练时：

- 每日 checkpoint；
- 保持有限 shuffle window，避免完全按时间导致短时分布相关；
- 不跨越标签归因窗口引入未成熟负例；
- 记录每日数据版本、feature version、embedding version 和归因版本。

## 6. 实验总路线

## Phase 0：数据正确性

目标：任何模型收益之前先排除泄漏和标签错误。

```text
P0.1 字段注册与 availability 审计
P0.2 用户历史构造
P0.3 candidate-isolated mask 单元测试
P0.4 conversion maturation / delay 统计
P0.5 时间回放与 train-serve parity
```

只使用小流量或数小时数据，追求正确性，不追求最终 AUC。

## Phase 1：静态与序列控制组

| ID | 模型 | 历史 | 目标 |
|---|---|---:|---|
| B0 | MLP | 0 | 静态基线 |
| B1 | 当前 RankMixer | 0 | 已有强基线 |
| B2 | Static-HSTU | 0 | 控制 HSTU block 本身 |
| B3 | HSTU strict | 64 | 验证序列收益 |
| B4 | HSTU strict | 128 | 论文式基线 |
| B5 | HSTU merged | 128 | event layout 消融 |
| B6 | HSTU merged | 256 | 生产基线 |

公平性：相同字段、相同标签、相同日期、相同采样；另提供相近 FLOPs 和相近训练时长两种对比。

## Phase 2：HSTU 核心组件消融

| 维度 | 候选值 |
|---|---|
| attention | HSTU SiLU / softmax Transformer |
| bias | position only / position + time |
| event layout | item-action interleaved / merged |
| pooling | candidate position / mean pooling（负对照） |
| history supervision | off / 0.1 / 0.2 |
| target mode | single / multi-candidate isolated |
| history length | 0/64/128/256/512/1024 |
| model width | 256/512 |
| depth | 4/6/8 |

不要做全笛卡尔积。先单变量筛选，再围绕最优区间局部搜索。

## Phase 3：创新方案单独验证

### 3.1 ES-HSTU

```text
B6
+ CTR head
+ CTCVR product constraint
+ clicked CVR auxiliary
+ history action loss
+ optional delay hazard
```

### 3.2 QC-HSTU

```text
B6
+ query prefix
+ query-conditioned bias
+ candidate-set training
+ optional context heads
```

### 3.3 FI-HSTU

```text
B6
+ semantic group gate
+ low-rank DCNv2
+ gated residual fusion
```

### 3.4 HM-HSTU

```text
B6
+ deterministic recent/high-intent channels
+ four channels
+ hierarchy/router/sparse attention in later stages
```

任何创新模块都必须与“只增加等量普通 FFN 参数”的 capacity control 比较，排除单纯参数量效应。

## Phase 4：组合

建议组合顺序：

```text
C1 = B6 + ES-HSTU
C2 = C1 + QC-HSTU
C3 = C2 + FI-HSTU
C4 = C2 + HM-HSTU
C5 = C2 + FI-HSTU + HM-HSTU（仅在各自通过时）
```

不默认 C5 一定最好。FI-HSTU 与 HM-HSTU 可能争夺容量或增加过多延迟。

## 7. 核心指标

### 7.1 离线概率指标

```text
AUC / GAUC
PR-AUC（稀疏转化更敏感）
LogLoss / Normalized Entropy
ECE / reliability diagram
Brier score
```

分别报告：

```text
CTR on entire impression space
CTCVR on entire impression space
CVR on clicked space
mature conversion window
fresh/partially observed windows（若有 delay model）
```

### 7.2 排序指标

同请求候选集合上：

```text
NDCG@K
MRR / Recall@K（视标签密度）
pairwise accuracy
```

CVR 极稀疏时，按请求计算的排序指标方差很大，应报告 bootstrap 置信区间。

### 7.3 校准

推广出价通常对概率绝对值敏感，必须评估：

- 全局 ECE；
- query 频次 bucket；
- item/creative 新旧；
- slot/位置；
- 价格/出价 bucket；
- 历史长度；
- 转化延迟 bucket；
- 国家、设备、流量入口。

模型上线前可做轻量校准，但校准不能掩盖模型在关键切片的系统性错误。

### 7.4 系统指标

```text
training examples/s
tokens/s
GPU utilization
embedding communication time
attention time
p50/p95/p99 training step time
HBM peak
checkpoint size/time
serving p50/p95/p99 latency
user-history cache hit rate
candidate throughput
```

## 8. 业务指标与在线实验

离线通过后，online A/B 至少关注：

```text
CVR / CTCVR
转化数
GMV / revenue / advertiser value
ROI / CPA guardrail
CTR（防止过度牺牲点击）
用户体验与负反馈
广告填充、竞价稳定性
p99 latency / timeout / cache miss
```

不得只以 CTR 或离线 AUC 决定 CVR 模型是否上线。

### 8.1 上线节奏

```text
离线回放
shadow serving
1% traffic
5% traffic
逐步扩量
```

每阶段需检查概率校准、竞价分布、超时和冷启动切片。模型 score 分布变化大时，先进行 bidding simulation。

## 9. 统计检验

- 按用户或请求聚类 bootstrap，避免把同用户大量曝光当作独立样本；
- 报告置信区间，而不只报告小数点后增益；
- 多实验并行时控制 false discovery；
- 线上按业务主指标预先注册停止规则；
- 不根据短期噪声反复调整实验终止时间。

## 10. 历史长度与算力曲线

需要画出：

```text
quality vs history length
quality vs training FLOPs
quality vs serving latency
quality vs HBM
```

建议长度：

```text
0, 64, 128, 256, 512, 1024
```

只有当 512/1024 继续产生稳定边际收益时，才进入 semi-local attention、context parallelism 或 10K 级历史工程。

## 11. 推理优化顺序

1. 多候选共享历史编码；
2. 用户历史 KV cache；
3. item/creative group token cache；
4. event item-action 合并；
5. candidate chunking；
6. jagged fused kernels；
7. attention truncation / semi-local attention；
8. FP8 GEMM；
9. INT4 embedding communication。

顺序原则：先消除重复计算，再改变精度或 attention 近似。

## 12. 线上一致性

必须版本化：

```text
feature registry version
embedding table version
tokenizer version
history selection version
mask/delay policy version
model checkpoint
calibration version
```

训练和服务共享：

- 同一字段默认值；
- 同一 hash；
- 同一时间 bucket；
- 同一历史可见性；
- 同一 candidate tokenizer；
- 同一 item-action 合并规则。

## 13. 故障与降级

线上提供层级回退：

```text
Level 0: 完整 HSTU + innovations
Level 1: HSTU without optional interaction/multi-sequence modules
Level 2: cached user representation + candidate tower
Level 3: existing RankMixer/MLP
```

缓存失效、序列过长或 kernel 异常时不能直接返回随机分数。

## 14. 决策门槛

每个阶段进入下一阶段前至少满足：

1. 正确性测试全部通过；
2. 关键离线指标在多个时间窗口方向一致；
3. 增益不是由泄漏、采样或参数量单独解释；
4. 概率校准没有不可接受恶化；
5. 系统成本在预算内，或存在明确优化路径；
6. 冷启动、tail query 和短历史没有严重回退；
7. 线上回退路径已验证。

## 15. 推荐的最小可行实验包

在资源有限时，优先完成以下 8 个实验：

```text
1. RankMixer static baseline
2. Static-HSTU control
3. HSTU strict T=64
4. HSTU strict T=128
5. HSTU merged T=256
6. merged HSTU + ESMM CTR/CTCVR
7. 实验 6 + query prefix/candidate isolation
8. 实验 7 + compact DCNv2 fusion
```

这组实验能够依次回答：

- HSTU block 本身是否有用；
- 时间历史是否有用；
- 原始交错是否值得额外计算；
- 全空间 CVR 目标是否有效；
- query-conditioned candidate modeling 是否有效；
- 静态字段交互是否与 HSTU 互补。

HM-HSTU 应在上述问题得到肯定答案后再投入。
