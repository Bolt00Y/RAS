# 6. 实验矩阵、评估与上线顺序

## 6.1 公平对比原则

所有模型必须固定：

- 同一份时间切分数据；
- 同一转化成熟窗口；
- 同一 sparse embedding 输入；
- 同一负采样与样本权重；
- 同一 global batch 或等价 token/sample throughput；
- 同一训练天数或总样本数；
- 同一 early-stop 规则；
- 同一 calibration 方法。

至少报告两类预算：

1. **等参数预算**：dense parameters 接近；
2. **等 FLOPs/延迟预算**：训练 FLOPs或在线 p99 接近。

否则无法判断收益来自结构还是纯扩容。

## 6.2 数据切分

建议按时间滚动：

```text
Train       : D-15 ~ D-2
Validation  : D-1（标签已成熟）
Test        : D（离线冻结，仅在标签成熟后统计）
```

若转化窗口为 7 天，则所有 split 都需向后等待 7 天成熟；不要让 test 特征使用未来统计。

同时保留 next-batch evaluation：

1. 对当前 batch 在 eval mode 记录预测；
2. 再用该 batch 更新模型；
3. 按天聚合 AUC/UAUC/LogLoss。

该方式更贴近持续训练系统，但最终仍需独立冻结测试集。

## 6.3 基线与实验编号

### 基础基线

| ID | 模型 |
|---|---|
| B0 | 当前 RankMixer（现网实现） |
| B1 | MLP/DCNv2 类轻量基线（用于 sanity check） |
| B2 | P0 Static OneTrans-S |

### 创新方案

| ID | 模型 |
|---|---|
| E1 | P1 Entity-Hierarchical OneTrans |
| E2 | P2 Funnel Task-Token OneTrans |
| E3 | P3 Search-Bias-Decoupled OneTrans |
| E4 | P4 Low-Rank Adaptive OneTrans |

### 组合方案

| ID | 模型 |
|---|---|
| C1 | P1 + P2 |
| C2 | P1 + P3 |
| C3 | P2 + P3 |
| C4 | P1 + P2 + P3 |
| C5 | C4 teacher -> P4 student |

不要一开始直接训练 C4。先确认每个增量模块的独立收益。

## 6.4 第一阶段最小实验矩阵

| 实验 | Tokenizer | Backbone | Objective | 目的 |
|---|---|---|---|---|
| X0 | 当前 | RankMixer | 当前 CVR | 现网锚点 |
| X1 | Global Auto-Split 12 | P0 | clicked BCE | OneTrans 静态基线 |
| X2 | Entity 5/9/1/1 | P1 | clicked BCE | 实体边界 |
| X3 | Entity + 3 task | P2 | ESMM + direct CTCVR | 全空间 CVR |
| X4 | Entity + heavy/bias split | P3 | pointwise + pairwise | 搜索偏置 |
| X5 | Entity low-rank | P4 | 与最佳目标一致 | 压缩 |
| X6 | 最佳完整方案 | full | best | teacher |
| X7 | X6 -> P4 | distilled | best | serving student |

## 6.5 模型规模 sweep

### Width/depth

```text
S:  d=192, layers=4, heads=3/6
M:  d=256, layers=6, heads=4
L:  d=384, layers=8, heads=6
```

优先加深再加宽，但在线受串行延迟限制。每个 sweep 保持 tokenizer 策略不变。

### Token 数

```text
P0: 8 / 12 / 16
P1: 12 / 16 / 20
```

关注 item token 数；835 个 item 字段远多于 creative。

### FFN ratio

```text
2 / 4 / 6
```

若使用完整 token-specific FFN，ratio=4 已可能占主要参数；不要默认越大越好。

## 6.6 指标

### 离线效果

必须同时报告：

- CVR AUC：点击空间；
- CVR UAUC：仅统计正负样本均存在的用户；
- CTCVR AUC/UAUC：整个曝光空间；
- LogLoss / NCE；
- PR-AUC：转化极稀疏时更敏感；
- ECE 与 calibration curve；
- request/query GAUC；
- 分桶 AUC、bias 和 calibration。

### 分桶

- CTR decile；
- user 活跃度；
- item/creative 冷启动；
- 广告主与 campaign 冷启动；
- query 频次和长度；
- position/page/slot；
- bid/price；
- conversion delay；
- 行业/类目；
- 新老素材。

### 系统

- dense params；
- train TFLOPs/sample；
- samples/s；
- MFU；
- peak HBM；
- online batch 1/32/100 的 p50/p95/p99；
- request 候选数变化下的延迟；
- prefix cache hit rate；
- 预测稳定性和数值溢出率。

## 6.7 显著性

5.5 亿样本会让极小离线差异也“统计显著”。上线前更关注：

- 多天方向一致；
- 用户级 bootstrap confidence interval；
- 不同 query/行业/广告主分桶一致；
- calibration 不恶化；
- 业务价值函数有实质变化。

不要只用单天全局 AUC 排序模型。

## 6.8 推荐优先级

### 有全曝光 click/convert 标签

```text
P0 -> P2 -> P1+P2 -> P3 -> P1+P2+P3 -> P4 distillation
```

P2 优先，因为它直接处理 CVR 的样本空间问题。

### 只有 clicked samples

```text
P0 -> P1 -> P3 -> 收集全曝光标签/补 CTR 任务 -> P2 -> P4
```

此时 P2 不能发挥 ESMM 的主要价值。

## 6.9 上线分阶段

1. Shadow：只记录预测、延迟、cache hit 与 calibration。
2. 1% 流量：验证错误率、p99、收益方向。
3. 5%/10%：观察 query/广告主/行业分桶。
4. 50%：确认预算、竞价和 pacing 不被异常扰动。
5. Full：保留 RankMixer 快速回滚开关。

对 P3，必须单独监控 bias logit；对 P2，必须单独监控 pCTR、pCVR、pCTCVR product/direct 的一致性。

## 6.10 最终推荐配置

在拥有全曝光标签的前提下，推荐最终 teacher：

```text
P1 + P2 + P3
d_model=256
layers=6
heads=4
tokens:
  user=5
  item=9
  creative=1
  tasks=3
Pre-RMSNorm
causal / structured block mask
token-specific QKV + FFN
ESMM product + direct CTCVR consistency
FiLM heavy-feature late fusion
additive bias net
```

验证效果后，用 P4：

```text
rank=16 or 32 adapters
fixed NS-query schedule
teacher distillation
```

构建在线 student。

## 6.11 Go / No-Go 标准

Go：

- CVR 与 CTCVR 指标同时不退化；
- calibration 可修复且稳定；
- 关键冷启动和长尾桶不退化；
- p99 满足线上预算；
- 多天/多流量段方向一致。

No-Go：

- 仅 clicked-space CVR 提升，exposure-space CTCVR 下降；
- 收益主要来自 position/bias shortcut；
- request 候选变化导致排序大幅不稳定；
- cache 失效后 p99 超预算；
- direct CTCVR 与 product CTCVR 长期严重分离；
- offline AUC 提升但 GMV/ROI/转化价值下降。
