# 03｜创新方案二：Entire-Space Causal Multi-Task HSTU

简称 **ES-HSTU**。

## 1. 动机

推广搜索 CVR 的主要困难通常不是网络表达能力不足，而是监督分布本身存在问题：

1. 条件 CVR 只在点击样本上有直接监督；
2. 点击样本占全曝光的小部分，转化又是点击样本的小部分；
3. 模型在线上需要对全候选空间推理，训练空间与推理空间不一致；
4. 转化存在数小时或数天延迟，新鲜未转化样本可能仍是潜在正例；
5. 点击、加购、下单、支付等任务相关但不完全一致，简单完全共享容易产生负迁移。

ES-HSTU 将 HSTU 的历史动作生成式监督与全曝光 CVR 因果链结合，重点提升监督有效样本量和概率一致性。

## 2. 基础概率链

### 2.1 二阶段链路

```text
impression -> click -> conversion
```

定义：

```text
p_ctr   = P(click | impression, x)
p_cvr   = P(conversion | click, impression, x)
p_ctcvr = P(click & conversion | impression, x)
```

约束：

```text
p_ctcvr = p_ctr × p_cvr
```

### 2.2 多阶段漏斗

若日志有加购、下单、支付，可扩展：

```text
impression -> click -> add_cart -> order -> pay
```

不建议直接用五个彼此独立 sigmoid，因为可能出现概率逆序。可以使用条件概率分解：

```text
p_click = σ(z_click)
p_cart_given_click = σ(z_cart)
p_order_given_cart_or_click = σ(z_order)
p_pay_given_order = σ(z_pay)

p_view_cart  = p_click × p_cart_given_click
p_view_order = p_view_cart × p_order_given_cart_or_click
p_view_pay   = p_view_order × p_pay_given_order
```

业务链路不严格时，应按实际可达图构造，不要强行假设每次支付都经过加购。

## 3. 共享 HSTU Backbone 与任务表示

对候选位置获得：

```text
h_j = HSTU(sequence)[candidate_j]
```

建议 backbone 共享，但任务 head 不完全共享：

```text
h_shared = MLP_shared(h_j)
h_ctr = Adapter_ctr(h_shared)
h_cvr = Adapter_cvr(h_shared)
h_value = Adapter_value(h_shared)
```

首版使用低秩 adapter 即可：

```text
Adapter_t(h) = h + A_t SiLU(B_t h), rank=16/32
```

只有在观察到显著任务冲突后，才引入 MMoE/PLE。不要首版就使用大规模 sparse MoE，因为 CVR 正例稀疏时路由容易失衡，且会混淆 HSTU 本身的收益。

## 4. 候选级损失

### 4.1 全曝光 CTR 与 CTCVR

```text
p_ctr = sigmoid(z_ctr)
p_cvr = sigmoid(z_cvr)
p_ctcvr = p_ctr * p_cvr

y_ctcvr = y_click * y_conversion
```

```text
L_ctr = BCE(p_ctr, y_click)
L_ctcvr = BCE(p_ctcvr, y_ctcvr)
```

这两个损失都在全曝光空间计算。

### 4.2 点击样本条件 CVR 辅助项

```text
L_cvr_clicked = y_click * BCE(p_cvr, y_conversion)
```

该项只作为校准辅助，不应替代 CTCVR 全空间约束。

### 4.3 正例稀疏处理

推荐优先使用：

- 按天/场景稳定的正负样本权重；
- logit adjustment 或 calibrated class weight；
- 全量曝光 + 分布式训练；

谨慎使用大比例正例过采样。若过采样，必须记录 sampling probability，并在 loss 或校准阶段还原先验。

总损失：

```text
L_candidate =
    λ_ctr L_ctr
  + λ_ctcvr L_ctcvr
  + λ_clicked L_cvr_clicked
```

起始值：

```yaml
lambda_ctr: 1.0
lambda_ctcvr: 1.0
lambda_clicked_cvr: 0.2
```

权重最终按各 loss 的梯度范数、校准和业务目标调节，而不是只让数值大小相近。

## 5. 历史动作生成式监督

HSTU 的重要优势是同一序列可在多个历史内容位置监督动作。对历史事件 `i` 的 HSTU 输出 `h_i` 预测：

```text
click_i
add_cart_i
order_i
conversion_i
dwell_bucket_i
negative_action_i
```

```text
L_history = Σ_i Σ_t w_{i,t} BCE(Head_t(h_i), y_{i,t})
```

其中 `w_{i,t}` 可考虑：

- 动作成熟度；
- 曝光 propensity；
- 不同历史位置的 recency；
- 行为可靠性。

最终：

```text
L = L_candidate + λ_history L_history
```

建议 `λ_history=0.1–0.2`。如果历史监督权重过大，backbone 可能更擅长重建过去而非预测当前推广搜转化。

## 6. 延迟反馈与删失

### 6.1 成熟样本基线

最稳妥方案是只在 `rank_time + attribution_window <= observation_time` 时计算 conversion negative：

```text
mature_mask = observation_age >= attribution_window
L_ctcvr = mature_mask * BCE(...)
```

但这会牺牲新鲜数据。

### 6.2 离散时间 Hazard Head

将转化延迟划为 `K` 个时间桶：

```text
0–10min, 10–30min, 30min–2h, 2–6h,
6–24h, 1–3d, 3–7d, >7d
```

输出每个 bucket 的条件 hazard：

```text
h_k = P(convert in bucket k | not converted before k, click, x)
```

累计转化概率：

```text
p_convert_by_K = 1 - Π_k (1 - h_k)
```

对已转化样本，在真实延迟 bucket 计算 event likelihood；对未成熟样本，只对已观察区间计算 survival likelihood。这样未成熟负样本不必被直接丢弃或错误标负。

### 6.3 简化版 Delay Calibration

若 Hazard Head 工程成本高，可先：

1. 主模型只用成熟标签；
2. 另建 delay correction head，输入 observation age；
3. 在线仍输出固定归因窗口下的校准概率。

必须避免将 `observation_age` 作为线上目标曝光输入，因为在线预测时未来观察年龄不存在；它只能服务于训练 likelihood 或标签校正。

## 7. Counterfactual Regularization（可选）

ESMM 的乘法约束改善全空间训练，但并不自动消除所有曝光/点击策略偏差。若日志有可靠 propensity 或可训练点击 propensity，可增加 IPS/DR 风格正则：

```text
w_i = clip(1 / propensity_i, 0, w_max)
L_ips = y_click * w_i * BCE(p_cvr, y_conversion)
```

为了降低高方差，建议：

- propensity clipping；
- self-normalized IPS；
- doubly robust residual；
- 仅作为小权重正则，不替代主 ESMM loss。

如果 propensity 不可靠，不应伪造无偏性结论。此时把该实验标为 sensitivity analysis。

## 8. 任务冲突与门控专家

### 8.1 先诊断再引入 MMoE

记录 backbone 共享参数上的任务梯度余弦：

```text
cos(g_ctr, g_cvr)
cos(g_click, g_pay)
```

若长期显著为负，且任务 adapter 无法缓解，再使用 2–4 个 dense experts：

```text
expert_e = FFN_e(h)
gate_t = softmax(W_t h)
h_t = Σ_e gate_t[e] expert_e(h)
```

每个任务拥有独立 gate。首版不使用 sparse top-k，以避免跨卡路由与负载均衡复杂度。

### 8.2 Funnel-aware experts

可设置具有归纳偏置的专家：

```text
engagement expert
purchase-intent expert
price/value expert
query-relevance expert
```

但专家名称只是一种初始化/正则假设，不能宣称模型自动具有可解释语义。需要通过 gate 分布与 feature attribution 验证。

## 9. 价值与转化联合建模

只优化二元 CVR 可能偏向低价值易转化商品。若有订单金额或利润：

```text
p_conv = conversion probability
v_cond = E[value | conversion]
expected_value = p_conv × softplus(v_cond)
```

回归 head 仅在真实转化样本上计算：

```text
L_value = y_conv * Huber(log1p(v_pred), log1p(v_true))
```

排序分数可由业务约束组合：

```text
score = bid × pCTR × pCVR × value_factor
```

模型训练和竞价公式必须分开管理，不能把线上出价或后处理反馈泄漏回标签。

## 10. 建议配置

```yaml
backbone:
  history_events: 128-256
  d_model: 256 or 512
  layers: 6 or 5
  heads: 4

candidate_tasks:
  ctr: true
  conditional_cvr: true
  ctcvr_constraint: true
  value: optional
  delay_hazard: phase_2

history_tasks:
  click: true
  conversion: true
  add_cart: if_available
  order: if_available

loss_weights:
  ctr: 1.0
  ctcvr: 1.0
  clicked_cvr: 0.2
  history: 0.2
  value: 0.1
  counterfactual: 0.0_in_phase_1
```

## 11. 实验矩阵

| ID | CTR | clicked CVR | CTCVR 约束 | 历史动作 | Delay | Task Adapter/MMoE |
|---|---:|---:|---:|---:|---:|---:|
| E0 | 否 | 是 | 否 | 否 | 成熟标签 | 否 |
| E1 | 是 | 是 | 否 | 否 | 成熟标签 | 否 |
| E2 | 是 | 是 | 是 | 否 | 成熟标签 | 否 |
| E3 | 是 | 是 | 是 | 是 | 成熟标签 | 否 |
| E4 | 是 | 是 | 是 | 是 | Hazard | Adapter |
| E5 | 是 | 是 | 是 | 是 | Hazard | MMoE（仅在冲突成立时） |

主要比较：

- clicked-space pCVR AUC/LogLoss/ECE；
- entire-space pCTCVR AUC/PR-AUC/NE；
- pCTR、pCVR、pCTCVR 的乘法一致性；
- 新鲜样本与成熟样本校准；
- 不同转化延迟 bucket；
- 新 item/creative 与低频 query。

## 12. 预期收益机制与可证伪假设

假设：

> 通过全曝光 CTR/CTCVR 约束和历史动作辅助监督，HSTU backbone 能从远多于转化正例的曝光与点击信号学习表示，并降低 clicked-only CVR 的样本选择偏差；增益应在低频 item、低频 creative、短训练窗口和稀疏转化场景更明显。

证伪条件：

- E2 相对 E1 不改善 entire-space CTCVR 校准；
- 增益只来自更高正例权重而非概率链；
- 关闭 HSTU 历史后增益完全不变；
- pCVR 提升但 pCTCVR 或线上转化价值恶化；
- delay head 只改善未成熟离线标签，却在成熟回放集上变差。

## 13. 实施优先级

这是四个创新方案中最建议优先实施的一个，因为：

1. 它直接针对 CVR 任务本身，而不是仅改变 backbone；
2. 可在不扩展序列长度的情况下增加有效监督；
3. 与 HSTU 的多位置动作预测自然兼容；
4. 失败时容易通过概率校准、loss 和切片定位原因；
5. 即使最终 HSTU 不上线，全空间多任务目标也可复用于 RankMixer 或其他静态模型。
