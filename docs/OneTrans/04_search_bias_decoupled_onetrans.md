# 4. P3：Search-Bias-Decoupled OneTrans——推广搜强信号保真与偏置解耦

## 4.1 创新动机

推广搜与普通 Feed 推荐不同：

- Query-item/广告匹配、历史统计率、bid、广告质量等强特征对排序非常关键；
- position、page、slot、流量入口、设备与投放策略会形成系统性偏置；
- 把所有特征都投成 token 并经过多层 attention，可能平滑掉高频、硬约束信号。

P3 以 OneTrans 为主干，但把特征分成三条路径：

```text
semantic/light features -> OneTrans backbone
heavy cross features    -> FiLM late fusion
bias/context features   -> additive bias net
```

整个方案不要求新增模态，但要求有字段 schema，能够从现有 1,234 个字段中标记三类索引。

## 4.2 字段划分

### Semantic/light

适合深层交互：

- user profile、兴趣、长期统计；
- item ID/category/brand/merchant 等；
- creative semantic 属性；
- query/item 的低维语义特征（若已在 item 或 creative 中）。

### Heavy cross

应保留分辨率：

- query-item exact/phrase match；
- user-item/广告主历史交互率；
- calibrated CTR/CVR prior；
- price/bid/quality 的显式组合；
- 规则或检索阶段产生的强 matching score。

### Bias/context

用于系统偏置：

- position、page index、slot；
- traffic source、device、time；
- campaign pacing、预算状态；
- 曝光策略、探索标记；
- 可能的 creative 展示形态。

若某字段同时具有语义和偏置作用，应复制到两条路径，而不是强制唯一归属；但 bias 分支对主干的梯度应受控。

## 4.3 主干 token

建议：

```text
User semantic     : 5 tokens
Item semantic     : 8 tokens
Creative semantic : 1 token
[BIAS_ANCHOR]     : 1 token
[CVR]             : 1 token
Total             : 16 tokens
```

顺序：

```text
[User ; BiasAnchor ; Item ; Creative ; CVR]
```

`BiasAnchor` 只读取 user/request/bias 特征，不读取 candidate semantic token，尽量保持 request-level 可复用。若 position 是 candidate-specific，则拆成 request bias 和 candidate bias 两部分。

## 4.4 FiLM late fusion

OneTrans 输出最后的 CVR 表示：

```text
h_cvr ∈ R^256
```

将 heavy cross features 独立投影：

```text
e_heavy = MLP(heavy_embeddings) ∈ R^256
u = concat(h_cvr, e_heavy) ∈ R^512
γ, β = MLP_film(e_heavy) ∈ R^512 × 2
v = (1 + 0.1*tanh(γ)) ⊙ u + β
z_main = MLP_main(v)
```

使用 `0.1*tanh(γ)` 限制早期调制幅度，避免 FiLM 分支吞噬 backbone。

这条路径使 exact match、统计率和 bid 等强信号不必经过 6 层 attention 才到达输出。

## 4.5 Context-aware bias net

```text
h_bias = final state of BiasAnchor
z_bias = MLP_bias(concat(h_bias, bias_raw_projection))
z_final = z_main + z_bias
p = sigmoid(z_final)
```

建议：

- `MLP_bias` 宽度不超过 64/128；
- bias 分支最后一层零初始化；
- 监控 `std(z_bias)`，防止它变成完整预测器；
- 对 bias 原始输入 stop-gradient 到 semantic tokenizer，避免 semantic 路径通过捷径复现 position。

## 4.6 Request 内 pairwise 目标

若训练样本保留同一次 request 的候选集合，可以增加：

```text
L_pair = -log sigmoid(z_main_pos - z_main_neg)
```

Pairwise loss 只作用于 `z_main`，不含 `z_bias`。同 request 内共享偏置在差分中自然抵消，使 semantic 主干更关注候选相对质量。

总损失：

```text
L = L_pointwise(z_main + z_bias)
  + λ_pair L_pair(z_main)
  + λ_orth L_orth(h_cvr, h_bias)
  + λ_bias ||z_bias||²
```

建议起点：

```text
λ_pair=0.1
λ_orth=0.01
λ_bias=1e-4
```

若采用 P2 的 ESMM 目标，pointwise 部分替换为 CTR/CTCVR 多任务损失；P3 与 P2 可以组合，但应先分别验证。

## 4.7 候选独立 mask

同 request 多候选并行时：

- User/request/bias prefix 可共享；
- 每个 candidate 的 item/creative/CVR token 只能读取共享 prefix 和自身候选 token；
- candidate 之间互不可见，防止列表泄漏和训练/serving 不一致。

这可在一个 attention kernel 中组织为 block-sparse mask，或先算共享 prefix cache，再逐候选计算后缀。

## 4.8 伪代码

```python
semantic_tokens = semantic_tokenizer(user_sem, item_sem, creative_sem)
bias_anchor = bias_tokenizer(request_bias)
x = assemble(user_tokens, bias_anchor, item_tokens, creative_token, cvr_token)
x = onetrans(x, candidate_independent_mask)

h_cvr = x[:, cvr_index]
h_bias = x[:, bias_index]

e_heavy = heavy_mlp(heavy_features)
gamma, beta = film_mlp(e_heavy).chunk(2, dim=-1)
u = concat([h_cvr, e_heavy], dim=-1)
v = (1.0 + 0.1 * tanh(gamma)) * u + beta

z_main = main_head(v)
z_bias = bias_head(concat([h_bias, bias_projection], dim=-1))
z = z_main + z_bias
```

## 4.9 必做消融

| 实验 | 变化 |
|---|---|
| P3-A | 全特征 tokenization vs semantic/heavy/bias 三分支 |
| P3-B | FiLM vs simple concat late fusion |
| P3-C | bias net on/off |
| P3-D | bias last layer zero-init on/off |
| P3-E | pointwise only vs + request pairwise |
| P3-F | candidate independence mask vs point-wise batch |
| P3-G | P3 单任务 vs P2+P3 联合 |

## 4.10 成功判据

- 全局 AUC/UAUC 提升；
- request/query GAUC 和 calibration 提升；
- position/page 分桶的预测偏差降低；
- exact-match、高 bid、冷广告、新 creative 等关键桶不退化；
- bias logit 幅度受控；
- 线上 GMV/ROI 改善且不会只靠 position 先验取得离线收益。
