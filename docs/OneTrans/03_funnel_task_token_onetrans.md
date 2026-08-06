# 3. P2：Funnel Task-Token OneTrans——面向全曝光 CVR 的任务原生建模

## 3.1 创新动机

推广搜 CVR 的主要难点通常不是 backbone 不够大，而是：

1. 传统 CVR 只在点击样本上训练；
2. 线上却对全部曝光候选预测；
3. 转化样本远少于点击样本；
4. CTR 与 CVR 共享信息时容易发生梯度干扰。

P2 将 ESMM 的全空间概率约束和 OneRank 的 task-private token/受控任务交互直接放进 OneTrans backbone，而不是在共享表示之后再挂多个 MLP tower。

## 3.2 数据要求

每条曝光样本至少包含：

```text
y_click   ∈ {0,1}
y_convert ∈ {0,1}
y_convert <= y_click
```

`y_convert=1` 表示该曝光最终发生点击并在定义窗口内转化。若转化有延迟，必须使用成熟标签。

若训练数据只有 clicked impressions，P2 不能实现 ESMM 的全空间目标；此时只可退化为多头 clicked-only 模型。

## 3.3 Token 组织

复用 P1 的 15 个数据 token：

```text
5 User + 9 Item + 1 Creative
```

追加 3 个 task tokens：

```text
[CTR] [CVR] [CTCVR]
```

总长度 18：

```text
[U1..U5 ; I1..I9 ; C1 ; CTR ; CVR ; CTCVR]
```

三个 task token 使用独立 QKV/FFN，并作为预测 query。

## 3.4 结构化可见性

前 4 层：

- 数据 token 按 entity causal 顺序交互；
- 每个 task token 可读取全部数据 token和自己；
- task token 之间互不可见，形成早期任务特化。

最后 2 层使用 funnel cascade：

```text
CTR    -> 只读数据和自身
CVR    -> 可读数据、CTR、自己
CTCVR  -> 可读数据、CTR、CVR、自己
```

为了减少下游稀疏任务反向破坏上游任务，CVR/CTCVR 读取其他 task token 时使用 off-diagonal gradient detach：

```python
kv_from_other_task = other_task_state.detach()
```

即前向允许知识传递，反向不让 CVR loss 更新 CTR 私有通道。

## 3.5 概率与损失

三个 token 输出：

```text
p_ctr          = sigmoid(z_ctr)
p_cvr          = sigmoid(z_cvr)
p_ctcvr_direct = sigmoid(z_ctcvr)
p_ctcvr_prod   = p_ctr * p_cvr
```

主损失：

```text
L_ctr     = BCE(p_ctr, y_click)                     # 全曝光
L_prod    = BCE(p_ctcvr_prod, y_convert)            # 全曝光，ESMM 约束
L_direct  = BCE(p_ctcvr_direct, y_convert)          # 全曝光，直接联合任务
L_cons    = symmetric_KL(p_ctcvr_direct, p_ctcvr_prod)
L_total   = L_ctr + λ1 L_prod + λ2 L_direct + λ3 L_cons
```

建议起点：

```text
λ1=1.0
λ2=0.5
λ3=0.05
```

可选 clicked-only 辅助项：

```text
L_clicked_cvr = BCE(p_cvr[y_click=1], y_convert[y_click=1])
```

建议在训练前 20% steps 不启用，之后以 `λ4=0.1~0.2` 渐进加入，避免稀疏 clicked-only 梯度主导共享特征。

最终线上：

- 若业务排序公式需要 post-click CVR，使用 `p_cvr`；
- 若直接优化曝光到转化，使用 `p_ctcvr_prod` 或与 `p_ctcvr_direct` 校准融合；
- 不要用 `p_ctcvr / p_ctr` 做在线除法，低 CTR 区间会数值不稳定。

## 3.6 为什么需要直接 CTCVR token

ESMM 只用乘积约束，结构简单且有效，但后续研究指出其估计仍可能存在偏差。增加 `[CTCVR]` token有三个用途：

1. 给联合事件一个直接监督通道；
2. 用 consistency loss 监控乘积约束是否失真；
3. 在极低 CTR 区间提供更稳定的曝光到转化估计。

它不是替代 ESMM，而是增加一个可诊断的直接联合头。

## 3.7 Task-specific scoring

基础版本：

```text
z_task = Linear(RMSNorm(h_task), 1)
```

增强版本可采用 OneRank 风格动态匹配：

```text
r_candidate = RMSNorm(mean(ItemTokens + CreativeToken))
z_task = dot(W_task h_task, W_cand_task r_candidate) / sqrt(d)
```

首次复现先用线性头；动态匹配作为独立 ablation，避免同时引入过多变量。

## 3.8 类别不平衡与校准

- 转化正例极少时，不建议简单正例复制而不做校准；
- 可对 `L_prod/L_direct` 使用正例权重，但线上需独立做 calibration；
- 评估必须同时看 CVR clicked-space 与 CTCVR exposure-space；
- 分桶检查低 CTR、冷 item、冷 creative、长尾 query 和新广告主。

## 3.9 伪代码

```python
data_tokens = entity_tokenizer(user_emb, item_emb, creative_emb)
task_tokens = stack([ctr_token, cvr_token, ctcvr_token]).expand(B, -1, -1)
x = concat([data_tokens, task_tokens], dim=1)

for layer_id, block in enumerate(blocks):
    mask = private_task_mask if layer_id < 4 else funnel_cascade_mask
    x = block(x, mask=mask, detach_cross_task=(layer_id >= 4))

h_ctr, h_cvr, h_ctcvr = x[:, -3], x[:, -2], x[:, -1]
z_ctr = ctr_head(h_ctr)
z_cvr = cvr_head(h_cvr)
z_ctcvr = ctcvr_head(h_ctcvr)

p_ctr = sigmoid(z_ctr)
p_cvr = sigmoid(z_cvr)
p_prod = p_ctr * p_cvr
p_direct = sigmoid(z_ctcvr)
```

## 3.10 必做消融

| 实验 | 变化 |
|---|---|
| P2-A | clicked-only CVR vs ESMM product |
| P2-B | 2 tokens（CTR/CVR）vs 3 tokens（+CTCVR） |
| P2-C | task mutually invisible vs fully visible |
| P2-D | cascade mask vs full cross-task attention |
| P2-E | cross-task detach on/off |
| P2-F | direct linear head vs dynamic matching |
| P2-G | direct CTCVR consistency weight 0/0.01/0.05/0.1 |

## 3.11 预期

P2 是最有可能产生稳定 CVR 收益的方案，原因是它同时利用 5.5 亿曝光级样本中的丰富 click 信号和稀疏 convert 信号。若你实际拥有的 5.5 亿样本只是 clicked samples，则该优势会显著减弱。
