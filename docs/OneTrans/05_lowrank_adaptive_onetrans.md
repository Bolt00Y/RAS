# 5. P4：Low-Rank Adaptive OneTrans——低秩 mixed parameterization 与 NS-query pyramid

## 5.1 创新动机

OneTrans 对每个 NS-token 分配完整 Q/K/V 和 FFN。该设计表达力强，但参数随 token 数线性增长。

以 `16 tokens, d=256, 6 layers, FFN ratio=4` 为例，仅 token-specific QKV+FFN 与共享输出投影约为：

```text
69.6M dense parameters
```

P4 用“共享主干 + token-specific 低秩残差”保留异构性，并把原论文只针对 S-token 的 query pyramid 扩展到静态 NS-token。

## 5.2 Factorized mixed parameterization

对任意 token-specific 矩阵：

```text
W_t = W_shared + g_t(x_t) · A_t B_t
```

其中：

```text
A_t ∈ R^(d_out × r)
B_t ∈ R^(r × d_in)
r = 16（起点）
g_t(x_t) = sigmoid(a_t^T RMSNorm(x_t))
```

对 Q/K/V：

```text
Q_t = x_t (W_Q_shared + g_t A^Q_t B^Q_t)
```

FFN 的两层也采用相同方式。Shared FFN 可用标准两层 MLP 或 SwiGLU；首次公平对比使用和 P1 相同的两层 FFN。

该设计保留：

- 所有 token 的共享统计学习；
- 每个 token 的独立低秩语义旋转；
- 输入相关的 adapter 强度；
- grouped GEMM 的实现可能。

## 5.3 参数估算

`d=256, r=16, 16 tokens`：

共享单层：

```text
QKV + W_O + FFN ≈ 0.786M
```

每 token 的 QKV 与 FFN adapters：

```text
≈ 0.0655M
```

16 token 单层：

```text
0.786M + 16 × 0.0655M ≈ 1.835M
```

6 层约：

```text
11.0M
```

相对完整 token-specific backbone 的 `69.6M`，block 参数减少约 84%。Tokenizer 和 embedding 不计入该比例。

建议同时测试：

```text
rank r ∈ {8,16,32,64}
```

## 5.4 NS-query pyramid

原论文 pyramid 逐层减少 S-token query，但仍保留完整 K/V。静态场景没有 S-token，因此提出 adaptive NS-query pyramid。

主配置 query 数：

```text
Layer 1: 16
Layer 2: 16
Layer 3: 12
Layer 4: 8
Layer 5: 4
Layer 6: 1  # 只更新 [CVR]
```

规则：

1. `[CVR]` 永远是 active query；
2. 其余 token 根据 saliency score 选 top-k；
3. 所有 token 都提供 K/V；
4. inactive token 直接 identity-carry 到下一层，不执行 attention output 更新和 FFN；
5. top-k 在一个 batch 内可固定为 token type 级别，避免每样本动态 shape；先学平均 saliency，再固化 serving schedule。

训练阶段：

```text
0~10% steps：所有 token active
10~30%：16 -> 12
30~50%：加入 8
50%以后：启用完整 schedule
```

避免从随机初始化开始就剪掉重要 token。

## 5.5 Saliency 与正则

```text
s_t = sigmoid(MLP_s(RMSNorm(x_t)))
```

训练时可用 soft gate，后期转为固定 top-k。增加：

```text
L_budget = |mean(active_ratio) - target_ratio|
L_entropy = -Σ s_t log s_t
```

`L_entropy` 只在早期用于防止所有 token 得分相同，后期逐步减小。

## 5.6 Prefix cache

P4 推荐基于 P1 的实体 token 顺序：

```text
[User prefix ; Item ; Creative ; CVR]
```

User prefix 使用共享+低秩参数并按 request 缓存。由于后层仅更新少量 candidate query，候选侧计算可进一步下降。

## 5.7 伪代码

```python
for layer_id, block in enumerate(blocks):
    scores = saliency[layer_id](x)             # [B, T]
    active = schedule[layer_id].select(scores)
    active.always_include(cvr_index)

    q = block.project_q(x, active, factorized=True)
    k = block.project_k(x, all_tokens=True, factorized=True)
    v = block.project_v(x, all_tokens=True, factorized=True)

    y_active = attention(q, k, v, causal_or_block_mask)
    x = identity_update_inactive(x, y_active, active)
    x = block.factorized_ffn_only_active(x, active)
```

## 5.8 关键实验

| 实验 | 变化 |
|---|---|
| P4-A | full token-specific vs shared-only vs low-rank adapters |
| P4-B | rank 8/16/32/64 |
| P4-C | static adapter gate vs input-dependent gate |
| P4-D | no pyramid vs fixed query schedule |
| P4-E | learned sample-level top-k vs token-type fixed top-k |
| P4-F | 16-16-12-8-4-1 vs 16-12-8-4-2-1 |
| P4-G | P1 full model蒸馏到 P4 |

## 5.9 蒸馏建议

P4 更适合作为 P1/P2/P3 的轻量 student：

```text
L = L_label
  + 0.5 * KL(student_prob, teacher_prob)
  + 0.1 * cosine(student_cvr_token, teacher_cvr_token)
```

先用完整 token-specific teacher 验证上限，再压缩，比直接从零训练 P4 更容易判断损失来自容量还是结构。

## 5.10 风险

- 低秩 residual 可能不足以表达 item token 之间差异；
- sample-level 动态 top-k 会导致 kernel 碎片和线上抖动；
- inactive token identity-carry 可能造成早期噪声长期保留；
- 参数减少不等于延迟必然减少，低秩小 GEMM 若未融合可能更慢。

因此最终以 p99、吞吐、MFU 和显存为准，不只看理论参数量。
