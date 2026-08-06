# 1. P0：Static OneTrans-S——静态约束下最贴近原论文的方案

## 1.1 目标

在没有行为序列的前提下，尽可能保留 OneTrans-S 的原始超参数与计算范式。该方案是后续创新方案的公平基线，不追求额外业务先验。

## 1.2 输入 tokenizer

将三个实体完整展平并拼接：

```text
x = concat(user_flat, item_flat, creative_flat)
x.shape = [B, 20,978]
```

使用单个低秩 Auto-Split MLP：

```text
[B, 20,978]
 -> Linear(20,978, 512)
 -> SiLU
 -> RMSNorm(512)
 -> Linear(512, 12 × 256)
 -> reshape
[B, 12, 256]
```

选择 12 个 NS-token，是为了对齐 OneTrans-S 顶层 12 个 NS-token。低秩中间层不是改变 Auto-Split 定义；原论文只要求“单个 MLP 后 split”，并未要求必须单层线性。这样可避免直接 `20,978 -> 3,072` 带来的约 64.4M tokenizer 权重。

Tokenizer 近似参数量：

```text
20,978 × 512 + 512 × 3,072 ≈ 12.31M
```

每个 token 增加可学习 position embedding。不要把输入先按每 1,311 维硬切，因为 1,311 不是 17 的倍数，会破坏字段 embedding 的完整性。

## 1.3 OneTrans block

配置对齐论文小模型：

```text
num_layers       = 6
d_model          = 256
num_heads        = 4
head_dim         = 64
ffn_ratio        = 4
norm             = Pre-RMSNorm
attention        = causal MHA
activation       = SiLU or GELU
dropout          = 0~0.05
tokens           = 12 NS tokens
```

由于 `L_S=0`，全部 token 都使用 token-specific 参数：

```text
Q_t = X_t W^Q_t
K_t = X_t W^K_t
V_t = X_t W^V_t
FFN_t(x) = W^2_t φ(W^1_t x)
```

建议 `W_O` 在同一层内共享；论文明确 token-specific 的是 Q/K/V 与 FFN，没有要求 output projection 也按 token 独立。

单层 token-specific 主体近似参数量：

```text
12 × (3d² + 8d²) + d²
= 12 × 11 × 256² + 256²
≈ 8.72M
```

6 层约 `52.30M`。加 tokenizer、norm 和 head 后，dense 参数约 `65–67M`，不含 sparse embedding table。

## 1.4 输出头

原论文只说明最终 token states 进入 task-specific head，没有规定唯一 pooling。为避免 causal 顺序导致早期 token 看不到后期 token，主配置使用全部 token：

```text
H = RMSNorm(final_tokens)                 # [B, 12, 256]
h = flatten(H)                            # [B, 3072]
z_cvr = MLP(3072 -> 512 -> 128 -> 1)
p_cvr = sigmoid(z_cvr)
```

必须做两个 ablation：

1. `flatten all tokens`：主配置；
2. 追加一个 learned `[CVR]` token，用最后 token 直接预测：更符合 autoregressive sink，但会从 12 增加到 13 token。

不要直接 mean pooling 作为唯一实现；mean pooling 会把所有 token 视为等价，而 OneTrans 特意给 NS-token 分配独立参数。

## 1.5 损失

若样本仅来自点击空间：

```text
L_cvr = BCEWithLogits(z_cvr, y_convert)
```

正负样本若有下采样，必须记录 sampling probability，并在校准时进行 logit correction；否则线上 CVR 会偏移。

若样本来自整个曝光空间，不建议把所有未点击样本直接标成 CVR 负样本。应切换 P2 的 ESMM 风格目标。

## 1.6 训练配置

建议起始配置：

```text
precision              = BF16
global_batch           = 16,384 或 32,768
dense_optimizer        = RMSProp(lr=0.005, alpha=0.99999, momentum=0)
weight_decay           = 0
dense_grad_clip        = 90
warmup_steps           = 2,000~5,000
lr_decay               = cosine 或按天 piecewise
activation_checkpoint  = 可选
```

这是尽量贴近原论文的起点。若现有训练平台更稳定地支持 AdamW，可增加同预算对照，但不要在首次对比中同时改 tokenizer、optimizer 与 backbone。

原论文每 GPU batch 为 2,048，并在 16 张 H100 上数据并行。你的 `B=2048` 若是 global batch，则每天约 26.9 万步，可能造成过高 optimizer 开销；优先通过数据并行提升 global batch。

## 1.7 推理与效率

Token 数只有 12，attention 的二次项很小；主要成本来自 token-specific FFN 和 tokenizer。此方案无法使用原论文的 sequence KV cache，因为所有输入在 Auto-Split 前已混合了 user/item/creative。

推理时可以：

- 融合 12 组 QKV GEMM 为 batched/grouped GEMM；
- 融合 token-specific FFN；
- 使用 BF16/FP16；
- 保持 token 数固定，避免动态 shape。

## 1.8 伪代码

```python
def forward(user_emb, item_emb, creative_emb):
    # user_emb: [B, 385, 17]
    # item_emb: [B, 835, 17]
    # creative_emb: [B, 14, 17]
    x = concat([
        user_emb.flatten(1),
        item_emb.flatten(1),
        creative_emb.flatten(1),
    ], dim=-1)                             # [B, 20978]

    x = linear_1(x)                        # [B, 512]
    x = rms_norm(silu(x))
    x = linear_2(x)                        # [B, 3072]
    x = x.view(B, 12, 256) + pos_emb

    for block in blocks:
        x = block.mixed_causal_attention(x, token_specific=True)
        x = block.mixed_ffn(x, token_specific=True)

    h = final_norm(x).flatten(1)
    logit = cvr_head(h)
    return logit
```

## 1.9 必做消融

| 实验 | 变化 |
|---|---|
| P0-A | causal vs full attention |
| P0-B | Auto-Split hidden 256/512/1024 |
| P0-C | 8/12/16 NS tokens |
| P0-D | flatten head vs `[CVR]` sink token |
| P0-E | shared QKV/FFN vs token-specific |
| P0-F | 4/6/8 layers，保持总 FLOPs 可比 |

## 1.10 预期与限制

优点：

- 最容易与论文结构对齐；
- 实现简单；
- 能验证 mixed parameterization 是否适合你的 1,234 个静态字段。

限制：

- Auto-Split 把 user/item/creative 彻底耦合，无法做 request-level user cache；
- causal token 顺序缺少显式语义；
- 无行为序列时，OneTrans 的核心“序列 + 交互统一”优势没有被完整利用；
- 推广搜中的强匹配、position、bid 等信号可能被深层 attention 稀释。

因此 P0 应作为基线，而不是最终上线方案。
