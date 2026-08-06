# 2. P1：Entity-Hierarchical OneTrans——实体保持、层次化 tokenization 与前缀复用

## 2.1 创新动机

P0 的全局 Auto-Split 会在 tokenizer 内提前混合 user、item 和 creative，使用户侧计算无法跨候选复用，也缺少推广搜中的实体边界。

P1 保留 OneTrans 的 causal backbone 和 token-specific 参数，但把 tokenizer 改造成“实体内自动分裂、实体间在 Transformer 中交互”。这是对 OneTrans + UniFormer 语义 tokenization + TMallGS 层次化输入校准的组合改造。

## 2.2 Token 配额

主配置使用 16 个 token：

```text
User      : 5 tokens
Item      : 9 tokens
Creative  : 1 token
[CVR]     : 1 learned task/sink token
Total     : 16 tokens
```

顺序为：

```text
[U1..U5 ; I1..I9 ; C1 ; CVR]
```

这个顺序有三个性质：

1. User token 不依赖 candidate，可按 request 缓存；
2. Item/Creative 能读取 User；
3. 最后的 `[CVR]` 能读取全部 token。

## 2.3 Field-wise Saliency Reweighting

在 flatten 前，每个字段 embedding 保持完整：

```text
E_user     : [B, 385, 17]
E_item     : [B, 835, 17]
E_creative : [B,  14, 17]
```

对每个实体单独生成字段权重。以 user 为例：

```text
z_u = mean(E_user, dim=-1)                 # [B, 385]
w_u = sigmoid(W_up(SiLU(W_down(z_u))))     # [B, 385]
E'_user = E_user * w_u[..., None]
```

建议 bottleneck：

```text
user/item saliency rank = 64
creative saliency rank  = 8
```

这样既不拆字段，又能在投影前压低无效字段噪声。初始时将 `W_up` 零初始化，使 gate 接近 0.5 或通过残差形式 `1 + 0.2*tanh(.)` 避免早期过度抑制。

## 2.4 实体内 Auto-Split

```text
User:
[B, 6545] -> Linear(6545, 512) -> SiLU -> Linear(512, 5×256)
          -> [B, 5, 256]

Item:
[B,14195] -> Linear(14195,512) -> SiLU -> Linear(512, 9×256)
          -> [B, 9, 256]

Creative:
[B,238] -> Linear(238,128) -> SiLU -> Linear(128,256)
        -> [B,1,256]
```

每个 token 加：

```text
token_position_embedding
+ entity_type_embedding(user/item/creative/task)
```

Tokenizer 总参数约 `12.5M`，与 P0 同一数量级。

## 2.5 Backbone

主配置：

```text
layers=6
d_model=256
heads=4
ffn_ratio=4
causal=True
Pre-RMSNorm
16 NS tokens
token-specific QKV + FFN
```

P1 保留完整 token-specific 参数，以便先单独验证 tokenizer 与缓存收益。预计 backbone dense 参数约 `69.6M`，加 tokenizer 与 head 后约 `83–86M`。

## 2.6 User-prefix KV cache

由于 User token 位于最前且 causal，不依赖 item/creative：

```text
Stage 1：每个 request 计算一次 U1..U5 的逐层 K/V
Stage 2：每个 candidate 仅计算 I1..I9、C1、CVR，并读取缓存
```

对一个 request 有 `C` 个候选时，user-prefix 部分从每候选重复计算变为一次计算。该缓存不是 OneTrans 原论文的行为序列 cache，而是静态 user-prefix cache；数学条件相同：前缀必须与候选无关，且后缀只能读取前缀，前缀不能读取后缀。

训练数据应按 `request_id` 聚合候选，至少在一个 micro-batch 内让同 request 样本连续，以便验证训练侧 prefix reuse。

## 2.7 输出

使用最后 `[CVR]` token：

```text
h_cvr = final_norm(X[:, -1, :])
z_cvr = MLP(256 -> 128 -> 1)
```

可附加一个浅层 residual head：

```text
z_res = Linear(concat(mean(U), mean(I), C), 1)
z = z_cvr + 0.1 * z_res
```

首次实验应关闭 residual head，避免掩盖 backbone 效果。

## 2.8 伪代码

```python
u = user_saliency(user_emb)
i = item_saliency(item_emb)
c = creative_saliency(creative_emb)

u_tok = user_tokenizer(u)         # [B, 5, 256]
i_tok = item_tokenizer(i)         # [B, 9, 256]
c_tok = creative_tokenizer(c)     # [B, 1, 256]
t_tok = cvr_token.expand(B, 1, 256)

x = concat([u_tok, i_tok, c_tok, t_tok], dim=1)
x = x + position_emb + entity_type_emb

for block in blocks:
    x = block(x, causal=True, token_specific=True)

logit = cvr_head(final_norm(x[:, -1]))
```

## 2.9 关键消融

| 实验 | 目的 |
|---|---|
| P1-A | P0 global Auto-Split vs entity Auto-Split |
| P1-B | 4/10/1/1、5/9/1/1、5/8/2/1 token 配额 |
| P1-C | saliency gate on/off |
| P1-D | entity type embedding on/off |
| P1-E | user-prefix cache on/off，测吞吐与 p99 |
| P1-F | causal vs block mask（User full attention、candidate 读 User） |

## 2.10 风险

- 实体边界可能损失全局 Auto-Split 在输入层的自由组合能力；
- 5/9/1 是工程初值，不代表最优；
- request 级 cache 依赖严格的特征归属，任何 candidate-dependent cross feature 都不能放入 User prefix；
- 若训练是完全 point-wise shuffle，cache 收益只能在线上出现，训练侧不会自然复用。

## 2.11 成功判据

相对 P0：

- CVR AUC/UAUC 不下降或提升；
- candidate batch 下吞吐明显提升；
- user-prefix 计算占比随候选数增加而摊薄；
- creative 冷启动和长尾 item 分桶不退化。
