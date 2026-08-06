# 04｜创新方案三：Field-Interaction Augmented HSTU

简称 **FI-HSTU**。

## 1. 动机

HSTU 擅长建模时间序列和候选相关的用户兴趣，但推广搜索 CVR 仍高度依赖当前曝光中的精细交叉，例如：

```text
query × item category
user purchasing power × price
item × creative template
query × creative wording
device × landing page type
traffic source × advertiser/shop
```

如果把 835 个 item 字段和 14 个 creative 字段压缩成一个 candidate token，部分细粒度交叉可能在进入 HSTU 前被不可逆地丢失。反过来，把所有 1,234 个字段作为长序列 token 又会造成高计算量，并混淆时间序列与静态字段交互。

FI-HSTU 使用两条互补分支：

1. HSTU sequence tower：建模历史、时间和候选条件化；
2. compact field-interaction tower：建模当前请求的 user-query-item-creative 显式交叉；
3. 用有初始恒等路径的门控残差融合，避免静态塔在训练早期破坏 HSTU 表示。

## 2. 总体架构

```text
historical sequence + target candidate
                  │
                  ▼
             HSTU tower
                  │
              h_seq ∈ R^d

current user/query/item/creative fields
                  │
                  ▼
     semantic tokenizer + field gate
                  │
                  ▼
       low-rank DCNv2 / bilinear tower
                  │
             h_cross ∈ R^d

h = h_seq + sigmoid(Wg[h_seq; h_cross]) ⊙ Ws(h_cross)
                  │
                  ▼
        CTR / CVR / CTCVR heads
```

## 3. 静态交互输入不能使用全部展平向量

直接将 20,978 维输入送入 DCNv2 会带来巨大的交叉矩阵和延迟。先将 1,234 个字段压缩为语义组 token。

### 3.1 推荐语义组数

起始配置约 24–32 个 group tokens：

```text
user:      6–8 groups
query:     2–4 groups
item:     10–14 groups
creative:  2–3 groups
context:   2–3 groups
```

每个 group 输出 32/64 维：

```text
G ∈ R[B, K, d_g]
K ≈ 24–32, d_g=32
```

展平后：

```text
x0 ∈ R[B, K*d_g] ≈ R[B, 768–1024]
```

这使显式交互塔的输入规模可控。

## 4. Field Gate：字段重要性建模

### 4.1 组内 gate

对语义组 `g` 的字段：

```text
s_f = w_g^T SiLU(W_g z_f)
α_f = softmax(s_f)
g_g = Σ_f α_f z_f
```

这类似 feature reweighting，但 gate 必须在组内计算，不能对 1,234 个字段做全局 softmax；否则大域会压制 creative 等小域。

### 4.2 跨组 gate

使用当前 query/request summary 条件化：

```text
r_g = sigmoid(MLP([g_g; q; g_g⊙q]))
g'_g = r_g ⊙ g_g
```

这样不同 query 可以选择不同 item/creative 语义组。

### 4.3 规避 gate 塌缩

训练早期可：

- gate bias 初始化为 0，使 sigmoid 约为 0.5；
- 对 gate 使用 0.05–0.1 dropout；
- 记录各组平均 gate 和方差；
- 不使用强稀疏正则，除非线上延迟要求明确进行特征裁剪。

## 5. 交互塔方案 A：Low-Rank DCNv2

令 `x0 ∈ R^D`，第 `l` 个 cross layer：

```text
x_{l+1} = x0 ⊙ (W_l x_l + b_l) + x_l
```

采用低秩分解：

```text
W_l ≈ U_l V_l
U_l ∈ R[D,r], V_l ∈ R[r,D]
```

推荐：

```yaml
D: 768-1024
cross_layers: 2-3
rank: 32-64
```

每层参数由 `D²` 降为约 `2Dr`。例如 `D=1024,r=32`：

```text
full matrix: 1,048,576
low-rank:       65,536
```

2–3 层即可覆盖到 3/4 阶显式交叉，不建议无限堆叠。

### 5.1 Mixture of Low-Rank Cross Experts（后续可选）

如果不同推广场景交互模式差异明显，可用 `E=4` 个低秩 cross experts：

```text
cross_l(x) = Σ_e gate_e(x) [U_{l,e} V_{l,e} x]
```

先使用 dense softmax gate。只有在单 cross network 的场景切片明显冲突时才增加该模块。

## 6. 交互塔方案 B：Bilinear Group Interaction

对 `K` 个 group token 计算受控的二阶交互：

```text
b_{ij} = (W_i g_i) ⊙ (W_j g_j)
```

不应枚举所有 `K(K-1)/2` 对。使用先验和数据筛选的边集合：

```text
query-user
query-item
query-creative
user-item
user-price
item-creative
creative-scene
```

设有效边 `|E|=40–80`，每条边输出 16/32 维，再聚合：

```text
h_bilinear = MLP(concat({b_ij}))
```

该方案更容易解释，但维护边集合需要业务知识。建议把 DCNv2 作为主方案，bilinear 作为对照或解释性增强。

## 7. 交互塔输出

并行使用 cross 与 deep 分支：

```text
x_cross = CrossNet(x0)
x_deep = MLP(x0): D -> 512 -> 256
h_cross = Linear([x_cross; x_deep]) -> d_model
```

采用 LayerNorm/RMSNorm，但不建议在输入 group tokens 上使用会跨样本混合统计的 BatchNorm，避免长尾稀疏域和分布式小 microbatch 不稳定。

## 8. 与 HSTU 的门控残差融合

### 8.1 推荐融合

```text
s = LayerNorm(h_cross)
g = sigmoid(W_g [h_seq; s; h_seq⊙s])
h_fused = h_seq + g ⊙ W_s s
```

初始化：

```text
W_s: normal small std
W_g bias: -2 to -1
```

使早期 `g≈0.12–0.27`，模型从 HSTU 基线平滑出发。

### 8.2 双向 FiLM 融合（后续可选）

让静态交互调节序列表征：

```text
γ, β = Linear(h_cross).chunk(2)
h_fused = (1 + 0.1*tanh(γ)) ⊙ h_seq + β
```

需要限制 `γ` 幅度，否则静态塔可能改变 HSTU 表示尺度并导致训练不稳。

### 8.3 不推荐简单拼接后大 MLP

```text
MLP([h_seq; h_cross])
```

虽然容易实现，但无法保证 HSTU 路径保留，难以定位收益来源；大 MLP 还可能仅靠静态特征拟合，掩盖序列塔退化。

## 9. 参数与算力预算

示例：

```text
K=28 groups
d_g=32
D=896
CrossNet: 3 layers, rank=32
DeepNet: 896 -> 512 -> 256
h_cross -> d_model=256
```

主要参数约：

```text
low-rank crosses: 3 × 2 × 896 × 32 ≈ 172K
deep MLP: 896×512 + 512×256 ≈ 590K
fusion/head: 数十万
```

不含字段投影和 embedding table，额外 dense 参数通常低于 2M，远低于直接对 20,978 维做大 MLP 或 full-rank cross。

## 10. 特征可缓存性与在线路径

将静态交互特征划分为：

- **user cache**：稳定用户画像、长期兴趣；
- **item/creative cache**：固有属性和预计算 group token；
- **request dynamic**：query、场景、实时价格/状态；
- **candidate cross**：在线只计算小型 gate 和 cross layers。

可缓存：

```text
preprojected user groups
preprojected item groups
preprojected creative groups
```

在线仅执行：

```text
query conditioning
selected group concatenation
2-3 low-rank cross layers
fusion
```

## 11. 训练策略

### 11.1 两阶段稳定训练

阶段 1：

```text
训练 HSTU baseline，得到稳定 checkpoint
```

阶段 2：

```text
加载 HSTU
新增 interaction tower 和 fusion
fusion gate 小值初始化
HSTU learning rate = new module LR 的 0.2–0.5
```

稳定后再统一学习率微调。

### 11.2 Branch dropout

以小概率屏蔽某一分支：

```text
P(drop h_cross) = 0.05–0.1
P(drop h_seq)   = 0.02–0.05
```

目的不是正则越大越好，而是避免任务头完全依赖单一路径。线上必须始终启用两分支。

## 12. 消融矩阵

| ID | HSTU | Group Gate | DCNv2 | Bilinear | Gated Fusion |
|---|---:|---:|---:|---:|---:|
| F0 | 是 | 否 | 否 | 否 | 否 |
| F1 | 是 | 是 | 否 | 否 | 是 |
| F2 | 是 | 是 | 是 | 否 | 是 |
| F3 | 是 | 是 | 否 | 是 | 是 |
| F4 | 是 | 是 | 是 | 是 | 是 |
| F5 | 否 | 是 | 是 | 否 | 不适用 |

`F5` 是关键控制组：如果 F2 与 F5 性能近似，说明交互塔承担了几乎全部收益，HSTU 历史可能没有被有效使用。

进一步消融：

- `K=16/24/32/48` 语义组；
- CrossNet 1/2/3/4 层；
- rank 16/32/64；
- concat、add、gated residual、FiLM；
- creative 独立组开/关；
- query-conditioned gate 开/关。

## 13. 重点指标与解释

FI-HSTU 的增益应主要出现在：

- 价格敏感用户；
- query 与 item/creative 组合差异大的场景；
- 同 item 多 creative；
- 冷启动 creative；
- 当前实时状态变化快、历史不能完全覆盖的候选；
- 短历史用户。

需要记录：

```text
fusion gate mean/std by slice
field/group gate distribution
cross layer norm
h_seq 与 h_cross 的 cosine similarity
gradient norm by branch
```

不要把 gate 权重直接解释为严格因果重要性，它只能作为模型内部诊断信号。

## 14. 风险

1. 语义分组错误导致有价值交叉在 tokenizer 前被合并；
2. 静态塔过强，模型忽略历史；
3. 当前统计特征包含未来信息，静态塔放大泄漏；
4. 交互塔增加在线 candidate-dependent 计算；
5. user/item/creative embedding 同时在两分支反向传播，梯度尺度失衡。

缓解：

- 使用 branch-specific LayerNorm；
- 监控两分支梯度并做 global norm clipping；
- 对共享 embedding 使用统一 optimizer group；
- 严格特征快照审计；
- 门控残差小值初始化；
- 在线仅保留通过消融的 group 和 cross 层。

## 15. 可证伪假设

> 在保持相同历史 HSTU 的条件下，紧凑的当前字段交互塔应主要改善细粒度 query-user-item-creative 组合、短历史和 creative 差异场景；若增益仅随参数量增加且不集中于这些切片，则没有证据表明显式交互是必要的。

FI-HSTU 的科学价值在于把“时间序列兴趣”和“当前曝光字段交叉”作为两种不同归纳偏置分别建模，并通过控制组验证二者是否互补，而不是简单堆叠两个大模型。
