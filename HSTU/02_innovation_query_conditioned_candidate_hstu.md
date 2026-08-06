# 02｜创新方案一：Query-Conditioned Candidate-Set HSTU

简称 **QC-HSTU**。

## 1. 动机

推广搜索与普通信息流推荐的关键差异是：

1. 同一用户在不同 query 下的短时意图可能完全不同；
2. 排序面对的是同一请求的一组候选，而不是互不相关的独立样本；
3. slot、位置、页面形态、竞价阶段等上下文在预排序或粗排时可能尚未完全确定；
4. session 内最新行为可能因在线日志延迟而不可见。

仅将 query embedding 拼进一个展平向量，难以让 query 同时影响历史选择、候选表示和时间相关性。QC-HSTU 将 query/request 作为结构化上下文，显式控制 attention bias、候选 mask 和输出 head。

## 2. 总体结构

```text
stable user context ──> user prefix tokens
query/request fields ──> query prefix tokens
historical events ─────> HSTU history tokens
candidate set ─────────> isolated target tokens

[user prefix, query prefix, history, candidate_1 ... candidate_M]
                              │
                              └─ HSTU with query-conditioned bias/mask
                                      │
                                      └─ candidate outputs
                                             │
                                             └─ context-conditioned heads
```

## 3. Query/Request Tokenizer

从当前 385 个 user 域字段中完成语义审计，找出实际属于检索请求的字段。建议构造 2–4 个 query token：

```text
q_semantic: query ID/text/category/intention embedding
q_scene:    page/surface/traffic source/device
q_time:     time, freshness, session state
q_market:   geo, auction/traffic context, business constraints
```

每个 token：

```text
q_k = Linear(GatedPool({field embeddings in group k})) ∈ R^d
```

如果有 query 文本 encoder 输出，不要直接与数百个 sparse fields 拼接；先投影到 `d`，并通过 modality/type embedding 标记。

## 4. Query-Conditioned History Relevance

### 4.1 Query-conditioned attention bias

标准 HSTU attention：

```text
A = SiLU(QK^T + B_pos + B_time) ⊙ Mask
```

加入 query-event relevance：

```text
r_i = MLP([q_summary; event_i; q_summary ⊙ event_i])
B_query[target_j, history_i] = w_j^T r_i
```

完整形式：

```text
A = SiLU(QK^T + B_pos + B_time + β B_query) ⊙ Mask
```

其中 `β` 可以是可学习标量，初始化为 0，确保模型从原始 HSTU 平滑开始训练。

### 4.2 低成本版本

不显式构造 `M×T×MLP`。先将 query 与历史事件映射到低维 `r=16/32`：

```text
q_r = Wq q_summary
k_r_i = Wk event_i
B_query[j,i] = <q_r_j, k_r_i>
```

若同一请求所有候选共享 query，则 `q_r` 只计算一次。可再加入 candidate-query 相关项：

```text
q_candidate_j = Wc [query; candidate_j]
B_query[j,i] = <q_candidate_j, k_r_i>
```

## 5. Candidate-Set Training

### 5.1 输入组织

同一请求追加 `M=8–32` 个候选：

```text
X = [context, query, history, τ_1, ..., τ_M]
```

候选可由以下部分组成：

- 实际曝光候选；
- 同请求中未点击候选；
- 高分但未曝光的 hard negatives（若日志可恢复）；
- 少量随机负例，防止只学局部边界。

### 5.2 Candidate-isolated mask

对候选 `τ_j`：

```text
can attend: context + query + history + self
cannot attend: τ_k, k != j
```

这保证候选顺序不影响分数，也避免某个候选标签或特征通过其它候选泄漏。

### 5.3 Listwise / pairwise 辅助目标

点式 BCE 仍是主目标。对于同一请求，可增加轻量 pairwise loss：

```text
L_pair = log(1 + exp(-(s_pos - s_neg)))
```

其中：

- `s_pos` 是转化或点击正例；
- `s_neg` 是同请求困难负例；
- 对没有正例的请求不计算该项。

总损失：

```text
L = L_pointwise + λ_pair L_pair
```

建议 `λ_pair=0.05–0.2` 起步，防止稀疏 CVR 正例导致 pairwise loss 主导。

## 6. Context-Conditioned Output Heads

推广搜中部分打分上下文可能在模型执行后才确定，例如最终 slot、广告位置、页面布局或竞价分支。若训练只使用实际展示位置，模型会把位置偏差混入候选质量。

设有限个离散上下文 `k=1...K`，建立多头：

```text
z_ctr^(k) = Head_ctr_k(h_candidate)
z_cvr^(k) = Head_cvr_k(h_candidate)
```

训练时仅对真实上下文 `k*` 回传：

```text
L = L(z^(k*), y)
```

服务时可：

- 在已知 slot 下选择对应 head；
- 在 slot 未知时输出全部 head，交给后续竞价/排序；
- 按预估 slot 分布加权。

### 6.1 参数共享

不要复制完整 tower。使用 shared trunk + 小型 adapter：

```text
h_shared = MLP(h_candidate)
h_k = h_shared + A_k(B_k h_shared)   # low-rank adapter
z_k = w_k^T h_k
```

`K` 较大时，可用上下文 embedding 条件化单一 hyper-head：

```text
z = MLP([h_candidate; context_embedding; interaction])
```

## 7. Session-Delay-Aware Mask

训练日志通常拥有目标请求之前所有事件，但线上特征链路可能存在延迟。若模型训练时读取线上不可见的几秒/几分钟内行为，就会形成 train-serve skew。

定义：

```text
visible(event_i, request_t) =
    event_time_i <= request_t - pipeline_delay(surface, event_type)
```

attention mask 加入：

```text
Mask[target, event_i] = 0, if not visible(...)
```

延迟可以按：

- 事件类型；
- 流量入口；
- 国家/机房；
- 特征服务版本；

分桶估计。首版使用保守固定 buffer，后续再细化。

## 8. Query 与 Creative 的三路交互

推广搜索中，转化可能由 `query × item × creative` 联合决定。建议在候选 tokenizer 内加入低成本三路门控：

```text
q = QueryEncoder(...)
i = ItemEncoder(...)
c = CreativeEncoder(...)

g_i = sigmoid(W_i [q; i; q⊙i])
g_c = sigmoid(W_c [q; c; q⊙c])

target = W_o [q; g_i⊙i; g_c⊙c; i⊙c]
```

这仍然只生成一个 candidate token，不增加 HSTU 序列长度。

## 9. 训练配置

建议从论文式 HSTU 基线初始化：

```yaml
history_events: 128
d_model: 256
layers: 6
heads: 4
context_tokens: 8
query_tokens: 2-4
candidates_per_request: 16
query_bias_rank: 32
candidate_isolation: true
pairwise_loss_weight: 0.1
```

若从零训练，`B_query` 系数从 0 开始，避免额外 bias 在早期破坏 attention 数值。

## 10. 消融实验

至少进行：

| 实验 | Query token | Query bias | Multi-candidate | Context head | Delay mask |
|---|---:|---:|---:|---:|---:|
| Q0 | 否 | 否 | 否 | 否 | 是 |
| Q1 | 是 | 否 | 否 | 否 | 是 |
| Q2 | 是 | 是 | 否 | 否 | 是 |
| Q3 | 是 | 是 | 是 | 否 | 是 |
| Q4 | 是 | 是 | 是 | 是 | 是 |
| Q5 | 是 | 是 | 是 | 是 | 否（仅用于量化泄漏） |

`Q5` 不能作为可上线模型，只用于测量线上延迟造成的理论上限与偏差。

## 11. 重点切片

QC-HSTU 应重点观察：

- head query 与 tail query；
- 同用户多意图 query；
- query-item 类目一致/不一致；
- creative 与 query 强/弱匹配；
- session 内发生过近期点击与无近期点击；
- 不同广告位、slot、流量入口；
- 同请求候选数不同的 bucket。

如果总体 AUC 提升但 tail query 或候选顺序一致性变差，应先检查 query embedding 质量和 candidate mask，而不是继续加深网络。

## 12. 风险与回退

### 风险

1. 385 个 user 字段中没有真正 query 信息；
2. 同请求候选日志不完整，候选集合存在强选择偏差；
3. query-conditioned bias 过强，使模型只关注字面匹配而忽视长期转化偏好；
4. context heads 数据分布不均，低频 head 欠拟合；
5. 线上 delay 分布变化导致 mask 失配。

### 回退策略

- 保留 query prefix，关闭 `B_query`；
- 退回单候选训练，但保留 KV cache；
- 多 context head 改为单 head + context embedding；
- 对 query bias 使用 dropout 或 `β` 上限约束；
- 对低频 context head 使用 shared adapter 或合并 bucket。

## 13. 科学性判断

本方案不是“为了增加 query 模块而增加”。其可证伪假设是：

> 与仅拼接 query 特征相比，让 query 直接影响历史相关性和候选目标位置，能够在相同候选与相同数据下改善推广搜索 CVR，且增益应集中在多意图用户、长历史和 query-item 相关性敏感切片。

若增益不集中于这些切片，或者关闭历史后仍获得同等增益，则收益更可能来自额外参数或 tokenizer，而不是 query-conditioned sequential modeling。
