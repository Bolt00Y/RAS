# MixFormer 论文详解：统一非序列特征交互与用户行为序列的 Co-Scaling 架构

> 论文：**MixFormer: Co-Scaling Up Dense and Sequence in Industrial Recommenders**  
> 作者：Xu Huang, Hao Zhang, Zhifang Fan, Yunwen Huang, Zhuoxing Wei, Zheng Chai, Jinan Ni, Yuchao Zheng, Qiwei Chen  
> 初始提交：2026-02-15；本文按 arXiv v2 阅读  
> 原文：[arXiv Abstract](https://arxiv.org/abs/2602.14110) · [HTML](https://arxiv.org/html/2602.14110v2) · [PDF](https://arxiv.org/pdf/2602.14110v2)

## 1. 论文定位

RankMixer 主要扩展非序列特征交互，用户行为序列通常仍由独立的 Transformer、DIN/DIEN 或 Cross-Attention 模块建模。工业模型因此常见如下结构：

```text
static / dense features -> feature interaction backbone
user behavior sequence  -> sequence model
                         -> late fusion
```

这种分离式设计产生一个 co-scaling 问题：

- 增加 dense backbone 容量会占用 sequence model 的计算预算；
- 增加序列长度和序列塔深度又会压缩非序列交互容量；
- 两类表示只在末端融合，无法在每一层深度交互；
- 同一个 item 候选下，用户侧和行为序列计算可能被重复执行。

MixFormer 的核心目标是：

> 用一个统一的 Transformer-style backbone，同时完成非序列 feature interaction 与序列聚合，使 dense capacity 和 sequence length 能在同一结构中协同扩展。

它不是 RankMixer 的简单 block 替换，而是把 RankMixer 风格的 Query Mixer 与 Cross-Attention 组合到每一层。

---

## 2. Figure 1：MixFormer 整体结构

<p align="center">
  <img src="https://arxiv.org/html/2602.14110v2/x1.png" width="94%" alt="MixFormer Figure 1">
</p>

*图源：MixFormer Figure 1。若图片加载失败，请打开论文 [HTML](https://arxiv.org/html/2602.14110v2) 或 [PDF 第 4 页](https://arxiv.org/pdf/2602.14110v2#page=4)。*

Figure 1 中，一个 MixFormer block 包含三部分：

```text
Query Mixer
    ↓
Cross Attention over behavior sequence
    ↓
Output Fusion
```

其中：

- Query 表示由非序列 user/item/context 特征形成；
- Key/Value 来自用户行为序列；
- Query Mixer 使用 RankMixer 风格的 HeadMixing；
- 每层 Cross-Attention 都允许高阶 query 表示重新聚合序列；
- Output Fusion 使用 Per-head SwiGLU 对交互结果进一步建模。

最终，非序列交互和序列建模不再是两个互不相干的模块，而是在每个 block 中反复耦合。

---

## 3. 输入表示

### 3.1 非序列特征

设 user、item、context 等非序列 embedding 拼接后为：

$$\mathbf e_{\mathrm{ns}} \in \mathbb R^F.$$

MixFormer 将其均匀切成 $N$ 个连续 head，并分别投影到隐藏维度 $D$ ：

$$\mathbf q_n^{(0)} = \mathbf W_n \mathbf e_{\mathrm{ns}}[a_n:b_n] + \mathbf b_n,$$

$$\mathbf Q^{(0)} = [\mathbf q_1^{(0)};\ldots;\mathbf q_N^{(0)}] \in \mathbb R^{N\times D}.$$

从 tokenizer 角度看，它与 RankMixer Autosplit 非常接近：

- 连续向量切分；
- 独立 projector；
- 形成固定数量的 query heads；
- 每个 head 拥有独立后续参数。

论文使用 “head” 而不是 “token”，是因为这些表示随后直接作为 Cross-Attention queries。

### 3.2 行为序列

用户行为序列表示为：

$$\mathbf S = [\mathbf s_1;\mathbf s_2;\ldots;\mathbf s_L] \in \mathbb R^{L\times D_s}.$$

每个行为位置可以包含 item、action、context 等现有序列特征。不同 block 对 sequence action vectors 使用独立变换：

$$\widetilde{\mathbf S}^{(l)} = \operatorname{SwiGLU}^{(l)}_{\mathrm{seq}} (\mathbf S).$$

这样每一层可以从不同表示空间理解同一段历史，而不是所有层共享一套固定 sequence encoding。

---

## 4. Query Mixer

### 4.1 HeadMixing

Query Mixer 的 HeadMixing 与 RankMixer Token Mixing 本质相同。输入：

$$\mathbf Q \in \mathbb R^{B\times N\times D}.$$

每个 query head 按 channel 切成 $N$ 片：

$$\mathbf Q \rightarrow \mathbb R^{B\times N\times N\times(D/N)}.$$

交换两个 $N$ 维轴，并重新拼接：

$$\operatorname{HeadMixing}(\mathbf Q) \in \mathbb R^{B\times N\times D}.$$

与 RankMixer 一样，它是无参数 reshape / transpose 操作，不生成 $N\times N$ attention score。

### 4.2 Per-head SwiGLU

HeadMixing 后，每个 query head 使用独立 SwiGLU：

$$\mathbf U_n = \operatorname{SwiGLU}_n \left( \operatorname{HeadMixing}(\mathbf Q)_n \right).$$

Query Mixer 的简化 residual 形式为：

$$\mathbf Q' = \mathbf Q + \operatorname{PHSwiGLU} \left( \operatorname{HeadMixing} \left( \operatorname{RMSNorm}(\mathbf Q) \right) \right).$$

其中 PHSwiGLU 表示 Per-head SwiGLU。

### 4.3 为什么还需要 Query Mixer

如果直接让原始 user/item query heads 对序列做 Cross-Attention，则每个 query 只能用较浅的局部信息选择行为。Query Mixer 先完成非序列特征高阶组合，使后续 query 可以表达：

```text
user preference × current item category
user value level × current item price
query intent × creative type
user-item compatibility × context
```

然后这些高阶 query 再决定历史序列中哪些行为最相关。

---

## 5. Cross Attention

对于第 $n$ 个 query head：

$$\mathbf q_n = \mathbf Q'_n\mathbf W_{Q,n}.$$

序列产生 head-specific key 和 value：

$$\mathbf K_n = \widetilde{\mathbf S} \mathbf W_{K,n}, \qquad \mathbf V_n = \widetilde{\mathbf S} \mathbf W_{V,n}.$$

Cross-Attention 为：

$$\mathbf c_n = \operatorname{softmax} \left( \frac{ \mathbf q_n \mathbf K_n^\top }{\sqrt{d_k}} + \mathbf M \right) \mathbf V_n.$$

其中 $\mathbf M$ 可以包含 padding mask 或其他序列约束。

### 5.1 与独立序列塔的区别

传统 late fusion：

```text
sequence tower -> one sequence vector
static backbone -> one dense vector
                  -> final fusion
```

MixFormer：

```text
Block 1 query semantics -> attend sequence
        ↓
Block 2 higher-order query semantics -> attend sequence again
        ↓
Block 3 richer query semantics -> attend sequence again
```

因此 sequence aggregation 是 query-dependent、layer-dependent 和 head-specific 的。

### 5.2 为什么每层 sequence transform 独立

若所有层共享同一 sequence transformation，深层 query 虽然变复杂，但 Key/Value 空间固定。论文消融显示，将每层 sequence FFN 改为共享版本会造成约 0.03% 的相对 AUC 下降，说明 layer-specific sequence views 有实际价值。

---

## 6. Output Fusion

Cross-Attention 输出需要与 query 表示融合。MixFormer 使用每个 head 独立的 SwiGLU：

$$\mathbf o_n = \operatorname{SwiGLU}^{\mathrm{out}}_n \left( [\mathbf Q'_n;\mathbf c_n] \right).$$

再通过 residual 形成下一层 query：

$$\mathbf Q^{(l+1)}_n = \mathbf Q'_n + \mathbf o_n.$$

Per-head Output Fusion 的作用是：

- 保留 query head 的差异；
- 让不同 user/item 子空间以不同方式吸收 sequence 信息；
- 避免所有 heads 共用一个融合 MLP 后重新同质化。

论文消融中，将 Per-head Output Fusion 改为共享 FFN，AUC 相对下降约 0.06%，是 MixFormer 局部消融中较明显的一项。

---

## 7. 完整 block 公式

简化表示一个 block：

$$\widetilde{\mathbf Q}^{(l)} = \mathbf Q^{(l)} + \operatorname{PHSwiGLU}^{(l)}_{\mathrm{query}} \left( \operatorname{HeadMixing} \left( \operatorname{RMSNorm}(\mathbf Q^{(l)}) \right) \right),$$

$$\widetilde{\mathbf S}^{(l)} = \operatorname{SwiGLU}^{(l)}_{\mathrm{seq}} \left( \operatorname{RMSNorm}(\mathbf S) \right),$$

$$\mathbf C^{(l)} = \operatorname{CrossAttn}^{(l)} \left( \widetilde{\mathbf Q}^{(l)}, \widetilde{\mathbf S}^{(l)} \right),$$

$$\mathbf Q^{(l+1)} = \widetilde{\mathbf Q}^{(l)} + \operatorname{PHSwiGLU}^{(l)}_{\mathrm{out}} \left( [\widetilde{\mathbf Q}^{(l)};\mathbf C^{(l)}] \right).$$

这种结构形成了交替循环：

```text
非序列特征高阶交互
        ↓
以当前高阶语义读取行为序列
        ↓
把序列证据反馈给 query heads
        ↓
进入下一层继续交互
```

---

## 8. Figure 2：UI-MixFormer

<p align="center">
  <img src="https://arxiv.org/html/2602.14110v2/x2.png" width="92%" alt="MixFormer Figure 2 UI decoupling">
</p>

*图源：MixFormer Figure 2。*

### 8.1 为什么需要 user-item 解耦

在一个推荐请求中，同一个用户通常要评估多个候选 item。若每个候选都重新执行完整 MixFormer：

- user static features 重复计算；
- user behavior sequence 重复变换；
- user-only query heads 重复执行；
- serving FLOPs 随候选数线性重复。

UI-MixFormer 将 query heads 分成：

$$N=N_U+N_I.$$

其中：

- $N_U$ 为 user-side heads；
- $N_I$ 为 item/general heads；
- 论文实践中可使用接近 1:1 的划分。

### 8.2 单向 mixing mask

UI-MixFormer 对 HeadMixing 增加结构 mask：

- user heads 不能接收 item heads 的信号；
- item heads 可以读取 user heads；
- user 表示保持候选无关；
- item 表示仍能获得 user 条件信息。

可以抽象为 block mixing matrix：

$$\mathbf M_{\mathrm{UI}} = \begin{bmatrix} \mathbf M_{UU} & \mathbf 0\\ \mathbf M_{IU} & \mathbf M_{II} \end{bmatrix}.$$

上右角为零，表示 item 信息不能反向污染 user-only path。

### 8.3 Request-level batching

对于同一请求的多个候选：

```text
user heads + sequence representations
        -> 计算一次并缓存
item heads
        -> 每个候选单独计算
item heads 读取 cached user information
```

论文报告 UI-MixFormer 相对原 MixFormer 可减少约 36% FLOPs，并获得超过 30% 的 serving speedup，同时基本保持精度。

---

## 9. Table 1：离线主结果

论文在大规模推荐数据上比较了不同的 dense/sequence 组合。

| 模型 | Finish AUC | Finish UAUC | Skip AUC | Skip UAUC | 参数量 | GFLOPs |
|---|---:|---:|---:|---:|---:|---:|
| TA → RankMixer | +0.95% | — | — | — | — | — |
| STCA → RankMixer | +1.12% | — | — | — | — | — |
| STCA ⊕ RankMixer | +1.11% | — | — | — | — | — |
| MixFormer-Small | +1.01% | — | — | — | 约 282M | 约 733 |
| MixFormer-Medium | +1.28% | +1.60% | +1.60% | +2.46% | 约 1.226B | 约 3503 |
| UI-MixFormer-Medium | 与 Medium 接近 | 与 Medium 接近 | 与 Medium 接近 | 与 Medium 接近 | 接近 | 约 2242 |
| OneTrans | +1.05% | +1.31% | — | — | — | 约 23371 |

其中：

- TA 可理解为独立的 Transformer-style attention sequence model；
- STCA 是更强的序列 Cross-Attention 模块；
- 箭头表示先序列建模再进入 RankMixer；
- 加号表示并行或后融合；
- MixFormer-Medium 在准确率和 FLOPs 之间取得更优 Pareto；
- UI 版本显著降低 FLOPs，并保持相近精度。

论文代表配置中：

- query head 数约为 16；
- Small 使用 4 blocks、hidden 约 386；
- Medium 使用 4 blocks、hidden 约 768；
- serving batch size 约 1500。

---

## 10. 结构消融

| 消融 | 相对 AUC 变化 | 解释 |
|---|---:|---|
| Query Mixer 去掉 HeadMixing | -0.03% | 非序列 heads 需要跨域交互 |
| HeadMixing 换成 Self-Attention | 约 0.00% | 动态 attention 未提供额外收益，但更贵 |
| Query Mixer 去掉 FFN | -0.04% | mixing 后非线性不可缺少 |
| CrossAttn 每层 FFN 改为共享 | -0.03% | sequence view 需要随层变化 |
| Output Fusion per-head 改为共享 | -0.06% | head-specific 融合最重要 |
| Pre-RMSNorm 改为 Post-LN | -0.01% | Pre-Norm 更稳定 |

这组结果说明 MixFormer 的收益不是仅由 Cross-Attention 产生，而来自三类异质能力的组合：

```text
固定高效 HeadMixing
+ per-head nonlinear capacity
+ layer-specific sequence aggregation
```

---

## 11. Figure 3：组件消融

<p align="center">
  <img src="https://arxiv.org/html/2602.14110v2/x3.png" width="88%" alt="MixFormer Figure 3 ablation">
</p>

*图源：MixFormer Figure 3。*

Figure 3 可以读出两个趋势：

1. 把结构改成更通用的共享模块，通常会下降；
2. 把 HeadMixing 换成更昂贵的 Self-Attention，并没有明显提高效果。

这与 RankMixer 的结论一致：工业 sparse feature heads 并不一定需要动态 $N\times N$ attention，固定 mixing 加独立非线性参数可能具有更高 ROI。

---

## 12. Figure 4 与 Figure 5：Co-Scaling 曲线

<p align="center">
  <img src="https://arxiv.org/html/2602.14110v2/x4.png" width="48%" alt="MixFormer Figure 4 dense scaling">
  <img src="https://arxiv.org/html/2602.14110v2/x5.png" width="48%" alt="MixFormer Figure 5 sequence scaling">
</p>

*图源：MixFormer Figure 4–5。*

### 12.1 固定序列长度下扩展 dense capacity

在 sequence length 固定为约 512 时，论文比较不同模型随 GFLOPs 增长的效果。MixFormer 同时具有：

- 更好的起始性能；
- 较好的 scaling slope；
- 在相似计算预算下优于解耦模型。

### 12.2 固定模型规模下扩展 sequence length

论文测试约 512、2048、8192 和 10000 的序列长度。MixFormer 随 sequence length 增长保持与强序列模型相近的增益斜率，同时整体性能更高。

### 12.3 Co-Scaling 的真正含义

Co-Scaling 不是简单同时增大 $D$ 和 $L$ ，而是统一参数化后，dense 与 sequence 能力在每层相互促进：

- dense query 更强，序列检索更精准；
- sequence evidence 更丰富，下一层 dense interaction 更有效；
- 计算预算不再需要在两个完全独立的大模型之间静态切分。

---

## 13. Figure 6：延迟与效率

<p align="center">
  <img src="https://arxiv.org/html/2602.14110v2/x6.png" width="86%" alt="MixFormer Figure 6 latency">
</p>

*图源：MixFormer Figure 6。*

Figure 6 主要展示 UI decoupling 后的效率收益。对多候选排序请求，用户侧复用使实际 serving cost 显著下降。

与只看单样本 FLOPs 不同，推荐系统在线成本取决于：

$$\mathrm{Cost/request} = \mathrm{UserCost} + K\times\mathrm{ItemCost},$$

其中 $K$ 是每个请求的候选数。

若不解耦：

$$\mathrm{Cost/request} = K \left( \mathrm{UserCost} + \mathrm{ItemCost} \right).$$

因此 UI-MixFormer 的价值随候选数增大而提高。

---

## 14. 线上结果

论文在 Douyin 与 Douyin Lite 上进行大规模 A/B 测试。相对强 STCA → RankMixer 基线，代表提升包括：

| 场景 | Active Days | Duration | Like | Finish | Comment |
|---|---:|---:|---:|---:|---:|
| Douyin | +0.0415% | +0.2799% | +0.1766% | +0.3897% | +0.7035% |
| Douyin Lite | +0.0252% | +0.4105% | +0.2125% | +0.2924% | +1.9097% |

论文说明这些结果具有统计显著性，并体现 unified dense-sequence backbone 在真实推荐系统中的价值。

---

## 15. 与 RankMixer 的逐项对比

| 维度 | RankMixer | MixFormer |
|---|---|---|
| 主要输入 | 非序列 sparse/dense features | 非序列 features + 行为序列 |
| 基础 mixing | Token Mixing | Query HeadMixing |
| 动态序列读取 | 无 | 每层 Cross-Attention |
| FFN | Per-token GELU FFN | Per-head SwiGLU |
| 序列模型 | 通常外挂独立模块 | 统一在每个 block 内 |
| Dense-sequence 交互 | 末端或外部融合 | 每层深度交互 |
| Scaling 重点 | token、width、depth、expert | dense capacity 与 sequence length co-scaling |
| 多候选复用 | 无特定设计 | UI decoupling + request batching |
| 计算结构 | 全部固定 mixing | query 固定 mixing + sequence attention |
| 适用场景 | 静态特征主导 ranking | 长行为序列与候选匹配都重要 |

---

## 16. 与 TokenMixer-Large 的关系

MixFormer 和 TokenMixer-Large 并不是线性替代关系：

- TokenMixer-Large 重点修复 RankMixer block 并扩展 dense capacity；
- MixFormer 重点统一 dense interaction 与 sequence modeling；
- 两者都使用 Per-token / Per-head SwiGLU 和 Pre-RMSNorm；
- MixFormer 的 Query Mixer 仍继承 RankMixer 固定 mixing 思想；
- TokenMixer-Large 的 Mixing & Reverting 理论上也可用于改进 Query Mixer residual；
- SP-MoE 可以进一步用于 MixFormer 的 Per-head FFN，但论文未直接验证该组合。

因此它们分别解决：

```text
TokenMixer-Large：怎样把 dense ranking backbone 做得更深、更大、更稳
MixFormer：怎样让 dense backbone 与行为序列在统一结构中一起扩展
```

---

## 17. 对当前电商推广搜 CVR 的意义

当前已知输入只有 user、item、creative sparse embeddings。若没有行为序列张量，则不能严格复现完整 MixFormer，因为 Cross-Attention 的 Key/Value 缺失。

但论文仍提供三类可迁移启示。

### 17.1 Query heads 而非最终 tokens

可以把当前 16 或 32 个 RankMixer tokens 重新解释为 candidate-conditioned query heads，并使用 task-aware pooling，而不是简单 Mean Pooling。

### 17.2 user-item 单向结构

即使暂时没有序列，也可以研究 UI-style masked mixing：

```text
user tokens 不读取 item / creative
item tokens 可以读取 user
creative token 可以读取 user + item
```

该结构与当前 Base 的层级 SENet 条件方向高度一致：

```text
user <- user
item <- user + item
creative <- user + item + creative
```

因此 UI-MixFormer 的单向 mask 可作为“RankMixer 版本的层级条件交互”研究方向。

### 17.3 请求级复用

推广搜通常同一请求包含多个候选。如果特征管线保留请求分组，可以将 user-side tokenization 和 user-side blocks 缓存一次，再对不同 item/creative 候选执行 item-side path。这不增加新数据，但需要 serving graph 支持。

### 17.4 严格迁移的前提

要完整研究 MixFormer，需要当前已有数据中本来就存在且可使用的行为序列；若没有，不能通过静态 user features 伪造 sequence，并宣称复现论文。

---

## 18. 局限与复现注意事项

1. Full MixFormer 需要行为序列，只有静态稀疏特征时不具备完整输入条件；
2. UI decoupling 的收益依赖多候选 request-level serving；
3. 序列长度增加会显著提高 Cross-Attention 计算和显存；
4. 论文使用内部工业数据，绝对增益不能直接迁移；
5. 统一模型可能比独立模块更难进行局部故障隔离；
6. request cache、padding、序列存储和在线 batching 是实际部署的重要成本；
7. Query Mixer 使用固定 mixing，仍存在 UniMixer 所指出的不可学习限制；
8. 论文没有验证 MixFormer 与 SP-MoE、RankUp tokenization 的完整组合。

---

## 19. 一句话总结

MixFormer 的核心贡献可以概括为：

> 把 RankMixer 风格的高效非序列 Query Mixing 与层层 Cross-Attention 融合在同一个 backbone 中，使高阶 user-item 语义能够反复读取行为序列，并通过 UI 解耦支持多候选请求级复用。

它是 RankMixer 发展谱系中“从静态 feature interaction 走向 dense 与 sequence 统一 co-scaling”的分支。
