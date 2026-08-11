# RankMixer 后续修改版本总结：架构变化、技术路线与适用边界

> 文档范围：以 RankMixer 为统一起点，梳理截至 2026 年 8 月公开的代表性后续方法，包括 TokenMixer-Large、RankUp、MixFormer / UI-MixFormer、UniMixer / UniMixing-Lite，以及 2026 年 5 月提出的 RankElastor。  
> 阅读目标：回答“每篇工作具体改了 RankMixer 的哪一部分、完整架构是什么、解决了什么问题、代价是什么，以及这些方法之间能否组合”。  
> 主要论文：[RankMixer](https://arxiv.org/html/2507.15551v3) · [TokenMixer-Large](https://arxiv.org/html/2602.06563v2) · [MixFormer](https://arxiv.org/html/2602.14110v2) · [UniMixer](https://arxiv.org/html/2604.00590v1) · [RankUp](https://arxiv.org/html/2604.17878v3) · [RankElastor](https://arxiv.org/html/2605.23191v1)

---

## 1. 一页结论：RankMixer 之后并不是一条线性演进路线

RankMixer 建立了“有限数量的特征 token + 无参数 Token Mixing + Per-token FFN”的硬件友好范式。后续工作没有简单地用一个新模型完全替代它，而是从五个不同方向修补其瓶颈：

```text
RankMixer
│
├── TokenMixer-Large
│   └── 修改 block、残差、归一化、深层监督和 MoE
│
├── RankUp
│   └── 修改输入 token 的生成方式和 token 类型，提升表示有效秩
│
├── MixFormer / UI-MixFormer
│   └── 将非序列特征交互与用户行为序列建模统一到同一 backbone
│
├── UniMixer / UniMixing-Lite
│   └── 将固定 Token Mixing 改成结构化、可学习的 mixing
│
└── RankElastor
    └── 用全参数细粒度 mixing 与 GLU-PFFN直接改善谱结构和表示坍缩
```

最关键的区别是：

- **TokenMixer-Large** 认为 RankMixer 的核心瓶颈是 block 设计和超大模型训练；
- **RankUp** 认为主要瓶颈是输入 token 冗余、低秩与几何自由度不足；
- **MixFormer** 认为主要瓶颈是静态特征与行为序列长期分离；
- **UniMixer** 认为主要瓶颈是 mixing pattern 被人工固定；
- **RankElastor** 认为固定 block-transpose mixing 的扩秩能力有上限，普通 P-FFN 又会缩秩。

因此，这些方法中有一部分是互补的，而不是互斥的。

---

## 2. 统一坐标系：RankMixer 原始架构到底固定了什么

### 2.1 输入 tokenization

RankMixer 将异构 embedding 按预定顺序拼接成一个长向量，再切成 $T$ 个 segment，每个 segment 使用独立 projector 映射到 $D$ 维：

$$\mathbf e_{\mathrm{all}}=[\mathbf e_1;\mathbf e_2;\ldots;\mathbf e_N]\in\mathbb R^F.$$

$$\mathbf x_t=\operatorname{Proj}_t\!\left(\mathbf e_{\mathrm{all}}[a_t:b_t]\right)\in\mathbb R^D.$$

$$\mathbf X_0=\operatorname{Stack}(\mathbf x_1,\ldots,\mathbf x_T)\in\mathbb R^{T\times D}.$$

你当前的两种论文基线正对应 RankMixer 的代表配置：

| 配置 | Token 数 | 隐藏维度 | Block 数 |
|---|---:|---:|---:|
| 小模型 | 16 | 768 | 2 |
| 大模型 | 32 | 1536 | 2 |

### 2.2 固定 Multi-head Token Mixing

每个 token 被均分为 $H$ 个 channel slices：

$$\mathbf x_t=[\mathbf x_t^{(1)}\|\mathbf x_t^{(2)}\|\cdots\|\mathbf x_t^{(H)}].$$

第 $h$ 个输出 mixed token 由所有输入 token 的第 $h$ 个 slice 拼接而成：

$$\mathbf s_h=\operatorname{Concat}\!\left(\mathbf x_1^{(h)},\mathbf x_2^{(h)},\ldots,\mathbf x_T^{(h)}\right).$$

原论文设置 $H=T$，使 mixing 前后都保持 $T$ 个 token，便于直接做残差。

### 2.3 Per-token FFN

每个 token 拥有独立的两层 FFN：

$$\operatorname{PFFN}_t(\mathbf s_t)=\mathbf W_{t,2}\operatorname{GELU}(\mathbf W_{t,1}\mathbf s_t+\mathbf b_{t,1})+\mathbf b_{t,2}.$$

原始 block 使用 Post-LayerNorm：

$$\mathbf S_l=\operatorname{LN}\!\left(\operatorname{Mix}(\mathbf X_{l-1})+\mathbf X_{l-1}\right).$$

$$\mathbf X_l=\operatorname{LN}\!\left(\operatorname{PFFN}(\mathbf S_l)+\mathbf S_l\right).$$

最后用 Mean Pooling：

$$\mathbf h=\frac{1}{T}\sum_{t=1}^{T}\mathbf x_{L,t}.$$

### 2.4 RankMixer 已经验证的核心能力

RankMixer-100M 的论文消融表明：

| 消融 | 相对 AUC 变化 |
|---|---:|
| 去掉 Multi-head Token Mixing | -0.50% |
| Per-token FFN 改为 shared FFN | -0.31% |
| 去掉 skip connection | -0.07% |
| 去掉 LayerNorm | -0.05% |
| Token Mixing 改为 Self-Attention | -0.03%，同时 FLOPs 显著增加 |

这说明后续方法即使修改 RankMixer，也普遍保留两个基础共识：

1. 不同 token 需要相互交换信息；
2. 不同 token 的非线性参数最好保持一定程度的隔离。

### 2.5 后续论文集中攻击的五个薄弱点

| RankMixer 原始设计 | 后续暴露的问题 |
|---|---|
| 固定 Autosplit / 语义顺序 | token 信息密度不均、相关特征集中、表示冗余 |
| 固定 block-transpose mixing | 对所有样本和所有层使用同一模式，表达自由度有限 |
| mixing 后直接与原 token 相加 | mixing 前后位置语义可能不对齐 |
| GELU Per-token FFN | 深层中可能缩秩，且乘法建模效率不足 |
| Post-Norm + 浅层结构 | 加深后梯度和数值稳定性变差 |
| 静态特征主干与序列塔分离 | dense capacity 与 sequence length 竞争预算 |
| Dense-train / sparse-infer MoE | 训练成本仍高，路由激活数不稳定 |
| Mean Pooling | token 数增大后可能稀释少数强信号 |

---

## 3. 总体架构差异表

| 方法 | Tokenizer | Mixing | FFN | Norm / Residual | Sequence | MoE | 主要目标 |
|---|---|---|---|---|---|---|---|
| RankMixer | 连续 Autosplit / 语义排序 | 固定 reshape-transpose | GELU Per-token FFN | Post-LN，直接残差 | 外挂 | ReLU 路由，DTSI | 高 MFU、低延迟扩容 |
| TokenMixer-Large | Semantic groups + Global Token | 固定 Mix + Revert | 两组 Per-token SwiGLU | Pre-RMSNorm、inter-residual、aux loss | 作为 raw token 接入 | Sparse-train / sparse-infer SP-MoE | 深层稳定与十亿级扩展 |
| RankUp | Random field split + Multi-embedding + 多类型 token | 基本沿用 MetaFormer / RankMixer | Per-token SwiGLU | PreNorm | 可加 Seq Token | 非主要贡献 | 提升 effective rank 与 token diversity |
| MixFormer | 非序列特征切成 query heads | 固定 Query HeadMixing + 动态 Cross-Attention | Per-head SwiGLU | Pre-RMSNorm，多子层残差 | 每层深度融合 | 可扩展但非论文核心 | Dense 与 Sequence Co-Scaling |
| UniMixer | 可沿用现有 tokenizer | Learnable global-local mixing | Per-token SwiGLU | SiameseNorm | 可扩展 | 可接 SP-MoE | 学习 mixing pattern，解除 $H=T$ |
| RankElastor | 基本沿用 tokenization | Parameterized Full Mixing | GLU-improved P-FFN | mixing 内残差与 LN | 另做泛化验证 | 非核心 | “Expand more, shrink less” |

---

# 4. TokenMixer-Large：修改的是整个 Block 与超大模型训练路径

<p align="center">
  <img src="https://arxiv.org/html/2602.06563v2/x1.png" width="92%" alt="TokenMixer-Large architecture">
</p>

*图源：TokenMixer-Large Figure 1。*

## 4.1 它认为 RankMixer 的根本问题是什么

TokenMixer-Large 指出四类直接问题：

1. mixing 后 token 的语义已经变化，却仍与原位置 token 直接相加；
2. 两层 RankMixer 可以训练，但继续加深时早期层梯度不足；
3. 原 RankMixer 的 MoE 是 Dense Train / Sparse Infer，训练阶段没有真正稀疏；
4. 历史模型中保留的 DCN、LHUC 等碎片化小算子会拖低整体 MFU。

## 4.2 新架构

```text
Semantic Group-wise Tokenizer
        + Global Token
        ↓
Pre-RMSNorm
        ↓
Mix
        ↓
Mixed-layout Per-token SwiGLU
        ↓
Revert
        ↓
Pre-RMSNorm
        ↓
Original-layout Per-token SwiGLU
        ↓
Aligned Residual
        ↓
可选 Inter-Residual + Auxiliary Loss
        ↓
可选 Sparse-Pertoken MoE
```

### 4.2.1 Semantic Group-wise Tokenizer 与 Global Token

不同语义组分别使用独立 MLP 对齐：

$$\mathbf x_i=\operatorname{MLP}_i\!\left(\operatorname{Concat}\{\mathbf e_j:\mathbf e_j\in G_i\}\right).$$

Global Token 读取全部语义组：

$$\mathbf x_G=\operatorname{MLP}_G\!\left(\operatorname{Concat}(G_1,\ldots,G_{T-1})\right).$$

最终仍保持总 token 数为 $T$：

$$\mathbf X=[\mathbf x_G;\mathbf x_1;\ldots;\mathbf x_{T-1}]\in\mathbb R^{T\times D}.$$

### 4.2.2 Mixing & Reverting

原 RankMixer 的 mixed token 属于新的坐标系。TokenMixer-Large 先在 mixed layout 中做非线性建模，再执行 inverse reshape，把表示恢复到 original-token layout，最后才与原始输入做跨 block residual：

```text
Original-layout X
        ↓ Mix
Mixed-layout H
        ↓ pSwiGLU
Enhanced mixed H'
        ↓ Revert
Original-layout R
        ↓ pSwiGLU + residual from X
Original-layout X_next
```

这一改变同时解除“每层必须严格 $H=T$ 才能连续堆叠”的强约束，因为 Revert 会把输出恢复到原输入坐标。

### 4.2.3 Per-token SwiGLU

普通 GELU FFN 被替换为：

$$\operatorname{pSwiGLU}_t(\mathbf x)=\mathbf W_{t,\mathrm{down}}\!\left(\operatorname{Swish}(\mathbf W_{t,\mathrm{gate}}\mathbf x)\odot(\mathbf W_{t,\mathrm{up}}\mathbf x)\right).$$

相较普通 FFN，它增加了一条乘法 gate 路径，并继续保留 token-specific 参数。

### 4.2.4 深层训练机制

- **Pre-RMSNorm**：避免原 Post-Norm 在大模型中数值爆炸；
- **Inter-Residual**：每隔若干层连接低层和高层；
- **Auxiliary Loss**：直接监督中间层输出；
- **Down-Matrix Small Init**：让残差分支初始接近恒等映射；
- **最后一层不接 interval residual**：避免低阶信息干扰最终抽象。

### 4.2.5 Sparse-Pertoken MoE

TokenMixer-Large 不再采用 RankMixer 的动态 ReLU 激活专家数，而是将已证明有效的 dense Per-token SwiGLU 先扩大，再切分成若干专家，使用固定 top-$k$ 稀疏路由：

```text
先扩大 dense pSwiGLU
        ↓
确认 dense capacity 带来收益
        ↓
将每个 token 的 pSwiGLU 切成多个 experts
        ↓
Sparse Train + Sparse Infer
```

同时增加：

- 每个 token 自己的 shared expert；
- Gate Value Scaling；
- Down projection small initialization。

## 4.3 关键消融证据

论文在 4B 模型上的消融为：

| 组件移除或替换 | 相对 AUC 变化 |
|---|---:|
| 去掉 Mixing & Reverting | -0.27% |
| Per-token SwiGLU 改为共享 SwiGLU | -0.21% |
| 去掉 Residual | -0.15% |
| Per-token SwiGLU 改为 Per-token FFN | -0.10% |
| 去掉 Inter-Residual 与 AuxLoss | -0.04% |
| 去掉 Global Token | -0.02% |

这说明 TokenMixer-Large 的主升级不是 Global Token，而是：

> 语义对齐的 Mix-Revert 残差路径 + Per-token SwiGLU。

## 4.4 相对 RankMixer 到底改了哪里

| 模块 | RankMixer | TokenMixer-Large |
|---|---|---|
| Tokenizer | Autosplit / 语义顺序 | Semantic groups + Global Token |
| Mixer | Mix 后直接残差 | Mix 后建模，再 Revert 后残差 |
| FFN | 一组 Per-token GELU FFN | mixed/original 两组 Per-token SwiGLU |
| Norm | Post-LayerNorm | Pre-RMSNorm |
| 深层训练 | 无专门设计 | inter-residual、aux loss、small init |
| MoE | DTSI + ReLU routing | Sparse-train / sparse-infer SP-MoE |
| 工程 | 1B 级为主 | FP8、Token Parallel、线上 7B、离线 15B |

## 4.5 对当前模型的意义

对于你当前 $T=32,D=1536,L=2$ 的大模型，TokenMixer-Large 是**最直接的后续版本**，因为它不要求新数据，也不要求重建任务定义。建议先做两层计算匹配版本，而不是立即加深：

$$h_{\mathrm{SwiGLU}}\approx\frac{kD}{3}.$$

若原 PFFN 扩张倍数 $k=4$，两个 pSwiGLU 的计算匹配 hidden size 约为：

$$h_{\mathrm{SwiGLU}}\approx2048.$$

---

# 5. RankUp：修改的是输入表示基底和 Token 生态

<p align="center">
  <img src="https://arxiv.org/html/2604.17878v3/x1.png" width="92%" alt="RankUp architecture">
</p>

*图源：RankUp Figure 1。*

## 5.1 它认为 RankMixer 的根本问题是什么

RankUp 不首先质疑 Token Mixing，而是提出：

> 参数量增加后，初始 tokens 仍可能高度相关或由低覆盖字段主导，导致有效表示容量没有随参数同步增长。

它观察到 RankMixer 的 effective rank 在层间呈现：

```text
Token Mixer 后上升
        ↓
Per-token FFN 后下降
        ↓
下一层 Mixer 再上升
        ↓
下一层 FFN 再下降
```

即阻尼振荡。

## 5.2 新架构

```text
Original sparse fields
├── Randomized field-level grouping
│       └── Local Sparse Tokens
├── Independent embedding view
│       └── Global Token
├── Independent embedding view / selected features
│       └── Interaction Token
├── Pre-trained user/item embeddings
│       └── Cross Token
├── Sequence encoder
│       └── Seq Token
└── Learnable parameters
        └── Task-Specific Tokens

All tokens
    ↓
PreNorm MetaFormer / RankMixer blocks
    ↓
Per-token SwiGLU
    ↓
Task-specific towers
```

## 5.3 Randomized Permutation Splitting

RankUp 对**字段索引**做随机排列，而不是对展开后的标量坐标随机打乱：

$$\mathcal F_\sigma=\{f_{\sigma(1)},f_{\sigma(2)},\ldots,f_{\sigma(M)}\}.$$

然后把完整字段 embedding 均衡分组、拼接并独立投影。其目标不是最大化任意噪声熵，而是：

- 分散高相关字段；
- 分散长尾或高缺失字段；
- 提高最低 token effective rank；
- 降低 token 间 mutual information；
- 让不同 PFFN 获得更均衡的有效梯度。

在工程实现中，一个模型版本应固定 permutation mapping；每 batch 动态重排会破坏 Per-token FFN 的 token identity。

## 5.4 Multi-embedding

同一原始字段可以进入多个独立 embedding 空间：

$$f_j\longrightarrow\{\psi_1(f_j),\psi_2(f_j),\ldots,\psi_K(f_j)\}.$$

这不是把同一向量复制多份，而是让 local token、global token 和 interaction token拥有独立参数与梯度路径。

## 5.5 Global、Cross 与 Task Tokens

Global Token：

$$\mathbf g=\operatorname{Func}\!\left(\operatorname{Pool}\{\operatorname{Embed}(f_i)\}_{i=1}^{M}\right).$$

Crossed Pre-trained Embedding Token：

$$\mathbf c=\operatorname{Proj}(\mathbf z_u\odot\mathbf z_i).$$

Task Token：

$$\{\mathbf q_1,\mathbf q_2,\ldots,\mathbf q_K\}.$$

它们分别提供：

- 全局上下文；
- 外部 user-item 匹配先验；
- 多任务私有汇聚位置。

## 5.6 关键消融证据

| 组件 | Order | Book | Add Service |
|---|---:|---:|---:|
| Randomized Permutation Split | +0.06% | +0.06% | +0.08% |
| Global Token + Multi-Embedding | +0.21% | +0.18% | +0.13% |
| Cross Embedding | +0.22% | +0.10% | +0.03% |
| Task Token | +0.09% | +0.02% | +0.02% |
| Full RankUp | +0.41% | +0.23% | +0.25% |

这些增益不能简单相加，说明组件之间存在能力重叠。

## 5.7 相对 RankMixer 到底改了哪里

| 模块 | RankMixer | RankUp |
|---|---|---|
| 字段切分 | 连续 Autosplit / 语义顺序 | field-level randomized split |
| Embedding | 通常单视角 | Multi-embedding |
| Token 类型 | 主要 local feature tokens | Global、Cross、Seq、Task 等多类型 tokens |
| Mixer | 基本保留 RankMixer / MetaFormer | 不是主要修改对象 |
| FFN | GELU PFFN | PreNorm + Per-token SwiGLU |
| 诊断指标 | AUC、MFU、延迟 | effective rank、MI、token redundancy |
| 主要问题 | 计算与扩容 | 表示容量利用率 |

## 5.8 对当前 1234 个字段的直接映射

当 $T=16$ 时：

$$1234=2\times78+14\times77.$$

当 $T=32$ 时：

$$1234=18\times39+14\times38.$$

因此可在不增加 local projector 主体参数的情况下，实现完整字段级 Random Split。严格 RankUp 的 Global、Cross 与 Task Token 是否能复现，则取决于你是否已有独立 embedding views、预训练 user/item 向量和多任务标签。

---

# 6. MixFormer / UI-MixFormer：修改的是“静态特征主干与序列塔分离”的系统结构

<p align="center">
  <img src="https://arxiv.org/html/2602.14110v2/x1.png" width="92%" alt="MixFormer architecture">
</p>

*图源：MixFormer Figure 1。*

## 6.1 它认为 RankMixer 的根本问题是什么

RankMixer 擅长非序列 feature interaction，但工业系统往往还需要一个独立 sequence tower：

```text
Non-sequential features -> RankMixer
Behavior sequence       -> Transformer / STCA / DIN
                          -> late fusion
```

这种架构存在：

- dense backbone 与 sequence model 争夺计算预算；
- 两路表示只在末端交互；
- user-item 高阶条件无法逐层指导序列聚合；
- 同一请求的多个候选可能重复计算用户侧序列。

## 6.2 新架构

```text
Non-sequential features
        ↓ split + projection
N query heads
        ↓
┌────────────────────────────────────┐
│ MixFormer Block × L                │
│                                    │
│ Query Mixer                        │
│   HeadMixing + Per-head SwiGLU     │
│        ↓                           │
│ Cross-Attention over sequence K/V  │
│        ↓                           │
│ Output Fusion                      │
│   Per-head SwiGLU                  │
└────────────────────────────────────┘
        ↓
Task heads
```

### 6.2.1 Query Mixer

非序列 query heads 先做 RankMixer 风格的固定 HeadMixing：

$$\mathbf P=\operatorname{HeadMixing}(\operatorname{Norm}(\mathbf X))+\mathbf X.$$

每个 head 使用独立 SwiGLU：

$$\mathbf q_i=\operatorname{SwiGLU}_i(\operatorname{Norm}(\mathbf p_i))+\mathbf p_i.$$

### 6.2.2 Cross-Attention

Query Mixer 输出的高阶 user-item-context 表示作为 queries，行为序列作为 keys / values：

$$\mathbf Z=\operatorname{CrossAttn}(\mathbf Q,\mathbf K(\mathbf S),\mathbf V(\mathbf S))+\mathbf Q.$$

这里使用 attention 是合理的，因为 query 与序列行为存在明确的匹配语义；论文并不是重新用 Self-Attention 处理所有异构静态字段。

### 6.2.3 Output Fusion

每个 head 独立融合非序列和序列证据：

$$\mathbf o_i=\operatorname{SwiGLU}_i(\operatorname{Norm}(\mathbf z_i))+\mathbf z_i.$$

下一层 Query Mixer 将读取已经包含序列信息的 heads，从而形成逐层深度耦合。

## 6.3 UI-MixFormer

UI-MixFormer 将 heads 拆为 user-side 与 item/general-side，并对 HeadMixing 加单向 mask：

```text
User heads:
    只能读取 user-side 信息
    可按 request 计算一次并缓存

Item heads:
    可以读取 user-side + item-side 信息
    每个候选单独计算
```

抽象 mixing mask 为：

$$\mathbf M_{\mathrm{UI}}=\begin{bmatrix}\mathbf M_{UU}&\mathbf 0\\\mathbf M_{IU}&\mathbf M_{II}\end{bmatrix}.$$

它保留 user $\rightarrow$ item 的条件化，但禁止 item $\rightarrow$ user 反向污染，使 user path 可跨候选复用。

论文中 MixFormer-medium 与 UI-MixFormer-medium 的离线指标基本一致，而 GFLOPs/Batch 从约 3503 降到约 2242；结合 request-level batching 后 serving speedup 超过 30%。

## 6.4 相对 RankMixer 到底改了哪里

| 模块 | RankMixer | MixFormer |
|---|---|---|
| 非序列交互 | Token Mixing + PFFN | Query HeadMixing + Per-head SwiGLU |
| 序列建模 | 外挂独立模块 | 每个 block 内 Cross-Attention |
| Dense / Sequence 融合 | 通常末端融合 | 每层深度融合 |
| 参数分配 | dense 与 sequence 两套模块 | 统一 backbone 参数 |
| 多候选复用 | 无专门约束 | UI decoupling + request batching |
| 主要 scaling 轴 | $T,D,L,E$ | dense width/depth + sequence length |

## 6.5 对当前场景的边界

若当前训练样本只有静态 user、item、creative embeddings，没有用户行为序列，则不能宣称严格复现 MixFormer。可以借鉴的仅有：

- user/item heads 分区；
- 单向 mixing mask；
- request-level user-side 复用；
- Per-head task-aware fusion。

---

# 7. UniMixer / UniMixing-Lite：修改的是固定 Mixing Operator

<p align="center">
  <img src="https://arxiv.org/html/2604.00590v1/x2.png" width="92%" alt="UniMixer architecture">
</p>

*图源：UniMixer Figure 2。*

## 7.1 它认为 RankMixer 的根本问题是什么

RankMixer Token Mixing 可以表示成一个固定 permutation matrix：

$$\operatorname{vec}(\operatorname{Mix}(\mathbf X))=\operatorname{vec}(\mathbf X)\mathbf W_{\mathrm{perm}}.$$

该矩阵：

- 没有参数；
- 对所有样本相同；
- 对所有层相同；
- mixing pattern 由人工 reshape 决定；
- 原实现通常要求 $H=T$。

UniMixer 的目标是将其推广为“可学习但仍结构化”的 static mixing。

## 7.2 Global Mixing + Local Mixing

先将 flatten 后的表示切为 $M$ 个 blocks，每个 block 宽度为 $B$：

$$M=\frac{L}{B},\qquad L=TD.$$

每个 block 使用自己的 local matrix：

$$\mathbf z_i=\mathbf x_i\mathbf W_B^{(i)}.$$

Global matrix 控制不同 blocks 的交换：

$$\widetilde{\mathbf z}_j=\sum_{i=1}^{M}W_{G,ij}\mathbf z_i.$$

因此完整 UniMixing 把两类能力解耦：

- $\mathbf W_B^{(i)}$：block 内部异构变换；
- $\mathbf W_G$：block 之间全局交互。

## 7.3 UniMixing-Lite

完整 global/local matrices 可能过大，Lite 版本使用：

1. **低秩 Global Mixing**

$$\mathbf W_G=\operatorname{Sinkhorn}(\mathbf A_G\mathbf B_G).$$

2. **Basis-composed Local Mixing**

$$\mathbf W_B^{(i)}=\operatorname{Sinkhorn}\!\left(\sum_{\ell=1}^{b}\omega_{\ell}^{(i)}\mathbf Z_\ell\right).$$

3. **Temperature Annealing + Warm-up**

早期保持 mixing 平滑，后期逐渐形成更尖锐、接近 permutation 的模式。

4. **SiameseNorm**

协调 learnable mixing 分支与 residual 分支的统计量，而不是简单沿用 Post-LN。

5. **Per-token SwiGLU / 可选 SP-MoE**

继续保留 token-specific 非线性容量。

## 7.4 新架构

```text
Input tokens / blocks
        ↓
Block-specific Local Mixing
        ↓
Learnable Global Mixing
        ↓
Residual + SiameseNorm
        ↓
Per-token SwiGLU
        ↓
可选 Sparse-Pertoken MoE
        ↓
Stack multiple UniMixer blocks
```

## 7.5 相对 RankMixer 到底改了哪里

| 模块 | RankMixer | UniMixer |
|---|---|---|
| Mixing pattern | 固定 permutation | 可训练 global-local matrices |
| 粒度 | 由 $T,H,D$ 固定 | 由 block size $B$ 控制 |
| $H=T$ | 通常必须满足 | 解除强约束 |
| Local transformation | Identity | block-specific learnable matrix |
| Global transformation | 固定 reshape | learnable，可低秩 |
| 结构约束 | 无学习 | Sinkhorn 双随机、temperature |
| Norm | Post-LN | SiameseNorm |
| Scaling 目标 | 增大 PFFN 参数 | 提高 mixing capacity 的 scaling ROI |

## 7.6 与 RankElastor 的根本区别

UniMixer 与 RankElastor 都“让 mixing 可学习”，但不是同一方案：

- UniMixer 使用结构化 global-local 分解、低秩和 basis，目标是生产级 scaling ROI；
- RankElastor 使用 $TD\times TD$ 的 full mixing，目标是最大化细粒度谱表达能力；
- UniMixer 的 mixing 更像受约束的可学习静态算子；
- RankElastor 更接近对 flatten 表示做完整线性层。

对于超大 $T,D$，UniMixing-Lite 通常比直接 Full Mixing 更现实。

---

# 8. RankElastor：最新的“核心算子级”高秩改造

<p align="center">
  <img src="https://arxiv.org/html/2605.23191v1/x11.png" width="92%" alt="RankElastor architecture">
</p>

*图源：RankElastor Figure 3，左侧为 RankMixer，右侧为 RankElastor。*

## 8.1 它与 RankUp 的问题意识相似，但修改位置不同

RankUp 主要通过输入 tokenization、多 embedding views 和附加 tokens 提高初始表示多样性。

RankElastor 则直接修改 RankMixer 的两个核心算子：

1. Token Mixing；
2. Per-token FFN。

它提出的口号是：

```text
Expand More:
    让 mixing 对 effective rank 的提升更强

Shrink Less:
    让 P-FFN 对 effective rank 的压缩更弱
```

## 8.2 注意：RankElastor 与 RankUp 的 effective rank 定义不同

RankUp 主要使用基于归一化奇异值熵的 effective rank：

$$\operatorname{erank}_{\mathrm{entropy}}(\mathbf X)=\exp\!\left(-\sum_i p_i\log p_i\right).$$

RankElastor 使用 stable-rank 形式：

$$\operatorname{erank}_{\mathrm{stable}}(\mathbf X)=\frac{\|\mathbf X\|_F^2}{\|\mathbf X\|_2^2}.$$

两者都衡量谱是否集中，但绝对值和数值范围不同，实验中不能直接混报或横向比较。

## 8.3 Parameterized Full Mixing

RankMixer 的固定 block-transpose 可以写为带 Kronecker 约束的 permutation：

$$\operatorname{vec}(\mathbf M^\top)=\operatorname{LN}\!\left((\mathbf P\otimes\mathbf I)\operatorname{vec}(\mathbf X^\top)+\operatorname{vec}(\mathbf X^\top)\right).$$

RankElastor 将它替换为 full mixing：

$$\operatorname{vec}(\mathbf M^\top)=\operatorname{LN}\!\left((\mathbf W+\mathbf I)\operatorname{vec}(\mathbf X^\top)\right).$$

其中：

$$\mathbf W\in\mathbb R^{TD\times TD}.$$

这意味着任意 token-channel 坐标都可以与任意其他坐标交互，不再受 block-transpose 或 Kronecker 结构限制。

## 8.4 GLU-improved P-FFN

RankElastor 的 token-specific FFN 为：

$$\mathbf Z_t=\left(\operatorname{GELU}(\mathbf M_t\mathbf W_1)\odot(\mathbf M_t\mathbf W_2)\right)\mathbf W_3+\mathbf M_t\mathbf W_r.$$

相比普通 PFFN：

- 增加一条乘法 gate；
- 增加可学习 residual mapping $\mathbf W_r$；
- 理论上可形成二阶多项式交互；
- 论文证明其对低秩输入具有更强的 rank recovery 能力。

## 8.5 新架构

```text
Tokenization
        ↓
Flatten T × D representation
        ↓
Parameterized Full Mixing: (TD) × (TD)
        ↓
Residual + LayerNorm
        ↓
Token-specific GLU-improved P-FFNs
        ↓
Stack L blocks
        ↓
Output projection
```

## 8.6 论文结果与消融

### 主结果

| 模型 | Criteo AUC | Criteo LogLoss | Avazu AUC | Avazu LogLoss |
|---|---:|---:|---:|---:|
| RankMixer | 0.81375 | 0.43799 | 0.79270 | 0.37218 |
| RankElastor | 0.81482 | 0.43730 | 0.79323 | 0.37196 |

### 模块消融

| 方案 | Criteo AUC | Avazu AUC |
|---|---:|---:|
| Full RankElastor | 0.81482 | 0.79323 |
| 去掉 Full Mixing | 0.81413 | 0.79289 |
| 改为 GELU FFN | 0.81349 | 0.79288 |
| RankMixer + GLU-style FFN | 0.81393 | 0.79286 |

重要结论是：只给原 RankMixer 换 GLU 收益有限；Full Mixing 与 GLU-PFFN 有明显协同。

## 8.7 复杂度与工业迁移风险

RankMixer mixing 的复杂度约为：

$$\mathcal O(TD).$$

RankElastor Full Mixing 的参数与计算复杂度约为：

$$\mathcal O(T^2D^2).$$

论文实验使用的尺寸很小：

- Criteo： $T=15,D=26$ ，Full Mixing 约 15.21 万参数；
- Avazu： $T=16,D=24$ ，Full Mixing 约 14.75 万参数。

而你的模型若直接照搬：

| 配置 | $TD$ | 单个 Full Mixing 矩阵参数 |
|---|---:|---:|
| $T=16,D=768$ | 12,288 | 150,994,944，约 151M |
| $T=32,D=1536$ | 49,152 | 2,415,919,104，约 2.416B |

这还只是**一个 mixing layer**，没有计算 optimizer state、激活、PFFN 和多层堆叠。因此 RankElastor 论文中“训练时间仅增加约 10%–15%”的结论不能直接外推到你的大模型。

对当前系统更可行的是 RankElastor-inspired 结构化近似：

$$\mathbf W\approx\mathbf A\mathbf B^\top,\qquad \operatorname{rank}(\mathbf W)=r\ll TD.$$

或者采用 block-sparse、Kronecker-sum、低秩 adapter；但这已经不是严格的 Full Mixing 复现，并会与 UniMixing-Lite高度接近。

---

# 9. 六种架构的最简计算图对照

## 9.1 RankMixer

```text
Autosplit Tokens
    -> Fixed Mix
    -> Add & Post-LN
    -> Per-token GELU FFN
    -> Add & Post-LN
    -> Mean Pooling
```

## 9.2 TokenMixer-Large

```text
Semantic Local Tokens + Global Token
    -> Pre-RMSNorm
    -> Mix
    -> Mixed-layout pSwiGLU
    -> Revert
    -> Pre-RMSNorm
    -> Original-layout pSwiGLU
    -> Aligned Residual
    -> optional Inter-Residual / AuxLoss / SP-MoE
```

## 9.3 RankUp

```text
Random Local Tokens
+ Multi-embedding Views
+ Global Token
+ Cross Token
+ Seq Token
+ Task Tokens
    -> PreNorm RankMixer / MetaFormer
    -> Per-token SwiGLU
    -> Task-specific Towers
```

## 9.4 MixFormer

```text
Non-sequential Query Heads
    -> Query HeadMixing + Per-head SwiGLU
    -> Cross-Attention over Behavior Sequence
    -> Per-head Output Fusion
    -> repeat L blocks
    -> Tasks
```

## 9.5 UniMixer

```text
Input Blocks
    -> Block-specific Local Mixing
    -> Learnable Global Mixing
    -> Residual + SiameseNorm
    -> Per-token SwiGLU / SP-MoE
    -> repeat L blocks
```

## 9.6 RankElastor

```text
Tokens
    -> Flatten
    -> Full (TD × TD) Learnable Mixing
    -> Residual + LN
    -> GLU-improved Token-specific FFN
    -> repeat L blocks
```

---

# 10. 各方法究竟解决 RankMixer 的哪一个瓶颈

| RankMixer 症状 | 最直接的后续方法 | 原因 |
|---|---|---|
| mixing 前后残差语义错位 | TokenMixer-Large | Revert 后再做 original-layout residual |
| 两层有效、加深后不稳定 | TokenMixer-Large | Pre-RMSNorm、inter-residual、aux loss |
| MoE 训练阶段仍然稠密 | TokenMixer-Large | Sparse train / sparse infer SP-MoE |
| token 信息量不均、长尾字段集中 | RankUp | Randomized field-level splitting |
| 参数很多但 effective rank 低 | RankUp / RankElastor | 分别改善输入基底与核心算子谱动态 |
| 固定 mixing pattern 不适配数据 | UniMixer / RankElastor | 结构化 learnable mixing / full mixing |
| $H=T$ 限制结构空间 | TokenMixer-Large / UniMixer | Revert 或一般化 mixing |
| 静态特征与行为序列割裂 | MixFormer | 每层 Query Mixer + Cross-Attention |
| 多候选重复计算 user/sequence | UI-MixFormer | 单向 mask + request-level reuse |
| Mean Pooling 稀释重要 token | 上述论文均未彻底统一解决 | 仍需独立做 task-aware pooling 消融 |

---

# 11. 方法之间的关系：哪些高度互补，哪些高度重叠

## 11.1 高度互补

### RankUp Tokenizer + TokenMixer-Large Block

```text
RankUp:
    改善输入 token basis 与 token diversity

TokenMixer-Large:
    改善 block residual、FFN 与深层优化
```

这是最自然的组合之一。应使用 2×2 factorial：

| 实验 | RankUp 输入 | TokenMixer-Large Block |
|---|---|---|
| F-00 | 无 | 无 |
| F-10 | 有 | 无 |
| F-01 | 无 | 有 |
| F-11 | 有 | 有 |

### RankUp Tokenizer + UniMixing-Lite

前者降低初始冗余，后者学习后续 mixing pattern，理论上也互补。但二者都可能改变 token geometry，因此必须先分别验证。

### MixFormer + TokenMixer-Large 的 FFN / Norm 设计

MixFormer 的 Query Mixer 仍继承 RankMixer 固定 HeadMixing，可借鉴：

- Pre-RMSNorm；
- Per-head SwiGLU；
- down projection small init；
- 深层 residual。

完整 Mixing & Reverting 如何嵌入 Query Mixer 需要重新定义，不能机械复制。

## 11.2 高度重叠

### UniMixer 与 RankElastor

二者都将固定 mixing 改为可学习 mixing：

- UniMixer 重点是结构化、低秩和 scaling ROI；
- RankElastor 重点是 full coordinate mixing 与谱理论。

首轮不建议同时加入，否则无法判断收益来自哪一种 learnable mixing。

### RankUp 与 RankElastor

两者都针对 effective rank，但修改位置不同：

- RankUp：输入层和 token 类型；
- RankElastor：block 内 mixing 与 PFFN。

可以组合，但应先确认是否存在独立增益。若 RankUp 已经让 tokenizer 输出和各层 erank 很高，Full Mixing 的额外价值可能下降。

---

# 12. 对你当前电商推广搜 CVR 模型的优先级

当前可用信息为：

```text
1234 sparse fields
embedding dim = 17
small: T=16, D=768, L=2
large: T=32, D=1536, L=2
no Sparse MoE
static user / item / creative features
```

## 12.1 第一优先级：TokenMixer-Large Block Lite

原因：

- 不增加新数据；
- 不改变 CVR 标签空间；
- 直接修复 RankMixer 被后续论文明确指出的 block 缺陷；
- 可通过计算匹配控制变量；
- 对大模型比对小模型更重要。

首轮：

```text
TM-0  原 RankMixer block
TM-1  只加 Mixing & Reverting
TM-2  TM-1 + compute-matched pSwiGLU
TM-3  TM-2 + Pre-RMSNorm
TM-4  TM-3 + down small init
```

当前只有两层时，不要先上 inter-residual 与 auxiliary loss。

## 12.2 第二优先级：RankUp Tokenization

首轮只改字段分组，不同时上 Multi-embedding、Cross Token 和 Task Token：

```text
RU-0  当前连续 Autosplit
RU-1  ordered field-aligned split
RU-2  fixed random field split，3 个 mapping seeds
RU-3  stratified random split
RU-4  T-1 local + 1 global
```

需要同时监控：

```text
minimum / mean / std token effective rank
pairwise token cosine similarity
token mutual information
per-token gradient norm
default-value / missing-rate distribution
```

## 12.3 第三优先级：UniMixing-Lite Adapter

先从最简单的可学习 $T\times T$ mixing matrix 开始，再逐步增加低秩、basis 和 Sinkhorn。不要首轮直接实现完整 UniMixer。

```text
UM-0  Fixed RankMixer Mix
UM-1  Learned T × T token mixing
UM-2  Low-rank token mixing
UM-3  Global-local mixing
UM-4  Sinkhorn + temperature schedule
```

## 12.4 第四优先级：RankElastor-inspired Mixing

不建议在 $T=32,D=1536$ 上直接使用完整 $49{,}152\times49{,}152$ 矩阵。可做小规模机制验证：

```text
RE-0  RankMixer
RE-1  仅 GLU-improved PFFN
RE-2  低秩 flatten-mixing adapter
RE-3  结构化 block full mixing
```

若 RE-1 只有很小收益，而 RE-2 + RE-1 有协同，才符合 RankElastor 论文机制。

## 12.5 条件路线：MixFormer

只有当前系统已经有行为序列时才进入严格 MixFormer。若没有序列数据，仅测试 UI-style user/item directional mixing，不应把模型命名为 MixFormer。

---

# 13. 推荐的统一消融矩阵

## Phase A：Block 与输入分开

```text
A0  RankMixer baseline
A1  Mixing & Reverting
A2  compute-matched pSwiGLU
A3  field-aligned tokenizer
A4  fixed random tokenizer × 3 seeds
A5  1 global + T-1 local
```

## Phase B：验证 learnable mixing

```text
B0  best fixed-mixing model
B1  learned T × T mixing
B2  low-rank global-local mixing
B3  RankElastor-inspired low-rank flatten mixing
B4  same-parameter MLP adapter control
```

## Phase C：只组合独立获益模块

```text
C0  best tokenizer + original block
C1  original tokenizer + best block
C2  best tokenizer + best block
C3  C2 + best learnable mixing
```

## Phase D：深度与 MoE

只有 C2 / C3 已经稳定优于对应两层模型后，才进行：

```text
D1  L=4
D2  L=4 + inter-residual
D3  L=4 + auxiliary loss
D4  dense enlargement
D5  Sparse-Pertoken MoE
```

遵循 TokenMixer-Large 的原则：

> First enlarge, then sparse。

---

# 14. 评价指标必须和论文声称解决的问题对应

| 方法 | 除 AUC / LogLoss 外必须记录的指标 |
|---|---|
| TokenMixer-Large | 深度学习曲线、梯度范数、NaN/Inf、update-to-weight ratio、MFU |
| RankUp | entropy effective rank、token MI、token redundancy、mapping-seed 方差 |
| MixFormer | sequence-length scaling、request cost、candidate reuse rate、P99 |
| UniMixer | learned matrix sparsity、temperature、global rank、basis 权重、scaling exponent |
| RankElastor | stable rank、mixing 前后谱变化、PFFN rank contraction、full-mixing 开销 |
| 所有方法 | AUC、GAUC/UAUC、LogLoss、校准、吞吐、HBM、P50/P95/P99 |

尤其要避免把 RankUp 的 entropy effective rank 与 RankElastor 的 stable rank 当成同一数值指标。

---

# 15. 工业视角下的最终评价

| 方法 | 科学价值 | 工业成熟度 | 当前静态 CVR 适配度 | 主要风险 |
|---|---|---|---|---|
| TokenMixer-Large | 很高 | 很高，已有多业务在线部署 | 很高 | 实现复杂、Grouped GEMM / Revert 优化 |
| RankUp | 很高 | 高，广告生产部署 | 很高 | Multi-embedding 内存、随机 mapping 稳定性 |
| MixFormer / UI | 很高 | 高，推荐线上部署 | 条件适用 | 必须已有行为序列与请求级候选 |
| UniMixer-Lite | 很高 | 有线上验证 | 中高 | mixing 优化、Sinkhorn 与温度调度复杂 |
| RankElastor | 理论价值很高 | 论文基准阶段 | 中，需结构化改造 | Full Mixing 在大 $T,D$ 下不可承受 |
| 原 RankMixer | 硬件基线价值很高 | 很高 | 已完成复现 | 固定 mixing、原 block 与 pooling 上限 |

---

## 16. 最终结论

RankMixer 后续演进可以概括为五句话：

1. **TokenMixer-Large**：保留固定 mixing 的硬件优势，但用 Mix-Revert、pSwiGLU、Pre-RMSNorm 与 SP-MoE 将 block 变得更深、更稳、更适合超大规模。
2. **RankUp**：不首先改 mixer，而是让进入 mixer 的 token 更独立、更多样、更高秩，并加入 Global、Cross、Sequence 与 Task 等信息载体。
3. **MixFormer**：把 RankMixer 风格的非序列 Query Mixer 与动态序列 Cross-Attention 放进同一个 block，实现 dense 与 sequence 的共同扩展。
4. **UniMixer**：把固定 reshape permutation 变成结构化可学习 global-local mixing，并通过低秩、basis 与 Sinkhorn控制成本。
5. **RankElastor**：从谱理论出发，用 Full Mixing 扩得更多、用 GLU-PFFN 缩得更少，但其完整 mixing 在当前大模型尺寸上必须结构化近似。

对你当前的电商推广搜 CVR，最稳健的演进顺序是：

```text
RankMixer
    ↓
TokenMixer-Large Block Lite
    ↓
RankUp field-level tokenizer / Global Adapter
    ↓
UniMixing-Lite learnable mixing
    ↓
必要时做 RankElastor-inspired 低秩谱增强
    ↓
已有序列时再进入 MixFormer / UI-MixFormer
```

这条路线能够把“输入表示”“block 质量”“mixing 自由度”“序列统一”和“稀疏扩容”分成可独立验证的科学变量，避免一次堆叠所有模块后无法归因。
