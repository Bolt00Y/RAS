# RankMixer 与可扩展 Mixer 排序模型文献综述

> 文献检索截止：**2026-08-31**<br>
> 适用任务：工业推荐/广告/搜索中的 CTR、CVR、pCVR 与多目标排序<br>
> 核心范围：RankMixer、TokenMixer-Large、MixFormer、UniMixer，以及从固定 Token Mixing 向可学习混合、序列统一、在线复用和表示秩优化发展的相关工作<br>
> 文档性质：公开文献综述 + 本仓库实现映射；论文结论与本地实验结论严格分开

---

## 0. 结论先行

RankMixer 家族不是“把视觉 MLP-Mixer 原样搬到推荐”这么简单。它真正形成了一套面向工业排序的设计范式：

1. 将数百到数千个异构稀疏字段压缩成少量、等宽且有固定身份的 Token；
2. 用规则化、矩阵友好的 Token Mixing 代替异构字段上的全量自注意力；
3. 用 **Per-token FFN/SwiGLU** 为不同语义子空间保留独立参数；
4. 通过宽度、深度、Token 数和专家数扩展 dense 容量，并同时优化 GPU 利用率与在线延迟。

截至检索日，最清晰的主干演进是：

```text
MLP-Mixer / MetaFormer
          │  提供“Token Mixer + Channel MLP”的通用骨架
          ▼
RankMixer (2025)
  固定无参 Mixing + Per-token FFN + 语义 Token + 可选 MoE
          │
          ├──────────────► MTmixAtt (2025)
          │                 自动 Token + 可学习逐头 Mixing + 多场景 MoE
          │
          ▼
TokenMixer-Large (2026)
  Mixing/Reverting + Pre-RMSNorm + 两段 pSwiGLU
  + 跨层残差/辅助损失 + Sparse-Pertoken MoE + 系统优化
          │
          ├──────────────► UG-Sep
          │                 分离用户/候选信息流，复用用户侧计算
          │
          ├──────────────► MixFormer / HyFormer / HeMix
          │                 将稠密交互与长行为序列更早、更深地结合
          │
          └──────────────► RankUp / RankElastor / DeRes
                            处理有效秩、深度和残差传播问题

RankMixer ──固定置换矩阵化──► UniMixer (2026)
                              可学习局部/全局双随机 Mixing
                              └── UniMixing-Lite ──► RoleMix 等应用
```

四篇核心论文的关系可以压缩为一句话：

- **RankMixer** 证明无参 Token Mixing + Token 独立 FFN 可以高效扩大工业排序模型；
- **TokenMixer-Large** 修复原始残差语义错位和深层训练问题，并把规模推进到 7B 在线、15B 离线；
- **MixFormer** 把 RankMixer 风格的 Query Mixer 与长序列 Cross-Attention 放进同一骨架，解决 dense/sequence 分开扩展的问题；
- **UniMixer** 把固定 Mixing 写成矩阵形式，再推广为可学习的局部/全局混合，尝试统一 Attention、TokenMixer 和 FM 三类交互。

对本仓库最重要的判断是：

- v1–v4 更接近原始 RankMixer；
- v5–v10 已采用与 TokenMixer-Large 相近的 Mixing/Reverting 和双段 Per-token SwiGLU，但**不是论文完整复现**；
- 本仓库已有 MixFormer 与 UniMixer/UniMixing-Lite 代码路线；
- 本地已有结果仍显示 Base 的 SENet + DCN-M + 深 MLP 最稳，不能用公开论文的工业结果替代本场景验证。

---

## 1. 检索范围、纳入标准与口径

### 1.1 三层文献范围

本文将文献分为三层，避免把名称相似但技术无关的论文混在一起。

| 层级 | 纳入标准 | 代表工作 |
|---|---|---|
| 核心家族 | 直接使用、修正或参数化 RankMixer/TokenMixer 的固定 Token Mixing 与 Token-specific FFN 范式 | RankMixer、TokenMixer-Large、UniMixer、MTmixAtt、UG-Sep、RankElastor |
| 结构后继 | 在工业排序中复用 Mixer 作为特征交互或查询增强模块，并加入长序列、异构交互、多场景或任务 Token | MixFormer、HyFormer、HeMix、RankUp、RoleMix、FA-RankMixer |
| 基础与邻近路线 | 奠定 Mixer/MetaFormer 概念，或是同一“推荐模型缩放”问题上的 Attention/FM/统一 Transformer 对照 | MLP-Mixer、MetaFormer、AutoInt、HiFormer、DHEN、Wukong、LONGER、OneTrans、DeRes |

### 1.2 检索方法

检索以原始论文页面和官方论文元数据为主，关键词包括：

- `RankMixer`、`TokenMixer`、`TokenMixer-Large`；
- `UniMixer`、`UniMixing-Lite`；
- `MixFormer recommender`、`token mixing industrial recommender`；
- 核心论文的参考文献、引用论文和作者后续工作。

优先使用 arXiv 论文正文、ACM DOI 页面和作者公开代码仓库。二手解读只用于发现线索，不作为本文技术结论的依据。

### 1.3 同名论文排除

本文中的 **MixFormer** 专指 2026 年工业推荐论文 *MixFormer: Co-Scaling Up Dense and Sequence in Industrial Recommenders*，不包括：

- 2022 年视觉跟踪论文 *MixFormer: End-to-End Tracking with Iterative Mixed Attention*；
- 2022 年视觉分类论文 *MixFormer: Mixing Features across Windows and Dimensions*。

本文中的 Mixer 也不包括 TimeMixer、MixerGAN 等面向时间序列、视觉或生成任务但与推荐排序谱系无直接关系的同名工作。

### 1.4 数字比较原则

不同论文的 `ΔAUC` 可能表示绝对 AUC 差、相对提升百分比或内部缩放后的差值；在线指标也对应不同业务、流量和基线。因此：

- 下文尽量保留论文原始记法；
- 每个数字同时注明基线和数据来源；
- **禁止依据跨论文数字直接给模型排总榜**；
- 私有工业数据上的结果应视为“作者报告”，不是独立复现结论。

---

## 2. 统一理解框架：一个 Mixer 排序模型到底由什么组成

大多数工作可以写成同一条流水线：

```text
稀疏/稠密/序列原始特征
        ↓
Embedding 与 Tokenization
        ↓
X ∈ R^(T×D)
        ↓
Token Mixer：跨 Token/子空间交换信息
        ↓
Channel Mixer：FFN / SwiGLU / MoE
        ↓
Residual + Norm，重复 L 层
        ↓
Mean / Global / Flatten / Task Token 读出
        ↓
CTR / CVR / 多目标任务头
```

其中真正决定一个方法属于哪条路线的，不是论文名称，而是下面四个问题。

### 2.1 Token 是怎么形成的

推荐输入不是天然序列。用户、Query、Item、上下文、统计量和行为序列具有不同字段数、维度和语义。常见 Tokenization 包括：

- **人工语义分组**：RankMixer、HyFormer、RoleMix 和本仓库 v3 以后采用；
- **一次投影后均匀切分 AutoSplit**：OneTrans、HeMix 的非序列部分采用或讨论；
- **可学习分组/选择**：MTmixAtt 的 AutoToken；
- **随机置换后切分**：RankUp 用于提升 Token 独立性与有效秩；
- **序列查询压缩**：MixFormer、HyFormer、HeMix、RoleMix 将长序列压成少量语义查询 Token。

Tokenization 不是普通预处理。固定 Mixing 本身不判断两个字段是否相关，因此 Token 边界直接定义了后续的归纳偏置。

### 2.2 Token 之间怎么 Mixing

设输入为 $X\in\mathbb{R}^{T\times D}$。不同方法的核心差异可写成混合矩阵的来源：

| 方法 | 全局混合权重 | 是否依赖当前样本 | 主要特点 |
|---|---|---:|---|
| RankMixer | 固定置换 $P$ | 否 | 无参数、无乘法 Mixing，最利于大矩阵 FFN 与 GPU 吞吐 |
| MTmixAtt | 每头可学习矩阵 $W_h$ | 否 | 比固定置换灵活，仍不计算 QK 相似度 |
| UniMixer | 学习到的局部矩阵 $W_B$ 与全局矩阵 $W_G$ | 否 | 可学习、近双随机、可解除 $H=T$ 约束 |
| Self-Attention | $\operatorname{softmax}(QK^\top)$ | 是 | 内容自适应，但异构字段上计算和建模未必最合适 |
| FM/Wukong | 由 $XX^\top$ 或其低秩形式产生 | 是 | 显式乘性交互，结构归纳偏置强 |

### 2.3 Token 内部用什么非线性

原始 Transformer 在所有位置共享 FFN；RankMixer 系列通常给每个 Token 独立参数：


\[
\operatorname{PFFN}(X)_t=f_t(X_t),\qquad t=1,\ldots,T.
\]

它带来两个效果：

- 保留用户、Item、上下文等不同语义位置的异质性；
- 参数量约随 $TD^2$ 增长，容易形成适合 GPU 的大 GEMM。

后续工作把普通 FFN 升级为 Per-token SwiGLU、Token 专属专家或 Sparse-Pertoken MoE。

### 2.4 深层信息怎么传播

Residual/Norm 并非实现细节，而是该家族持续演进的主线：

- RankMixer 使用两次 Add & LayerNorm，但固定 Mixing 后的行语义与原 Token 不完全对齐；
- TokenMixer-Large 引入 Reverting，把残差重新对齐到原 Token 空间，并采用 Pre-RMSNorm；
- UniMixer 使用 SiameseNorm 双流连接；
- DeRes 进一步把恒等残差和跨层可学习残差拆开；
- RankElastor 从有效秩角度分析 Mixing 扩张与 FFN 收缩的拉锯。

---

## 3. 核心论文一：RankMixer

**论文**：[RankMixer: Scaling Up Ranking Models in Industrial Recommenders](https://arxiv.org/abs/2507.15551)<br>
**作者/机构**：Jie Zhu 等，ByteDance<br>
**版本与发表**：arXiv:2507.15551；CIKM 2025；DOI [10.1145/3746252.3761507](https://doi.org/10.1145/3746252.3761507)<br>
**核心贡献**：为工业排序设计 GPU 友好的统一交互骨架，并展示约十亿 dense 参数的低延迟部署。

### 3.1 出发点

传统 DLRM 往往拼接 DCN、FM、Attention、DIN、MLP 等模块。它们可以有效，但算子碎片多、矩阵规模小、GPU MFU 低；简单加宽 MLP 或增加人工 Cross 也容易收益递减。RankMixer 的目标不是追求更复杂的单次交互，而是构造一个可重复堆叠、可预测扩展且硬件效率高的基本块。

### 3.2 语义 Tokenization

输入字段先按业务语义形成 $T$ 组，每组由独立投影映射到同一宽度 $D$：

\[
x_t=\operatorname{MLP}_t(\operatorname{Concat}(g_t))\in\mathbb{R}^{D},
\quad X=[x_1;\ldots;x_T]\in\mathbb{R}^{T\times D}.
\]

语义分组的意义在于：每一行都有稳定身份，后续 Per-token FFN 才能学习“这个位置代表什么”。论文也讨论自动切分，但主线设计强调 coherent semantic groups。

### 3.3 无参 Multi-head Token Mixing

把每个 Token 的 $D$ 维切成 $H$ 个 head，$d_h=D/H$：

\[
X:[T,D]\rightarrow[T,H,d_h]
\xrightarrow{\text{transpose}}[H,T,d_h]
\rightarrow[H,Td_h].
\]

当 $H=T$ 时，输出仍为 $[T,D]$，可以进入固定形状的残差和 Per-token FFN。这个操作没有 Q/K/V、没有 $T^2$ 注意力权重，本质上是对 $\operatorname{vec}(X)$ 施加固定置换矩阵 $P$：

\[
\operatorname{vec}(\operatorname{Mix}(X))=P\operatorname{vec}(X).
\]

每个混合后 Token 都拿到所有原 Token 的同一 head 子空间，从而用非常低的算子成本实现全局信息交换。

### 3.4 Per-token FFN 与 Block

论文的 dense block 使用 Token 独立 FFN，并在 Mixing、FFN 后分别做残差与 LayerNorm。若 FFN expansion ratio 为 $k$，忽略 bias/norm 后：

\[
\text{Params}\approx2kLTD^2,
\qquad
\text{Forward FLOPs}\approx4kLTD^2.
\]

这说明 RankMixer 的扩展主要由 $T,D,L$ 控制。其参数和计算集中在规则的大矩阵乘上，固定 Mixing 自身几乎不增加参数。

### 3.5 Sparse MoE 扩展

论文还给出 Sparse MoE 版本：路由器选择少量专家，训练阶段用动态路由缓解专家负载和训练不足，并采用 dense-train/sparse-inference 思路提高激活参数 ROI。这个方案后来被 TokenMixer-Large 的 Sparse-Pertoken MoE 重新设计，因为训练/推理稀疏口径不一致会带来效率与一致性问题。

### 3.6 规模与结果

论文在日均万亿级样本、300+ 特征、连续两周的 Douyin 工业数据上报告结果。代表配置包括：

- RankMixer-100M：$D=768,T=16,L=2$；
- RankMixer-1B：$D=1536,T=32,L=2$；
- 全流量部署模型约 1.1B dense 参数。

相对论文 DLRM 基线，作者报告：

| 模型 | Finish AUC | Finish UAUC | Skip AUC | Skip UAUC |
|---|---:|---:|---:|---:|
| RankMixer-100M | +0.64% | +0.72% | +0.86% | +1.33% |
| RankMixer-1B | +0.95% | +1.22% | +1.25% | +1.82% |

关键消融包括：

- 去掉 Token Mixing：AUC 下降约 0.50%；
- Per-token FFN 改成共享 FFN：下降约 0.31%；
- 用 Self-Attention 替换 Mixing：效果略降约 0.03%，同时参数约增 16%、FLOPs 约增 71.8%；
- 去掉 skip connection 或 LayerNorm 也有小幅下降。

系统侧，论文将 dense 参数从约 15.8M 扩到 1.1B，报告 MFU 从约 4.47% 提升到 44.57%，在线延迟约从 14.5 ms 到 14.3 ms。在线 A/B 中，作者报告 Douyin 活跃天数 +0.2908%、使用时长 +1.0836%；广告场景 AUC +0.73%、ADVV +3.90%。

### 3.7 关键局限

后续文献集中指出四个问题：

1. Mixing 改变行语义后直接与原输入残差相加，存在语义错位；
2. $H=T$ 是为了保持形状而引入的刚性约束；
3. 固定置换不随数据学习，表达力上限可能导致有效秩增长不足；
4. 原始论文主要验证宽度/参数扩展，深度扩展和稀疏专家训练仍不充分。

---

## 4. 核心论文二：TokenMixer-Large

**论文**：[TokenMixer-Large: Scaling Up Large Ranking Models in Industrial Recommenders](https://arxiv.org/abs/2602.06563)<br>
**作者/机构**：Yuchen Jiang、Jie Zhu 等，ByteDance<br>
**版本**：arXiv:2602.06563v2，2026<br>
**核心贡献**：系统修复 RankMixer 的残差、深度、MoE 和系统效率问题，将 TokenMixer 扩至 15B 离线和 7B 在线。

### 4.1 对 RankMixer 的问题诊断

TokenMixer-Large（下文简称 TML）明确把 RankMixer 中的基本块称为 TokenMixer，并归纳四类瓶颈：

- 残差路径没有对齐原 Token 语义；
- 深层模型梯度和早层表示利用不足；
- 原始算子较碎，未充分利用训练/推理硬件；
- MoE 训练与推理稀疏模式不一致，规模探索停留在约 1B。

### 4.2 Mixing → Reverting 的双空间块

TML 的关键变化是先在 mixed space 做 Token 独立非线性，再通过逆置换回到 original-token space：

\[
M=\operatorname{Mix}(X),
\]

\[
\widehat M=M+\operatorname{pSwiGLU}(\operatorname{RMSNorm}(M)),
\]

\[
R=\operatorname{Revert}(\widehat M),
\]

\[
X'=X+\operatorname{pSwiGLU}(\operatorname{RMSNorm}(R)).
\]

Reverting 是 Mixing 的精确逆变换，恢复 $T\times D$ 的原 Token 坐标。这样长残差 $X\rightarrow X'$ 的两端具有相同语义，而 mixed-space FFN 仍能处理跨 Token 拼接后的子空间。

论文还允许 Mixing 的 head 数与 Token 数不同，再由 Reverting 恢复形状，因此从设计上解除原始 $H=T$ 的强约束。

### 4.3 从普通 FFN 到 Per-token SwiGLU

TML 使用 Token 独立 SwiGLU：

\[
f_t(x)=W_{down,t}
\left[\operatorname{SiLU}(xW_{gate,t})\odot(xW_{up,t})\right].
\]

论文建议 down projection 使用较小初始化尺度（报告值 0.01），使新增残差分支在训练早期接近恒等映射，提高大模型稳定性。

### 4.4 深度扩展：跨层残差与辅助损失

仅堆叠更多 block 并不自动带来收益。TML 每隔约 2–3 个 block 加入 interval/inter-layer residual，并在中间层接辅助预测头。作用分别是：

- 提供跨越多个 block 的短梯度路径；
- 让低层直接接收任务监督；
- 减少深层中原始特征和早期交互被稀释。

论文消融中，去掉标准残差约损失 0.15%，去掉跨层残差与辅助损失约损失 0.04%。

### 4.5 Sparse-Pertoken MoE

Sparse-Pertoken MoE（SP-MoE）不是把一个共享 MoE 放到所有 Token 上，而是保留 Token 身份：

- 每个 Token 有自己的稀疏专家集合；
- 路由采用 top-k softmax；
- 每个 Token 有始终激活的 shared expert；
- 使用 gate scaling 调节路由输出；
- 训练和推理都保持稀疏，避免 dense-train/sparse-serve 不一致。

论文报告 1:2 的激活比例在其在线场景具有较好 ROI。4B SP-MoE 模型只激活约 2.3B 参数，并达到与 4B dense 版本相近的作者报告增益。

### 4.6 系统共同设计

TML 把架构和系统优化写在同一条扩展路径中：

- 把大量 Per-token 小算子组织成 grouped operators；
- FP8 E4M3 推理报告约 1.7× 加速；
- Token Parallel 将通信模式由近似 $4L$ 降到 $2L+1$；
- 4-way Token Parallel 报告吞吐 +29.2%，再结合通信重叠可达 +96.6%。

这部分很关键：该家族的“可扩展”不仅来自 FLOPs 公式，也来自算子融合、精度格式和并行策略。

### 4.7 规模与结果

论文覆盖广告、电商、直播等多个 ByteDance 私有数据集，并报告：

- 离线最大规模：广告 15B、电商 7B、直播 4B；
- 在线部署规模：7B、4B、2B；
- 电商实验中，RankMixer 500M 的作者报告 ΔAUC 为 +0.84%，TML 500M 为 +0.94%，4B 为 +1.14%，7B 为 +1.20%；
- 在线相对 RankMixer 基线，电商订单 +1.66%、GMV +2.98%，广告 ADSS +2.0%，直播收入/付费约 +1.4%。

消融中，去掉 Global Token 约 -0.02%，去掉 Mixing/Reverting 约 -0.27%，Per-token SwiGLU 改共享 SwiGLU 约 -0.21%，改普通 PFFN 约 -0.10%。

论文还显示“模型变大必须配更多数据”：同一模型在 30M、60M、90M 等不同训练数据规模上的收益并不等价。参数 Scaling Law 不能脱离数据量讨论。

### 4.8 关键局限

- 主要证据来自私有工业数据与内部系统，外部难以完全复现；
- Per-token SwiGLU/SP-MoE 参数和激活巨大，需要 grouped kernel 才能获得论文级效率；
- 固定 Mixing 的表达上限仍存在，后续 UniMixer、MTmixAtt 和 RankElastor 正面处理这一问题；
- 论文证明的是一套“模型 + 训练 + 系统”组合，单独复制网络结构不等于复制其吞吐和收益。

---

## 5. 核心论文三：MixFormer

**论文**：[MixFormer: Co-Scaling Up Dense and Sequence in Industrial Recommenders](https://arxiv.org/abs/2602.14110)<br>
**作者/机构**：Xu Huang 等，ByteDance<br>
**版本与发表**：arXiv:2602.14110v2；KDD 2026，DOI [10.1145/3770855.3818447](https://doi.org/10.1145/3770855.3818447)<br>
**核心贡献**：在一个 decoder-style block 中联合扩展非序列特征交互和长行为序列建模。

### 5.1 为什么它属于扩展家族，但不是简单“RankMixer v3”

RankMixer/TML 主要解决 dense heterogeneous feature interaction；LONGER/STCA 等主要解决长行为序列。常见生产结构是先压缩序列，再把少量序列摘要 Token 送入 RankMixer。这种两阶段结构存在：

- 序列与非序列只在较晚层交互；
- dense 和 sequence 各自占用预算，难以统一扩展；
- 用户序列侧计算在多个候选间重复或融合不足。

MixFormer 保留 RankMixer 风格的无参 HeadMixing，但把它定位为 **Query Mixer**，再让增强后的非序列 Query 去 Cross-Attend 行为序列。

### 5.2 一个 MixFormer Block

给定 $N$ 个非序列 Token $X\in\mathbb{R}^{N\times D}$ 和行为序列 $S$：

1. **Query Mixer**：

\[
P=\operatorname{HeadMixing}(\operatorname{Norm}(X))+X,
\]

\[
q_i=\operatorname{SwiGLUFFN}_i(\operatorname{Norm}(p_i))+p_i.
\]

2. **Sequence Cross-Attention**：混合后的高阶 Query 作为 Q，行为序列表示作为 K/V；
3. **Output Fusion**：再用一层 Per-head/Per-token SwiGLU 融合序列信息和非序列信息。

这里的建模判断很明确：Self-Attention 在异构字段 Token 上未必合适，但 Query 到同质行为序列的 Cross-Attention 仍然有效。因此 MixFormer 不是“完全去 Attention”，而是把不同交互算子放到适合的位置。

### 5.3 UI-MixFormer：用户—物品解耦

在线一次请求通常有同一用户对应多个候选 Item。UI-MixFormer 将 Token 分成 user-side 和 item/group-side，并对 HeadMixing 加掩码：

- 用户 Token 不接收候选 Item 信号，保持可复用；
- Item Token 可以读取用户上下文；
- 结合 request-level batching 复用同一用户的序列和中间表示。

论文报告约 36% FLOPs 降低和超过 30% 的 serving speedup，同时保持主要离线指标。

### 5.4 规模与结果

论文在万亿级样本、300+ 特征、两周 Douyin 数据上比较：

| 结构 | Dense 参数 | 每 batch FLOPs | 说明 |
|---|---:|---:|---|
| STCA → RankMixer | 1,255M | 6,736G | 两阶段强基线 |
| MixFormer-medium | 1,226M | 3,503G | 更低计算且指标更高 |
| UI-MixFormer-medium | 约同级 | 2,242G | 用户/物品解耦后进一步降算力 |

MixFormer-medium 相对论文基础模型报告 Finish AUC/UAUC +1.28%/+1.60%，Skip AUC/UAUC +1.60%/+2.46%。在线相对超过 1B 参数的 STCA→RankMixer 基线：

- Douyin：活跃天数 +0.0415%，使用时长 +0.2799%；
- Douyin Lite：活跃天数 +0.0252%，使用时长 +0.4105%。

### 5.5 关键局限

- 它解决的是“dense + long sequence”联合扩展，不适合用来单独判断纯 dense Mixer 谁更强；
- UI 解耦收益依赖“一用户多候选”的请求形态；
- 主要数据和生产优化不可公开复现；
- 对没有长行为序列或序列已在上游固定压缩的任务，额外 Cross-Attention 可能得不偿失。

---

## 6. 核心论文四：UniMixer

**论文**：[UniMixer: A Unified Architecture for Scaling Laws in Recommendation Systems](https://arxiv.org/abs/2604.00590)<br>
**作者/机构**：Mingming Ha 等，Kuaishou<br>
**版本**：arXiv:2604.00590v2，2026<br>
**核心贡献**：把固定 TokenMixer 写成矩阵形式并参数化，提出可学习的局部/全局 Mixing 与轻量 UniMixing-Lite，同时统一解释 Attention、TokenMixer 和 FM。

### 6.1 从固定置换到可学习 Mixing

把 $X\in\mathbb{R}^{T\times D}$ 展平为长度 $L=TD$ 的向量。RankMixer 的 reshape/transpose 等价于固定置换矩阵：

\[
y=P\operatorname{vec}(X),\qquad P\in\{0,1\}^{L\times L}.
\]

直接学习完整 $L\times L$ 矩阵的参数与计算均不可接受。UniMixer 将长度 $L$ 划成 $G=L/B$ 个、每个宽度为 $B$ 的块：

- 每个块使用局部矩阵 $W_B^{(g)}\in\mathbb{R}^{B\times B}$；
- 同一块内位置再通过全局矩阵 $W_G\in\mathbb{R}^{G\times G}$ 跨块交互。

计算复杂度约为：

\[
\mathcal O(LB+L^2/B),
\]

避免直接构造 $L^2$ 全连接 Mixing，并解除 RankMixer 的 $H=T$ 限制。

### 6.2 对称、近双随机与温度退火

论文对可学习矩阵施加三个归纳偏置：

- 通过 $(W+W^\top)/2$ 形成对称性；
- 指数化确保非负；
- 用 Sinkhorn–Knopp 交替归一化，使行和、列和近似为 1。

训练温度从约 1.0 逐步降至 0.05：高温时 Mixing 较平滑，有利于探索；低温时更尖锐、接近置换/稀疏结构。论文消融显示，去掉温度机制、对称性或温度 warmup 都会降低效果。

### 6.3 UniMixing-Lite

完整版仍需保存 $W_G$ 和大量 $W_B^{(g)}$。Lite 版本进一步分解：

\[
W_G=UV^\top,
\]

\[
W_B^{(g)}=\sum_{k=1}^{K}\alpha_{gk}B_k.
\]

即全局矩阵用低秩因子表示，局部矩阵由少量共享 basis 加块专属系数组合。它保留全局可学习性和局部差异，同时显著减少参数。

### 6.4 统一三类交互

UniMixer 把主流扩展架构都解释为“局部变换 + 全局混合”：

| 架构 | 局部变换 | 全局混合 |
|---|---|---|
| Self/heterogeneous Attention | $XW_V$ | $\operatorname{softmax}(QK^\top)$，样本依赖 |
| TokenMixer | $X$ 或固定子空间 | 固定置换 $P$ |
| FM | 投影后的局部特征 | $XX^\top$ 型乘性交互 |
| UniMixer | 可学习块内变换 | 可学习、结构化的 $W_G$ |

这不是说三种模型完全等价，而是提供一个比较其“局部表示”和“全局路由”来源的共同坐标系。

### 6.5 FFN、归一化与稀疏化

UniMixer 完整模型组合：

- Pertoken SwiGLU；
- SiameseNorm 双残差流；
- Sparse-Pertoken MoE；
- RMSNorm 与温度退火。

SiameseNorm 的目标是同时保留 PreNorm 的梯度稳定性和 PostNorm 的表征校准。论文的深度实验中，RankMixer 从 2 层加到 4 层出现下降，而 UniMixer-Lite 从 2、4 到 8 层仍可提升，说明可学习 Mixing 与残差结构共同改善深度扩展。

### 6.6 结果与 Scaling Law

论文在快手广告留存场景、超过 0.7B 用户样本的私有数据上报告约 100M 规模对比：

| 模型 | Dense 参数 | AUC |
|---|---:|---:|
| RankMixer | 135.5M | 0.749329 |
| TokenMixer-Large | 103.3M | 0.748410 |
| UniMixer | 67.5M | 0.749770 |
| UniMixer | 101.5M | 0.750238 |
| UniMixer-Lite | 42.4M | 0.751121 |
| UniMixer-Lite | 76.2M | 0.751401 |
| UniMixer-Lite，4 blocks | 84.5M | 0.752718 |

这些点的 FLOPs、深度和配置并不完全相同，适合说明论文内部趋势，不适合视为统一排行榜。

论文拟合的参数 Scaling 指数约为：RankMixer 0.1160、UniMixer 0.1320、UniMixer-Lite 0.1419；FLOPs Scaling 指数约为 0.1166、0.1257、0.1353。指数仅对该数据、损失定义和拟合区间有效。

在线实验作者报告 Kuaishou Ads 的 1–30 日累计广告消耗（CAD）平均提升超过 15%。该指标与 CTR/CVR AUC 不同，不能和其他论文的 GMV、时长直接比较。

### 6.7 关键局限

- Sinkhorn、对称化和显式 Mixing 矩阵带来额外计算与数值稳定要求；
- 低秩全局矩阵可能限制表达力，rank、block size、basis 数需随 $TD$ 调节；
- 论文中的最佳配置是整体组合，不能把收益只归因于“可学习矩阵”；
- 仍以私有数据为主，外部可比实验有限。

---

## 7. 直接后继与重要扩展

### 7.1 时间线总表

| 时间 | 工作 | 与 RankMixer/TokenMixer 的关系 | 主要新增能力 | 公开状态 |
|---|---|---|---|---|
| 2025-07 | [RankMixer](https://arxiv.org/abs/2507.15551) | 起点 | 固定无参 Mixing、Per-token FFN、语义 Token、MoE | CIKM 2025；工业私有数据 |
| 2025-10 | [MTmixAtt](https://arxiv.org/abs/2510.15286) | 直接改造固定 Mixing | AutoToken、逐头可学习 Mixing、多场景 MoE、MLoRA | arXiv；工业私有数据 |
| 2025-10 | [OneTrans](https://arxiv.org/abs/2510.26104) | 邻近统一路线 | 统一 S/NS Token、混合参数化 Transformer、金字塔裁剪、KV Cache | arXiv；工业私有数据 |
| 2026-01 | [HyFormer](https://arxiv.org/abs/2601.12681) | 使用 RankMixer 风格 Query Boosting | Query Decoding 与 Query Boosting 交替，长序列和异构特征逐层交互 | arXiv；工业私有数据 |
| 2026-02 | [TokenMixer-Large](https://arxiv.org/abs/2602.06563) | 官方核心后继 | Reverting、深层训练、SP-MoE、FP8/Token Parallel | arXiv；工业私有数据 |
| 2026-02 | [UG-Sep](https://arxiv.org/abs/2602.10455) | TML serving 扩展 | 用户/候选 Token 分离、计算复用、信息补偿、W8A16 | arXiv；多业务线上实验 |
| 2026-02 | [HeMix](https://arxiv.org/abs/2602.09387) | 异构 Mixer 后继 | 混合查询兴趣提取、HeteroMixer、HeteroFFN | arXiv；工业私有数据 |
| 2026-02 | [MixFormer](https://arxiv.org/abs/2602.14110) | Mixer + 长序列统一 | Query Mixer、Cross-Attention、UI 解耦 | KDD 2026；工业私有数据 |
| 2026-04 | [UniMixer](https://arxiv.org/abs/2604.00590) | 固定 Mixing 参数化 | 局部/全局双随机 Mixing、Lite、统一理论 | arXiv；工业私有数据 |
| 2026-04 | [RankUp](https://arxiv.org/abs/2604.17878) | MetaFormer/RankMixer 表示增强 | 随机置换切分、多 Embedding、Global/Cross/Task Token | arXiv；腾讯线上实验 |
| 2026-05 | [RankElastor](https://arxiv.org/abs/2605.23191) | RankMixer 谱分析与直接修正 | 参数化 Full Mixing、GLU-PFFN、有效秩理论 | KDD 2026；有公开代码 |
| 2026-06 | [DeRes](https://arxiv.org/abs/2606.07980) | 可插入 TML/UniMixer 的残差扩展 | 恒等 + 跨块注意力双路径、SiLU Pointwise AttnRes | arXiv；含公开数据实验 |
| 2026-07 | [FA-RankMixer](https://arxiv.org/abs/2607.15590) | 直接应用改造 | 多域 DIN、语义 Token、浅/深双流、双线性融合 | KDD Cup 方案；有代码 |
| 2026-07 | [RoleMix](https://arxiv.org/abs/2607.22700) | UniMixing-Lite 应用 | 角色保持 Token、层次窗口序列压缩、统一 PCVR 交互 | KDD Cup 方案 |

### 7.2 MTmixAtt：把固定置换改成可学习逐头 Mixing

**论文**：[MTmixAtt: Integrating Mixture-of-Experts with Multi-Mix Attention for Large-Scale Recommendation](https://arxiv.org/abs/2510.15286)

它对 RankMixer 做了两项直接修改：

1. **AutoToken** 用可学习特征选择矩阵做 top-k 分组，减少人工语义分桶；
2. 每个 head 使用可学习 $W_h\in\mathbb{R}^{T\times T}$，替代纯 reshape/transpose 的固定 Mixing。

Block 同时包含共享 dense experts、场景专属 sparse experts 和多任务低秩适配头。论文在 Meituan TRec 数据上报告，约 14M 的 RankMixer CTR AUC 为 0.7781，约 15M 的 MTmixAtt 为 0.7792，1B 版本为 0.7811；线上首页 Payment PV +3.62%、Actual Payment GTV +2.54%。

值得注意的工程结论：可学习 Mixing 的初始化很重要；全 1 初始化可能造成 rank-1 式退化，论文实验中正交初始化最好。其 PostNorm_R 在本场景优于 PreNorm，和 TML 的 PreNorm 结论不同，再次说明 Norm 不能跨数据直接照搬。

### 7.3 HyFormer：Query Decoding 与 Query Boosting 交替

**论文**：[HyFormer: Revisiting the Roles of Sequence Modeling and Feature Interaction in CTR Prediction](https://arxiv.org/abs/2601.12681)

HyFormer 将传统“LONGER 压缩序列 → RankMixer 做特征交互”改为逐层交替：

- Query Decoding：由非序列特征和序列池化信息生成 Global Query，对各行为序列的 K/V 做 Cross-Attention；
- Query Boosting：把解码 Query 与非序列 Token 拼接，再执行与 RankMixer 相同的 MLP-Mixer 式固定 Token Mixing 和 Per-token FFN；
- 下一层用增强后的 Query 再读取序列。

在 70 天、30 亿样本的 Douyin Search 数据上，论文报告 HyFormer 418M/3.9T FLOPs 的 AUC 为 0.6489，相对 LONGER+RankMixer 的 386M/3.5T、AUC 0.6478 提升约 0.17%。这是“逐层统一交互”的证据，但仍是单一私有数据和作者实现。

### 7.4 HeMix：异构序列 Tokenization + HeteroMixer

**论文**：[Query-Mixed Interest Extraction and Heterogeneous Interaction: A Scalable CTR Model for Industrial Recommender Systems](https://arxiv.org/abs/2602.09387)

HeMix 的 Query-Mixed Interest Extraction 同时使用：

- 来自非序列特征的动态 Query，建模候选/上下文相关兴趣；
- 可学习固定 Query，建模候选无关的稳定兴趣；
- 全局长序列与当日实时序列两路输入。

HeteroMixer 将交互分成多 Token 融合、异构 mixed-token 交互、按组对齐重建，并使用 HeteroFFN。论文把模型从约 100M 扩到 1.5B，在相同 Tokenization 下，HeteroMixer 的 CTR/CVR AUC 高于 RankMixer block 和 Transformer block。

在线数字需要区分基线：正文表格报告相对 DLRM 为 GMV +3.61%、PV_CTR +2.78%、UV_CVR +2.12%；相对 RankMixer 为约 +0.61%、+2.32%、+0.81%。摘要和贡献段看似给出不同 GMV，实质上对应不同基线，引用时必须写清楚。

### 7.5 UG-Sep：让用户侧 Token 真正可复用

**论文**：[Compute Only Once: UG-Separated TokenMixer for Efficient Large Recommendation Models](https://arxiv.org/abs/2602.10455)

固定 Token Mixing 会在每层把 user-side 与 item/group-side 信息完全缠在一起，导致同一用户面对多个候选时不能缓存中间结果。UG-Sep：

- 将 Token 分为 U/G 两类；
- 对 Mixing 和 residual 施加方向性掩码，使一部分 U Token 始终保持纯用户信息；
- 允许 G Token 读取 U 信息，保持打分所需的用户→候选条件化；
- 用 Information Compensation 恢复被掩蔽的交互；
- 用 W8A16 缓解分离后暴露的权重带宽瓶颈。

论文在 Douyin、Hongguo、Chuanshanjia、Qianchuan 四个业务报告 11.5%–22.0% serving latency 降低，AUC 变化大致在 0.002% 到 -0.024% 区间；有用户级聚合训练数据时，训练吞吐最高报告 +14.8%。它是系统扩展，不是单纯追求更高 AUC 的新 backbone。

### 7.6 RankUp 与 RankElastor：从“规模”转向“表示有效秩”

**RankUp**：[RankUp: Towards High-rank Representations for Large Scale Advertising Recommender Systems](https://arxiv.org/abs/2604.17878)

RankUp 使用随机置换切分、多套 Embedding、Global Token、预训练 user/item cross token 和 task-specific token，目标是避免不同 Token 过度相关、任务间相互污染。作者报告在 Weixin Video Accounts、Official Accounts、Moments 的 GMV 分别 +3.41%、+4.81%、+2.12%。

需要注意，RankUp 所述 RankMixer 有效秩“阻尼振荡”不是原 RankMixer 论文的结论，而是建立在同期谱分析工作之上。它应被视为 2026 年提出的新诊断假设，而不是已经被多组独立工业实验共同确认的事实。

**RankElastor**：[Expand More, Shrink Less: Shaping Effective-Rank Dynamics for Dense Scaling in Recommendation](https://arxiv.org/abs/2605.23191)，代码：[GitHub](https://github.com/vasile-paskardlgm/RankElastor)，KDD 2026。

RankElastor 对 RankMixer 的 block-transpose 和 PFFN 做谱分析，观察到：

- 固定 Mixing 只能有限扩张 effective rank；
- GELU 型 PFFN 往往收缩 effective rank；
- 两者交替造成随深度衰减的振荡。

它用完整可学习 $W\in\mathbb{R}^{TD\times TD}$ 做 Parameterized Full Mixing，并用 GLU-improved PFFN 和可学习残差增强谱稳定性。代价是 Mixing 从 RankMixer 的近似 $\mathcal O(TD)$、零参数，升到 $\mathcal O(T^2D^2)$ 参数/计算，因此它优化的是表达力和可复现研究，不是同一硬件效率目标。

在 Criteo/Avazu 的 10 次重复实验中，论文报告 Criteo AUC 由 RankMixer 0.81375 提升到 0.81482，Avazu 由 0.79270 提升到 0.79323；另在 KuaiVideo/TaobaoAd 上也优于 RankMixer。由于有公开代码和公开数据协议，它是当前家族中较适合外部复核的一篇。

### 7.7 DeRes：深度扩展首先可能是残差问题

**论文**：[DeRes: Decoupling Residual Stability and Adaptivity for Scalable CTR Prediction](https://arxiv.org/abs/2606.07980)

DeRes 不是新的 Token Mixer，而是可插到 OneTrans、TML、UniMixer 等 backbone 上的双路径残差：

- Identity path 固定保留原信号和梯度通道；
- Block Attention Residual 从此前多个 block 的压缩状态中选择信息；
- vector-wise gate 合并两路；
- Pointwise AttnRes 用 SiLU 代替 softmax，使多个历史 block 可同时激活，也允许负权重抑制过时信号。

论文在 3.31 亿工业交互、Criteo 和 Avazu 上报告最高约 +0.32% AUC，额外 FLOPs 小于 5%；其 8 层模型在作者实验中可达到 16 层 OneTrans 的水平。它提示：当深度不再带来收益时，不应只增大 FFN，需检查跨层信息流。

### 7.8 RoleMix 与 FA-RankMixer：竞赛场景中的可复现应用

**RoleMix**：[RoleMix: Unifying Sequential and Non-Sequential Features via Semantic Tokenization for Post-Click Conversion Rate Prediction](https://arxiv.org/abs/2607.22700)

RoleMix 把用户、Item、pairwise、dense、context、cross 等角色编码为显式语义 Token；长行为域用两阶段层次窗口注意力压成 item-aware/context-aware query Token，再用 UniMixing-Lite 统一交互。KDD Cup 2026 Tencent UniRec Challenge 中报告官方 online leaderboard AUC 83.648%，比官方基线高 1.953 个百分点；这里的“online”指竞赛在线榜评测，不是生产 A/B。消融显示语义 Tokenization 是最大单项增益。

**FA-RankMixer**：[Field-Aware RankMixer with Dual-Stream Bilinear Fusion for the Tencent UNI-REC Challenge](https://arxiv.org/abs/2607.15590)，代码：[GitHub](https://github.com/PixelCookie-zyf/TAAC-2026-SeRankMixer)

它用多域 target-aware DIN 提取兴趣，形成按字段/行为域组织的语义 Token，经 RankMixer block 交互；同时保留浅 MLP 流，再用 group-wise bilinear 模块融合深浅表示。最终在官方榜单排名第 9。两篇竞赛论文的价值主要是展示 Mixer 如何适配公开任务和更具体的 PCVR 特征，而非证明普适工业 Scaling Law。

---

## 8. 基础文献与邻近扩展路线

这些工作不是 RankMixer 的直接版本，但理解它们可以避免把所有提升都归结为 Token Mixing。

| 工作 | 关键思想 | 与 Mixer 家族的关系 |
|---|---|---|
| [MLP-Mixer](https://arxiv.org/abs/2105.01601) | 交替进行跨 patch 的 Token-mixing MLP 和逐位置 Channel-mixing MLP | 提供“两轴分解 Mixing”的直接概念来源；原任务是视觉，不是推荐 |
| [MetaFormer](https://arxiv.org/abs/2111.11418) | 把 Transformer 抽象为 Token Mixer + Channel MLP + Residual/Norm | RankMixer/TML/UniMixer 都可放入此骨架理解 |
| [AutoInt](https://arxiv.org/abs/1810.11921) | 多头自注意力显式学习高阶字段交互 | Attention 型推荐特征交互基础对照 |
| [HiFormer](https://arxiv.org/abs/2311.05884) | 异构自注意力、低秩近似和剪枝 | 说明标准共享 Attention 不适合异构字段，并提供工业低延迟对照 |
| [DHEN](https://arxiv.org/abs/2203.11014) | 分层堆叠多种异构交互模块 | 同样追求深度/规模，但用 ensemble 而非统一 Mixer |
| [Wukong](https://arxiv.org/abs/2403.02545) | 堆叠 FM Block + Linear Compression，显式产生任意阶交互 | FM 型 Scaling Law 代表，是 UniMixer 统一框架中的另一极 |
| [LONGER](https://arxiv.org/abs/2505.04421) | Global Token、Token Merge、混合注意力、KV Cache | 长序列 Scaling 代表；MixFormer/HyFormer 的序列侧基础 |
| [OneTrans](https://arxiv.org/abs/2510.26104) | S/NS Token 放入同一 causal Transformer；序列共享参数、NS Token 独立参数 | 与 MixFormer/HyFormer 并行的“统一 backbone”路线 |

### 8.1 MLP-Mixer 与 RankMixer 的关键区别

MLP-Mixer 的 Token-mixing 本身是可学习 MLP，并通常在所有 channel 间共享；RankMixer 的 Mixing 是固定置换，主要学习能力放在 Token 独立 FFN。二者共同点是把“跨位置”和“位置内通道”解耦，差别是参数放在哪里、Token 是否异构、是否以工业 GPU 友好为第一目标。

### 8.2 Attention 路线为何没有被彻底淘汰

RankMixer 论文说明 Self-Attention 直接用于异构 dense Token 的 ROI 不高，但 MixFormer、HyFormer、HeMix 仍大量使用 Cross-Attention。原因在于：

- dense 字段 Token 固定且异构，内容依赖的全量两两注意力可能浪费；
- 行为序列位置较同质，Query→Sequence 的内容检索恰好是 Attention 的强项；
- 最优结构往往是“固定/可学习 Mixer 处理异构 dense，Attention 处理序列检索”，而不是二选一。

### 8.3 Wukong/FM 路线为何仍重要

Token Mixing 擅长全局重排和大容量非线性映射，但没有显式乘法 Cross。Wukong 通过 $XX^\top$ 及低秩优化构造明确的高阶交互；本仓库 Base 的 DCN-M 也属于显式 Cross 路线。RankMixer 与 Wukong、DCN 的关系更接近互补而非替代，这与本仓库 v8/v9 重新引入显式交叉的方向一致。

---

## 9. 核心方法横向对照

### 9.1 架构矩阵

| 维度 | RankMixer | TokenMixer-Large | MixFormer | UniMixer / Lite |
|---|---|---|---|---|
| 主要对象 | 异构 dense 特征 | 异构 dense 特征的大规模深层扩展 | dense 特征 + 长行为序列 | 通用推荐特征 Mixing |
| Tokenization | 语义分组为主 | 语义组 + Global Token | 非序列 Token + 序列表示 | 语义 Token；论文配置可调 |
| 跨 Token 操作 | 固定 reshape/transpose | Mixing + exact Reverting | Query HeadMixing + Sequence Cross-Attention | 可学习局部 $W_B$ + 全局 $W_G$ |
| 样本依赖 Mixing | 否 | 否 | HeadMixing 否；Cross-Attention 是 | 否 |
| $H=T$ 约束 | 是 | 设计上解除 | Query Mixer 通常保持等形 | 无 |
| Channel 模块 | Per-token FFN | 两段 Per-token SwiGLU / SP-MoE | Per-head SwiGLU + sequence FFN | Pertoken SwiGLU / Sparse-Pertoken MoE |
| Norm/Residual | Post-LN 型两次 Add&Norm | Pre-RMSNorm，原空间长残差，跨层残差 | PreNorm 式 Query/Attention/Fusion 残差 | RMSNorm + SiameseNorm |
| Global Token | 非核心必需项 | 是 | Query 本身承担全局接口 | 可配，应用论文常使用 |
| 长序列 | 通常先外部压缩 | 仍以 dense 主干为主 | Block 内原生 Cross-Attention | 核心论文非专门长序列；RoleMix 扩展 |
| 稀疏专家 | 动态 MoE 扩展 | Sparse-Pertoken MoE | 论文主线不是 MoE | 完整模型支持 Sparse-Pertoken MoE |
| 在线复用 | 原始结构困难 | 可结合 UG-Sep | UI-MixFormer 原生解耦 | 需额外掩码/结构设计 |
| 最大作者报告规模 | 约 1.1B 在线 | 15B 离线、7B 在线 | 约十亿级 | 论文重点是同预算 ROI，不以最大 B 数为主 |

### 9.2 关键设计分歧

#### 语义分组还是自动切分

- RankMixer、HyFormer、RoleMix 和本仓库 v3 支持语义分组；
- OneTrans 在其数据上 AutoSplit 优于人工 Group-wise Tokenizer；
- RankUp 的随机置换切分提升有效秩；
- MTmixAtt 尝试让模型学习分组。

结论不是“语义分组永远最好”，而是 Tokenization 必须作为核心变量做等参数消融。字段顺序、分桶规则和投影方式不同，结论可以反转。

#### 固定还是可学习 Mixing

- 固定 Mixing：零参数、确定性、硬件效率最好，但表达受限；
- 逐头小矩阵：适度增加参数，兼顾灵活性；
- UniMixing-Lite：低秩全局 + basis 局部，适合受控增加表达；
- Full Mixing：表达最强但复杂度最高，更适合小 $TD$ 或研究验证。

#### PreNorm、PostNorm 还是 SiameseNorm

- TML 在其大规模实验中 Pre-RMSNorm 更稳定，PostNorm 曾出现 NaN；
- MTmixAtt 在其数据上 PostNorm_R 最优；
- UniMixer 使用 SiameseNorm；
- 本仓库 v10 改成 LayerNorm，是本地受控候选而非 TML 原设置。

因此 Norm 选择应在同一初始化、学习率、深度和数据上比较，不能仅按论文结论替换。

#### Dense 还是 MoE

MoE 只有在专家 kernel、路由平衡、通信和在线 sparse execution 同时可用时才可能提高 ROI。若本地仍未证明 dense Mixer 超过强 Base，直接加入 MoE 会显著增加归因难度，应后置。

---

## 10. 如何解读论文证据

### 10.1 证据强弱

| 证据类型 | 能说明什么 | 不能说明什么 |
|---|---|---|
| 同论文同数据等预算消融 | 某模块在该实验设置下的净贡献 | 跨业务、跨实现的普适收益 |
| 私有工业离线结果 | 大规模场景可行性和作者系统内部排序 | 外部绝对复现、和另一论文直接排名 |
| 在线 A/B | 组合方案在指定业务流量上的真实价值 | 单一模块因果贡献、其他业务收益 |
| 参数/FLOPs Scaling 曲线 | 在拟合区间内的边际收益趋势 | 任意更大规模仍按同一指数增长 |
| 公开代码 + 公开数据多 seed | 架构机制和统计稳定性较容易复核 | 工业吞吐、千亿样本和在线价值 |

### 10.2 当前文献的共性限制

1. 核心工业论文大多使用私有数据，特征、负采样、训练时长和基础模型不完全公开；
2. 多数只发布网络公式，没有生产 grouped kernel、并行运行时和完整训练配方；
3. 2026 年多篇工作仍是 arXiv 版本，正文存在会议模板占位符、拼写或指标基线表述问题；
4. AUC 的万分位差在工业中可能重要，但单次结果容易受 seed、数据漂移和 checkpoint 影响；
5. “参数更多但延迟不变”依赖 batch、硬件、精度和算子融合，不能从静态 FLOPs 自动推出。

### 10.3 相对可复现资源

- RankElastor 提供[代码仓库](https://github.com/vasile-paskardlgm/RankElastor)和 Zenodo 归档，且含 Criteo/Avazu/FuxiCTR 实验；
- FA-RankMixer 提供[竞赛代码](https://github.com/PixelCookie-zyf/TAAC-2026-SeRankMixer)；
- MLP-Mixer、MetaFormer、AutoInt 等基础工作有公开实现；
- RankMixer、TML、MixFormer、UniMixer 的论文页未提供完整生产实现入口，复现时应明确“结构复现”和“系统复现”的边界。

---

## 11. 与本仓库实现的对应关系

本节只说明代码结构相似性，不把本地模型命名等同于公开论文的完整实现。

### 11.1 总体映射

| 公开路线 | 本仓库入口 | 已对齐部分 | 尚未对齐/不可声称部分 |
|---|---|---|---|
| RankMixer | [`cvr_bn_rankmixer_v1.py`](../src/models/rankmixer/cvr_bn_rankmixer_v1.py) 至 [`cvr_bn_rankmixer_v4.py`](../src/models/rankmixer/cvr_bn_rankmixer_v4.py) | 固定无参 Mixing、$H=T$、Per-token FFN、Add&Norm；v2/v3 语义 Token 化 | 生产级 billion-scale 系统、论文 MoE、完全相同 tokenizer/训练数据 |
| TokenMixer-Large 风格 | [`cvr_bn_rankmixer_v5.py`](../src/models/rankmixer/cvr_bn_rankmixer_v5.py) 至 [`cvr_bn_rankmixer_v10.py`](../src/models/rankmixer/cvr_bn_rankmixer_v10.py) | Mixing/Reverting、双空间 Per-token SwiGLU、Global Token、PreNorm（v5–v9 以 RMSNorm 为主） | 跨层 residual、auxiliary loss、SP-MoE、grouped op、FP8、Token Parallel；v10 使用 LayerNorm |
| MixFormer | [`cvr_fst_last_norpy.py`](../src/models/mixformer/cvr_fst_last_norpy.py) | Query HeadMixing、Per-head SwiGLU、Sequence Cross-Attention、Output Fusion | UI-MixFormer 的用户/物品解耦与 request-level reuse 未在该文件中出现 |
| UniMixer | [`unimixer.py`](../src/models/unimixer/unimixer.py)、[`cvr_bn_unimixer_v1.py`](../src/models/rankmixer/cvr_bn_unimixer_v1.py) | UniMixing-Lite、低秩 $W_G$、basis $W_B$、Sinkhorn、温度退火、pSwiGLU、SiameseNorm | 完整 SP-MoE、论文全部数据/系统设置；本地 v1 是三桶 fst_CVR 适配 |

### 11.2 v1–v4：更接近原始 RankMixer

- v1 将 20,978 维直接切成 16 Token，存在切断 17 维字段和跨桶边界的问题；
- v2 改成字段安全分组并恢复 SENet 等本地模块；
- v3 采用 16 个固定业务语义组，是当前最清晰的本地正向 Tokenization 证据；
- v4 在 v3 上加入 Query→Item Cross，但当前实现未带来收益。

公开 RankMixer 的“语义 coherent group”原则与 v3 方向一致，但本地 SENet、Bucket Cross、gated pooling 和训练生命周期都是项目适配，不属于论文原始标准块。

### 11.3 v5–v10：TML 风格但不等于 TML

v5–v10 的共同核心是：

```text
31 Local + 1 Global Token
→ parameter-free Mixing
→ mixed-space Per-token SwiGLU
→ exact Reverting
→ original-space Per-token SwiGLU
→ 两层堆叠
```

其中：

- v5 使用 $T=H=32,D=1024$，容量最大；
- v6 将宽度收敛为 $D=512$，保留 RMSNorm 与增强读出；
- v8/v9 在 Token 化前重新引入 Masked DCN 或 Base 同构 DCN-M，验证显式 Cross + Mixer；
- v10 使用 LayerNorm + PureFlat 单路径读出。

这些版本仍固定 $H=T=32$，也没有 TML 论文的跨层残差、辅助损失和 SP-MoE。因此最准确的描述是“**TML-inspired dense block adaptation**”。

### 11.4 本地 MixFormer

[`src/models/mixformer/cvr_fst_last_norpy.py`](../src/models/mixformer/cvr_fst_last_norpy.py) 已实现论文主 block 的三个阶段：

1. `query_mixer`：RMSNorm → HeadMixing residual → Per-head SwiGLU residual；
2. `seq_cross_attention`：序列共享 SwiGLU + 非序列 Query 对序列 K/V 的 Cross-Attention；
3. `output_fusion`：再做一次 Per-head SwiGLU residual。

当前文件未检索到 UI-MixFormer 的方向性 user/item mask、跨候选用户侧缓存或 request-level batching。因此不能把论文报告的 30%+ serving speedup 直接归入本地实现。

### 11.5 本地 UniMixer v1

[`cvr_bn_unimixer_v1.py`](../src/models/rankmixer/cvr_bn_unimixer_v1.py) 使用 32 个互斥语义 Token、$D=512$、两层 UniMixing-Lite、Pertoken SwiGLU、SiameseNorm 和 PureFlat 深任务头。其设计与论文的核心算子一致，但做了明确的本地适配：

- 只使用 common/item/creative 三桶；
- 每个语义组独立 Linear + Token-specific BN；
- 不使用论文完整 MoE；
- 只预测 `fst_CVR`；
- 固定 dense 参数约 150.758M，目前是静态就绪、训练未验证状态。

详见 [`unimixer_v1_introduction.md`](unimixer_v1_introduction.md)。

### 11.6 本地结果不能被公开论文覆盖

截至本仓库 2026-08-28 汇总：

- Base 仍是唯一多日验证整体最优方案；
- v3 的语义分组是最清晰的 RankMixer 单变量正向证据；
- v5/v6 缩小差距但没有稳定超过 Base；
- v8/v9/v10、UniMixer v1 的证据状态各不相同，部分只有静态实现或结果归属待确认。

完整本地证据见 [`current_progress_technical_summary_2026-08-28.md`](current_progress_technical_summary_2026-08-28.md)。公开论文只能提供候选假设，不能替代相同特征、相同日期、相同初始化与多 seed 的本地检验。

---

## 12. 对当前项目最有价值的研究假设

### 12.1 P0：先测“表示是否真的扩张”

RankElastor/RankUp 给出了一个比继续堆参数更可诊断的方向：对每层输出 $X_l\in\mathbb{R}^{B\times T\times D}$ 记录：

- effective rank / fractional effective rank；
- Token 两两 cosine similarity；
- singular value spectrum；
- 每个 Token 的梯度范数与更新幅度；
- Mixing 前、mixed-space FFN 后、Reverting 后、original-space FFN 后的谱变化。

若 v5/v6 的大容量主要在 FFN 后发生 rank contraction，就能解释为什么参数增加但本地 AUC 未超过 Base。

### 12.2 P0：把 Tokenization 做成等参数对照

至少比较：

1. v3 固定业务语义组；
2. 字段安全 AutoSplit；
3. 固定随机置换后均衡分组；
4. 低成本可学习选择/AutoToken。

保持 Token 数、$D$、FFN、读出、初始化和训练协议完全相同。当前跨论文对 Tokenization 的结论互相冲突，这正说明本地单变量实验价值高。

### 12.3 P0：固定 Mixing 与 UniMixing-Lite 等预算比较

建议建立三点曲线，而不是只比两个大模型：

- 固定无参 Mixing；
- 逐 head 小型可学习 $T\times T$ Mixing；
- UniMixing-Lite 的低秩全局 + basis 局部 Mixing。

需同时匹配或报告 dense 参数、forward FLOPs、实际 step time、峰值内存和测试 AUC。若可学习 Mixing 只提高训练 AUC而扩大泛化间隙，应优先加正则/约束，而非继续升 rank。

### 12.4 P1：深度实验必须带残差方案

不要把 $L=2\rightarrow4\rightarrow8$ 作为孤立变量。建议同时比较：

- 当前短残差；
- 每 2 层 interval residual；
- 中间辅助 loss；
- SiameseNorm；
- 轻量 DeRes/跨层加权残差。

主要观察 NaN、梯度、有效秩和多日 AUC，而不仅是最终单日点。

### 12.5 P1：显式 Cross + Mixer 是本项目特有的高优先级

公开论文倾向用统一 Mixer 替代旧交互模块，但本地最强 Base 依赖全维 DCN-M，v9 也正是 Raw/Cross 双视图 + Mixer。建议把问题定义为：

> 在固定总计算预算下，多少容量给显式乘性交互，多少容量给 Token-specific 非线性最优？

可比较 Base、Pure Mixer、DCN-M→Mixer、并行 DCN/Mixer 四类，并让任务头参数量对齐。

### 12.6 P1：读出不能和 backbone 混在一次改动里

Mean、Global Token、conditioned pooling、PureFlat、Task Token、DCNM Shortcut 对信息瓶颈影响很大。v5–v10 同时改 backbone 和读出会降低归因能力，应使用同一最终 Token 张量做离线冻结或严格单变量读出对照。

### 12.7 P2：MoE 和在线复用的进入条件

只有满足以下条件后再推进 MoE：

- dense Mixer 已稳定优于或至少达到 Base；
- grouped per-token kernel 可用；
- 路由负载、激活专家比例和通信可观测；
- 训练和 serving 都能真正稀疏执行。

UG-Sep/UI-MixFormer 则应先确认线上请求是否存在稳定的“一用户多候选”批结构，以及用户侧特征在候选间是否完全相同。若没有这两个条件，理论复用不会变成真实延迟收益。

---

## 13. 推荐阅读顺序

### 13.1 只读四篇

1. [RankMixer](https://arxiv.org/abs/2507.15551)：理解基本 Token Mixing 和 Per-token FFN；
2. [TokenMixer-Large](https://arxiv.org/abs/2602.06563)：理解残差修复、SwiGLU、深度和系统共同设计；
3. [MixFormer](https://arxiv.org/abs/2602.14110)：理解 dense 与长序列联合扩展；
4. [UniMixer](https://arxiv.org/abs/2604.00590)：理解固定 Mixing 的矩阵化和可学习推广。

### 13.2 做结构研究

依次补读 [MTmixAtt](https://arxiv.org/abs/2510.15286)、[RankElastor](https://arxiv.org/abs/2605.23191)、[RankUp](https://arxiv.org/abs/2604.17878)、[DeRes](https://arxiv.org/abs/2606.07980)。

### 13.3 做长序列/统一模型

先读 [LONGER](https://arxiv.org/abs/2505.04421) 和 [OneTrans](https://arxiv.org/abs/2510.26104)，再读 [HyFormer](https://arxiv.org/abs/2601.12681)、[MixFormer](https://arxiv.org/abs/2602.14110)、[HeMix](https://arxiv.org/abs/2602.09387)。

### 13.4 做工程部署

重点读 TML 的 grouped operators/FP8/Token Parallel、MixFormer 的 UI decoupling、[UG-Sep](https://arxiv.org/abs/2602.10455) 的用户侧复用与 W8A16。

---

## 14. 文献清单与简要注释

### A. 核心与直接后继

1. **Zhu, J. et al. (2025). RankMixer: Scaling Up Ranking Models in Industrial Recommenders.** CIKM 2025. [arXiv:2507.15551](https://arxiv.org/abs/2507.15551)，[DOI](https://doi.org/10.1145/3746252.3761507)。固定无参 Mixing + Per-token FFN 的起点。
2. **Jiang, Y. et al. (2026). TokenMixer-Large: Scaling Up Large Ranking Models in Industrial Recommenders.** [arXiv:2602.06563](https://arxiv.org/abs/2602.06563)。核心官方后继。
3. **Huang, X. et al. (2026). MixFormer: Co-Scaling Up Dense and Sequence in Industrial Recommenders.** KDD 2026. [arXiv:2602.14110](https://arxiv.org/abs/2602.14110)，[DOI](https://doi.org/10.1145/3770855.3818447)。dense/sequence 统一扩展。
4. **Ha, M. et al. (2026). UniMixer: A Unified Architecture for Scaling Laws in Recommendation Systems.** [arXiv:2604.00590](https://arxiv.org/abs/2604.00590)。固定 Mixing 的可学习矩阵推广。
5. **Qi, X. et al. (2025). MTmixAtt: Integrating Mixture-of-Experts with Multi-Mix Attention for Large-Scale Recommendation.** [arXiv:2510.15286](https://arxiv.org/abs/2510.15286)。自动 Token、学习 Mixing、多场景专家。
6. **Lu, H. et al. (2026). Compute Only Once: UG-Separated TokenMixer for Efficient Large Recommendation Models.** [arXiv:2602.10455](https://arxiv.org/abs/2602.10455)。TokenMixer serving 复用扩展。
7. **Li, G. et al. (2026). Expand More, Shrink Less: Shaping Effective-Rank Dynamics for Dense Scaling in Recommendation.** KDD 2026. [arXiv:2605.23191](https://arxiv.org/abs/2605.23191)，[DOI](https://doi.org/10.1145/3770855.3818049)，[代码](https://github.com/vasile-paskardlgm/RankElastor)。RankElastor 与有效秩分析。

### B. Mixer 与序列/异构交互融合

8. **Huang, Y. et al. (2026). HyFormer: Revisiting the Roles of Sequence Modeling and Feature Interaction in CTR Prediction.** [arXiv:2601.12681](https://arxiv.org/abs/2601.12681)。Query Decoding/Boosting 交替。
9. **Wang, F. et al. (2026). Query-Mixed Interest Extraction and Heterogeneous Interaction: A Scalable CTR Model for Industrial Recommender Systems.** [arXiv:2602.09387](https://arxiv.org/abs/2602.09387)。HeMix/HeteroMixer。
10. **Chen, J. et al. (2026). RankUp: Towards High-rank Representations for Large Scale Advertising Recommender Systems.** [arXiv:2604.17878](https://arxiv.org/abs/2604.17878)。多 Token 类型与高秩表示。
11. **Wang, W. et al. (2026). RoleMix: Unifying Sequential and Non-Sequential Features via Semantic Tokenization for Post-Click Conversion Rate Prediction.** [arXiv:2607.22700](https://arxiv.org/abs/2607.22700)。UniMixing-Lite 的 PCVR 应用。
12. **Zhang, Y. et al. (2026). Field-Aware RankMixer with Dual-Stream Bilinear Fusion for the Tencent UNI-REC Challenge.** [arXiv:2607.15590](https://arxiv.org/abs/2607.15590)，[代码](https://github.com/PixelCookie-zyf/TAAC-2026-SeRankMixer)。竞赛级 RankMixer 应用。
13. **Cheng, W. et al. (2026). DeRes: Decoupling Residual Stability and Adaptivity for Scalable CTR Prediction.** [arXiv:2606.07980](https://arxiv.org/abs/2606.07980)。深层残差扩展。

### C. 统一模型、长序列和推荐 Scaling 对照

14. **Zhang, Z. et al. (2025). OneTrans: Unified Feature Interaction and Sequence Modeling with One Transformer in Industrial Recommender.** [arXiv:2510.26104](https://arxiv.org/abs/2510.26104)。统一 causal Transformer 路线。
15. **Chai, Z. et al. (2025). LONGER: Scaling Up Long Sequence Modeling in Industrial Recommenders.** RecSys 2025. [arXiv:2505.04421](https://arxiv.org/abs/2505.04421)。长序列基础。
16. **Zhang, B. et al. (2024). Wukong: Towards a Scaling Law for Large-Scale Recommendation.** [arXiv:2403.02545](https://arxiv.org/abs/2403.02545)。堆叠 FM Scaling 路线。
17. **Gui, H. et al. (2023). Hiformer: Heterogeneous Feature Interactions Learning with Transformers for Recommender Systems.** [arXiv:2311.05884](https://arxiv.org/abs/2311.05884)。异构 Attention 工业对照。
18. **Zhang, B. et al. (2022). DHEN: A Deep and Hierarchical Ensemble Network for Large-Scale Click-Through Rate Prediction.** [arXiv:2203.11014](https://arxiv.org/abs/2203.11014)。异构交互模块深层集成。

### D. 概念基础

19. **Tolstikhin, I. et al. (2021). MLP-Mixer: An all-MLP Architecture for Vision.** [arXiv:2105.01601](https://arxiv.org/abs/2105.01601)。Token/Channel 两轴 Mixing 的代表性起点。
20. **Yu, W. et al. (2022). MetaFormer Is Actually What You Need for Vision.** [arXiv:2111.11418](https://arxiv.org/abs/2111.11418)。通用 MetaFormer 骨架。
21. **Song, W. et al. (2019). AutoInt: Automatic Feature Interaction Learning via Self-Attentive Neural Networks.** [arXiv:1810.11921](https://arxiv.org/abs/1810.11921)。推荐字段自注意力交互基础。

---

## 15. 核心四篇 BibTeX

```bibtex
@inproceedings{zhu2025rankmixer,
  title     = {RankMixer: Scaling Up Ranking Models in Industrial Recommenders},
  author    = {Jie Zhu and Zhifang Fan and Xiaoxie Zhu and Yuchen Jiang and others},
  booktitle = {Proceedings of the 34th ACM International Conference on Information and Knowledge Management},
  year      = {2025},
  doi       = {10.1145/3746252.3761507},
  eprint    = {2507.15551},
  archivePrefix = {arXiv},
  url       = {https://arxiv.org/abs/2507.15551}
}

@misc{jiang2026tokenmixerlarge,
  title     = {TokenMixer-Large: Scaling Up Large Ranking Models in Industrial Recommenders},
  author    = {Yuchen Jiang and Jie Zhu and Xintian Han and Hui Lu and others},
  year      = {2026},
  eprint    = {2602.06563},
  archivePrefix = {arXiv},
  primaryClass = {cs.IR},
  url       = {https://arxiv.org/abs/2602.06563}
}

@inproceedings{huang2026mixformer,
  title     = {MixFormer: Co-Scaling Up Dense and Sequence in Industrial Recommenders},
  author    = {Xu Huang and Hao Zhang and Zhifang Fan and Yunwen Huang and others},
  booktitle = {Proceedings of the 32nd ACM SIGKDD Conference on Knowledge Discovery and Data Mining},
  year      = {2026},
  doi       = {10.1145/3770855.3818447},
  eprint    = {2602.14110},
  archivePrefix = {arXiv},
  url       = {https://arxiv.org/abs/2602.14110}
}

@misc{ha2026unimixer,
  title     = {UniMixer: A Unified Architecture for Scaling Laws in Recommendation Systems},
  author    = {Mingming Ha and Guanchen Wang and Linxun Chen and Xuan Rao and others},
  year      = {2026},
  eprint    = {2604.00590},
  archivePrefix = {arXiv},
  primaryClass = {cs.IR},
  url       = {https://arxiv.org/abs/2604.00590}
}
```

> BibTeX 中核心论文以 `and others` 缩写作者，正式投稿前应从 arXiv/ACM 页面重新导出完整作者列表和最终 proceedings 信息。

---

## 16. 最终判断

这条路线当前已经从“一个高效固定 Mixer”发展成四个相互交叉的问题族：

1. **表达力**：固定置换是否足够，还是需要 MTmixAtt/UniMixer/RankElastor 式可学习 Mixing；
2. **可训练深度**：Reverting、跨层残差、辅助损失、SiameseNorm、DeRes 如何避免深层退化；
3. **信号统一**：MixFormer/HyFormer/HeMix/RoleMix 如何让序列和非序列特征逐层交互；
4. **真实成本**：SP-MoE、grouped kernel、Token Parallel、UG-Sep/UI decoupling 如何把理论容量转成可部署吞吐。

对当前仓库，最稳妥的路线不是继续无条件放大 Per-token SwiGLU，而是按顺序回答三个问题：

1. 现有 Tokenization 是否保留字段语义并产生高秩、互补的 Token；
2. 固定 Mixing、低秩可学习 Mixing 和显式 DCN-M 在等预算下各贡献多少；
3. 只有 dense 结构稳定胜出后，才引入深度、MoE 和在线复用优化。

这也解释了为什么公开论文中“更大必然更好”的趋势，不能直接覆盖本仓库 Base 当前仍最优的事实：Scaling Law 依赖架构、数据量、训练配方、系统 kernel 和任务分布同时成立，而不只是参数计数。
