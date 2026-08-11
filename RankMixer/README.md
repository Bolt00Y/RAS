# RankMixer 研究分支

本目录用于在已完成 RankMixer 论文基线复现的基础上，系统研究适用于推广搜索 CVR 预估的改进方案，并持续梳理 RankMixer 方法谱系。

## 当前基线

- 日训练数据：约 5.5 亿条；
- Batch size：2048；
- 稀疏特征：user 385 个、item 835 个、creative 14 个，共 1234 个字段；
- 每个字段 embedding 维度：17；
- 展开输入： $\mathbb{R}^{B\times20{,}978}$ ；
- 小模型： $T=16$ 、 $D=768$ 、2 个 blocks；
- 大模型： $T=32$ 、 $D=1536$ 、2 个 blocks；
- Autosplit 后每个 segment 独立投影；
- 输出：Mean Pooling + CVR 二分类头；
- 当前实现不使用 Sparse MoE。

当前 Autosplit 被保留为论文复现主基线。随机划分、语义划分、字段对齐划分只作为独立消融，不被视为对基线的“纠错”。

## RankMixer 方法谱系阅读

建议先阅读总览，再按对应技术问题进入单篇详解。

1. [RankMixer 后续修改版本总结：架构变化、技术路线与适用边界](docs/13_rankmixer_post_variants_architecture_summary.md)
2. [RankMixer 方法谱系总览：TokenMixer-Large、RankUp、MixFormer 与 UniMixer](docs/11_rankmixer_family_evolution_overview.md)
3. [RankMixer 论文详解：硬件友好扩展架构](docs/07_rankmixer_paper_detailed_review.md)
4. [TokenMixer-Large 论文详解：第二代大规模排序主干](docs/08_tokenmixer_large_paper_detailed_review.md)
5. [RankUp 论文详解：高秩表示学习](docs/12_rankup_paper_detailed_review.md)
6. [MixFormer 论文详解：Dense 与 Sequence Co-Scaling](docs/09_mixformer_paper_detailed_review.md)
7. [UniMixer 论文详解：结构化可学习 Mixing](docs/10_unimixer_paper_detailed_review.md)
8. [RankUp Figure 1–5 扩展图解与复现分析](docs/04_rankup_paper_walkthrough.md)

### 方法定位

| 方法 | 主要解决的问题 | 相对 RankMixer 的关键变化 |
|---|---|---|
| RankMixer（2025-07-21） | 工业排序模型难以低延迟、高 MFU 地扩展 | 固定 Token Mixing + Per-token FFN |
| TokenMixer-Large（2026-02-06） | residual 语义错位、深层训练和 MoE 不完整 | Mixing & Reverting、pSwiGLU、Pre-RMSNorm、SP-MoE |
| MixFormer（2026-02-15） | 非序列交互与行为序列彼此割裂 | Query Mixer + 每层 Cross-Attention + UI 解耦 |
| UniMixer（2026-04-01） | 固定 mixing 不可学习且受 $H=T$ 约束 | Learnable global-local mixing + Lite 压缩与 Sinkhorn |
| RankUp（2026-04-20） | 参数增长没有转化为有效表示维度 | Random Split、Multi-embedding、Global/Cross/Task Tokens |
| RankElastor（2026-05-22） | 固定 mixing 扩秩有限，普通 PFFN 反复缩秩 | Parameterized Full Mixing + GLU-improved P-FFN |

## 当前业务研究文档

1. [文献综述与基线诊断](docs/01_literature_and_diagnosis.md)
2. [详细改进方案](docs/02_modification_schemes.md)
3. [实验与统计检验协议](docs/03_experiment_protocol.md)
4. [电商推广搜 CVR：RankMixer 与强 Base 的差距诊断及改进路线](docs/05_ecommerce_cvr_gap_diagnosis_and_recovery_plan.md)
5. [大参数 RankMixer 若仍弱于 Base：系统消融实验方案](docs/06_large_rankmixer_underperformance_ablation_plan.md)
6. [机器可读实验矩阵](configs/experiment_matrix.yaml)

## 当前问题驱动的推荐优先级

| 优先级 | 方案 | 主要目的 | 首版风险 |
|---|---|---|---|
| P0 | Base-preserving zero-init RankMixer residual | 保住强 Base，只让 RankMixer 学习剩余误差 | 中 |
| P0 | 复用 Base 的 BN + hierarchical SENet 前端 | 恢复样本条件的 user/item/creative 特征选择 | 低到中 |
| P0 | Base-matched MLP head + task-aware pooling | 避免 Mean Pooling 稀释强 token | 低 |
| P0/P1 | DCNv2 并行支路或低秩 adapter | 恢复显式乘法交叉 | 中 |
| P1 | TokenMixer-Large Block Lite | 修复 residual 语义错位，并为大模型扩展做准备 | 中 |
| P1 | RankUp-Lite：固定随机划分 + Global Adapter | 提升输入表示秩、降低 token 冗余 | 低到中 |
| P1 | UniMixer-Lite Adapter | 验证固定 mixing 是否限制当前数据 | 中 |
| 条件 P1 | MixFormer / UI-MixFormer | 仅在已有行为序列和请求级多候选条件下研究 | 中高 |
| P1 | Base teacher distillation | 将强 Base 的决策边界迁移到单分支 RankMixer | 中 |
| P2 | RankElastor-inspired 低秩 flatten mixing | 验证核心算子谱增强，不直接使用完整 Full Mixing | 高 |
| P2 | 原样扩大到 $T=32$ 、 $D=1536$ | 验证纯容量 scaling | 高，必须配合利用率诊断 |
| P3 | Sparse-Pertoken MoE | 在 dense scaling 已获益后提高容量效率 | 高 |

## 研究原则

- 所有方案必须与原始 RankMixer 基线进行单变量对照；
- 当前 Base 是同数据上的最强结构证据，新模型应先继承其有效归纳偏置；
- 同时报告参数匹配、FLOPs 匹配和线上延迟匹配三类结果；
- 不把多个模块一次性串联后只报告最终指标；
- 不以参数增加本身作为创新，必须说明新增模块补足了哪类建模缺口；
- 任何收益都必须同时检查 AUC/GAUC、LogLoss、校准、吞吐、MFU 和 P99 延迟；
- RankUp 类方案必须同时检查 entropy effective rank 与 token redundancy；
- RankElastor 类方案必须单独记录 stable rank，不能与 RankUp 的 effective rank 数值混报；
- UniMixer 类方案必须加入固定随机 mixing 和同参数 MLP 对照；
- MixFormer 只有在当前已有行为序列时才属于严格复现；
- 先验证单模块，再组合两个已独立获益的模块；
- 对转化延迟、点击条件样本和全曝光样本空间进行严格区分。

## 建议首轮实验

```text
A0  当前 Base 与 RankMixer 的多 seed / checkpoint 复现
A1  RankMixer + Base bucket-wise BN
A2  RankMixer + Base-matched MLP head
A3  Mean + Attention residual pooling
A4  calibrated Base / RankMixer logit blend
B1  BN + hierarchical SENet + RankMixer
B2  Base + zero-init RankMixer residual
B3  RankMixer + low-rank DCNv2 adapter
C1  RankMixer + Mixing & Reverting
C2  ordered field-aligned / fixed random split
C3  simple learned token mixing
```

若大参数模型仍弱于 Base，优先执行文档 06 中的 Base 反向消融、规模轴 2×2 拆分、pooling/head、BN/SENet 与 DCNv2 对照，再进入完整 RankUp、TokenMixer-Large、UniMixer、RankElastor-inspired mixing、MoE 或更大规模训练。
