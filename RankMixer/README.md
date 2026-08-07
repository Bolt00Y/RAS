# RankMixer 研究分支

本目录用于在已完成 RankMixer 论文基线复现的基础上，系统研究适用于推广搜索 CVR 预估的改进方案。

## 当前基线

- 日训练数据：约 5.5 亿条；
- Batch size：2048；
- 稀疏特征：user 385 个、item 835 个、creative 14 个，共 1234 个字段；
- 每个字段 embedding 维度：17；
- 展开输入： $\mathbb{R}^{B\times20{,}978}$ ；
- Autosplit：16 个连续 segment；
- 每个 segment 独立投影到 768 维；
- RankMixer：2 个 block， $T=H=16$ ；
- 输出：Mean Pooling + CVR 二分类头；
- 当前实现不使用 Sparse MoE。

当前 Autosplit 被保留为论文复现主基线。随机划分、语义划分、字段对齐划分只作为独立消融，不被视为对基线的“纠错”。

## 文档

1. [文献综述与基线诊断](docs/01_literature_and_diagnosis.md)
2. [详细改进方案](docs/02_modification_schemes.md)
3. [实验与统计检验协议](docs/03_experiment_protocol.md)
4. [RankUp 论文图解与复现分析](docs/04_rankup_paper_walkthrough.md)
5. [电商推广搜 CVR：RankMixer 与强 Base 的差距诊断及改进路线](docs/05_ecommerce_cvr_gap_diagnosis_and_recovery_plan.md)
6. [机器可读实验矩阵](configs/experiment_matrix.yaml)

## 当前问题驱动的推荐优先级

| 优先级 | 方案 | 主要目的 | 首版风险 |
|---|---|---|---|
| P0 | Base-preserving zero-init RankMixer residual | 保住强 Base，只让 RankMixer 学习剩余误差 | 中 |
| P0 | 复用 Base 的 BN + hierarchical SENet 前端 | 恢复样本条件的 user/item/creative 特征选择 | 低到中 |
| P0 | Base-matched MLP head + task-aware pooling | 避免 Mean Pooling 稀释强 token | 低 |
| P0/P1 | DCNv2 并行支路或低秩 adapter | 恢复显式乘法交叉 | 中 |
| P1 | RankUp-Lite：固定随机划分 + Global Adapter | 提升输入表示秩、降低 token 冗余 | 低到中 |
| P1 | TokenMixer-Large Block Lite | 修复 residual 语义错位，并为大模型扩展做准备 | 中 |
| P1 | Base teacher distillation | 将强 Base 的决策边界迁移到单分支 RankMixer | 中 |
| P2 | 原样扩大到 $T=32,D=1536$ | 验证纯容量 scaling | 高，当前不宜直接给完整预算 |
| P3 | Sparse-Pertoken MoE | 在 dense scaling 已获益后提高容量效率 | 高 |

## 研究原则

- 所有方案必须与原始 RankMixer 基线进行单变量对照；
- 当前 Base 是同数据上的最强结构证据，新模型应先继承其有效归纳偏置；
- 同时报告参数匹配、FLOPs 匹配和线上延迟匹配三类结果；
- 不把多个模块一次性串联后只报告最终指标；
- 不以参数增加本身作为创新，必须说明新增模块补足了哪类建模缺口；
- 任何收益都必须同时检查 AUC/GAUC、LogLoss、校准、吞吐、MFU 和 P99 延迟；
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
```

先确认能否缩小或闭合当前 Base 差距，再进入 Random Split、TokenMixer-Large、大模型扩展或 Sparse-Pertoken MoE。
