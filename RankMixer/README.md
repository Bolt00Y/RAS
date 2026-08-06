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
4. [机器可读实验矩阵](configs/experiment_matrix.yaml)

## 推荐优先级

| 优先级 | 方案 | 主要目的 | 首版风险 |
|---|---|---|---|
| P0 | RankUp-Lite：固定随机划分 + Global Token | 提升输入表示秩、降低 token 冗余 | 低 |
| P0 | TokenMixer-Large Block Lite | 修复 mixing 前后残差语义错位，并为加深网络做准备 | 中 |
| P1 | 低秩 DCNv2 显式交叉支路 | 补充受控的显式乘法交互 | 中 |
| P1 | 实例条件 Token-Channel Gate | 动态选择当前请求的重要 token/channel | 低 |
| P1/P2 | UniMixer-inspired Learnable Mixing Adapter | 让固定 mixing 模式可学习，同时保持结构化约束 | 中 |
| P2 | Sparse-Pertoken MoE | 在近似固定激活 FLOPs 下扩大容量 | 高 |
| 条件方案 | ESMM + 任务特定池化 | 处理全曝光 CVR 的样本选择偏差和稀疏性 | 取决于标签 |

## 研究原则

- 所有方案必须与原始 RankMixer 基线进行单变量对照；
- 同时报告参数匹配、FLOPs 匹配和线上延迟匹配三类结果；
- 不把多个模块一次性串联后只报告最终指标；
- 不以参数增加本身作为创新，必须说明新增模块补足了哪类建模缺口；
- 任何收益都必须同时检查 AUC/GAUC、LogLoss、校准、吞吐、MFU 和 P99 延迟；
- 先验证单模块，再组合两个已独立获益的模块；
- 对转化延迟、点击条件样本和全曝光样本空间进行严格区分。

## 建议首轮实验

```text
RM-B0  原始 RankMixer 基线
RM-R1  固定随机 feature permutation，T=16
RM-R2  15 local tokens + 1 global token
RM-T1  Mixing & Reverting，计算量匹配版
RM-D1  低秩 DCNv2 并行支路
RM-G1  输入层 Factorized Token-Channel Gate
```

先在完全一致的数据窗口和训练步数下完成以上实验，再决定是否进入深层 TokenMixer-Large、UniMixing 或 Sparse-Pertoken MoE。
