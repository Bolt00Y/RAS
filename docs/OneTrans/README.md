# OneTrans 在推广搜 CVR 中的建模探索

本目录针对以下固定输入约束，设计 OneTrans 在推广搜（广告搜索/商业搜索排序）中的 CVR 预测方案：

- 日训练样本：约 `5.5 × 10^8`
- User：`385 × 17 = 6,545` 维
- Item：`835 × 17 = 14,195` 维
- Creative：`14 × 17 = 238` 维
- 单样本总输入：`20,978` 维
- 目标：预测 post-click CVR；若具备曝光级 click/convert 标签，则同时建议建模 CTR 与 CTCVR

## 核心结论

当前约束只包含非序列特征，没有 OneTrans 原论文中的多行为序列。因此不能严格复现其 S-token、timestamp-aware sequence fusion、S-token pyramid 和原始 cross-request sequence KV cache。最接近论文的可执行基线应当把全部输入视为 NS 特征，保留：

1. Auto-Split tokenizer；
2. 6 层、`d_model=256`、4 头；
3. Pre-RMSNorm；
4. causal attention；
5. NS-token-specific Q/K/V 与 FFN；
6. BF16、梯度裁剪和时间顺序评估。

在此基线上，本目录进一步提供四个创新方案。

| 编号 | 方案 | 主要创新 | 优先级 |
|---|---|---|---|
| P0 | Static OneTrans-S | 静态条件下最贴近原论文的复现基线 | 必做 |
| P1 | Entity-Hierarchical OneTrans | 保留 user/item/creative 边界，支持 user-prefix KV cache | 高 |
| P2 | Funnel Task-Token OneTrans | ESMM 约束 + CTR/CVR/CTCVR task tokens + 受控任务信息流 | 最高（需曝光级标签） |
| P3 | Search-Bias-Decoupled OneTrans | 重特征 FiLM late fusion + bias logit 解耦 + request 内 pairwise loss | 高（推广搜） |
| P4 | Low-Rank Adaptive OneTrans | 低秩 token-specific 参数 + NS-query pyramid，降低参数和延迟 | 中高（工程化） |

## 文件索引

- [`00_problem_and_literature.md`](./00_problem_and_literature.md)：约束、论文对照与设计边界
- [`01_faithful_static_onetrans.md`](./01_faithful_static_onetrans.md)：最贴近原论文的静态 OneTrans
- [`02_entity_hierarchical_onetrans.md`](./02_entity_hierarchical_onetrans.md)：创新方案一
- [`03_funnel_task_token_onetrans.md`](./03_funnel_task_token_onetrans.md)：创新方案二
- [`04_search_bias_decoupled_onetrans.md`](./04_search_bias_decoupled_onetrans.md)：创新方案三
- [`05_lowrank_adaptive_onetrans.md`](./05_lowrank_adaptive_onetrans.md)：创新方案四
- [`06_experiment_and_rollout.md`](./06_experiment_and_rollout.md)：实验矩阵、指标和上线顺序
- [`../../configs/onetrans_search_cvr.yaml`](../../configs/onetrans_search_cvr.yaml)：建议配置

## 推荐落地顺序

1. 先实现 P0，和现有 RankMixer 在相同样本、相同 dense 参数/FLOPs 下比较。
2. 若训练集含全部曝光及 click/convert 标签，优先切换 P2；CVR 的样本选择偏差通常比骨干网络的小结构差异更重要。
3. 再实现 P1，验证实体边界、prefix cache 和 Auto-Split 的收益。
4. 在推广搜场景实现 P3，单独处理 position/page/slot/bid/query-item match 等强信号。
5. 当效果确认后，用 P4 压缩 token-specific 参数并降低 serving 成本。

> 本目录给出的参数量均为 dense 部分的近似值，不含上游 sparse embedding table；真实 FLOPs 与延迟需要以具体 kernel、候选组织方式和硬件为准。
