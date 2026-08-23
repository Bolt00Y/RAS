### v8 日报

今日完成 RankMixer v8 方案设计。设计思路是：现有 RankMixer 直接从原始特征学习交互的难度较大，因此在语义 Token 化之前增加两层全字段 Masked Low-Rank DCN，通过动态 Mask 筛选有效的显式交叉，同时保留原始 Global Token，避免原始信息被交叉层完全覆盖。整体采用“分层 SENet → Masked Low-Rank DCN → 31 个交叉语义 Token + 原始 Global Token → 两层 TokenMixer → 三路增强读出 → CVR 任务塔”的架构，参数量控制在 200M 以内。

### v9 日报

今日完成 RankMixer v9-Small 方案设计。设计思路是：结合现有实验结果，单纯扩大 RankMixer 容量并未超过 Base，而 Base 的分层 SENet 和 DCNM 显式交叉仍是当前最可靠的有效结构，因此 v9 保留精确两层 DCNM500，并通过 Raw/Cross 双视图 Token 同时利用原始信息和交叉信息；另外增加 DCNM Shortcut，降低 Token 化过程中关键信息损失的风险。整体采用“分层 SENet → Base DCNM → Raw/Cross 语义 Token → 两层 TokenMixer → 三路增强读出 + DCNM Shortcut → 深层 CVR 任务塔”的架构，参数量约 199M，较 v5 降低 42.8%，用于快速验证显式交叉与隐式交互融合能否超过 Base。