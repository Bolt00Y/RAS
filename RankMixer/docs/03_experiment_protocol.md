# RankMixer 改进实验与统计检验协议

## 1. 目的

本协议用于回答三个不同的问题，不能混为一谈：

1. **固定训练数据与训练步数时，结构是否更有效？**
2. **固定参数量或活跃 FLOPs 时，结构是否更有效？**
3. **固定线上延迟预算时，结构是否更有业务价值？**

因此，每个候选方案至少需要给出：

- `same-data / same-step` 对照；
- `parameter-matched` 或 `active-FLOPs-matched` 对照；
- `latency-matched` 或明确的延迟代价说明。

单纯以“参数更多、AUC 更高”不能证明结构改进有效。

---

## 2. 基线冻结

正式实验前冻结 `RM-B0`：

```text
Fields: user 385 + item 835 + creative 14
Embedding dim: 17
Flatten dim: 20978
Autosplit: 16 continuous segments
Projection dim: 768
RankMixer blocks: 2
Mixing heads: 16
Pooling: Mean Pooling
Task: CVR binary classification
Batch size: 2048
Sparse MoE: disabled
```

同时记录：

- field schema 版本与顺序 hash；
- embedding table 配置 hash；
- tokenizer segment 映射；
- 代码 commit；
- 数据抽取 SQL / pipeline 版本；
- label maturation 规则；
- optimizer、LR schedule、weight decay、warmup、gradient clipping；
- BF16/FP16/FP32 设置；
- 随机 seed；
- 训练集、验证集和测试集的绝对时间区间。

基线至少重复训练 3 次，用于估计训练噪声。若基线自身波动已经接近候选方案的收益，不能直接进入线上实验。

---

## 3. 数据切分与标签完整性

## 3.1 必须使用时间切分

推荐采用连续时间窗口：

```text
Train:      [t0, t1)
Validation: [t1, t2)
Test:       [t2, t3)
```

不要随机打散样本后切分，以避免：

- 同一 user/request/session 跨集合泄漏；
- 同一 item/creative 短期状态泄漏；
- 未来统计特征进入过去样本；
- 转化延迟造成标签污染。

最终日期由生产数据周期确定，但所有模型必须使用完全相同的绝对窗口。

## 3.2 转化延迟与右删失

CVR 标签必须达到既定成熟窗口。例如业务定义为点击后 7 天内转化，则测试样本也必须等待完整 7 天后再计算最终指标。

需要同时记录：

- label delay 分布；
- 不同 delay bucket 的正例占比；
- immature label 占比；
- 因等待成熟窗口而丢弃的样本量。

严禁把尚未回流的正例当作负例。若线上训练必须使用未成熟标签，需要另行研究 delayed-feedback / survival correction，不能把该问题归因于 RankMixer backbone。

## 3.3 clicked-only 与 entire-space 必须区分

- clicked-only CVR：目标是 `P(conversion | click, x)`；
- entire-space 训练：需要曝光、点击和转化标签，并可采用 ESMM 类目标；
- 两种样本空间的 AUC 和 LogLoss 不应直接横向比较。

每份实验报告必须在第一页明确写出样本空间。

---

## 4. 分阶段实验规模

## 4.1 Stage A：管线正确性

使用固定的小规模时间窗口或 1% 数据：

- 检查张量形状、参数量和 mask；
- 进行 overfit-small-batch 测试；
- 验证 loss 可下降；
- 验证新模块 identity/zero-init 时输出与基线近似一致；
- 检查 NaN/Inf、梯度爆炸和 gate/router collapse；
- 对 Random Split 检查训练与推理映射完全相同。

该阶段只判断实现正确性，不用于宣称效果提升。

## 4.2 Stage B：中等规模筛选

建议使用完整数据分布的固定子集，而不是只抽热门流量：

- 训练数据量为全量的 5%~10%；
- 保持相同 epoch 或相同 seen examples；
- 每个方案至少 2 个 seed；
- Random Split 至少 3 个固定 permutation seed；
- 淘汰明显劣化或效率不达标的方案。

## 4.3 Stage C：全量离线训练

候选方案进入 5.5 亿条/天的完整训练窗口：

- 与基线使用相同数据、步数和调度；
- 再补充参数/FLOPs 匹配实验；
- 至少复现 2~3 次；
- 完成总体和分桶指标；
- 完成吞吐、显存和延迟 profiling。

## 4.4 Stage D：线上实验

只有离线收益稳定、校准无明显恶化、工程预算达标后进入线上：

- 小流量 shadow/canary；
- 再进行随机流量 A/B；
- 使用预先定义的 guardrail；
- 观察至少覆盖主要工作日/周末模式和转化成熟周期；
- 线上结果按实验单元进行显著性检验，不能把单条曝光当成完全独立样本。

---

## 5. 统一训练控制变量

首轮单模块实验固定：

```text
Data window
Seen examples
Batch size
Embedding tables
Tokenizer output D=768
T=16（除非该方案明确研究 token 数）
Optimizer and LR schedule
Warmup ratio
Weight decay
Gradient clipping
Loss definition and sample weights
Negative sampling
Precision
Checkpoint selection rule
```

若某方案必须改变其中一项，报告中必须写出原因，并增加对应的控制实验。

### 5.1 训练时长的三种公平口径

每个方案至少采用前两种：

1. **same steps / same examples**：衡量样本效率；
2. **same wall-clock budget**：衡量生产训练预算下的最终效果；
3. **train-to-convergence**：衡量理论上限，但必须报告额外训练成本。

不要只选择最有利于候选模型的口径。

---

## 6. 指标体系

## 6.1 主效果指标

根据业务预先指定一个 primary metric，推荐从以下选择：

- AUC；
- GAUC / UAUC（按 user、request 或业务单元聚合）；
- LogLoss；
- NCE；
- PR-AUC（正例极稀疏时尤其重要）。

AUC 提升不能代替 LogLoss 和校准检查。

## 6.2 校准指标

至少报告：

$$
\text{CVR Bias}=\frac{\sum_i \hat p_i}{\sum_i y_i}.
$$

并报告：

- ECE；
- Brier Score；
- reliability diagram；
- 按预测分位数的 `predicted / observed`；
- 不同转化率分桶的 bias。

如果候选模型 AUC 提升但明显过估/低估，需先完成独立校准或重新训练，再讨论上线。

## 6.3 分桶指标

至少按以下维度评估：

- user 活跃度、新老用户；
- item 热度、类目、价格区间、新老商品；
- creative 类型、新老素材；
- query 频次、意图类别、长度或复杂度；
- 正负样本、转化延迟；
- 流量入口、设备、地域和时间段；
- head / torso / tail 特征覆盖；
- 样本权重或出价区间（若适用）。

分桶边界必须由基线数据预先确定，不能在看到候选结果后反复调整。

## 6.4 系统指标

训练侧：

- 总参数、dense 参数、embedding 参数；
- 每样本理论 FLOPs；
- 激活参数/FLOPs；
- examples/s、steps/s；
- MFU；
- HBM 峰值；
- 通信量与 all-to-all 时间；
- 数据加载占比；
- forward/backward/optimizer 时间。

推理侧：

- batch=1 和线上真实 batch 分布；
- P50/P95/P99 latency；
- QPS / GPU；
- HBM/显存；
- kernel 数量；
- 小算子占比；
- compile/fusion 成功率；
- MoE dispatch 与 expert imbalance。

参数量小但 kernel 数显著增加的方案不能被视为“低成本”。

## 6.5 结构诊断指标

### Effective Rank

对单样本 token matrix `H_b ∈ R^[16,768]`：

$$
p_i=\frac{\sigma_i}{\sum_j\sigma_j},\qquad
\operatorname{erank}(H_b)=\exp\left(-\sum_i p_i\log(p_i+\epsilon)\right).
$$

建议在以下位置采样：

- tokenizer 输出；
- 每层 mixing 后；
- 每层 PFFN 后；
- 最终输出。

### Token redundancy

计算 token 两两 cosine similarity 的：

- 平均值；
- 最大值；
- 上分位数；
- 不同层的变化。

### 优化诊断

记录：

- 每个 token projector/PFFN 的梯度范数；
- 参数更新比 `||ΔW||/||W||`；
- activation RMS；
- dead ReLU / gate saturation；
- MoE expert load、router entropy 和 dropped tokens；
- Gate 均值、方差和分位数。

---

## 7. 参数、FLOPs 与延迟匹配

## 7.1 参数匹配

当候选方案增加参数时，至少增加一个基线宽度对照，使总 dense 参数接近。例如：

```text
Candidate: RankMixer + DCN branch
Control:   slightly wider RankMixer PFFN
```

若候选效果仅与参数增加相关，宽化基线可能取得同等收益。

## 7.2 活跃 FLOPs 匹配

MoE 必须同时报告：

- 总容量参数；
- 每样本实际激活参数；
- router/dispatch 开销；
- 理论 GEMM FLOPs；
- 实测 wall-clock。

TokenMixer-Large Lite 的双 pSwiGLU 必须与单 PFFN 的计算量匹配版先对照，再测试容量增强版。

## 7.3 延迟匹配

线上预算未知时，报告完整 Pareto frontier：

```text
x-axis: P99 latency or GPU cost
 y-axis: primary quality metric
```

不要以一个任意的参数阈值替代真实 SLA。

可使用以下初始筛选 guardrail，但必须由生产 SLA 覆盖：

- 低风险 adapter/gate：训练吞吐下降不超过约 5%；
- 小型 side branch：线上 P99 增幅尽量控制在约 3%；
- MoE/深层模型：按完整收益-成本曲线评估，不使用统一阈值。

这些只是研究阶段默认值，不是最终上线标准。

---

## 8. 统计检验

## 8.1 预注册

在训练前写明：

- primary metric；
- secondary metrics；
- 主要分桶；
- 最小实际有意义提升（MDE）；
- seed 数；
- 训练停止规则；
- checkpoint 选择规则；
- 淘汰 guardrail。

防止在大量指标中只挑选偶然为正的结果。

## 8.2 置信区间

离线推荐使用 paired bootstrap：

1. 以 user、request 或 day 作为重采样单元；
2. 同一 bootstrap sample 同时计算基线和候选；
3. 对差值构造 95% CI；
4. 报告绝对变化和相对变化。

不要把 5.5 亿条样本全部视为独立 Bernoulli 样本；同一用户、请求和时间段存在相关性。

## 8.3 多重比较

当同时测试大量方案/分桶时，使用 Benjamini-Hochberg FDR 或预先指定少量 confirmatory hypotheses。

探索性结果应标记为 exploratory，不能与预注册主结果同等解读。

## 8.4 统计显著不等于业务显著

在超大样本下，极小差异也可能有很小 p-value。最终决策同时要求：

- CI 不跨 0；
- 超过预先定义的 MDE；
- LogLoss/校准和 guardrail 不恶化；
- 收益能够覆盖计算与工程成本；
- 关键分桶不存在不可接受的退化。

---

## 9. 每类方案的专用检查

## 9.1 Random Split / Global Token

- 至少 3 个 permutation seed；
- 映射文件 checksum；
- token coverage、基数和非零率分布；
- effective rank 与 token redundancy；
- Global Token 与 local tokens 的 cosine similarity；
- 减少一个 local token 是否损失 creative/item 细节。

## 9.2 TokenMixer-Large Lite

- compute-matched 与 capacity-enhanced 分开；
- Mixing/Reverting round-trip 单元测试；
- residual layout 检查；
- PreNorm/PostNorm 独立消融；
- 深层版本梯度与 activation RMS；
- auxiliary head 在推理图中确认被移除。

## 9.3 Learnable Mixing

- `gamma=0` 时严格回归基线；
- mixing matrix 行/列和；
- 温度退火轨迹；
- matrix entropy 与稀疏度；
- 是否退化为全局平均；
- 固化 mixing matrix 后的推理一致性。

## 9.4 DCNv2 支路

- Cross branch zero-init 输出一致性；
- rank `r` 和层数独立消融；
- 与“同参数宽化 PFFN”对照；
- side branch kernel 数和 MFU；
- cross 输出与 RankMixer 输出的相关性；
- 若高度相关且收益很小，说明支路冗余。

## 9.5 Factorized Gate

- gate 初始值为 1；
- token/channel gate 分开消融；
- gate saturation；
- tail token 是否长期被抑制；
- gate 对不同样本群体是否有合理差异；
- identity regularization 敏感性。

## 9.6 Sparse-Pertoken MoE

- first-enlarge-then-sparse 三阶段；
- 总容量与活跃容量；
- 每个 token 内 expert load；
- shared expert 消融；
- routing alpha 消融；
- load-balance loss；
- dispatch/communication 时间；
- P99 尾延迟；
- 专家是否学到可区分表示，而非简单复制。

## 9.7 ESMM / 多任务

- 成熟曝光、点击、转化标签；
- CTR、CTCVR、CVR 分别评估；
- task-specific pooling 权重分布；
- 任务 seesaw；
- pCTR、pCVR 和乘积的校准；
- 不给未点击样本伪造 CVR 标签。

---

## 10. 推荐实验顺序与晋级门槛

## Phase 0：基线复现

```text
RM-B0 × 3 seeds
```

晋级条件：指标和吞吐稳定，所有配置可复现。

## Phase 1：低风险、高信息量

```text
RM-R1 固定随机划分
RM-R2 15 Local + 1 Global
RM-G1 Token-only Gate
```

晋级条件：至少一个全量指标稳定为正，且效率满足低风险 guardrail。

## Phase 2：核心 block

```text
RM-T1 Mixing & Reverting，compute matched
RM-T2 + pSwiGLU / Pre-RMSNorm
RM-T4 L=4 + inter-residual
```

仅当 `L=2` block 升级已经获益时研究深层版本。

## Phase 3：互补交互

```text
RM-D2 低秩 DCNv2
RM-U1 简单 learned token mixing
RM-U3 UniMixing-Lite adapter
```

先用简单 learned `16×16` mixing 判断问题是否存在，再上结构更复杂版本。

## Phase 4：容量 scaling

```text
RM-M1 dense pSwiGLU
RM-M2 dense enlarged
RM-M3 Sparse-Pertoken MoE
```

只有 dense enlarged 明确优于 compute-matched dense 时，才进入 sparse MoE。

## Parallel Track：目标层

```text
RM-E1 ESMM + task-specific pooling
```

仅在 entire-space 标签条件满足时运行。

---

## 11. 组合实验规则

只有两个组件均满足以下条件时才允许组合：

1. 各自相对 `RM-B0` 独立获益；
2. 收益超过基线 seed 波动；
3. 无明显校准和关键分桶退化；
4. 效率成本可接受；
5. 两者建模假设具有互补性。

组合后使用 2×2 factorial design：

| A | B | 实验 |
|---|---|---|
| 0 | 0 | Baseline |
| 1 | 0 | A only |
| 0 | 1 | B only |
| 1 | 1 | A + B |

从而判断：

- 增益是否可加；
- 是否存在互相替代；
- 是否出现负交互；
- 组合收益是否只是其中一个模块贡献。

---

## 12. 单次实验报告模板

```markdown
# Experiment RM-XX

## Hypothesis
新增模块补足的具体能力是什么？

## Change
与 RM-B0 相比只改变了什么？

## Data and Labels
绝对时间区间、样本空间、成熟窗口、样本量。

## Model Budget
总参数、活跃参数、理论 FLOPs、实测吞吐、显存、延迟。

## Training
optimizer、LR、steps/examples、seed、checkpoint rule。

## Overall Results
AUC/GAUC、LogLoss、PR-AUC、calibration、95% CI。

## Segment Results
预注册分桶。

## Structural Diagnostics
effective rank、token redundancy、gate/router/gradient。

## Efficiency
examples/s、MFU、P50/P95/P99。

## Failure Analysis
不符合假设的现象和可能原因。

## Decision
Reject / Iterate / Full-scale / Online A/B。
```

---

## 13. 最终决策原则

一个可接受的 RankMixer 改进应同时满足：

- 改动对应明确且可检验的建模缺口；
- 在严格控制变量下优于基线；
- 收益不仅来自更多参数或更多训练时间；
- 具有稳定的统计证据和实际意义；
- 校准和关键分桶没有不可接受退化；
- 训练和推理成本处于可接受 Pareto 前沿；
- 实现可以稳定复现、灰度和回滚。

未满足以上条件的结构变化，即使形式新颖，也不应进入生产主干。