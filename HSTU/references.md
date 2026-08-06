# 参考文献与官方实现

本研究尽量使用论文原文、作者官方实现或机构官方页面。检索与核对日期：2026-08-06。

## 1. HSTU 原论文与官方代码

### [R1] Actions Speak Louder than Words: Trillion-Parameter Sequential Transducers for Generative Recommendations

- Jiaqi Zhai et al., ICML 2024.
- arXiv: https://arxiv.org/abs/2402.17152
- HTML: https://arxiv.org/html/2402.17152
- 作用：本目录的论文式基线来源。
- 关键点：
  - 将推荐改写为 sequential transduction；
  - 内容与动作交错输入；
  - target-aware ranking；
  - HSTU 的 `U,V,Q,K`、SiLU pointwise attention、位置/时间 bias 与 gated output；
  - jagged fused attention、Stochastic Length 和多位置监督。

### [R2] Meta Generative Recommenders — Official Repository

- 官方仓库：https://github.com/meta-recsys/generative-recommenders
- 作用：核对 HSTU/STU、jagged kernels、KV cache、公开配置和 DLRM-HSTU 生产形态。

重要文件：

- STU/HSTU layer：
  https://github.com/meta-recsys/generative-recommenders/blob/main/generative_recommenders/modules/stu.py
- DLRM-HSTU：
  https://github.com/meta-recsys/generative-recommenders/blob/main/generative_recommenders/modules/dlrm_hstu.py
- DLRMv3 configs：
  https://github.com/meta-recsys/generative-recommenders/blob/main/generative_recommenders/dlrm_v3/configs.py
- ML-1M HSTU config：
  https://github.com/meta-recsys/generative-recommenders/blob/main/configs/ml-1m/hstu-sampled-softmax-n128-final.gin
- ML-1M HSTU-large config：
  https://github.com/meta-recsys/generative-recommenders/blob/main/configs/ml-1m/hstu-sampled-softmax-n128-large-final.gin

官方 DLRMv3 配置可见一个生产化参考形态：`d_model=512`、4 heads、5 layers、target-aware causal HSTU、上下文 preprocessor、位置/时间 encoder、独立 item tower 和 multi-task head。该配置是量级参考，不应不经调参直接复制到本数据。

## 2. HSTU 扩展与超长序列

### [R3] Scaling Generative Recommendations with Context Parallelism on Hierarchical Sequential Transducers

- arXiv: https://arxiv.org/abs/2508.04711
- 作用：长序列 HSTU 的 jagged context parallelism 与分布式扩展参考。

### [R4] Bending the Scaling Law Curve in Large-Scale Recommendation Systems

- ULTRA-HSTU, Meta, 2026.
- arXiv: https://arxiv.org/abs/2602.16986
- HTML: https://arxiv.org/html/2602.16986
- 作用：生产版 event layout、超长序列和系统优化参考。
- 关键点：
  - 将 item/action 合并为单事件并对当前候选 action 置零；
  - Load-Balanced Stochastic Length；
  - Semi-Local Attention；
  - attention truncation 与 Mixture of Transducers；
  - BF16/FP8/INT4 协同和 jagged memory optimization。

本文档仍保留原始 item/action 交错版作为论文忠实基线，再把合并事件作为生产消融，避免混淆“复现原论文”和“采用后续优化”。

### [R5] Beyond the Flat Sequence: Hierarchical and Preference-Aware Generative Recommendations

- arXiv: https://arxiv.org/abs/2603.00980
- HTML: https://arxiv.org/html/2603.00980
- 作用：session hierarchy、偏好感知稀疏 attention 与“平铺序列”局限的参考。

### [R6] CMSL: Constructive Multi-Sequence Learning for Recommendation Systems

- arXiv: https://arxiv.org/abs/2606.28533
- HTML: https://arxiv.org/html/2606.28533
- 作用：多上下文序列构造、序列压缩和 CTR/CVR 融合参考。
- 注意：本目录 HM-HSTU 先使用确定性通道，并通过固定 token/FLOP 预算消融，避免把所有增益归因于更高计算量。

### [R7] TWIN: TWo-stage Interest Network for Lifelong User Behavior Modeling in CTR Prediction at Kuaishou

- arXiv: https://arxiv.org/abs/2302.02352
- 作用：长历史中先检索目标相关行为、再精细建模的两阶段思想参考。

## 3. 推广/广告场景的生成式排序

### [R8] CADET: Context-Conditioned Ads CTR Prediction With a Decoder-Only Transformer

- LinkedIn, 2026.
- arXiv: https://arxiv.org/abs/2602.11410
- HTML: https://arxiv.org/html/2602.11410
- 作用：QC-HSTU 的广告场景设计参考。
- 关键点：
  - 静态用户 prefix + impression/action 序列；
  - 多候选追加与 candidate-isolated mask；
  - session-aware delay mask，降低 train-serve leakage；
  - context-conditioned heads，应对打分时未知的后置上下文；
  - 辅助动作任务和 pairwise ranking loss；
  - packed token buffer 与 sequence chunking。
- 注意：该论文直接研究的是 Ads CTR，而本项目目标是 CVR；本目录仅迁移其结构思想，不将其 CTR 结果当作 CVR 证据。

## 4. CVR 全空间建模与偏差

### [R9] Entire Space Multi-Task Model: An Effective Approach for Estimating Post-Click Conversion Rate

- Xiao Ma et al., SIGIR 2018.
- arXiv: https://arxiv.org/abs/1804.07931
- 作用：ES-HSTU 的 `pCTCVR = pCTR × pCVR` 概率链。
- 关键点：
  - clicked-only 训练与全曝光推理造成 sample selection bias；
  - CVR 正例具有 data sparsity；
  - CTR 与 CTCVR 在全曝光空间训练；
  - 通过乘法形式约束 CVR，而不是不稳定地用除法恢复。

### [R10] ESCM²: Entire Space Counterfactual Multi-Task Model for Post-Click Conversion Rate Estimation

- arXiv: https://arxiv.org/abs/2204.05125
- 作用：提醒 ESMM 并不自动解决所有策略/选择偏差，并为可选 counterfactual regularization 提供依据。
- 使用原则：只有 propensity 或相关估计足够可靠时，才启用 IPS/DR 正则，并采用 clipping 与方差诊断。

## 5. 静态字段交互与多任务

### [R11] DCN V2: Improved Deep & Cross Network and Practical Lessons for Web-scale Learning to Rank Systems

- arXiv: https://arxiv.org/abs/2008.13535
- 作用：FI-HSTU 中低秩显式交叉网络的依据。
- 关键点：
  - 显式、有限阶 feature crosses；
  - low-rank cross layers；
  - 可选 mixture of low-rank experts；
  - 面向 web-scale learning-to-rank 的表达力/成本折中。

### [R12] FiBiNET: Combining Feature Importance and Bilinear Feature Interaction for Click-Through Rate Prediction

- arXiv: https://arxiv.org/abs/1905.09433
- 作用：语义组内 feature reweighting 与受控 bilinear interaction 的参考。
- 注意：本目录不直接对 1,234 个字段做全局 SENet；采用分域/分组 gate，防止大字段域压制 creative 小域。

### [R13] Modeling Task Relationships in Multi-task Learning with Multi-gate Mixture-of-Experts

- Google Research: https://research.google/pubs/modeling-task-relationships-in-multi-task-learning-with-multi-gate-mixture-of-experts/
- 作用：当 CTR、CVR、支付等任务出现明确梯度冲突时，作为 task-specific gate 的参考。
- 使用原则：先用共享 backbone + 低秩 task adapters；只有诊断到负迁移后再引入 MMoE。

## 6. 设计对应关系

| 本目录设计 | 主要来源 | 本项目中的改造 |
|---|---|---|
| Paper-Faithful HSTU-CVR | R1, R2 | 内容/动作历史 + 候选位置 CVR/CTR 多任务 |
| Merged Event Production HSTU | R4 | item+action 合并，候选 action 置零 |
| QC-HSTU | R8, R1 | query-conditioned bias、多候选隔离、context heads |
| ES-HSTU | R9, R10, R2 | CTR/CTCVR 全空间约束 + HSTU 历史动作监督 |
| FI-HSTU | R11, R12 | 紧凑语义组、低秩交叉、门控残差融合 |
| HM-HSTU | R4, R5, R6, R7 | 确定性多通道、session hierarchy、router、SLA |
| 大规模训练系统 | R2, R3, R4 | jagged packing、KV cache、负载均衡、混合精度 |

## 7. 文献使用边界

1. 不把公开小数据配置直接视为推广搜最优超参数；
2. 不把 CTR 论文的实验结果当成 CVR 增益证据；
3. 不把 2026 年论文中的生产收益外推到本数据；
4. 所有创新方案都需要与参数量/FLOP 控制组比较；
5. 对尚未在本业务数据验证的设计统一使用“假设”“建议”“待消融”，不作确定收益承诺。
