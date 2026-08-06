# 0. 问题定义、论文对照与设计边界

## 0.1 输入与规模

上游 sparse embedding 已输出：

```text
user      : [B, 385, 17] -> flatten -> [B,  6,545]
item      : [B, 835, 17] -> flatten -> [B, 14,195]
creative  : [B,  14, 17] -> flatten -> [B,    238]
concat    :                         -> [B, 20,978]
```

总字段数为 `1,234`。Item 字段占约 67.7%，User 占约 31.2%，Creative 占约 1.1%。

日训练样本约 5.5 亿。OneTrans v3 的工业数据日均曝光为 1.182 亿，因此你的每日样本量约为其 4.65 倍；但总训练天数、用户覆盖、标签成熟度和累计样本未知，不能据此判断总体有效数据一定更多。

若 global batch 为：

| Global batch | 每 5.5 亿样本约需 optimizer steps |
|---:|---:|
| 2,048 | 268,555 |
| 16,384 | 33,569 |
| 32,768 | 16,785 |
| 65,536 | 8,393 |

因此数据量足以支持 `d=256/384`、6–8 层的 dense backbone，真正的瓶颈更可能是标签稀疏、样本空间、特征组织和在线延迟。

## 0.2 OneTrans 原论文的关键结构

[OneTrans](https://arxiv.org/abs/2510.26104) 将输入分为：

- S-tokens：多行为序列事件；同类 token 共享 Q/K/V 与 FFN；
- NS-tokens：用户、候选、上下文等非序列特征；每个 token 使用独立 Q/K/V 与 FFN。

统一输入顺序为：

```text
[S-tokens ; NS-tokens]
```

核心是 Pre-RMSNorm causal Transformer。NS-token 位于 S-token 之后，因此能够读取全部历史及其之前的 NS-token。原论文还使用：

- Auto-Split 或 group-wise NS tokenizer；
- timestamp-aware 多行为融合；
- pyramid stack：逐层减少 S-token query，K/V 仍覆盖完整前缀；
- cross-candidate / cross-request KV cache；
- FlashAttention-2、BF16/FP16、activation recomputation。

论文的小模型设置为：

```text
layers=6
d_model=256
heads=4
dense params≈91M
S-query length: 1190 -> 12
NS tokens=12
```

论文的消融显示：

- Auto-Split 优于人工 group-wise tokenizer；
- timestamp-aware fusion 优于无时间融合；
- NS token-specific QKV/FFN 优于全 token 参数共享；
- causal 与 full attention 效果接近，但 causal 才能使用标准 KV cache；
- pyramid 基本不损失效果，却显著减少计算。

## 0.3 当前输入为何不能“严格复现” OneTrans

当前只给出 `user/item/creative` 三组静态 embedding，没有任何行为事件序列、时间戳或行为类型。因此：

1. `L_S=0`，所有输入只能作为 NS-tokens；
2. 不存在 S-token 参数共享；
3. 原论文的 timestamp-aware merge 无法实现；
4. 原论文只裁剪 S-token query，因此原始 pyramid 不适用；
5. 原始 sequence KV cache 不存在；
6. OneTrans“统一序列建模与特征交互”的主要优势只能复现后半部分，即异构特征交互。

这不是数据量不足，而是输入模态不同。本文档将 P0 定义为“在静态约束下最贴近原论文”，而不是声称严格复现完整 OneTrans。

## 0.4 CVR 样本空间必须先确定

### 情况 A：只有点击样本

训练目标是：

```text
pCVR = P(convert=1 | click=1, x)
```

可直接用 clicked-only BCE，但线上通常对全曝光候选推理，会存在训练/推理样本空间不一致。

### 情况 B：有全部曝光样本，且同时有 click 与 convert 标签

优先使用整个曝光空间的多任务建模。ESMM 的约束为：

```text
pCTCVR = pCTR × pCVR
```

在所有曝光上训练 CTR 和 CTCVR，可以减轻 clicked-only CVR 的 sample selection bias 与数据稀疏。创新方案 P2 专门把该约束内生到 OneTrans task token 中。

### 情况 C：convert 标签有延迟

必须使用已成熟的转化窗口；训练日与标签截断日应错开。否则结构改进可能只是在拟合 label censoring。

## 0.5 相关一手文献及本设计吸收的内容

1. [OneTrans, WWW 2026](https://arxiv.org/abs/2510.26104)  
   统一 tokenizer、mixed parameterization、causal backbone、pyramid 与 KV cache。

2. [RankMixer, CIKM 2025](https://arxiv.org/abs/2507.15551)  
   语义 token、per-token FFN、硬件友好 scaling；作为现有基线。

3. [ESMM, SIGIR 2018](https://arxiv.org/abs/1804.07931)  
   通过 `pCTCVR=pCTR×pCVR` 在整个曝光空间学习，解决 CVR 的样本选择偏差和数据稀疏。

4. [OneRank, KDD 2026](https://arxiv.org/abs/2606.16838)  
   task token、任务互不可见、级联任务信息流、cross-task gradient detachment 与动态匹配打分。

5. [TMallGS, KDD 2026](https://arxiv.org/abs/2607.13398)  
   推广搜/电商搜索中的层次化 tokenization、field saliency、per-field QKV、FiLM late fusion、context-aware bias net。

6. [UniFormer, KDD 2025](https://arxiv.org/abs/2606.27058)  
   语义 tokenization、user-item 解耦、request 级复用、feature/task space 分离和 multi-view FFN。

## 0.6 统一设计原则

- 不把 17 维字段 embedding 从中间切开；
- 任何 token 顺序都必须有业务含义或可学习含义；
- User/request 前缀应尽量与候选解耦，支持 request-level cache；
- CVR 优先解决样本空间与标签链路，再比较 backbone；
- 强匹配、统计率、position/bid 等高频信号不应全部交给深层 attention 平滑；
- 所有方案都需要和 RankMixer 在相同训练样本、标签窗口、dense 参数/FLOPs 下比较；
- 先做可解释、可回滚的小模型，再扩宽或加深。
