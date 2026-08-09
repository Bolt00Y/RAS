# RankMixer 论文详解：面向工业推荐排序的硬件友好扩展架构

> 论文：**RankMixer: Scaling Up Ranking Models in Industrial Recommenders**  
> 作者：Jie Zhu, Zhifang Fan, Xiaoxie Zhu, Yuchen Jiang, Hangyu Wang, Xintian Han, Haoran Ding, Xinmin Wang, Wenlin Zhao, Zhen Gong, Huizhi Yang, Zheng Chai, Zhe Chen, Yuchao Zheng, Qiwei Chen, Feng Zhang, Xun Zhou, Peng Xu, Xiao Yang, Di Wu, Zuotao Liu  
> 初始提交：2025-07-21；本文按 arXiv v3 阅读  
> 原文：[arXiv Abstract](https://arxiv.org/abs/2507.15551) · [HTML](https://arxiv.org/html/2507.15551v3) · [PDF](https://arxiv.org/pdf/2507.15551v3)

## 1. 论文定位

RankMixer 解决的并不是“怎样设计一个更复杂的特征交叉算子”，而是一个更工业化的问题：

> 在推荐排序模型中，如何把模型规模从千万级扩展到十亿级，同时保持高吞吐、低延迟和较高 GPU 利用率？

传统排序模型通常由多个异构模块拼接而成，例如 DCN、FM、attention、LHUC、MLP 和各种业务交叉塔。这些模块可能在较小规模上有效，但容易产生：

- 大量小算子和不规则访存；
- 不同模块之间的串行依赖；
- 低 Model FLOPs Utilization；
- 难以形成统一、可预测的 scaling 路径；
- 参数增加后，推理延迟增长过快。

RankMixer 的核心思想是保留 Transformer 中最适合 GPU 的规则矩阵计算，但把二次复杂度的 Self-Attention 替换为无参数的 Token Mixing，再通过 Per-token FFN 保留不同特征子空间的独立建模能力。

---

## 2. Figure 1：整体架构

<p align="center">
  <img src="https://arxiv.org/html/2507.15551v3/x1.png" width="92%" alt="RankMixer Figure 1">
</p>

*图源：RankMixer Figure 1。若图片无法加载，请打开论文 [HTML](https://arxiv.org/html/2507.15551v3) 或 [PDF 第 3 页](https://arxiv.org/pdf/2507.15551v3#page=3)。*

Figure 1 可以分成五个阶段：

```text
原始特征 embedding
        ↓
语义排序与连续拼接
        ↓
Autosplit + 独立投影，形成 T 个 tokens
        ↓
L 个 RankMixer blocks
        ↓
Mean Pooling + 多任务预测塔
```

每个 RankMixer block 又由两部分组成：

```text
Multi-head Token Mixing
        ↓
Per-token FFN
```

两部分都带残差与 LayerNorm。

---

## 3. 输入如何变成 token

### 3.1 原始 embedding 拼接

设共有 $N$ 个输入特征，每个特征经过 embedding 后得到向量 $\mathbf e_i$ 。RankMixer 首先按预先确定的特征顺序进行拼接：

$$\mathbf e_{\mathrm{input}} = [\mathbf e_1;\mathbf e_2;\ldots;\mathbf e_N] \in\mathbb R^F.$$

论文强调可以利用业务知识把语义相关的特征排列在相邻位置，但后续切分发生在拼接后的连续向量上，并不要求每个切分边界严格对齐字段边界。

### 3.2 Autosplit

把长度为 $F$ 的连续向量切成 $T$ 个 segment。第 $t$ 个 segment 记为 $\mathbf s_t$ ：

$$\mathbf s_t = \mathbf e_{\mathrm{input}}[a_t:b_t].$$

每个 segment 使用独立投影矩阵映射到统一维度 $D$ ：

$$\mathbf x_t = \mathbf W_t\mathbf s_t+\mathbf b_t, \qquad \mathbf x_t\in\mathbb R^D.$$

最后堆叠得到：

$$\mathbf X_0 = [\mathbf x_1;\mathbf x_2;\ldots;\mathbf x_T] \in\mathbb R^{T\times D}.$$

对 batch 输入：

$$\mathbf X_0 \in \mathbb R^{B\times T\times D}.$$

这一设计的重要性质是：

- token 数 $T$ 与隐藏维度 $D$ 都可以独立扩展；
- 每个 token 的输入 projector 独立；
- 规则的大矩阵投影适合 GPU；
- 不需要在千级字段上执行二次复杂度 attention。

---

## 4. Multi-head Token Mixing

### 4.1 操作过程

输入为：

$$\mathbf X \in \mathbb R^{B\times T\times D}.$$

每个 token 沿 channel 维切成 $H$ 个 head，每个 head 宽度为 $D/H$ ：

$$\mathbf X \rightarrow \mathbb R^{B\times T\times H\times(D/H)}.$$

然后交换 token 轴与 head 轴：

$$\mathbb R^{B\times T\times H\times(D/H)} \rightarrow \mathbb R^{B\times H\times T\times(D/H)}.$$

最后把后两维拼接：

$$\operatorname{Mix}(\mathbf X) \in \mathbb R^{B\times H\times(TD/H)}.$$

论文设置 $H=T$ ，于是输出重新变为：

$$\operatorname{Mix}(\mathbf X) \in \mathbb R^{B\times T\times D}.$$

因此可以与输入做逐位置残差相加。

### 4.2 直观解释

假设有 4 个 token，每个 token 分成 4 个 channel head：

```text
Token 1 = [h11, h12, h13, h14]
Token 2 = [h21, h22, h23, h24]
Token 3 = [h31, h32, h33, h34]
Token 4 = [h41, h42, h43, h44]
```

Mixing 后：

```text
New Token 1 = [h11, h21, h31, h41]
New Token 2 = [h12, h22, h32, h42]
New Token 3 = [h13, h23, h33, h43]
New Token 4 = [h14, h24, h34, h44]
```

每个新 token 都包含所有原 token 的一个 channel 子空间，因此实现了跨 token 信息交换。

### 4.3 与 Self-Attention 的区别

Self-Attention 需要根据输入动态计算 $QK^\top$ ，复杂度包含 $T^2D$ 项；RankMixer 的 Token Mixing 只是 reshape、transpose 和 concat：

- 无可学习参数；
- 无 softmax；
- 不生成 $T\times T$ attention score；
- mixing pattern 对所有样本固定；
- 算子规则，易于 fusion 和高效实现。

代价是它不能像 attention 一样针对不同样本动态改变 token-to-token 权重。

---

## 5. Per-token FFN

Token Mixing 负责信息交换，真正的非线性建模主要由 Per-token FFN 完成。

对于第 $t$ 个 token：

$$\operatorname{PFFN}_t(\mathbf x_t) = \mathbf W_{2,t} \operatorname{GELU} ( \mathbf W_{1,t}\mathbf x_t+\mathbf b_{1,t} ) + \mathbf b_{2,t}.$$

每个 token 拥有独立参数：

$$\operatorname{PFFN}_i \neq \operatorname{PFFN}_j, \qquad i\neq j.$$

这使模型同时具备：

- Token Mixing 提供跨子空间交互；
- Per-token FFN 保留不同 token 的独立表达能力；
- 参数量可以随 $T$ 、 $D$ 、FFN 扩张倍数线性或二次扩展；
- 各 token 的大 GEMM 可以组织为 grouped GEMM。

若 FFN 扩张倍数为 $k$ ，单层 dense PFFN 的主要参数量近似为：

$$P_{\mathrm{PFFN/layer}} \approx 2kTD^2.$$

$L$ 层总参数近似为：

$$P_{\mathrm{PFFN}} \approx 2kLTD^2.$$

前向与反向主要计算量近似随 $LTD^2$ 增长，因此论文将 $T$ 、 $D$ 、 $L$ 和 expert 数视为主要 scaling 轴。

---

## 6. RankMixer block 与输出

原论文采用 Post-LayerNorm 风格。简化表达为：

$$\mathbf S_l = \operatorname{LN} \left( \operatorname{Mix}(\mathbf X_{l-1}) + \mathbf X_{l-1} \right),$$

$$\mathbf X_l = \operatorname{LN} \left( \operatorname{PFFN}(\mathbf S_l) + \mathbf S_l \right).$$

最后使用 Mean Pooling：

$$\mathbf h = \frac{1}{T} \sum_{t=1}^{T} \mathbf x_{L,t}.$$

再送入一个或多个任务塔。

Mean Pooling 的优势是简单、稳定和硬件友好；它的局限是默认所有 token 具有相同先验权重，这后来成为 RankUp、MixFormer 以及实际业务改造中重点讨论的问题之一。

---

## 7. Sparse MoE 与 DTSI

RankMixer 进一步把 Per-token FFN 扩展为 Sparse MoE。每个 token 拥有自己的专家集合，路由器为样本选择少量专家。

一般形式为：

$$\mathbf y_t = \sum_{e=1}^{E} g_{t,e}(\mathbf x_t) \operatorname{Expert}_{t,e}(\mathbf x_t).$$

论文提出 Dense-Training/Sparse-Inference 思路：训练早期让更多专家获得梯度，缓解路由不充分和专家失衡；推理时仅激活少量专家，以控制 FLOPs 和延迟。

这个方向的核心目标不是单纯提高上限，而是提高参数量与激活计算量之间的 ROI：

- 总参数量可以随 expert 数增长；
- 单样本激活 FLOPs 增长较慢；
- per-token expert 保持局部参数隔离；
- 需要额外处理负载均衡、通信和 grouped GEMM 效率。

TokenMixer-Large 后续将这一思路升级为 Sparse-Pertoken MoE，并取消 Dense-Training/Sparse-Inference 的不一致。

---

## 8. Figure 2：Scaling 结果

<p align="center">
  <img src="https://arxiv.org/html/2507.15551v3/x2.png" width="88%" alt="RankMixer Figure 2 scaling curves">
</p>

*图源：RankMixer Figure 2。*

论文展示了模型效果随参数量和 FLOPs 增长而持续改善。核心结论不是某个单点模型赢了多少，而是 RankMixer 能在较大范围内形成相对平滑的 scaling 曲线。

论文中的两个代表配置为：

| 模型 | Token 数 | Hidden | Block 数 | 参数量级 |
|---|---:|---:|---:|---:|
| RankMixer-100M | 16 | 768 | 2 | 约 107M |
| RankMixer-1B | 32 | 1536 | 2 | 约 1.1B |

在主要离线任务上，论文报告的相对增益包括：

| 模型 | Finish AUC | Finish UAUC | Skip AUC | Skip UAUC |
|---|---:|---:|---:|---:|
| RankMixer-100M | +0.64% | +0.72% | +0.86% | +1.33% |
| RankMixer-1B | +0.95% | +1.22% | +1.25% | +1.82% |

这些数值来自论文内部工业数据，不能直接外推到其他业务，但证明了该架构具备明确的规模增益趋势。

---

## 9. 关键消融表

### 9.1 主结构消融

论文对 1B 模型的主要消融结论如下：

| 变体 | 相对 AUC 变化 | 说明 |
|---|---:|---|
| 去掉 skip connection | -0.07% | 残差对优化和信息保留必要 |
| 去掉 Token Mixing | -0.50% | 跨 token 信息交换是最关键组件 |
| 去掉 LayerNorm | -0.05% | 归一化仍有稳定作用 |
| Per-token FFN 改为共享 FFN | -0.31% | token-specific 参数隔离具有明显价值 |

这张表说明，RankMixer 的能力并不是只来自“大 FFN”：Token Mixing 与 Per-token 参数隔离缺一不可。

### 9.2 Routing 方式消融

| Routing 结构 | 相对 AUC 变化 | 参数 / FLOPs 影响 |
|---|---:|---|
| RankMixer | 基准 | 基准 |
| All-Concat | -0.18% | 失去 token 分区结构 |
| All-Share | -0.25% | 失去 token-specific FFN |
| Self-Attention | -0.03% | 参数约 +16%，FLOPs 约 +71.8% |

Self-Attention 略低于 RankMixer，并显著增加计算，支持论文“固定 mixing 在工业 ranking 中具有更高 ROI”的结论。

---

## 10. 线上结果与系统效率

论文最有工业意义的结果是：参数量大幅增加时，推理延迟几乎不变。

| 指标 | 原系统 | RankMixer-1B |
|---|---:|---:|
| Dense 参数 | 15.8M | 约 1.1B |
| 主要 FLOPs | 约 107G | 约 2106G |
| MFU | 约 4.47% | 约 44.57% |
| 推理延迟 | 约 14.5 ms | 约 14.3 ms |

参数约扩大 70 倍、FLOPs 约扩大 20.7 倍，但通过大 GEMM、kernel fusion 和减少异构模块，GPU 利用率显著提高。

论文报告的代表性线上收益包括：

- 推荐场景用户活跃天数约 +0.2908%；
- 应用使用时长约 +1.0836%；
- 广告场景 AUC 约 +0.73%；
- 广告价值指标约 +3.90%。

这些结果说明，RankMixer 的主要贡献是建立了一条可以真实部署的“大 ranking model”路线，而不只是离线 AUC 改进。

---

## 11. RankMixer 相对传统方法的变化

| 维度 | 传统异构排序网络 | Transformer / Attention | RankMixer |
|---|---|---|---|
| 特征交互 | 多种人工模块拼接 | 动态全局 attention | 固定 Token Mixing + PFFN |
| 复杂度 | 模块依赖强，难统一 | token 数二次复杂度 | 主要为规则 reshape 与 GEMM |
| 参数隔离 | 依赖模块设计 | 常用共享 FFN | 每 token 独立 FFN |
| 硬件效率 | 小算子多，MFU 低 | attention 与长 token 成本高 | 大 GEMM、易 fusion |
| Scaling 轴 | 不统一 | 宽度、深度、head | token、宽度、深度、expert |
| 动态交互 | 由人工模块决定 | 强 | mixing 本身固定，动态性来自 FFN |

---

## 12. 论文的主要局限

### 12.1 Mixing pattern 固定

Token Mixing 对所有样本使用同一个置换模式，不能根据当前 user-item 请求动态调整 token 交互强度。UniMixer 后续正是针对这一点，把固定 mixing 参数化并可学习化。

### 12.2 强制 $H=T$

原始结构依靠 $H=T$ 保证 mixing 前后形状一致，这限制了 token 数、head 数和 hidden 维度的组合。TokenMixer-Large 的 Mixing & Reverting 与 UniMixer 都尝试解除这一约束。

### 12.3 Residual 语义错位

Mixing 后的第 $t$ 个 token 已经不再对应原始第 $t$ 个特征子空间，却仍与原位置残差相加。TokenMixer-Large 将其定义为 residual semantic misalignment，并引入 Reverting 修复。

### 12.4 Mean Pooling 可能稀释强 token

所有 token 等权聚合。在 user、item、creative 信息量严重不均衡时，关键 token 可能被平均稀释。

### 12.5 参数量不等于有效表示容量

RankUp 发现 Token Mixer 后 effective rank 上升，而 Per-token FFN 后可能再次下降；继续增参并不保证表示维度被充分利用。

### 12.6 静态特征与行为序列仍然分离

RankMixer 主要解决非序列特征交互。若另接一个序列模型，dense feature interaction 与 sequence modeling 仍由不同模块承担。MixFormer 后续提出统一 backbone 进行 co-scaling。

---

## 13. 对当前推广搜 CVR 模型的对应关系

当前小模型：

$$T=16, \qquad D=768, \qquad L=2.$$

当前大模型：

$$T=32, \qquad D=1536, \qquad L=2.$$

两者都与论文代表配置高度一致。需要特别注意：

1. 论文证明的是架构在其数据和工程栈上的 scaling 能力，不保证在任何特征体系上自动优于 SENet + DCNv2；
2. 当前 Base 具有层级条件 SENet、显式交叉和较强 MLP head，属于不同归纳偏置；
3. 当大模型仍不如 Base 时，应拆分 token 数、width、PFFN 容量、pooling 与输入前端，而不是继续盲目增参；
4. RankMixer 最值得保留的价值可能是高效 residual branch，而不一定是完全替换强 Base；
5. 后续 TokenMixer-Large、RankUp、MixFormer 和 UniMixer 分别从 block、表示秩、序列统一与可学习 mixing 四个不同方向补足 RankMixer。

---

## 14. 一句话总结

RankMixer 的核心创新可以概括为：

> 用无参数、硬件友好的跨 token 重排替代 Self-Attention，用独立 Per-token FFN 承担高容量非线性建模，从而把工业排序模型变成一条可统一扩展、可高 MFU 部署的规则计算主干。

它建立了方法谱系的起点，但后续工作表明，固定 mixing、残差语义、表示有效秩、sequence-dense 分离和任务聚合仍然存在改进空间。
