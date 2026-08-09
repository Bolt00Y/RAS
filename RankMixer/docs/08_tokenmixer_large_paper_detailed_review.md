# TokenMixer-Large 论文详解：从 RankMixer 基础块到十亿级工业排序主干

> 论文：**TokenMixer-Large: Scaling Up Large Ranking Models in Industrial Recommenders**  
> 作者：Yuchen Jiang, Jie Zhu, Xintian Han, Hui Lu, Kunmin Bai, Mingyu Yang, Shikang Wu, Ruihao Zhang, Wenlin Zhao, Shipeng Bai, Sijin Zhou, Huizhi Yang, Tianyi Liu, Wenda Liu, Ziyan Gong, Haoran Ding, Zheng Chai, Deping Xie, Zhe Chen, Yuchao Zheng, Peng Xu  
> 初始提交：2026-02-06；本文按 arXiv v2 阅读  
> 原文：[arXiv Abstract](https://arxiv.org/abs/2602.06563) · [HTML](https://arxiv.org/html/2602.06563v2) · [PDF](https://arxiv.org/pdf/2602.06563v2)

## 1. 论文定位

TokenMixer-Large 是 RankMixer 的直接后续工作。它并不否定 RankMixer 的硬件友好路线，而是指出原始 TokenMixer / RankMixer block 在继续扩展时存在四类结构瓶颈：

1. mixing 后的 token 语义发生变化，却仍与原位置做残差相加；
2. 网络加深后，早期层获得的有效梯度不足；
3. 原 RankMixer 的 MoE 训练与推理路由不一致，稀疏化不完整；
4. 原始工作主要验证到十亿级，尚未建立更大规模的稳定训练与部署体系。

TokenMixer-Large 的目标是：

> 保留 RankMixer 的大 GEMM 与无 attention-score 优势，同时重新设计 residual、FFN、归一化、深层监督和 MoE，使模型能够稳定扩展到在线 7B、离线 15B。

它可以被看成“RankMixer block 的第二代工程化与规模化版本”。

---

## 2. Figure 1：整体架构

<p align="center">
  <img src="https://arxiv.org/html/2602.06563v2/x1.png" width="94%" alt="TokenMixer-Large Figure 1">
</p>

*图源：TokenMixer-Large Figure 1。若图片加载失败，请打开论文 [HTML](https://arxiv.org/html/2602.06563v2) 或 [PDF 第 3 页](https://arxiv.org/pdf/2602.06563v2#page=3)。*

从输入到输出，TokenMixer-Large 包含：

```text
Semantic Group-wise Tokenizer
        + Global Token
        ↓
多个 TokenMixer-Large Blocks
        ↓
Original-layout tokens
        ↓
Pooling / task towers
```

每个 block 的关键路径是：

```text
Pre-RMSNorm
   ↓
Mixing
   ↓
Mixed-layout Per-token SwiGLU
   ↓
Reverting
   ↓
Original-layout Per-token SwiGLU
   ↓
Residual / Inter-residual
```

相较 RankMixer，最大的结构变化不是简单把 GELU 换成 SwiGLU，而是引入了 **Mixing & Reverting**：模型先在 mixed token 语义中交互，再恢复到原 token 语义后做跨层残差。

---

## 3. RankMixer 的 residual semantic misalignment

### 3.1 原始问题

RankMixer 输入：

$$\mathbf X \in \mathbb R^{B\times T\times D}.$$

Token Mixing 后：

$$\mathbf M = \operatorname{Mix}(\mathbf X).$$

虽然 $\mathbf M$ 与 $\mathbf X$ 形状相同，但第 $t$ 个 mixed token 已经由所有原 token 的某个 channel slice 拼接而成，因此：

$$\operatorname{Semantics}(\mathbf M_t) \neq \operatorname{Semantics}(\mathbf X_t).$$

原 RankMixer 仍直接执行：

$$\mathbf Y_t = \operatorname{LN}(\mathbf M_t+\mathbf X_t).$$

这在张量维度上合法，却在 token 语义上不严格对齐。两层模型可能仍能训练，但随着深度增加，跨层 residual 会不断混合“原布局语义”和“mixed 布局语义”。

### 3.2 为什么这会限制扩展

残差连接的价值通常建立在主分支与 shortcut 表达相同坐标系的前提上。若语义不对齐：

- shortcut 不再是清晰的恒等映射；
- 深层梯度沿 residual 路径传播时可能携带错位信息；
- token-specific FFN 的身份更难稳定；
- 增加层数后，优化收益可能迅速饱和。

TokenMixer-Large 把这一问题定义为 Token Semantic Alignment 问题。

---

## 4. Mixing & Reverting

### 4.1 Mixing

与 RankMixer 相同，先把每个 token 按 channel 切成若干片段，并跨 token 重新组合：

$$\mathbf M = \operatorname{Mix}(\mathbf X).$$

此时 $\mathbf M$ 位于 mixed layout。

### 4.2 Mixed-layout Per-token SwiGLU

每个 mixed token 使用独立 SwiGLU：

$$\operatorname{SwiGLU}_t(\mathbf m_t) = \left( \operatorname{SiLU}(\mathbf m_t\mathbf W_{g,t}) \odot \mathbf m_t\mathbf W_{u,t} \right) \mathbf W_{d,t}.$$

其中：

- $\mathbf W_{g,t}$ 是 gate projection；
- $\mathbf W_{u,t}$ 是 up projection；
- $\mathbf W_{d,t}$ 是 down projection；
- 不同 token 仍使用独立参数。

### 4.3 Reverting

对 mixed representation 执行逆置换，恢复原 token 布局：

$$\mathbf R = \operatorname{Revert}(\mathbf M').$$

理想情况下：

$$\operatorname{Revert} \left( \operatorname{Mix}(\mathbf X) \right) = \mathbf X.$$

在经过 mixed-layout 非线性变换后，Revert 输出不再等于输入，但每个位置重新对应原始 token 子空间。

### 4.4 Original-layout Per-token SwiGLU

恢复原布局后，再由第二组 Per-token SwiGLU 建模：

$$\mathbf O_t = \operatorname{SwiGLU}^{\mathrm{orig}}_t(\mathbf R_t).$$

最终 residual 发生在同一语义布局：

$$\mathbf X_{l+1} = \mathbf X_l + \mathbf O.$$

### 4.5 相比 RankMixer 的本质变化

RankMixer：

```text
Original layout
   -> Mix
   -> PFFN in mixed layout
   -> residual with original layout
```

TokenMixer-Large：

```text
Original layout
   -> Mix
   -> pSwiGLU in mixed layout
   -> Revert
   -> pSwiGLU in original layout
   -> semantically aligned residual
```

因此它同时建模：

- mixed token 子空间中的跨域组合；
- original token 子空间中的局部精炼；
- 语义对齐的跨层恒等路径。

---

## 5. Per-token SwiGLU

RankMixer 使用两层 GELU FFN：

$$\operatorname{FFN}(\mathbf x) = \mathbf W_2 \operatorname{GELU}(\mathbf W_1\mathbf x).$$

TokenMixer-Large 改为 SwiGLU：

$$\operatorname{SwiGLU}(\mathbf x) = \left( \operatorname{SiLU}(\mathbf x\mathbf W_g) \odot \mathbf x\mathbf W_u \right) \mathbf W_d.$$

SwiGLU 的乘法 gate 提供输入条件的通道选择，可以比纯加性 GELU FFN 更高效地表达乘法和条件交互。

论文仍然保留 per-token 参数隔离。其 4B 消融显示，把 Per-token SwiGLU 换成共享 SwiGLU，AUC 相对下降约 0.21%；换回 Per-token 普通 FFN，下降约 0.10%。这说明：

- token-specific 参数仍是核心；
- SwiGLU gate 在大模型中提供了独立增益；
- 不能把性能简单归因于参数量增加。

### 5.1 计算匹配

普通扩张倍数为 $k$ 的 FFN，主要参数为：

$$P_{\mathrm{FFN}} \approx 2kD^2.$$

若一个 block 中有两个 SwiGLU，每个 hidden size 为 $h$ ，主要参数为：

$$P_{\mathrm{2\ SwiGLU}} \approx 6Dh.$$

计算匹配时可令：

$$h \approx \frac{kD}{3}.$$

例如 $k=4$ ：

$$\begin{aligned} D=768 &:\quad h\approx1024,\\ D=1536 &:\quad h\approx2048. \end{aligned}$$

这也是迁移到当前 RankMixer 时最合理的首版对照。

---

## 6. Pre-RMSNorm 与小初始化

### 6.1 Pre-RMSNorm

原 RankMixer 采用 Post-LayerNorm。TokenMixer-Large 使用 Pre-RMSNorm：

$$\operatorname{RMSNorm}(\mathbf x) = \frac{\mathbf x} {\sqrt{\frac{1}{D}\sum_{j=1}^{D}x_j^2+\epsilon}} \odot \boldsymbol\gamma.$$

简化 block 为：

$$\mathbf X_{l+1} = \mathbf X_l + \mathcal F \left( \operatorname{RMSNorm}(\mathbf X_l) \right).$$

相较 Post-Norm，Pre-Norm 为梯度提供更直接的 residual 路径。论文指出 Post-Norm 在大规模实验中可能短暂带来小幅收益，但存在最终数值发散和 NaN 风险，因此使用 Pre-Norm 作为稳定 scaling 方案。

### 6.2 Down projection small initialization

论文只缩小 SwiGLU down projection 的初始化尺度，而不同时缩小 gate 和 up projection：

$$\operatorname{Std}(W_d) = 0.01 \times \operatorname{Std}(W_u).$$

这样 residual 分支初始输出较小，网络起点更接近恒等映射。消融中，去掉该策略相对下降约 0.03%。

这与 ReZero 类方法的思想相近：先让新分支温和接入，再逐渐学习有效残差。

---

## 7. Inter-residual 与 Auxiliary Loss

随着层数增加，仅依赖相邻层 residual 仍可能使早层梯度不足。TokenMixer-Large 增加间隔 residual：

$$\mathbf X_l = \mathcal B_l(\mathbf X_{l-1}) + \mathbf X_{l-r},$$

其中 $r$ 可以是 2 或 3 层间隔。

同时在中间层接辅助预测头：

$$\mathcal L = \mathcal L_{\mathrm{main}} + \lambda \mathcal L_{\mathrm{aux}}.$$

辅助损失的目标是：

- 给中间层提供更直接监督；
- 避免深层模型只有最后几层获得强任务梯度；
- 改善早期层的有效更新；
- 缩短超大模型收敛周期。

论文 4B 消融中，去掉 inter-residual 和 auxiliary loss，AUC 相对下降约 0.04%。该组件对两层模型不是第一优先级，但在扩展到 4 层以上时更重要。

---

## 8. Figure 2：Sparse-Pertoken MoE

<p align="center">
  <img src="https://arxiv.org/html/2602.06563v2/x2.png" width="92%" alt="TokenMixer-Large Figure 2 Sparse-Pertoken MoE">
</p>

*图源：TokenMixer-Large Figure 2。*

### 8.1 从 RankMixer MoE 到 SP-MoE

RankMixer 的 MoE 使用 Dense-Training/Sparse-Inference，训练和推理激活模式不完全一致。TokenMixer-Large 改为 Sparse-Pertoken MoE：

- 训练和推理都执行 sparse routing；
- 每个 token 保留独立专家集合；
- 每个 token 增加 always-on shared expert；
- 同时激活少量 routed experts；
- 路由权重保留原始 gate value scaling。

对于 token $t$ ：

$$\mathbf y_t = \alpha_{t,s} \operatorname{Expert}_{t,s}(\mathbf x_t) + \sum_{e\in\operatorname{TopK}(t)} \alpha_{t,e} \operatorname{Expert}_{t,e}(\mathbf x_t).$$

其中 shared expert 始终激活，routed expert 由 Top-K 选择。

### 8.2 First enlarge, then sparse

论文强调：

> 稀疏化不是用来修复一个无效的 dense architecture，而是先证明增加 dense capacity 有价值，再把已经有效的容量转化为 sparse experts。

因此正确顺序是：

```text
扩大 dense Per-token SwiGLU
        ↓
确认更大容量继续提升
        ↓
切分为多个专家
        ↓
保持较低 active FLOPs
```

### 8.3 Shared expert 与 gate scaling

消融表中：

| 变体 | 相对 AUC 变化 |
|---|---:|
| 去掉 shared expert | -0.02% |
| 去掉 gate value scaling | -0.03% |
| 去掉 down small init | -0.03% |
| SP-MoE 换回普通 Sparse MoE | -0.10% |

shared expert 提供稳定公共路径，减少所有样本都被强制路由到离散专家的风险；gate value scaling 则保留路由器置信度，不只使用归一化后的相对排序。

---

## 9. 语义分组 Tokenizer 与 Global Token

TokenMixer-Large 不再只把 tokenizer 当作无关紧要的预处理，而是明确加入：

- semantic group-wise tokenizer；
- Global Token。

Local tokens 表达不同特征组，Global Token 提供全局摘要：

$$\mathbf g = \operatorname{MLP} \left( \operatorname{Pool} \left( \{\mathbf e_i\}_{i=1}^{N} \right) \right).$$

初始 token 序列为：

$$\mathbf X_0 = [\mathbf g;\mathbf x_1;\ldots;\mathbf x_T].$$

4B 消融中，去掉 Global Token 下降约 0.02%，说明它有增益，但重要性低于 Mixing & Reverting 和 Per-token SwiGLU。

对于当前要求 $H=T$ 的 RankMixer，可用 $T-1$ 个 local tokens 加 1 个 global token，保持总 token 数不变。

---

## 10. 大规模结果

论文把模型扩展到：

- 在线广告场景约 7B；
- 在线电商场景约 4B；
- 离线实验约 15B；
- Sparse-Pertoken MoE 以较少 active parameters 保持接近 dense 模型效果。

电商数据上的代表结果如下：

| 模型 | 主要参数量 | 相对 AUC 增益 |
|---|---:|---:|
| RankMixer | 约 90M | +0.84% |
| TokenMixer-Large | 约 500M | +0.94% |
| TokenMixer-Large | 约 4B | +1.14% |
| TokenMixer-Large | 约 7B | +1.20% |
| 4B SP-MoE | 约 4.6B 总参数，2.3B active | +1.14% |

这些结果体现：

- 升级后的 block 在相近规模上优于 RankMixer；
- dense scaling 在较大范围内仍有效；
- SP-MoE 能在较少 active capacity 下接近 dense 4B；
- 超大模型需要更长训练周期。

---

## 11. 收敛周期表

论文给出的收敛现象非常重要：

| 规模 | 训练周期 | 相对 AUC 增益 |
|---|---:|---:|
| 约 90M | 14 天 | +0.94% |
| 约 500M | 30 天 | +0.62% |
| 约 2.3B active | 30 天 | +0.41% |
| 约 2.3B active | 60 天 | +0.70% |

大模型在固定训练天数下可能暂时落后，但继续训练后仍显著提升。这说明评估 scaling 时至少要同时看：

```text
相同 seen examples
相同 wall-clock
各自训练至近似收敛
```

不过，训练周期更长不能成为无限延长失败实验的理由。需要结合学习曲线斜率、训练 AUC、验证 AUC、参数 update ratio 和 effective rank 判断模型是否仍在有效学习。

---

## 12. 关键 4B 消融表

| 消融 | 相对 AUC 变化 | 说明 |
|---|---:|---|
| 去掉 Global Token | -0.02% | 全局上下文有小幅贡献 |
| 去掉 Mixing & Reverting | -0.27% | 最大结构贡献之一 |
| 去掉 residual | -0.15% | 稳定优化不可缺少 |
| 去掉 inter-residual 与 auxiliary loss | -0.04% | 深层监督有增益 |
| Per-token SwiGLU 改为 shared SwiGLU | -0.21% | token-specific 参数很关键 |
| Per-token SwiGLU 改为 Per-token FFN | -0.10% | gated FFN 有独立价值 |

该表给出的优先级非常清楚：

```text
Mixing & Reverting
≈ token-specific parameterization
> residual
> SwiGLU
> deep supervision
> Global Token
```

---

## 13. 工程优化

TokenMixer-Large 的规模能力不只来自数学结构，还依赖系统协同：

- Grouped GEMM 组织 per-token / per-expert 计算；
- 自定义 fused operators；
- FP8 训练与推理；
- Token Parallel；
- 减少 DCN、LHUC 等碎片化模块；
- 提升大矩阵计算占比；
- 在广告 backbone 中把 MFU 提升到约 60%。

论文提出“pure model”方向：当统一 backbone 足够大时，部分传统外挂模块的边际收益会下降。但这属于论文自身数据上的规模现象，不能直接推导所有业务都应删除 DCN 或 SENet。

---

## 14. 线上结果

论文报告 TokenMixer-Large 已部署于字节跳动多个推荐与广告场景，代表结果包括：

- 电商订单量约 +1.66%；
- 预览支付 GMV per capita 约 +2.98%；
- 广告 ADSS 约 +2.0%；
- 直播收入约 +1.4%。

线上收益说明 upgraded backbone 不仅改善离线 AUC，也能在极大规模和严格延迟约束下部署。

---

## 15. 与 RankMixer 的逐项对比

| 维度 | RankMixer | TokenMixer-Large |
|---|---|---|
| Tokenizer | Autosplit / semantic order | semantic group-wise + Global Token |
| Mixing | 固定 reshape mixing | 固定 mixing + explicit reverting |
| Mixed layout FFN | Per-token GELU FFN | Per-token SwiGLU |
| Original layout FFN | 无 | 第二组 Per-token SwiGLU |
| Residual | mixed token 与原位置直接相加 | 恢复原语义后残差 |
| Norm | Post-LayerNorm | Pre-RMSNorm |
| 深层优化 | 相邻 residual | inter-residual + auxiliary loss |
| MoE | DTSI Sparse MoE | train/serve 一致的 SP-MoE |
| 初始化 | 常规 | down projection small init |
| 已验证规模 | 在线约 1B | 在线约 7B，离线约 15B |
| 主要目标 | 建立硬件友好 scaling 主干 | 修复 block 缺陷并扩展到更大规模 |

---

## 16. 对当前电商推广搜 CVR 的意义

当前 RankMixer 大模型为：

$$T=32, \qquad D=1536, \qquad L=2.$$

若原始大模型仍弱于 Base，TokenMixer-Large 提供的最合理消融顺序是：

1. 只加入 Mixing & Reverting，保持 FFN 类型和主要计算量不变；
2. 换成计算匹配的 Per-token SwiGLU；
3. 把 Post-LN 改为 Pre-RMSNorm；
4. 对 down projection 使用 0.01 small initialization；
5. 两层版本获益后，再扩展到 4 层；
6. 只有 dense hidden enlargement 已经证明有效后，再研究 SP-MoE。

不建议直接复制完整 4B / 7B 配置，因为当前业务还存在：

- Base 的层级 SENet 与 DCNv2 明显更强；
- 当前只有静态 user、item、creative 输入；
- Mean Pooling 可能仍是主要瓶颈；
- 大模型收敛周期和优化超参数尚未验证；
- 结构收益必须与参数和 FLOPs 匹配对照。

最有科学价值的首组实验是：

```text
BL-0  原始大 RankMixer
BL-1  + Mixing & Reverting
BL-2  + compute-matched Per-token SwiGLU
BL-3  + Pre-RMSNorm
BL-4  + down projection 0.01 initialization
```

---

## 17. 局限与复现注意事项

1. 论文结果来自字节跳动内部数据和高度优化的系统栈；
2. 部分超大模型收益依赖更长训练周期；
3. 自定义 grouped GEMM、FP8 和 token parallel 可能是规模效率的重要组成部分；
4. “pure model 可替代 DCN”是规模和数据依赖结论，不是普遍定理；
5. Global Token 的构造细节、完整 token 分组和部分超参数没有全部公开；
6. 原论文中的相对 AUC 数值不能直接映射为当前业务的绝对 AUC 预期；
7. SP-MoE 的通信、路由均衡和线上 P99 成本必须单独测量。

---

## 18. 一句话总结

TokenMixer-Large 的核心贡献可以概括为：

> 在 RankMixer 的硬件友好 Token Mixing 基础上，通过 Mixing & Reverting 恢复 token 语义对齐，以 Per-token SwiGLU、Pre-RMSNorm、深层 residual 和 Sparse-Pertoken MoE 建立可稳定扩展到十亿级以上的第二代排序主干。

它是 RankMixer 发展谱系中“修 block、加深度、做稀疏化和极致工程优化”的分支。
