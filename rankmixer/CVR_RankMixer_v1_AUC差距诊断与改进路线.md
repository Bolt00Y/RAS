# CVR RankMixer v1 的 0.003 AUC 差距：基于真实 SENet+DCNM Base 的修正版诊断

> 真实 Base：<code>code/cvr_bn_senet_dcnm.py</code>。<br>
> RankMixer v1：<code>code/cvr_bn_rankmixer_v1.py</code>。<br>
> Base 运行参数：<code>set-x.args.resolved.txt</code>、<code>0721args.txt</code>。<br>
> RankMixer 运行参数：<code>code/set-xcal.txt</code>。<br>
> 完整模块方案：[CVR_RankMixer_全量改进方案_按模块分类.md](CVR_RankMixer_全量改进方案_按模块分类.md)

本文替换了旧版错误证据链。<code>cvr_fst_last_norpy.py</code> 不是 Base，只是一次类似训练尝试；
其中的 dense、DIN、gattr、last 等结构不能再用于解释 Base 0.865 与 RankMixer v1 0.862 的差距。

---

## 1. 执行结论

### 1.1 这次 0.003 比旧文档认为的更接近“结构差异”

真实 Base 与 RankMixer v1 实际共享：

- Common、Item、Creative 三桶输入；
- 17 维 embedding 与相同 feature version；
- 三桶独立 BN；
- first-only 监督；
- wide、last、multi-task、delay 全部关闭；
- <code>flood_adam</code>、$2\times10^{-5}$ 学习率、batch 2048；
- 概率式 BCE、logit clip 和未真正使用的 grad clipping。

所以不能再说 v1 因为少了 dense、DIN、gattr 或 last 才低 0.003。

### 1.2 真实结构差异

~~~mermaid
flowchart LR
    X["相同三桶 + 相同 BN"] --> B["真实 Base"]
    X --> R["RankMixer v1"]

    B --> S["层级 SENet"]
    S --> D["2× DCNM<br/>20978→500→20978"]
    D --> M["MLP<br/>20978→2048→2048→256"]
    M --> HB["first head"]

    R --> T["错误标量切片<br/>1311×15+1313"]
    T --> P["16×768 Token"]
    P --> RM["2× hybrid RM"]
    RM --> MEAN["Mean readout"]
    MEAN --> HR["first head"]
~~~

根因优先级：

| 优先级 | 根因 | 证据强度 |
|---:|---|---|
| P0 | v1 没有 Base 实际开启的层级 SENet | 源码 + resolved 参数直接证明 |
| P0 | tokenizer 切断所有内部字段边界 | 维度计算直接证明 |
| P0/P1 | v1 删除 Base 两层 DCNM 全局乘性交叉 | 源码直接证明 |
| P1 | mean-only readout 丢失 token 身份 | 源码直接证明 |
| P1 | hybrid block 拓扑未经严格对照 | 源码直接证明 |
| P1 | 167.3M RM vs 90.3M Base 的参数与收敛不公平 | 源码估算；图变量需复核 |
| P1 | checkpoint 成熟度可能不对称 | scope 支持；需 restore 日志 |

### 1.3 最推荐的第一组实验

~~~text
R0 当前 v1
R1 R0 + exact Base SENet
R2 R1 + 完整字段 tokenizer
R3 R2 + mean + low-rank flatten
R4 R3 的 k=2 参数匹配版
R5 保留 Base SENet+DCNM，只用 RM 替换 Base MLP
~~~

这组实验比直接换 SwiGLU 或增加 DIN 更能解释 0.003。

---

## 2. 为什么旧 Base 判断错了

### 2.1 类代码、运行参数与其他实验不能混为一谈

Base 类默认打开 last/wide/multi-task，但真实 resolved 参数明确关闭：

~~~text
enable_wide_cvr = false
enable_mlt_loss = false
enable_last_cvr = false
enable_delay_train_mode = false
~~~

Base 类还会打印 dense/seq/gattr 配置规模，但 <code>model_fn()</code> 在 920 行明确标注：

~~~text
冷启动精简版：dense/seq/gattr/din 均为 0，仅保留 common/item/creative
~~~

随后 938-953 行确实只收集三桶。

### 2.2 cvr_fst_last_norpy.py 的正确定位

该文件实现了 dense、DIN、gattr 和更丰富任务，可作为未来研究参考，但：

- 不能证明真实 Base 使用这些输入；
- 不能证明 0.865 来自这些模块；
- 不能把“恢复这些模块”称为 parity；
- 任何增益都必须当作新增能力单独消融。

---

## 3. 真实运行配置对照

| 配置 | 真实 Base | RankMixer v1 | 是否相同 |
|---|---|---|---|
| Model | SENet+DCNM+MLP | RankMixer v1 | 否 |
| Feature version | v10_base_cold | v10_base_cold | 是 |
| Input | U/I/C | U/I/C | 是 |
| Embedding | 17 | 17 | 是 |
| Input BN | true | true | 是 |
| SENet | true + SENet BN | false | **否** |
| DCNM | 2 层，bottleneck 500 | 无 | **否** |
| Main tower | 2048,2048,256 | T16,D768,L2,k4 | **否** |
| Head task | first | first | 是 |
| last/wide/mlt/delay | false | false | 是 |
| Optimizer | flood_adam | flood_adam | 是 |
| LR | 2e-5 | 2e-5 | 是 |
| Batch | 2048 | 2048 | 是 |
| Checkpoint | 提供历史 checkpoint | 提供历史 checkpoint | 路径模式相似，加载量待核验 |

Base 参数证据：<code>set-x.args.resolved.txt:27-76</code>。<br>
RankMixer 参数证据：<code>set-xcal.txt:248-298</code>。

---

## 4. 真实 Base 逐层拆解

### 4.1 输入

相同 feature version 下：

$$
U:6545=385\times17,
\quad
I:14195=835\times17,
\quad
C:238=14\times17.
$$

三桶分别经过 BN，再进入 SENet。

### 4.2 层级 SENet

SENet 先对每字段 17 维取均值，再生成 $(0,2)$ gate：

$$
g_U=f(s_U),
\quad
g_I=f([s_U,s_I]),
\quad
g_C=f([s_U,s_I,s_C]).
$$

这使 Item 权重依赖用户，Creative 权重依赖全部三域。v1 完全缺少这一步。

### 4.3 两层 DCNM

每层：

$$
x_{l+1}
=\mathrm{LN}
\left(
x_0\odot W_{l,2}W_{l,1}x_l+x_l
\right),
$$

其中 $W_{l,1}:20978\to500$，$W_{l,2}:500\to20978$。

Base 在任何强压缩前就完成两次全局条件乘性交叉。

### 4.4 三层 MLP

DCNM 输出进入：

~~~text
20978 -> 2048 -> 2048 -> 256 -> 1
~~~

运行时 MLP BN 打开，激活为 GELU。只构造 first head，因为 last/wide/mlt 均关闭。

---

## 5. RankMixer v1 逐层拆解

### 5.1 相同输入，不同前处理

v1 也只收集三桶并分别 BN，但没有 SENet，直接拼接。

### 5.2 所有内部切分边界都错误

$$
20978//16=1311,
\qquad
1311\bmod17=2.
$$

第 $k$ 个边界相对字段边界偏移 $2k\bmod17$；$k=1,\dots,15$ 时没有一个为 0。

因此：

- 每个 token 包含字段残片；
- token 5 跨 Common/Item；
- token 16 混合 Item/Creative；
- PFFN 独立参数无法形成稳定业务职责。

### 5.3 过早压缩

Tokenizer 在任何可学习全局交叉前把 20978 维压为：

$$
16\times768=12288,
$$

总宽度减少约 41.4%。Base 则先在 20978 维完成 SENet+DCNM，再压到 2048。

### 5.4 Hybrid block 和 mean

v1 每层三次 LN，并把 fixed-mixed 坐标与原坐标相加；最后只做 mean：

$$
r=\frac1{16}\sum_{t=1}^{16}h_t.
$$

Base 的 MLP 在第一层前仍可区分全部输入坐标；v1 readout 主动抹掉 token 身份。

---

## 6. 参数量与训练公平性

### 6.1 真实 Base

| 模块 | 权重主项 |
|---|---:|
| SENet | 0.521M |
| 2×DCNM | 41.956M |
| MLP | 47.682M |
| 总 trainable 近似 | 90.3M |

### 6.2 当前 v1

| 模块 | trainable 近似 |
|---|---:|
| Tokenizer | 16.123M |
| 2×PFFN | 151.118M |
| 总计 | 167.3M |

### 6.3 最简单的参数匹配

保持 $T=H=16,D=768,L=2$，把 expansion $k=4$ 改为 $k=2$：

$$
P_{\mathrm{RM},k=2}\approx91.7\mathrm M.
$$

它与 Base 90.3M 只差约 1.6%，是比当前 167M 更干净的结构对照。参数匹配仍不等于 FLOPs 和 kernel
效率匹配，因此两者都要报告。

---

## 7. 修正后的六条改进路线

### 7.1 路线一：Exact-SENet

~~~text
当前 v1 + 完全相同 Base SENet
~~~

只增加 Base SENet，不改 tokenizer、block、readout。回答缺失动态字段选择造成多少差距。

### 7.2 路线二：Field-Safe Tokenizer

~~~text
Exact-SENet
 -> 以完整 17 维字段为原子
 -> 业务语义分成 16 组
 -> 每组一层 Dense+GELU
~~~

首轮不改两层 tokenizer MLP，避免混入容量。

### 7.3 路线三：Dual Readout

~~~text
Mean Pool
 + Low-rank Flatten
 -> Fusion
 -> first head
~~~

用于恢复 token 位置身份。

### 7.4 路线四：Strict Backbone Swap

~~~mermaid
flowchart TD
    X["相同三桶 + BN"] --> S["相同 Base SENet"]
    S --> D["相同 2×DCNM"]
    D --> B["Base: MLP"]
    D --> R["实验: Field-safe RM"]
    B --> H1["相同 first loss"]
    R --> H2["相同 first loss"]
~~~

这是回答“MLP 与 RankMixer 谁更好”的最严格方案。

### 7.5 路线五：参数与初始化公平

- 当前 167M 与 $k=2$ 的 91.7M 同时报告；
- 输出 loaded/missing/random-init scope；
- 做都冷启与各自等成熟度热启；
- 比较同样本、同 FLOPs、同 wall-clock。

### 7.6 路线六：Aligned SwiGLU Hybrid

在路线一至五稳定后，再增加：

- company-aligned residual；
- 等参数 Pertoken SwiGLU；
- 小型 DCNM late branch；
- Base teacher distillation。

不能一次全部打开。

---

## 8. 哪些属于后续研究，不属于追回 Base

| 方案 | 正确定位 |
|---|---|
| Dense/Gattr | 新增输入 |
| DIN/Sequence/MixFormer | 新增序列能力 |
| last/wide/multi-task | 新增监督 |
| Task/Global/Cross Token | 表示扩展 |
| Soft-to-Hard/RankUp | Tokenizer 研究 |
| UniMixer/RankElastor | Mixing 研究 |
| Sparse MoE | 容量与系统研究 |

任何这些方案得到增益，都不能倒推当前 v1 的 0.003 来自对应模块缺失。

---

## 9. 推荐实验表

| 阶段 | 唯一主要变化 | 结论 |
|---|---|---|
| E0 | 多 seed 复现 Base/v1 | 差距方差 |
| E1 | + exact SENet | Base gate 贡献 |
| E2 | + field-safe tokenizer | 字段切断损失 |
| E3 | + dual readout | mean 瓶颈 |
| E4 | strict/aligned/reverting | block 拓扑 |
| E5 | $k=2$ 参数匹配 | 容量/收敛公平 |
| E6 | SENet+DCNM 后 MLP→RM | backbone 因果对照 |
| E7 | 等参 SwiGLU | 非线性门控 |
| E8 | DCNM late/global | 交叉组合 |
| E9 | Base KD | 收敛迁移 |
| E10+ | 序列、多任务、learned mixing、MoE | 新能力 |

### 9.1 淘汰规则

~~~text
1k-step 正确性
 -> 1-2 日小窗
 -> 5 日中窗
 -> 完整训练窗
 -> 不重叠时间窗
 -> 在线灰度
~~~

不能仅用首日 AUC 淘汰 167M 冷启模型。

---

## 10. 必须报告的证据

### 10.1 运行配置

- resolved args；
- feature version、日期、采样；
- checkpoint 路径；
- active branch 开关；
- git commit 和 token schema hash。

### 10.2 Restore

- 每个 scope 的 loaded、missing、random-init 参数量；
- Base SENet/DCNM/MLP 是否实际加载；
- RM tokenizer/PFFN/head 是否完全随机。

### 10.3 效果

- AUC/GAUC、PR-AUC、LogLoss、COPC、ECE；
- 多 seed 或 paired bootstrap；
- 冷启动、长尾、Creative 切片。

### 10.4 成本

- trainable params、FLOPs；
- samples/s、peak memory；
- checkpoint 大小和恢复时间；
- serving P50/P95/P99。

---

## 11. 源码证据

| 事实 | 位置 |
|---|---|
| Base resolved 参数 | <code>set-x.args.resolved.txt:27-76</code> |
| RM 启动参数 | <code>code/set-xcal.txt:248-298</code> |
| Base 三桶输入 | <code>cvr_bn_senet_dcnm.py:915-953</code> |
| Base BN+SENet | <code>cvr_bn_senet_dcnm.py:955-980</code> |
| Base SENet 数学路径 | <code>cvr_bn_senet_dcnm.py:830-905</code> |
| Base DCNM | <code>cvr_bn_senet_dcnm.py:783-828</code> |
| Base MLP/head | <code>cvr_bn_senet_dcnm.py:988-1016,1097-1108</code> |
| Base first loss | <code>cvr_bn_senet_dcnm.py:527-548</code> |
| v1 三桶/BN | <code>cvr_bn_rankmixer_v1.py:889-930</code> |
| v1 tokenizer | <code>cvr_bn_rankmixer_v1.py:774-799,938-962</code> |
| v1 block/readout | <code>cvr_bn_rankmixer_v1.py:801-980</code> |
| v1 first loss | <code>cvr_bn_rankmixer_v1.py:409-420</code> |

---

## 12. 最终判断

现有证据不能证明“RankMixer 不适合 CVR”，只能证明当前 v1 在一次实验中：

1. 删除了真实 Base 的层级 SENet；
2. 删除了两层 DCNM；
3. 用错误字段切分构造 token；
4. 用 mean-only 丢失 token 身份；
5. 使用约 1.85 倍参数的新主干，但未证明获得同等训练成熟度。

最有价值的下一步不是恢复 Base 从未使用的 DIN/last，而是：

> 补回 exact SENet，修复字段 tokenizer，构造 91.7M 参数匹配版本，再用“保留 SENet+DCNM、只替换 MLP”的实验严格判断 RankMixer 主干价值。
