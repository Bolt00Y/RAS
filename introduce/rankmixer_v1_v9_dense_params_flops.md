# RankMixer v1–v9 Dense 参数量与 FLOPs 静态核算

> 核算日期：2026-08-28  
> 代码范围：`src/models/rankmixer/cvr_bn_rankmixer_v1.py` 至 `cvr_bn_rankmixer_v9.py`，并单列 `cvr_bn_rankmixer_v1_lrfix.py`。  
> 配置范围：优先采用 `bash/set-rankmixer-v*-args.txt` 中的实际启动参数；v1 使用当前源码与历史启动脚本中的 `T=16、D=768、L=2、k=4` 配置。

## 1. 结论汇总

下表中的 **Dense 参数量** 是固定稠密可训练参数，不含动态稀疏 Embedding；**扩展 FLOPs** 是本文用于版本比较的主结果，表示单样本一次推理前向的固定算术量。

| 版本 | 核心结构 | Dense 可训练参数 | 线性主项 FLOPs/样本 | 扩展 FLOPs/样本 | Batch=2,048 扩展前向 |
|---|---|---:|---:|---:|---:|
| RankMixer v1 | `T/H=16`，`D=768`，`L=2`，PFFN `k=4` | **167,293,157（167.293157M）** | 334,213,632 | **335,944,329（0.335944329G）** | 0.688013986 TFLOPs |
| RankMixer v1_lrfix | 前向结构与 v1 相同，仅修正学习率里程碑逻辑 | **167,293,157（167.293157M）** | 334,213,632 | **335,944,329（0.335944329G）** | 0.688013986 TFLOPs |
| RankMixer v2 | `16×768`，SENet，PFFN `k=2`，Gated Pool + Bucket Cross | **95,809,126（95.809126M）** | 191,362,206 | **192,547,107（0.192547107G）** | 0.394336475 TFLOPs |
| RankMixer v3 | v2 主干，改为硬编码业务语义分组 | **95,809,126（95.809126M）** | 191,362,206 | **192,547,107（0.192547107G）** | 0.394336475 TFLOPs |
| RankMixer v4 | v3 + 两条 Query→Item Low-rank Cross | **96,439,272（96.439272M）** | 192,608,926 | **193,818,155（0.193818155G）** | 0.396939581 TFLOPs |
| RankMixer v5 | `32×1024`，4 个 Per-token SwiGLU Stage，增强读出 + 深任务头 | **348,432,486（348.432486M）** | 703,530,158 | **705,277,790（0.705277790G）** | 1.444408914 TFLOPs |
| RankMixer v6 | v5 型主干，`D=512`、SwiGLU `M=704`，语义均衡分组 | **177,217,126（177.217126M）** | 357,528,750 | **358,638,942（0.358638942G）** | 0.734492553 TFLOPs |
| RankMixer v7 | v3 主干，线性输出头替换为 `[2048,2048,256]` 深任务头 | **102,113,126（102.113126M）** | 203,944,094 | **205,185,571（0.205185571G）** | 0.420220049 TFLOPs |
| RankMixer v8 | `32×512`，SwiGLU `M=512`，前置 2 层 Masked Low-rank DCN | **192,242,606（192.242606M）** | 387,421,278 | **388,878,806（0.388878806G）** | 0.796423795 TFLOPs |
| RankMixer v9 | 精确 2 层 DCNM500 + Raw/Cross Token + DCNM Shortcut | **199,445,658（199.445658M）** | 401,828,334 | **403,287,434（0.403287434G）** | 0.825932665 TFLOPs |

主要结论：

- 参数量和 FLOPs 最小的是 v2/v3；两者只改变字段分组方式，所以理论复杂度完全相同。
- v1_lrfix 没有修改前向网络，因此参数量和 FLOPs 与 v1 完全相同。
- v4 相对 v3 增加 630,146 个参数和 1,271,048 FLOPs/样本，增幅均约 0.66%。
- v5 是最大版本，348.432M Dense 参数、0.705278 GFLOPs/样本；主要成本来自 4 个按 Token 独立的 SwiGLU Stage。
- v6 相对 v5 减少 49.14% 参数和 49.15% FLOPs。
- v7 不是 v6 的顺序增量，而是从 v3 主干分叉并加深任务头。
- v8 相对 v6 增加 Masked Low-rank DCN，同时把 SwiGLU 中间维度从 704 降到 512，最终为 192.243M 参数、0.388879 GFLOPs/样本。
- v9 相对 v8 增加 7.203M 参数和 0.014409 GFLOPs/样本，固定 Dense 参数仍低于 200M。

## 2. 统计范围与统一输入

### 2.1 纳入和排除的文件

纳入主表：

```text
cvr_bn_rankmixer_v1.py
cvr_bn_rankmixer_v1_lrfix.py
cvr_bn_rankmixer_v2.py
...
cvr_bn_rankmixer_v9.py
```

`cvr_bn_unimixer_v1.py` 虽然位于同一目录，但实现的是 UniMixer，不是 RankMixer 的编号版本，因此没有混入本表。测试文件和 `model_utils.py` 也不属于独立模型版本。

### 2.2 输入维度

各启动脚本统一使用 `data.cvr.cvr_fea_v10_base_cold` 和 17 维字段 Embedding：

| 特征桶 | 字段数 | 单字段维度 | 展平维度 |
|---|---:|---:|---:|
| Common | 385 | 17 | 6,545 |
| Item | 835 | 17 | 14,195 |
| Creative | 14 | 17 | 238 |
| **合计** | **1,234** | 17 | **20,978** |

后文记总输入维度 `E=20,978`，字段数 `F=1,234`，SENet hidden `S=128`。

### 2.3 Dense 参数口径

计入：

- Dense/Linear 的 kernel 与 bias；
- SENet、DCNM、Token Projection、PFFN/SwiGLU、Pooling、Readout 和任务头的可训练权重；
- BatchNorm 的 `gamma/beta`；
- LayerNorm 的 `gamma/beta`；
- RMSNorm 的 `gamma`；
- 可训练 scalar gate。

不计入：

- 动态稀疏 Embedding 表；
- BatchNorm moving mean/variance；
- optimizer slot、梯度、指标状态、`global_step` 和学习率状态；
- 只读诊断张量。

若训练完成后所有稀疏字段实际保留的不同 key 总数为 `U`，完整可训练参数量为：

```text
P_total(version) = 17U + P_dense(version)
```

`U` 依赖真实 checkpoint，不能由 1,234 个字段数静态推导。

### 2.4 FLOPs 口径

FLOPs 按 **单样本、一次推理前向** 计算，1 次乘法和 1 次加法分别计 1 FLOP，即 1 MAC = 2 FLOPs。

主表同时给出两个口径：

1. **线性主项 FLOPs**：只统计参数化 Dense/Linear、PFFN/SwiGLU 投影及 SENet MatMul，适合与只报告矩阵主项的工具对齐。
2. **扩展 FLOPs**：在线性主项上继续计入激活、归一化、门控、残差、Pooling、显式乘积等算术操作，是本文用于比较的主结果。

扩展口径的基础规则：

| 算子 | FLOPs 规则 |
|---|---:|
| 带 bias 的 `m→n` Dense | `2mn` |
| 无 bias MatMul | 每个输出点积 `2m-1` |
| 推理 BatchNorm | 每元素 4 |
| 长度 `n` 的 LayerNorm | `8n+2` |
| 长度 `n` 的 RMSNorm | `4n+2` |
| 源码中的 tanh 近似 GELU (`gelu_2`) | 每元素 9 |
| ReLU、sigmoid、exp、rsqrt | 每元素/标量调用 1 |
| 长度 `n` 的 softmax | `3n-1` |
| `reshape/split/concat/stack/transpose` | 0 |

稀疏 Embedding lookup 本身按内存访问处理，不计固定算术 FLOPs。若一个字段在某样本中有 `a_j` 个活跃 key 并做 sum pooling，还需增加：

```text
F_lookup = 17 × Σ_j max(a_j - 1, 0)
```

因此完整单样本前向为 `F_lookup + 表中固定扩展 FLOPs`。本文不把 backward、优化器更新、参数服务器通信或设备内存搬运折算进 FLOPs。

## 3. v1、v2、v3、v4、v7 的逐模块核算

这五个版本均使用 16 个 768 维 Token。v1 使用 `k=4`；v2/v3/v4/v7 使用 `k=2`，并启用 SENet、Gated Pool 和 Bucket Cross。

### 3.1 Dense 参数分解

| 模块 | v1 / v1_lrfix | v2 | v3 | v4 | v7 |
|---|---:|---:|---:|---:|---:|
| 三桶 Input BN | 41,956 | 41,956 | 41,956 | 41,956 | 41,956 |
| Hierarchical SENet | 0 | 522,112 | 522,112 | 522,112 | 522,112 |
| 16 个 Token 投影 | 16,123,392 | 16,123,392 | 16,123,392 | 16,123,392 | 16,123,392 |
| 两层 RankMixer Block | 151,127,040 | 75,577,344 | 75,577,344 | 75,577,344 | 75,577,344 |
| Gated Pool | 0 | 768 | 768 | 768 | 768 |
| Bucket Cross | 0 | 3,541,249 | 3,541,249 | 3,541,249 | 3,541,249 |
| Query→Item Cross | 0 | 0 | 0 | 630,146 | 0 |
| Fusion LayerNorm | 0 | 1,536 | 1,536 | 1,536 | 1,536 |
| 输出/任务头 | 769 | 769 | 769 | 769 | 6,304,769 |
| **合计** | **167,293,157** | **95,809,126** | **95,809,126** | **96,439,272** | **102,113,126** |

关键等式：

```text
Token projection = E×D + T×D
                 = 20,978×768 + 16×768
                 = 16,123,392

v1 one block = 16×[2×4×768² + (4+1)×768] + 3×2×768
             = 75,563,520

v2/v3/v4/v7 one block
             = 16×[2×2×768² + (2+1)×768] + 2×2×768
             = 37,788,672
```

v2 与 v3 的每个字段都只进入一个 Token 投影，全部投影输入宽度之和仍为 `E`，所以语义组边界变化不改变参数量。

v4 的新增 630,146 个参数由以下部分构成：

```text
Query LN + Query projection
+ 2 × (Item LN + Item projection + Pair hidden + Gate + Output projection)
= 630,146
```

v7 用 6,304,769 参数的深任务头替换 v3 的 769 参数线性头，因此净增 6,304,000。

### 3.2 扩展 FLOPs 分解

| 模块 | v1 / v1_lrfix | v2 | v3 | v4 | v7 |
|---|---:|---:|---:|---:|---:|
| 三桶 Input BN | 83,912 | 83,912 | 83,912 | 83,912 | 83,912 |
| Hierarchical SENet | 0 | 1,087,414 | 1,087,414 | 1,087,414 | 1,087,414 |
| Token 投影 + GELU | 32,332,800 | 32,332,800 | 32,332,800 | 32,332,800 | 32,332,800 |
| 两层 RankMixer Block | 303,513,792 | 151,879,808 | 151,879,808 | 151,879,808 | 151,879,808 |
| Pooling | 12,288 | 48,415 | 48,415 | 48,415 | 48,415 |
| Bucket Cross | 0 | 7,106,307 | 7,106,307 | 7,106,307 | 7,106,307 |
| v4 QI Cross 新增量 | 0 | 0 | 0 | 1,271,048 | 0 |
| Fusion Add + LayerNorm | 0 | 6,914 | 6,914 | 6,914 | 6,914 |
| 输出/任务头 + sigmoid | 1,537 | 1,537 | 1,537 | 1,537 | 12,640,001 |
| **合计** | **335,944,329** | **192,547,107** | **192,547,107** | **193,818,155** | **205,185,571** |

单个 v1 Block：

```text
4kTD² + 9TkD + 2TD + 3T(8D+2)
= 151,756,896 FLOPs，k=4
```

单个 v2/v3/v4/v7 Block：

```text
4kTD² + 9TkD + 2TD + 2T(8D+2)
= 75,939,904 FLOPs，k=2
```

v4 的 1,271,048 新增 FLOPs 包含两条 QI Cross 的投影、乘积、差值、GELU、Gate、输出映射，以及 QI residual 和 Bucket/QI residual 的合并加法。

## 4. v5、v6、v8、v9 的逐模块核算

这四个版本使用 31 个 Local Token + 1 个 Global Token，`T=H=32`，每个 Block 包含 Mixed-space 和 Original-space 两个独立 Per-token SwiGLU Stage。

### 4.1 实际配置

| 配置 | v5 | v6 | v8 | v9 |
|---|---:|---:|---:|---:|
| Hidden `D` | 1,024 | 512 | 512 | 512 |
| SwiGLU hidden `M` | 704 | 704 | 512 | 512 |
| Block 数 `L` | 2 | 2 | 2 | 2 |
| Pool query/key `Q` | 128 | 128 | 128 | 128 |
| Flatten readout `R` | 512 | 512 | 512 | 256 |
| 任务头输入宽度 | 2,560 | 1,536 | 1,536 | 1,792 |
| 显式 Cross | 无 | 无 | 2×Masked Low-rank DCN | 2×精确 DCNM500 |
| Local Token 输入 | Raw | Raw | Cross | Raw + Cross |
| Global Token 输入 | Raw | Raw | Raw | Cross |
| DCNM Shortcut | 无 | 无 | 无 | `20,978→512` |

### 4.2 Dense 参数分解

| 模块 | v5 | v6 | v8 | v9 |
|---|---:|---:|---:|---:|
| 三桶 Input BN | 41,956 | 41,956 | 41,956 | 41,956 |
| Hierarchical SENet | 522,112 | 522,112 | 522,112 | 522,112 |
| 显式 Cross | 0 | 0 | 52,823,368 | 42,082,868 |
| Local Tokenizer | 21,544,960 | 10,772,480 | 10,772,480 | 21,513,216 |
| Global Token MLP | 22,533,120 | 11,004,416 | 11,004,416 | 11,004,416 |
| 两层 TokenMixer | 277,266,432 | 138,723,328 | 100,925,440 | 100,925,440 |
| Final RMSNorm | 32,768 | 16,384 | 16,384 | 16,384 |
| Global-conditioned Pool | 262,400 | 131,328 | 131,328 | 131,328 |
| Gated Flatten Readout | 16,253,953 | 8,127,489 | 8,127,489 | 4,063,745 |
| DCNM Shortcut | 0 | 0 | 0 | 10,742,272 |
| CVR Task Head | 9,974,785 | 7,877,633 | 7,877,633 | 8,401,921 |
| **合计** | **348,432,486** | **177,217,126** | **192,242,606** | **199,445,658** |

说明：v8 的 52,823,368 个显式 Cross 参数由 42,082,868 个 Low-rank DCN 主体参数和 10,740,500 个 Mask MLP 参数组成。v9 的 Local Token 同时拼接 Raw 与 Cross 两个视图，因此仅投影 kernel 从 `ED` 变为 `2ED`；每个 Token 的 bias 和 RMSNorm gamma 仍各只有一份，所以 v9 Local Tokenizer 为 `2ED+2ND=21,513,216`，并不等于 v8 整个模块的简单两倍。

### 4.3 扩展 FLOPs 分解

| 模块 | v5 | v6 | v8 | v9 |
|---|---:|---:|---:|---:|
| 三桶 Input BN | 83,912 | 83,912 | 83,912 | 83,912 |
| Hierarchical SENet | 1,087,414 | 1,087,414 | 1,087,414 | 1,087,414 |
| 显式 Cross | 0 | 0 | 105,811,064 | 84,331,564 |
| Local Tokenizer | 43,375,678 | 21,687,870 | 21,687,870 | 43,169,342 |
| Global Token MLP | 45,073,410 | 22,012,418 | 22,012,418 | 22,012,418 |
| 两层 TokenMixer | 554,574,080 | 277,422,336 | 201,851,136 | 201,851,136 |
| Final RMSNorm | 131,136 | 65,600 | 65,600 | 65,600 |
| Global-conditioned Pool | 8,459,100 | 4,233,564 | 4,233,564 | 4,233,564 |
| Gated Flatten Readout | 32,513,027 | 16,260,099 | 16,260,099 | 8,130,051 |
| DCNM Shortcut | 0 | 0 | 0 | 21,488,128 |
| CVR Task Head + sigmoid | 19,980,033 | 15,785,729 | 15,785,729 | 16,834,305 |
| **合计** | **705,277,790** | **358,638,942** | **388,878,806** | **403,287,434** |

### 4.4 通用复算公式

对 v5/v6/v8/v9，记 Local Token 数 `N=31`、总 Token 数 `T=32`、Hidden 为 `D`、SwiGLU hidden 为 `M`、Pool 投影为 `Q`、Flatten 输出为 `R`。

Local Tokenizer：

```text
Raw 单视图：2ED + 9ND + N(4D+2)
Raw+Cross 双视图：4ED + 9ND + N(4D+2)
```

Global Token MLP：

```text
2ED + 2D² + 9D + (4D+2)
```

单个 RMSNorm + Per-token SwiGLU Stage：

```text
T × [6DM + 3M + (4D+2)]
```

单个 RankMixer Block 含两个 Stage 和两个 Residual Add：

```text
2 × F_stage + 2TD
```

Global-conditioned Pool：

```text
2(N+1)DQ          # Query/Key Dense
+ 2NQ             # Q·K、reduce、scale
+ (3N-1)          # softmax
+ (2N-1)D         # 对 Local Token 加权求和
```

Gated Flatten Readout：

```text
2NDR + 9R + (4R+2) + 1 + R
```

v8 单层 Masked Low-rank DCN（`E=20,978`、rank `r=500`、mask hidden `K=250`）：

```text
4Er + 2EK + 2Kr + K + r + 10E + 2
= 52,905,532 FLOPs
```

两层为 105,811,064 FLOPs。

v9 单层精确 DCNM500：

```text
4E×500 + 2E + (8E+2)
= 42,165,782 FLOPs
```

两层为 84,331,564 FLOPs。

## 5. 版本间增量

| 对比 | Dense 参数变化 | 扩展 FLOPs/样本变化 | 结构原因 |
|---|---:|---:|---|
| v2 vs v1 | -71,484,031（-42.73%） | -143,397,222（-42.69%） | PFFN `k=4→2`，虽新增 SENet/Cross，但总体大幅下降 |
| v3 vs v2 | 0 | 0 | 仅改变语义分组，不改变总投影宽度与主干 |
| v4 vs v3 | +630,146（+0.66%） | +1,271,048（+0.66%） | 两条 Query→Item Low-rank Cross |
| v6 vs v5 | -171,215,360（-49.14%） | -346,638,848（-49.15%） | `D=1024→512` |
| v7 vs v3 | +6,304,000（+6.58%） | +12,638,464（+6.56%） | 线性头改为深任务头 |
| v8 vs v6 | +15,025,480（+8.48%） | +30,239,864（+8.43%） | 新增 Masked DCN，同时 `M=704→512` 抵消部分成本 |
| v9 vs v8 | +7,203,052（+3.75%） | +14,408,628（+3.71%） | 精确 DCNM、双视图 Token 和 Shortcut，Flatten `512→256` 抵消部分成本 |

## 6. 核验说明与使用限制

- v1–v3 的结果与仓库已有 `introduce/model_parameters_flops_analysis.md` 逐项交叉核对一致。
- v4 以 v3 为基底，独立复算 Query→Item Cross 的所有 kernel/bias、3 个 LayerNorm、2 个 gate 和 residual 合并。
- v5、v6 的参数总数与各自介绍文档中的逐模块精确值一致。
- v7 使用源码中的固定深任务头公式复算，结果与 `rankmixer_v7_introduction.md` 一致。
- v8、v9 的参数总数与源码 `_calculate_dense_trainable_params()`/静态校验值一致。
- 当前本地环境没有生产 TensorFlow 1.x、Flood 与 Cayman 运行时，因此 FLOPs 是按当前源码和启动参数做的静态算术核算，不是生产图 profiler 实测。矩阵主项是确定值；激活、归一化和超越函数在不同 profiler 中可能采用不同口径，所以主表同时保留“线性主项”和“扩展”两列。
- FLOPs 不等于延迟。Transpose、Concat、稀疏查表、参数服务器通信和内存带宽虽然在本文中不计算术 FLOPs，仍可能显著影响实际训练与在线推理耗时。
