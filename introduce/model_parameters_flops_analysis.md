# 四种 CVR 模型的参数量与 FLOPs 逐模块计算

本文对以下四个实现进行统一口径的静态复杂度核算：

1. `src/models/seq_model/cvr_bn_senet_dcnm_fst.py`
2. `src/models/rankmixer/cvr_bn_rankmixer_v1.py`
3. `src/models/rankmixer/cvr_bn_rankmixer_v2.py`
4. `src/models/rankmixer/cvr_bn_rankmixer_v3.py`

计算采用当前训练脚本中的实际模型配置，而不是只看构造函数中未被训练脚本采用的回退值。四个模型统一使用 `data.cvr.cvr_fea_v10_base_cold`、17 维 embedding，以及相同的 `common/item/creative` 三桶输入。

## 1. 结论汇总

设 `U_j` 为训练完成后第 `j` 个稀疏字段在参数服务器中实际保留的不同 key 数，`U = Σ_j U_j`。由于这里使用动态稀疏 embedding，完整 embedding 参数量不是 `字段数 × 17`，而是 `17U`。在相同特征、相同训练日期、相同初始 checkpoint 和相同淘汰策略下，四个模型的 `U` 相同。

| 算法 | 固定稠密可训练参数 | 含稀疏 embedding 的可训练参数总量 | 仅 MatMul/Linear FLOPs/样本 | 本文扩展口径 FLOPs/样本 |
|---|---:|---:|---:|---:|
| SENet + DCN-M + MLP 基线 | **90,341,785（90.342M）** | **`17U + 90,341,785`** | 180,316,654 | **180,923,051（0.180923 GFLOPs）** |
| RankMixer v1 | **167,293,157（167.293M）** | **`17U + 167,293,157`** | 334,213,632 | **335,944,329（0.335944 GFLOPs）** |
| RankMixer v2 | **95,809,126（95.809M）** | **`17U + 95,809,126`** | 191,362,206 | **192,547,107（0.192547 GFLOPs）** |
| RankMixer v3 | **95,809,126（95.809M）** | **`17U + 95,809,126`** | 191,362,206 | **192,547,107（0.192547 GFLOPs）** |

主要比较结果：

- v1 的固定参数量是基线的 **1.8518 倍**，FLOPs 是基线的 **1.8568 倍**。
- v2/v3 的固定参数量是基线的 **1.0605 倍**，FLOPs 是基线的 **1.0642 倍**。
- v2/v3 的固定参数量约为 v1 的 **57.27%**，FLOPs 约为 v1 的 **57.32%**。
- v2 与 v3 的语义分组不同，但每个字段只进入一个投影、token 数相同、投影后维度相同，所以二者的理论参数量和算术 FLOPs **完全相同**。

## 2. 统一输入与默认配置

### 2.1 特征维度

当前冷启动特征配置只保留三桶，且三桶字段均使用默认的 17 维 embedding。

| 特征桶 | 字段数 | 单字段维度 | 展平维度 |
|---|---:|---:|---:|
| common | 385 | 17 | `385 × 17 = 6,545` |
| item | 835 | 17 | `835 × 17 = 14,195` |
| creative | 14 | 17 | `14 × 17 = 238` |
| 合计 | **1,234** | 17 | **`E = 20,978`** |

后文使用以下记号：

| 记号 | 数值 | 含义 |
|---|---:|---|
| `F_c, F_i, F_a` | `385, 835, 14` | common、item、creative 字段数 |
| `F` | `1,234` | 总字段数 |
| `e` | `17` | embedding 维度 |
| `E` | `20,978` | 三桶展平后的总维度 |
| `S` | `128` | SENet 隐层维度 |
| `M` | `500` | DCN-M bottleneck 维度 |
| `T` | `16` | RankMixer token 数 |
| `H` | `16` | RankMixer head 数 |
| `D` | `768` | RankMixer hidden 维度 |
| `L` | `2` | RankMixer block 数 |

### 2.2 实际采用的模型参数

| 算法 | 参与计算的主要配置 |
|---|---|
| 基线 | `use_senet=true`，`use_senet_bn=true`，`S=128`，`cross_num=2`，`M=500`，`cvr_layers=[2048,2048,256]`，MLP BN 开启，激活为 `gelu_2`，wide/last/aux/delay 塔关闭 |
| v1 | `T=H=16`，`D=768`，`L=2`，`k=4`，投影和 PFFN 激活为 `gelu_2`，`rm_proj_ln=false`，SENet 未接入主塔 |
| v2 | `T=H=16`，`D=768`，`L=2`，`k=2`，token 数按桶为 `[5,10,1]`，SENet 和 SENet BN 开启，gated pool 与 bucket cross 开启 |
| v3 | 除 token 的硬编码业务语义分组外，与 v2 配置相同 |

训练日期不会改变固定稠密参数量或单样本主塔 FLOPs，只会影响动态 embedding 的 `U` 和每个样本的活跃稀疏 key 数。仓库内不同启动脚本的日期文本并不完全一致；本文按照“实际实验已统一日期”的前提计算，并把日期相关部分保留为 `17U`。

## 3. 计算口径

### 3.1 参数量

- 主表统计 **可训练模型参数**。
- Dense 层参数为 `m × n + n`，其中后一项为 bias。
- BatchNorm 计入可训练的 `gamma + beta = 2n`。
- LayerNorm 计入可训练的 `gamma + beta = 2n`。
- 不计 optimizer slot、梯度、`global_step`、学习率状态和 BatchNorm 的 moving mean/variance。
- 动态稀疏 embedding 单独写成 `17U`，不能根据 1,234 个字段直接得到唯一数值。

### 3.2 FLOPs

FLOPs 按 **单样本、一次推理前向** 计算：

- 一次乘法或加法各计 1 FLOP，因此一次 MAC 计 2 FLOPs。
- 带 bias 的 `m → n` Dense：每个输出包含 `m` 次乘、`m-1` 次累加和 1 次 bias 加法，合计 `2mn` FLOPs。
- 无 bias 的 MatMul：`n(2m-1)` FLOPs。
- 推理 BatchNorm：按每元素 4 FLOPs 计。
- LayerNorm：按每个长度为 `n` 的向量 `8n+2` FLOPs 计。
- 代码中的 tanh 近似 GELU `0.5x[1+tanh(c(x+0.044715x³))]` 按每元素 9 FLOPs 计；`tanh`、`sigmoid`、`exp` 各按 1 FLOP 计。不同 profiler 对超越函数的计数可能不同，因此同时给出了只统计 MatMul/Linear 的结果。
- `reshape/split/concat/stack/transpose` 记为 0 算术 FLOPs；它们仍会产生内存访问和运行时间。
- 稀疏 embedding 查表本身是内存访问，不记算术 FLOPs。若 lookup 对一个字段的 `a_j` 个活跃 key 做 sum pooling，还需增加 `17·max(a_j-1,0)` 次加法。因此完整 FLOPs 为表中固定值再加数据相关的 `F_lookup`。
- 不计算反向传播；训练 FLOPs 不能在缺少算子反向图和通信实现时简单声明为一个精确常数。

## 4. 算法一：SENet + DCN-M + MLP 基线

对应文件：`src/models/seq_model/cvr_bn_senet_dcnm_fst.py`。

### 4.1 参数量

| 模块 | 按默认参数设定的计算 | 该模块参数量 |
|---|---|---:|
| 动态稀疏 embedding | `Σ_j(U_j × 17)` | **`17U`** |
| 三桶输入 BatchNorm | `2E = 2 × 20,978` | 41,956 |
| SENet：6 个无 bias 权重矩阵 | `F_cS + SF_c + (F_c+F_i)S + SF_i + FS + SF_a` | 521,344 |
| SENet 内部 3 个 BatchNorm | `3 × 2S = 3 × 256` | 768 |
| DCN-M cross 0：`E→M→E` + LN | `(EM+M) + (ME+E) + 2E` | 21,041,434 |
| DCN-M cross 1：`E→M→E` + LN | `(EM+M) + (ME+E) + 2E` | 21,041,434 |
| MLP0：`20,978→2,048` + BN | `(20,978×2,048+2,048) + 2×2,048` | 42,969,088 |
| MLP1：`2,048→2,048` + BN | `(2,048×2,048+2,048) + 2×2,048` | 4,200,448 |
| MLP2：`2,048→256` + BN | `(2,048×256+256) + 2×256` | 525,056 |
| 输出头：`256→1` | `256×1+1` | 257 |
| **固定稠密参数合计** | 上述固定模块求和 | **90,341,785** |
| **完整可训练参数总量** | 稀疏 embedding + 固定稠密参数 | **`17U + 90,341,785`** |

SENet 的 521,344 个权重进一步展开为：

| SENet 权重 | 形状 | 参数量 |
|---|---:|---:|
| common input | `385 × 128` | 49,280 |
| common output | `128 × 385` | 49,280 |
| common+item input | `1,220 × 128` | 156,160 |
| item output | `128 × 835` | 106,880 |
| all-bucket input | `1,234 × 128` | 157,952 |
| creative output | `128 × 14` | 1,792 |
| **合计** |  | **521,344** |

### 4.2 FLOPs

| 模块 | 单样本 FLOPs 计算 | 该模块 FLOPs |
|---|---|---:|
| 动态 embedding lookup/pooling | 查表为内存访问；多 key pooling 为数据相关 `F_lookup` | **`F_lookup`** |
| 三桶输入 BatchNorm | `4E` | 83,912 |
| SENet | 字段均值 + 6 个 MatMul + 3 个 BN + tanh/sigmoid + 重标定 | 1,087,414 |
| DCN-M cross 0 | `4EM + 2E + (8E+2)` | 42,165,782 |
| DCN-M cross 1 | `4EM + 2E + (8E+2)` | 42,165,782 |
| MLP0 | `2E×2,048 + 4×2,048 + 9×2,048` | 85,952,512 |
| MLP1 | `2×2,048×2,048 + 4×2,048 + 9×2,048` | 8,415,232 |
| MLP2 | `2×2,048×256 + 4×256 + 9×256` | 1,051,904 |
| 输出头 + sigmoid | `2×256×1 + 1` | 513 |
| **固定主塔 FLOPs 合计** | 上述固定模块求和 | **180,923,051** |
| **完整单样本 FLOPs** | 固定主塔 + 稀疏 pooling | **`F_lookup + 180,923,051`** |

SENet FLOPs 的完整计算式为：

```text
字段均值：E
6 个无 bias MatMul：
S(2F_c-1) + F_c(2S-1)
+ S[2(F_c+F_i)-1] + F_i(2S-1)
+ S(2F-1) + F_a(2S-1)
内部 BN：3×4S
tanh：3S
sigmoid 与乘 2：2F
embedding 重标定：E

合计 = 1,087,414 FLOPs
```

## 5. 算法二：RankMixer v1

对应文件：`src/models/rankmixer/cvr_bn_rankmixer_v1.py`。

v1 把 20,978 维长向量切为 `[1,311]×15 + [1,313]`，再分别投影为 16 个 768 维 token。每个 block 使用 `k=4` 的 token 独立 PFFN，并实际执行 3 次 LayerNorm。

### 5.1 参数量

| 模块 | 按默认参数设定的计算 | 该模块参数量 |
|---|---|---:|
| 动态稀疏 embedding | `Σ_j(U_j × 17)` | **`17U`** |
| 三桶输入 BatchNorm | `2E` | 41,956 |
| 16 个 token 投影 | 权重 `ED` + 16 个 bias `TD` | 16,123,392 |
| Block 0：token mixing | reshape/transpose，无权重 | 0 |
| Block 0：16 个独立 PFFN | `T[2kD²+(k+1)D]`，`k=4` | 75,558,912 |
| Block 0：3 个 LayerNorm | `3×2D` | 4,608 |
| Block 1：token mixing | reshape/transpose，无权重 | 0 |
| Block 1：16 个独立 PFFN | `T[2kD²+(k+1)D]`，`k=4` | 75,558,912 |
| Block 1：3 个 LayerNorm | `3×2D` | 4,608 |
| Mean pooling | 无权重 | 0 |
| 输出头：`768→1` | `D+1` | 769 |
| **固定稠密参数合计** | 上述固定模块求和 | **167,293,157** |
| **完整可训练参数总量** | 稀疏 embedding + 固定稠密参数 | **`17U + 167,293,157`** |

token 投影参数可交叉验证为：

```text
15 × [(1,311×768) + 768]
+ 1 × [(1,313×768) + 768]
= 16,123,392
```

### 5.2 FLOPs

| 模块 | 单样本 FLOPs 计算 | 该模块 FLOPs |
|---|---|---:|
| 动态 embedding lookup/pooling | 数据相关 | **`F_lookup`** |
| 三桶输入 BatchNorm | `4E` | 83,912 |
| 16 个 token 投影 + GELU | `2ED + 9TD` | 32,332,800 |
| Block 0 | `4kTD² + 9TkD + 2TD + 3T(8D+2)` | 151,756,896 |
| Block 1 | 同 Block 0 | 151,756,896 |
| Mean pooling | `TD` | 12,288 |
| 输出头 + sigmoid | `2D + 1` | 1,537 |
| **固定主塔 FLOPs 合计** | 上述固定模块求和 | **335,944,329** |
| **完整单样本 FLOPs** | 固定主塔 + 稀疏 pooling | **`F_lookup + 335,944,329`** |

单个 v1 block 的 FLOPs 分解如下：

| Block 子模块 | FLOPs |
|---|---:|
| 两个 PFFN MatMul | `4kTD² = 150,994,944` |
| PFFN 隐层 GELU | `9TkD = 442,368` |
| 两次残差相加 | `2TD = 24,576` |
| 3 次 token-wise LayerNorm | `3T(8D+2) = 295,008` |
| **单 block 合计** | **151,756,896** |

## 6. 算法三：RankMixer v2

对应文件：`src/models/rankmixer/cvr_bn_rankmixer_v2.py`。

v2 在完整字段边界上将三桶划分为 `[5,10,1]` 个 token，启用层级 SENet，把 PFFN 扩展倍数降为 `k=2`，每个 block 使用 2 次 Add&Norm，并启用 gated pooling 与 bucket-cross residual。

### 6.1 v2 token 投影分组

| 桶 | 字段分组 | 分组输入维度 | 投影参数量 |
|---|---|---|---:|
| common | `[77,77,77,77,77]` | `[1,309]×5` | `5×(1,309×768+768) = 5,030,400` |
| item | `[84]×5 + [83]×5` | `[1,428]×5 + [1,411]×5` | 10,909,440 |
| creative | `[14]` | `[238]` | 183,552 |
| **合计** | 16 个 token | 总输入维度 20,978 | **16,123,392** |

### 6.2 参数量

| 模块 | 按默认参数设定的计算 | 该模块参数量 |
|---|---|---:|
| 动态稀疏 embedding | `Σ_j(U_j × 17)` | **`17U`** |
| 三桶输入 BatchNorm | `2E` | 41,956 |
| SENet：6 个权重矩阵 | 与基线相同 | 521,344 |
| SENet 内部 3 个 BatchNorm | `3×2S` | 768 |
| 16 个语义 token 投影 | `ED+TD` | 16,123,392 |
| Block 0：PFFN | `T[2kD²+(k+1)D]`，`k=2` | 37,785,600 |
| Block 0：2 个 LayerNorm | `2×2D` | 3,072 |
| Block 1：PFFN | `T[2kD²+(k+1)D]`，`k=2` | 37,785,600 |
| Block 1：2 个 LayerNorm | `2×2D` | 3,072 |
| Gated pooling score | `D×1`，无 bias | 768 |
| Bucket cross：`6D→D` 投影 | `6D²+D` | 3,539,712 |
| Bucket cross：LayerNorm | `2D` | 1,536 |
| Bucket cross：标量 gate | 1 | 1 |
| Fusion LayerNorm | `2D` | 1,536 |
| 输出头：`D→1` | `D+1` | 769 |
| **固定稠密参数合计** | 上述固定模块求和 | **95,809,126** |
| **完整可训练参数总量** | 稀疏 embedding + 固定稠密参数 | **`17U + 95,809,126`** |

### 6.3 FLOPs

| 模块 | 单样本 FLOPs 计算 | 该模块 FLOPs |
|---|---|---:|
| 动态 embedding lookup/pooling | 数据相关 | **`F_lookup`** |
| 三桶输入 BatchNorm | `4E` | 83,912 |
| SENet | 与基线相同 | 1,087,414 |
| 16 个语义 token 投影 + GELU | `2ED + 9TD` | 32,332,800 |
| Block 0 | `4kTD² + 9TkD + 2TD + 2T(8D+2)` | 75,939,904 |
| Block 1 | 同 Block 0 | 75,939,904 |
| Gated pooling | score Linear + softmax + 加权求和 | 48,415 |
| Bucket cross residual | 三桶均值 + 两两乘积 + `6D→D` + GELU + LN + gate | 7,106,307 |
| Fusion：残差相加 + LN | `D + (8D+2)` | 6,914 |
| 输出头 + sigmoid | `2D+1` | 1,537 |
| **固定主塔 FLOPs 合计** | 上述固定模块求和 | **192,547,107** |
| **完整单样本 FLOPs** | 固定主塔 + 稀疏 pooling | **`F_lookup + 192,547,107`** |

单个 v2 block 的 FLOPs 分解如下：

| Block 子模块 | FLOPs |
|---|---:|
| 两个 PFFN MatMul | `4kTD² = 75,497,472` |
| PFFN 隐层 GELU | `9TkD = 221,184` |
| 两次残差相加 | `2TD = 24,576` |
| 2 次 token-wise LayerNorm | `2T(8D+2) = 196,672` |
| **单 block 合计** | **75,939,904** |

Gated pooling FLOPs：

```text
无 bias score Linear：T(2D-1) = 24,560
softmax：3T-1 = 47
权重乘法与 token 求和：(2T-1)D = 23,808
合计 = 48,415
```

Bucket cross FLOPs：

```text
三桶 token 均值：TD = 12,288
三个两两向量乘积：3D = 2,304
6D→D Dense：12D² = 7,077,888
GELU：9D = 6,912
LayerNorm：8D+2 = 6,146
标量 sigmoid 与缩放：1+D = 769
合计 = 7,106,307
```

## 7. 算法四：RankMixer v3

对应文件：`src/models/rankmixer/cvr_bn_rankmixer_v3.py`。

v3 把 v2 的桶内均衡连续分组改为硬编码业务语义组。静态校验结果为：所有 1,234 个字段均恰好出现一次，无遗漏、无未知字段、无组内重复，也没有跨桶字段。

### 7.1 v3 语义组

| 桶 | 语义组字段数 | 对应输入维度 |
|---|---|---|
| common | `[16,90,92,85,102]` | `[272,1,530,1,564,1,445,1,734]` |
| item | `[98,71,58,60,126,73,46,134,33,136]` | `[1,666,1,207,986,1,020,2,142,1,241,782,2,278,561,2,312]` |
| creative | `[14]` | `[238]` |

虽然各 token 的输入宽度与 v2 不同，但每桶的总输入维度和 token 数相同，因此：

```text
投影权重总量 = E×D
投影 bias 总量 = T×D
投影参数总量 = ED+TD = 16,123,392
```

### 7.2 参数量

| 模块 | 按默认参数设定的计算 | 该模块参数量 |
|---|---|---:|
| 动态稀疏 embedding | `Σ_j(U_j × 17)` | **`17U`** |
| 三桶输入 BatchNorm | `2E` | 41,956 |
| SENet：6 个权重矩阵 | 与基线/v2 相同 | 521,344 |
| SENet 内部 3 个 BatchNorm | `3×2S` | 768 |
| 16 个硬编码语义 token 投影 | `ED+TD` | 16,123,392 |
| Block 0：PFFN + 2 个 LN | `37,785,600 + 3,072` | 37,788,672 |
| Block 1：PFFN + 2 个 LN | `37,785,600 + 3,072` | 37,788,672 |
| Gated pooling score | `D` | 768 |
| Bucket cross：投影 + LN + gate | `(6D²+D)+2D+1` | 3,541,249 |
| Fusion LayerNorm | `2D` | 1,536 |
| 输出头 | `D+1` | 769 |
| **固定稠密参数合计** | 上述固定模块求和 | **95,809,126** |
| **完整可训练参数总量** | 稀疏 embedding + 固定稠密参数 | **`17U + 95,809,126`** |

### 7.3 FLOPs

| 模块 | 单样本 FLOPs 计算 | 该模块 FLOPs |
|---|---|---:|
| 动态 embedding lookup/pooling | 数据相关 | **`F_lookup`** |
| 三桶输入 BatchNorm | `4E` | 83,912 |
| SENet | 与基线/v2 相同 | 1,087,414 |
| 16 个硬编码语义 token 投影 + GELU | `2ED+9TD` | 32,332,800 |
| Block 0 | `4kTD²+9TkD+2TD+2T(8D+2)` | 75,939,904 |
| Block 1 | 同 Block 0 | 75,939,904 |
| Gated pooling | 与 v2 相同 | 48,415 |
| Bucket cross residual | 与 v2 相同 | 7,106,307 |
| Fusion：残差相加 + LN | `D+(8D+2)` | 6,914 |
| 输出头 + sigmoid | `2D+1` | 1,537 |
| **固定主塔 FLOPs 合计** | 上述固定模块求和 | **192,547,107** |
| **完整单样本 FLOPs** | 固定主塔 + 稀疏 pooling | **`F_lookup + 192,547,107`** |

## 8. Batch size 为 2,048 时的前向 FLOPs

启动脚本的训练 batch size 为 2,048。忽略跨样本算子融合带来的 profiler 差异，固定主塔前向 FLOPs 近似按 batch 线性放大：

| 算法 | 单样本固定 FLOPs | Batch=2,048 的固定前向 FLOPs |
|---|---:|---:|
| 基线 | 180,923,051 | 370,530,408,448（0.370530 TFLOPs） |
| v1 | 335,944,329 | 688,013,985,792（0.688014 TFLOPs） |
| v2 | 192,547,107 | 394,336,475,136（0.394336 TFLOPs） |
| v3 | 192,547,107 | 394,336,475,136（0.394336 TFLOPs） |

这些数值只是一次 forward，不包含 backward、优化器更新、稀疏参数服务器通信和 `F_lookup`。

## 9. BatchNorm 非训练状态量

如果报告工具把 BatchNorm 的 moving mean 和 moving variance 也称为“参数”，可在固定可训练参数之外加入下表。它们是 checkpoint 状态，但不是可训练参数。

| 算法 | BN moving 状态量 | 固定模型状态总量（可训练参数 + BN moving） |
|---|---:|---:|
| 基线 | 51,428 | 90,393,213 |
| v1 | 41,956 | 167,335,113 |
| v2 | 42,724 | 95,851,850 |
| v3 | 42,724 | 95,851,850 |

若还要统计 Adam/Adagrad/Flood optimizer slot，结果会随优化器及稀疏参数服务器实现变化，不属于通常论文中报告的模型参数量。

## 10. 如何得到包含 embedding 的最终纯数字

当前工作区没有这批训练日期对应的 Flood 参数服务器 checkpoint 或逐字段 retained-key 统计，因此不能从源码可靠推导 `U`。要把 `17U + P_dense` 变成纯数字，只需从相同最终 checkpoint 读取每张稀疏表的实际保留 key 数：

```text
U = Σ_j retained_key_count(j)
P_sparse = 17U
P_total(base) = 17U + 90,341,785
P_total(v1)   = 17U + 167,293,157
P_total(v2)   = 17U + 95,809,126
P_total(v3)   = 17U + 95,809,126
```

这里必须使用实际 retained-key 数，不能用字段数、shard 数、span 数或一次样本中取出的 1,234 个向量代替。由于四个算法使用相同特征和日期，`17U` 是共同项，不影响四个稠密主塔之间的复杂度排序。

## 11. 核验说明

- 已从 `cvr_fea_v10_base.py` 静态解析得到 `385/835/14` 三桶字段数。
- 已逐项校验 v3 的 5/10/1 个语义组对三桶字段集合的完整覆盖、唯一性和桶归属。
- 参数表与 FLOPs 表均使用独立算式复算总和。
- 当前本地环境缺少生产使用的 TensorFlow 1.x、Flood 和 Cayman 运行时，因此这里是源码级静态核算，而不是对生产图执行 profiler 后伪称为实测结果。MatMul 主项与可训练参数总数是确定的；低阶激活/归一化 FLOPs 会因 profiler 口径产生少量差异。
