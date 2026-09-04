# RankMixer v6-E2-Small-1：MeanPool 与 mature_v1 紧凑任务头消融

模型代码：`src/models/rankmixer/cvr_bn_rankmixer_v6_e2_small_1.py`。
启动参数：`bash/set-rankmixer-v6-e2-small-1-args.txt`。
结构测试：`src/models/rankmixer/tests/test_rankmixer_v6_e2_small_1.py`。

## 1. 实验目的与边界

本版本以 `cvr_bn_rankmixer_v6_e2_small.py` 为唯一主干基线，只替换
RankMixer 之后的最终归一化、读出和 CVR 任务头，用于测试：

```text
PureFlat(8192) + [2048, 2048, 256]
                    ↓
MeanPool(256) + mature_v1-style [256, 128]
```

这是一个联合消融：同时改变了读出方式和任务头容量，因此实验结果
不能单独归因给 mean pooling 或小任务头中的某一项。

本版本不复制 mature_v1 的 creative bypass。Small 中仍保留 1 个 creative local
token，且 global token 仍由 SENet 后的 common、item、creative 三桶共同生成；
如果再引入 creative bypass，就会叠加改变 creative 的路由和容量分配。

## 2. 保持不变的 Small 主干

| 模块 | v6-E2-Small-1 配置 |
|---|---|
| 输入字段 | common 385、item 835、creative 14，Embedding 维度 17 |
| 输入处理 | 三桶独立 Flood/Riemann BN + 字段级 hierarchical SENet |
| Local Token | 10 common + 20 item + 1 creative，共 31 个 |
| Global Token | SENet 后的 common + item + creative 投影为 1 个 |
| Token 形状 | 31 Local + 1 Global，`[B,32,256]` |
| RankMixer | `T=H=32`、`D=256`、`L=2`、SwiGLU `M=704` |
| Block 归一化 | v6 per-token RMSNorm，`epsilon=1e-6` |
| Token 投影优化 | 保留 `rm_optimize_tokenize=true` 的 Unpack 执行路径 |

字段列表、分组顺序、checksum、Token 投影、Global Token、SENet、RankMixer block、
损失、优化器、数据读取、指标、导出和 checkpoint hook 均与 Small 保持一致。

## 3. 新的末端结构

```text
RankMixer hidden_tokens                         [B,32,256]
    ↓ mature_v1 Final LayerNorm, shared gamma/beta
final_tokens                                    [B,32,256]
    ↓ reduce_mean(axis=1), 32 Token 等权平均
context                                         [B,256]
    ↓ Dense(256) → Flood/Riemann BN → GELU2
hidden_0                                        [B,256]
    ↓ Dense(128) → Flood/Riemann BN → GELU2
hidden_1                                        [B,128]
    ↓ Dense(1), linear
logit                                           [B,1]
    ↓ reshape → clip[-50,50] → sigmoid
fst_cvr prediction                              [B]
```

### 3.1 Final LayerNorm

与 `cvr_senet_mature_rankmixer_v1` 一致，每个 Token 沿最后的 256 维计算均值和方差：
对应变量 scope 为 `rm_final_layer_norm`。

\[
y_{b,t,d}=\gamma_d
\frac{x_{b,t,d}-\mu_{b,t}}
{\sqrt{\sigma^2_{b,t}+10^{-8}}}+\beta_d.
\]

`gamma` 和 `beta` 的形状都是 `[256]`，由 32 个 Token 位置共享。这与 Small
原来的最终 RMSNorm 不同：原实现不减均值、没有 beta，且 gamma 形状为
`[32,256]`。Block 内部的 RMSNorm 不变，只替换读出前的最终归一化。

### 3.2 Mean pooling

\[
c_{b,d}=\frac{1}{32}\sum_{t=1}^{32}y_{b,t,d}.
\]

池化不含参数，不在池化后另外增加 LayerNorm、RMSNorm、BatchNorm 或
dropout。第一个 Dense 之后的 BN 负责任务头隐层归一化。

### 3.3 mature_v1 风格任务头

`cvr_layers=[256,128]`，1 维输出层仍由 `_task_head` 单独创建。两个隐层的
操作顺序严格为：

```text
Dense(linear) → Flood/Riemann BN → GELU2
```

BN 配置与 mature_v1 对齐：

- `batch_norm_decay=0.9`；
- `use_riemann_bn=true`；
- `embed_use_renorm=false`，`embed_renorm_decay=0.99`；
- BN 启用 `center` 和 `scale`，训练/预测分支仍由 `ModelBase.batch_norm_layer_v2`
  处理。

GELU2 为：

\[
\operatorname{GELU2}(x)=\frac{x}{2}\left[1+\tanh\left(
\sqrt{\frac{2}{\pi}}(x+0.044715x^3)\right)\right].
\]

权重使用 mature_v1 的 `Normal(stddev=1/sqrt(fan_in))` 初始化，bias 为零。两个隐层
和最终输出层均挂载 `l2_deep=1e-6` 的 L2 regularizer。输出层不做 BN，
也不在 Dense 内做 sigmoid；它输出线性 logit，然后沿用 Small 的 logit 裁剪和
sigmoid 路径。

## 4. 参数量

以源码 `_calculate_dense_trainable_params()` 的口径计算：

| 模块 | 可训练参数 |
|---|---:|
| 三桶输入 BN | 41,956 |
| 字段级 SENet | 522,112 |
| Local Token 投影与归一化 | 5,386,240 |
| Global Token | 5,436,672 |
| 2 个 RankMixer Block | 69,451,776 |
| Final LayerNorm | 512 |
| 256 → 128 → 1 任务头 | 99,585 |
| **合计** | **80,938,853** |

原 Small 为 102,356,069，本版本减少 21,417,216，约 20.92%。该口径不含稀疏
Embedding 表、优化器状态、BN moving mean/variance 和指标变量。

## 5. 服务器与 checkpoint 兼容性

本文件是完整独立的模型实现，没有导入或继承旧 RankMixer 模型，且不增加任何
第三方库。它仍使用项目已有的 Python 3、TensorFlow 1.x、Flood/Riemann BN、
FloodOptimizer 和 PS 路径，因此不需要更改服务器运行库版本。

启动参数保留：

```text
--ignore_dense_checkpoint=True
--ignore_sparse_checkpoint=False
```

由于 Final Norm 变量和任务头形状已经改变，首次实验应使用独立任务/模型
目录，不能恢复 Small 的 Dense checkpoint。稀疏 Embedding 仍按原启动参数热启动。

## 6. 验证边界

配套测试检查：

- 新模型与 Small 使用完全相同的服务器 import；
- 除构造参数、参数量计算、最终归一化、读出和任务头外，主干方法 AST
  与 Small 一致；
- 字段分组、Token 路由和 checksum 不变，且不存在 creative bypass；
- 任务头顺序为 Dense → BN → GELU2，隐藏层为 `[256,128]`；
- 解析参数量和固定预期值均为 80,938,853；
- 环境存在 TensorFlow 时，使用真实 TensorFlow kernel 对比新 Final LayerNorm 与
  mature_v1 的输出和 mean-pool 输出。

本地如未安装 TensorFlow/Flood，可完成语法、AST、参数文件和参数量检查；真实
TensorFlow/Flood 全链路建图和训练仍应在原服务器环境进行。
