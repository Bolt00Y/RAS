# RankMixer v6-E2-Small：256 维与严格等价的 Token 构造优化

本版本对应 `src/models/rankmixer/cvr_bn_rankmixer_v6_e2_small.py`，启动参数为
`bash/set-rankmixer-v6-e2-small-args.txt`。原版 v6-E2 的代码和参数文件保持原样。

“结果一致”的比较对象是：相同 D=256、相同权重、相同输入下的参考执行与优化执行。
D=512 缩到 D=256 本身会改变容量，不能据此承诺与 512 维版本的预测或 AUC 相同。

**模型规模**

| 配置 | 原版 v6-E2 | v6-E2-Small |
|---|---:|---:|
| 字段数 / Embedding 维度 | 1234 / 17 | 1234 / 17 |
| Local / Global Token | 31 / 1 | 31 / 1 |
| Token 维度 D | 512 | 256 |
| Mixing H / 每个 head 维度 | 32 / 16 | 32 / 8 |
| Block 数 | 2 | 2 |
| SwiGLU 中间维度 M | 704 | 704 |
| PureFlat 输出宽度 | 16384 | 8192 |
| 任务塔隐藏层 | 2048, 2048, 256 | 2048, 2048, 256 |
| Dense 可训练参数量 | 199,367,013 | 102,356,069 |

Dense 参数减少 97,010,944，即 48.66%。此口径不含稀疏 Embedding、优化器状态和
BN moving statistics。源码同时保留解析计算和建图后的参数总量校验。

**优化内容与等价性**

每个投影 family 的输出为 `Y[B,N,256]`。原路径对每个 Token 分别执行：

```python
tokens[j] = Y[:, j, :]
```

优化路径改为一次：

```python
family_tokens = tf.unstack(Y, num=N, axis=1)
```

随后仍按原有的 `token_index` 放回冻结的 31 个语义组位置。字段顺序、Token 顺序、
投影权重与 bias、GELU2、RMSNorm、初始化公式和参数命名均保持一致。

对于各输出收到的梯度 `g_j[B,256]`，两条路径的输入梯度满足：

```text
dL/dY = sum_j scatter_to_token_j(g_j)
      = stack([g_0, ..., g_(N-1)], axis=1)
```

不同 Token 的写入位置互不重叠。因此可以用一次 Pack 梯度替代 N 次完整形状的
StridedSliceGrad 及其相加。五个 family 的大小为 1、5、5、5、15；测试图中的
31 个 StridedSliceGrad 被消除。

按 B=2048、D=256、FP32 估算，原切片梯度的逻辑张量总量为：

```text
2048 × 256 × 4 × (1² + 5² + 5² + 5² + 15²) = 602 MiB
```

改为按 family 拼接梯度后为：

```text
2048 × 256 × 4 × (1 + 5 + 5 + 5 + 15) = 62 MiB
```

这是这些中间梯度张量的总量估算，不是实测峰值内存或整日训练的内存降幅；执行器的
张量复用、图优化和并发调度会影响实际内存。

投影前后的 10 次 family 转置保持原样。原因是修改前向布局可能触发 TensorFlow
Grappler 对 GELU 乘法的不同结合顺序。在本地严格检查中，该类布局改写曾出现
约 4.8e-7 的输出差异，因此没有纳入交付代码。SwiGLU 和主干的所有算术操作沿用
原实现。

**服务器环境与训练协议**

模型导入的 Python 模块与原版一致，使用现有 TensorFlow/Flood API；无需新增库、
自定义 C++/CUDA 算子、升级运行时或调整硬件。原有三维批量 MatMul、FP32、Flood BN、
FloodAdam、SENet、RMSNorm、损失、学习率调度、采样、评估及导出代码均被保留。

新旧 args 的差异只有：

```text
模型入口：models.rankmixer.cvr_bn_rankmixer_v6_e2_small.MLPModel
model_args.rm_hidden_dim：512 → 256
model_args.rm_optimize_tokenize：新增 true
```

外层参数逐行相同，包括原有线程配置、`fast_matmul`、PS 配置、NUMA、训练与测试日期、
checkpoint 来源和 BN 配置。其他 model_args 逐项相同。

首次启动 Small 应使用独立的任务/模型目录。提供的参数保留
`ignore_dense_checkpoint=True` 和原来的稀疏热启动配置。512 维 Dense checkpoint
与 Small 的变量形状不同，不能用于 Small 的 Dense 热启动。后续逐日训练只能沿
Small 自己的 checkpoint 续训，并沿用原有日期和续训协议。

**开关与回退**

```json
{"rm_hidden_dim":256,"rm_optimize_tokenize":true}
```

`rm_optimize_tokenize=true` 是默认值。改为 JSON 布尔值 `false` 会使用原来的
逐 Token 切片执行路径，模型仍保持 256 维。开关不引入任何可训练变量；同一个
Small checkpoint 可以在两种模式间切换。开启和关闭开关时应保持输入和其他配置相同。

**已经完成的验证**

本地使用隔离的 Python 3.11.16、TensorFlow 2.16.2 `compat.v1`、FP32 CPU 运行；
该验证环境未安装到服务器。配置为 intra-op=32、inter-op=8。

`test_rankmixer_v6_e2_small.py` 的 8 个测试全部通过：

- 静态比较确保除了构造配置及 `_semantic_tokenize`，其他所有方法与原版在去掉日志
  名称差异后 AST 相同；包含整个主干、损失、数据、优化器和导出路径。
- 冻结字段列表、分组 checksum、变量声明以及新旧 args 差异通过校验。
- 数值参考直接提取原版 v6-E2 的真实 TensorFlow Token 构造方法，将其计算维度设为
  256。仅绕过构造函数固定 512 的检查，不重写参考计算公式。
- 在 1 和 3 个最大变量分片下，覆盖 batch 1、7、17 及 batch 3 的全零输入。
- 同一 checkpoint 下逐数组严格比较 Token 输出、随机线性探针 loss、输入梯度和
  全部 Token 投影参数梯度，`max_abs_error=0`，不使用 atol/rtol 容差。
- 使用原生 TensorFlow Adam 做连续 3 步更新探针，逐步比较权重及优化器状态；
  包括从已训练的 checkpoint 恢复非零 moment 状态，比较全部通过。
- 关闭优化开关后也与 256 维参考路径严格一致。

这里的 Adam 是验证程序的更新探针，生产代码继续使用原来的 FloodAdam。数值验证覆盖
被改动的 Token 构造组件；它不是 HDFS/稀疏 PS/Flood BN 的整链路训练、实际 BCE/AUC
或服务器吞吐实测。运行库的图改写行为与分布式更新时序应在服务器验证。

**在原服务器环境核对**

在服务器项目根目录直接使用训练环境已有的 Python 执行，无需安装依赖：

```bash
python src/models/rankmixer/tools/verify_rankmixer_v6_e2_small.py \
  --output /tmp/rankmixer-v6-e2-small-verify.json
```

工具默认在 CPU 上使用 32/8 线程，输出 TensorFlow/Python 版本、严格比较结果、
变量分片、梯度节点数量和 checkpoint 检查结果。TF1 使用原生 API；TF2 使用 compat.v1。
CPU 验证不代表 GPU 内核或整套 Flood 训练已经验证。

核对参考开关：

```bash
python src/models/rankmixer/tools/verify_rankmixer_v6_e2_small.py \
  --reference-mode --output /tmp/rankmixer-v6-e2-small-reference.json
```

任何零容差比较失败都会抛出错误，不会自动放宽阈值或把失败记录为通过。此时可使用
`rm_optimize_tokenize=false` 的 256 维参考执行路径，再针对该服务器运行时定位差异。

需要测量 Token 构造局部耗时时：

```bash
python src/models/rankmixer/tools/verify_rankmixer_v6_e2_small.py \
  --partitions 1 --benchmark-steps 30 --benchmark-batch-size 2048 \
  --warmup-steps 5 --output /tmp/rankmixer-v6-e2-small-token-benchmark.json
```

此计时包含 Token 构造、反向、原生 Adam 和输入 feed 拷贝，不包含其余网络、
Flood 优化、HDFS 或参数服务器开销，不能用它直接推算 560 分钟会缩短到多少。
整日性能应在相同资源、数据和评估协议下记录实际训练耗时与吞吐。
