# RankMixer v6-E2-Small-2：三层 RankMixer 深度消融

模型代码：`src/models/rankmixer/cvr_bn_rankmixer_v6_e2_small_2.py`。
启动参数：`bash/set-rankmixer-v6-e2-small-2-args.txt`。
结构测试：`src/models/rankmixer/tests/test_rankmixer_v6_e2_small_2.py`。

## 1. 实验定义

本版本以 `cvr_bn_rankmixer_v6_e2_small_1.py` 为严格基线，唯一的算法改动是：

```text
rm_layer_num: 2 → 3
```

即在原来两个同构 RankMixer Block 后再堆叠一个相同 Block。这与
`cvr_senet_mature_rankmixer_v1` 的 `L=3` 在深度上对齐，但不搬运 mature_v1 的
RankMixer Block 实现：Small-2 的三层仍全部是 v6 风格的
Mixing/Reverting + 双 Per-token SwiGLU Block。因此该实验只测试 v6 主干深度。

## 2. 与 Small-1 的严格对照

| 配置 | Small-1 | Small-2 |
|---|---:|---:|
| Local / Global Token | 31 / 1 | 31 / 1 |
| Token 形状 | `[B,32,256]` | `[B,32,256]` |
| `T / H / D` | 32 / 32 / 256 | 32 / 32 / 256 |
| RankMixer Block 数 | 2 | **3** |
| 每层 SwiGLU 中间维度 | 704 | 704 |
| Block 内归一化 | per-token RMSNorm | per-token RMSNorm |
| Final Norm | mature_v1 LayerNorm | mature_v1 LayerNorm |
| Readout | MeanPool | MeanPool |
| CVR 任务头 | 256 → 128 → 1 | 256 → 128 → 1 |

除 `rm_layer_num` 和因新增 Block 产生的参数量外，以下内容均不变：

- 385 common、835 item、14 creative 字段及 17 维 Embedding；
- 三桶 Flood/Riemann BN 和字段级 hierarchical SENet；
- 10 common + 20 item + 1 creative 的 31 个 Local Token 顺序与 checksum；
- 由 SENet 后 common + item + creative 生成的 Global Token；
- Token 投影、`rm_optimize_tokenize=true`、`D=256`、`H=T=32`、`M=704`；
- 每个 Block 内的 Mixing、Reverting、两个 per-token RMSNorm、两个 SwiGLU
  和长残差路径；
- 末端共享 `[256]` gamma/beta 的 mature_v1 Final LayerNorm；
- 32-token 等权 MeanPool，不增加 creative bypass；
- `Dense 256 → BN → GELU2 → Dense 128 → BN → GELU2 → Linear 1`；
- 损失、优化器、学习率、数据、采样、评估、导出和 checkpoint 路径。

## 3. 三层执行流程

```text
input_tokens [B,32,256]
    ↓ rm_block_0: v6 Mixing/Reverting + double per-token SwiGLU
hidden_1    [B,32,256]
    ↓ rm_block_1: v6 Mixing/Reverting + double per-token SwiGLU
hidden_2    [B,32,256]
    ↓ rm_block_2: v6 Mixing/Reverting + double per-token SwiGLU
hidden_3    [B,32,256]
    ↓ mature_v1 Final LayerNorm
final       [B,32,256]
    ↓ MeanPool(axis=1)
context     [B,256]
    ↓ Dense 256 / Riemann BN / GELU2
    ↓ Dense 128 / Riemann BN / GELU2
    ↓ Linear 1 / clip[-50,50] / sigmoid
prediction  [B]
```

`_rm_stack()` 仍使用 `for block_idx in range(self.rm_layer_num)`，因此配置
`rm_layer_num=3` 会创建相互独立的 `rm_block_0`、`rm_block_1`、`rm_block_2`
变量 scope，不会在三层之间共享参数。

## 4. 参数量变化

单个 v6 D256/M704 RankMixer Block 包含：

```text
2 组 per-token RMSNorm
+ 2 个独立 per-token SwiGLU
= 34,725,888 可训练参数
```

| 模块 | Small-1 | Small-2 |
|---|---:|---:|
| 输入 BN + SENet + Token 投影 | 11,386,980 | 11,386,980 |
| RankMixer Block | 69,451,776 | **104,177,664** |
| Final LayerNorm | 512 | 512 |
| 256 → 128 → 1 任务头 | 99,585 | 99,585 |
| **合计** | **80,938,853** | **115,664,741** |

Small-2 比 Small-1 增加 34,725,888 个可训练参数，增幅约 42.90%。口径不含
稀疏 Embedding、优化器状态、BN moving statistics 和指标变量。

## 5. Bash 与运行环境

`set-rankmixer-v6-e2-small-2-args.txt` 由 Small-1 参数文件复制，仅修改：

```text
模型入口：models.rankmixer.cvr_bn_rankmixer_v6_e2_small_2.MLPModel
model_args.rm_layer_num：2 → 3
```

它保留 `--ignore_dense_checkpoint=True` 和 `--ignore_sparse_checkpoint=False`。第三个
RankMixer Block 在 Small-1 checkpoint 中不存在，首次训练应使用独立任务/模型目录；
稀疏 Embedding 仍按原参数热启动。

模型不新增任何 import 或第三方依赖，继续使用项目原有的 Python 3、
TensorFlow 1.x、Flood/Riemann BN、FloodOptimizer 和 PS 接口。

## 6. 验证边界

配套测试将校验：

- Small-2 与 Small-1 的 import、特征分组和所有方法 AST 保持一致；
- 构造参数中只有 `rm_layer_num` 的固定值从 2 改为 3；
- Bash 除模型入口和 `rm_layer_num` 外逐项一致；
- 三层 `_rm_stack()` 调用次数、scope 和输出形状符合预期；
- 参数解析计算与固定值均为 115,664,741。

本地如没有 TensorFlow/Flood，可完成语法、AST、Bash JSON 和参数量验证；完整
TensorFlow/Flood 建图与训练仍应在原服务器环境进行。
