# RankMixer v6-E2-Small-3：保留 PureFlat 任务头的三层版本

模型代码：`src/models/rankmixer/cvr_bn_rankmixer_v6_e2_small_3.py`。
启动参数：`bash/set-rankmixer-v6-e2-small-3-args.txt`。
模型入口：`models.rankmixer.cvr_bn_rankmixer_v6_e2_small_3.MLPModel`。

## 1. 实验定义

本版本以 [v6-E2-Small](rankmixer_v6_e2_small_introduction.md) 的
`cvr_bn_rankmixer_v6_e2_small.py` 为基线，唯一的模型结构改动是：

```text
rm_layer_num: 2 → 3
```

在原有两个 RankMixer Block 后增加一个同构 Block，并同步更新固定结构校验和参数量
校验。三层均使用 v6 的 Mixing/Reverting、双 per-token RMSNorm 和双 per-token SwiGLU。

名称中的 `Small-3` 是本次新版本名。它保持原 Small 的 Final RMSNorm、PureFlat
和 `[2048,2048,256]` 任务头；现有 Small-1/Small-2 使用的 Final LayerNorm、MeanPool
及较小任务头不属于本次配置。

## 2. 与原 Small 的对照

| 配置 | v6-E2-Small | v6-E2-Small-3 |
|---|---:|---:|
| 字段数 / Embedding 维度 | 1234 / 17 | 1234 / 17 |
| Local / Global Token | 31 / 1 | 31 / 1 |
| Token 形状 | `[B,32,256]` | `[B,32,256]` |
| `T / H / D` | 32 / 32 / 256 | 32 / 32 / 256 |
| RankMixer Block 数 | 2 | **3** |
| SwiGLU 中间维度 M | 704 | 704 |
| Block 内归一化 | per-token RMSNorm | per-token RMSNorm |
| Final Norm | per-token RMSNorm | per-token RMSNorm |
| Readout | PureFlat | PureFlat |
| 展平后宽度 | 8192 | 8192 |
| CVR 任务塔隐藏层 | `[2048,2048,256]` | `[2048,2048,256]` |
| Dense 可训练参数量 | 102,356,069 | **137,081,957** |

三桶 Flood/Riemann BN、hierarchical SENet、冻结特征分组及 checksum、Token 投影、
`rm_optimize_tokenize=true`、长残差路径、激活、初始化、损失、优化器、学习率调度、
采样、评估和导出逻辑均沿用原 Small。新模型保留独立的完整实现，不导入或继承其他
RankMixer 模型文件，也不新增第三方依赖。

## 3. 三层执行流程

```text
三桶 BN → hierarchical SENet → 31 Local Token + 1 Global Token
input_tokens [B,32,256]
    ↓ rm_block_0: Mixing/Reverting + 双 per-token SwiGLU
hidden_1    [B,32,256]
    ↓ rm_block_1: Mixing/Reverting + 双 per-token SwiGLU
hidden_2    [B,32,256]
    ↓ rm_block_2: Mixing/Reverting + 双 per-token SwiGLU
hidden_3    [B,32,256]
    ↓ Final per-token RMSNorm
final       [B,32,256]
    ↓ PureFlat
context     [B,8192]
    ↓ Dense 2048 / BN / GELU2
    ↓ Dense 2048 / BN / GELU2
    ↓ Dense 256 / BN / GELU2
    ↓ Linear 1 / clip[-50,50] / sigmoid
prediction  [B]
```

`_rm_stack()` 沿用 `for block_idx in range(self.rm_layer_num)`。
`rm_layer_num=3` 对应 `rm_block_0`、`rm_block_1`、`rm_block_2` 三个独立变量 scope，
各层不共享参数。源码的默认值、固定配置约束和 Bash 参数均设为 3。

## 4. 参数量核对

对于 `T=32, D=256, M=704`，单个 per-token SwiGLU 包含三组权重及对应 bias：

```text
单个 SwiGLU = T × (3 × D × M + 2 × M + D)
             = 17,354,752

单个 Block = 2 × T × D + 2 × 单个 SwiGLU
            = 16,384 + 34,709,504
            = 34,725,888
```

| 模块 | v6-E2-Small | v6-E2-Small-3 |
|---|---:|---:|
| 输入 BN + SENet + Token 投影及其 RMSNorm | 11,386,980 | 11,386,980 |
| RankMixer Block | 69,451,776 | 104,177,664 |
| Final per-token RMSNorm | 8,192 | 8,192 |
| PureFlat 后的任务头 | 21,509,121 | 21,509,121 |
| **合计** | **102,356,069** | **137,081,957** |

Small-3 增加 **34,725,888** 个 Dense 可训练参数，增幅约 **33.93%**。
该口径包含任务头 BN 的可训练参数，不含稀疏 Embedding 表、优化器状态、BN moving
statistics 和指标变量。代码保留解析计算和完整建图后的参数总量校验，固定期望值为
`137081957`。训练耗时、显存及 AUC 变化需要通过服务器实际实验评估。

## 5. 上传与 Bash 参数

上传到服务器项目的对应相对路径：

- `src/models/rankmixer/cvr_bn_rankmixer_v6_e2_small_3.py`
- `bash/set-rankmixer-v6-e2-small-3-args.txt`
- `introduce/rankmixer_v6_e2_small_3_introduction.md`（实验说明）

继续使用现有训练提交方式，将 args 文件切换为
`bash/set-rankmixer-v6-e2-small-3-args.txt`。该文件由原 Small 的 args 复制，仅修改：

```text
第一行模型入口：models.rankmixer.cvr_bn_rankmixer_v6_e2_small_3.MLPModel
model_args.rm_layer_num：2 → 3
```

其余 model_args 和外层训练配置沿用原 Small，包括线程、PS、NUMA、BN、优化器、
`fast_matmul`、数据和 checkpoint 设置。日期仍保留原文件的值：

| 参数 | 保留值 |
|---|---|
| `train_dates` | `2026-08-14:2026-08-14` |
| `test_date` | `2026-08-15:2026-08-15` |
| `additional_checkpoint_dates` | `2026-08-13:2026-08-13` |
| `checkpoint_import_dir` | 原路径的 `pt=2026-07-01/checkpoint` |

提交前应按实际训练批次调整训练日期、测试日期、附加 checkpoint 日期和来源路径，
确认训练平台指向 **Small-3 独立的实验/模型输出目录**，避免与两层 Small 共用输出。

首次运行保留：

```text
--ignore_dense_checkpoint=True
--ignore_sparse_checkpoint=False
```

原两层 checkpoint 不含新增的 `rm_block_2`，首次 Small-3 训练按上述设置初始化
Dense 参数，并沿用原配置进行稀疏 Embedding 热启动。后续续训应使用 Small-3 自己的
checkpoint，按现有训练平台的日期推进与恢复协议配置 Dense 恢复。

## 6. 验证范围

本地 Python 3.9.6 已通过新模型语法检查，以及 args 的 shell 引号和 4 个 JSON 参数
解析检查。在项目根目录可复查语法，并运行配套的离线结构测试：

```bash
python -m py_compile src/models/rankmixer/cvr_bn_rankmixer_v6_e2_small_3.py
python -m unittest discover -s src/models/rankmixer/tests \
  -p 'test_rankmixer_v6_e2_small_3.py' -v
```

Small-3 的 7 项离线回归测试全部通过，覆盖：默认层数及固定约束为 3、Bash JSON 中 `rm_layer_num=3`、
`_rm_stack()` 顺序调用三个独立 Block、与原 Small 保持相同的 Final RMSNorm / PureFlat
任务头，以及解析参数量为 `137081957`。测试还实际执行构造器，确认显式传入两层或四层
会被拒绝，并逐方法比较源码 AST，确保除层数和日志名称外的模型逻辑与原 Small 一致。

原 Small / Small-1 / Small-2 的 25 项测试中，22 项通过，3 项 TensorFlow 数值测试因本地
缺少运行库而跳过。

本地未安装 TensorFlow/Flood；本说明中的参数量由源码公式计算。语法和结构检查不能替代
原服务器 TensorFlow/Flood、HDFS、稀疏 PS 的完整建图与训练验证；服务器实验结果应以
实际训练日志为准。
