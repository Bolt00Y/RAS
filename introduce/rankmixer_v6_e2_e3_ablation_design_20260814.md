# RankMixer v6-E2/E3 消融实验设计与服务器运行说明

> **日报摘要：** 本次完成 v6→v10 第一批严格消融实验改造。E2 在保持 v6 Core、RMSNorm 和 `[2048,2048,256]` 任务塔不变的前提下，仅将增强 Readout 替换为 PureFlat；E3 在 E2 基础上仅将 RMSNorm 替换为 LayerNorm。训练统一采用 `2026-08-14` 数据进行 Dense 冷启动，参数量只用于记录和实现校验，不作为实验控制条件。
>
> 状态：E2/E3 两份完整独立模型及服务器参数已实现；未在本地训练。
>
> 训练：`2026-08-14` Dense 随机冷启动；测试：`2026-08-15`。
>
> 目标：用两个严格单变量桥接点拆解 v6→v10：E1→E2 只替换完整 Readout，E2→E3 只替换 Norm。参数量只记录，不作为控制条件。

## 1. 四个任务分别是什么

| ID | 实际模型 | Dense 参数 | 在本批中的角色 |
|---|---|---:|---|
| E0_BASE | 现有 `SENet + 2×DCNM500 + MLP` Base | 90,341,785 | 同日端到端绝对锚点 |
| E1_V6 | 现有 `cvr_bn_rankmixer_v6` | 177,217,126 | v6 结构锚点 |
| E2_TML_FLAT_RMS | v6 TokenMixer-Large Core + RMSNorm + PureFlat，任务塔保持 2048 | 199,367,013 | Readout-only 桥接点 |
| E3_FLAT_LN | E2 全链路 RMSNorm→LayerNorm，任务塔保持 2048 | 199,275,877 | Norm-only 桥接点，即 v10 结构端点 |

本轮只需要新增并同时提交 E2、E3。E0/E1 已有完全相同训练日、测试日、特征和 Dense 冷启动口径的结果，直接作为既有对照，不重复训练。

E2/E3 按仓库现有风格分别实现为完整独立文件，不继承、不互相导入旧 RankMixer 模型。Python 模块名不能包含连字符，因此任务名 `v6-E2/v6-E3` 对应的合法入口为：

```text
models.rankmixer.cvr_bn_rankmixer_v6_e2.MLPModel
models.rankmixer.cvr_bn_rankmixer_v6_e3.MLPModel
```

两份完整代码通过 AST 等价测试约束除 Norm 实现、参数断言和必要日志名称外的模型结构一致，避免人工复制后出现隐性架构漂移。

## 2. E0～E3 共同控制什么

E1～E3 的 RankMixer 输入与 Core 固定为：

```text
1,234 个字段（Common 385 + Item 835 + Creative 14）
→ 三桶 Input BN
→ v6 Hierarchical SENet
→ v6 冻结语义分组：31 Local Token（10 + 20 + 1）
→ 1 Global Token
→ [B,32,512]
→ 2 个 TokenMixer-Large Block
   每个 Block：Mixing / Reverting + 两套 Per-token pSwiGLU(M=704)
→ fst_CVR 单任务 BCE
```

E0～E3 的服务器控制变量统一为：

- 相同训练、测试和附加 Sparse 日期；
- `ignore_dense_checkpoint=True`、`ignore_sparse_checkpoint=False`；
- 相同 `feature_version`、Embedding 维度 17、optimizer、LR、batch、epoch、数据文件与过滤口径；
- 相同 worker/PS 数量和硬件；
- `save_predict_result=True`，保留同一测试样本的 label/prediction 用于 paired AUC；
- 四个任务使用互不相同且事先确认为空的模型目录，禁止 `auto_load_cp` 命中旧任务。

Base 与 RankMixer 的网络结构本来就不同，因此 E1−E0 只能用于端到端位置判断，不是单因素模块消融。

## 3. E1 → E2：只替换完整 Readout 接口

### 3.1 E1 v6 保留什么

E1 的 `[B,32,512]` 最终 Token 经过 v6 增强读出：

```text
分支 A：Global Token                         → 512
分支 B：Global Query / Local Key softmax Pool → 512
分支 C：31 Local Token Flatten → FC           → 512
三路 concat                                  → 1536
任务头                                       → 2048 → 2048 → 256 → 1
```

对应读出与任务头参数：

| 组件 | 参数量 |
|---|---:|
| Global-conditioned Q/K Pool | 131,328 |
| Gated Flatten 15872→512 | 8,127,489 |
| 1536→2048→2048→256→1 任务头 | 7,877,633 |
| 合计 | 16,136,450 |

### 3.2 E2 具体改哪里

E2 保留 E1 的字段、SENet、Token、Global Token、两个 Block、Mixing/Reverting、pSwiGLU、Residual、全部 RMSNorm 和初始化，只替换最终读出：

```text
[B,32,512]
→ 固定 Token 顺序直接 Flatten 为 [B,16384]
→ 2048 → 2048 → 256 → 1
```

三个 FC 后仍使用与 v6/v10 一致的 Task BatchNorm 和 GELU2。任务头共 38,286,337 个参数，其中包括三层 BN 的 8,704 个 trainable `gamma/beta`。任务塔隐藏层宽度、BN、激活、初始化和输出层均与 E1 保持一致；第一层输入维度由 Readout 输出接口自然决定。

E2 明确删除：

- `_global_conditioned_pool`；
- Local Token 轴上的 Q/K score 和 softmax；
- 31-Token 压缩 Flatten 分支；
- flatten scalar gate；
- 三路 context concat。

参数变化：

```text
E2 − E1 = 199,367,013 − 177,217,126 = +22,149,887（+12.499%）
```

该参数变化不是实验控制目标，而是完整 Readout 输出从 1536 维变为 16384 维后，任务塔第一层权重形状变化的自然结果。解释必须写成：

> 在固定 v6 Core、RMSNorm 和 `[2048,2048,256]` 任务塔配置时，将完整增强 Readout 替换为 PureFlat 的端到端效应。

不能写成“Q/K Pool 的独立效应”或“排除容量后的纯 Readout 机制效应”，因为本实验替换的是包含输出接口在内的整套 Readout。若要单独识别 Pool，需另做同输出维度的专门实验，本批不做。

## 4. E2 → E3：只改变 Norm

E2 与 E3 加载两份独立完整 Python 模型。两份参数文件的模块入口因文件名不同而不同；去掉入口名称后，模型参数只有下列一项不同：

```text
E2: "rm_norm_type":"rms_norm"
E3: "rm_norm_type":"layer_norm"
```

为保持参数文件除目标变量外完全一致，E3 仍接收 `rm_rms_epsilon`，但标准 LayerNorm 路径不读取该值。

改变发生在五类位置：

| 位置 | 调用次数 | E2：v6 RMSNorm 参数形状 | E3：v10 LayerNorm 参数形状 | `E3−E2` 参数 |
|---|---:|---|---|---:|
| 31 个 Local Token 投影后 | 1 | `gamma[31,512]` | `gamma[512]+beta[512]` | −14,848 |
| Global Token 投影后 | 1 | `gamma[512]` | `gamma[512]+beta[512]` | +512 |
| 两个 Block、每层两个 PreNorm | 4 | 每次 `gamma[32,512]` | 每次 `gamma[512]+beta[512]` | −61,440 |
| 最终 Token Norm | 1 | `gamma[32,512]` | `gamma[512]+beta[512]` | −15,360 |
| **总计** | 7 次实际调用 |  |  | **−91,136** |

除了 Norm 算法、由其定义决定的 `gamma/beta` 形状以及纯日志/入口名称外，E2/E3 的 Tensor 输入、Core 权重尺寸、Residual、PureFlat、任务头、训练配置和数据完全相同。因此：

```text
Δnorm = AUC(E3) − AUC(E2)
```

可以解释为“仓库 v10 全链路 LayerNorm 相对 v6 RMSNorm 在当前一天冷启动中的本地效应”。它不是一般性的 Norm 文献复现实验，而是为了拆解仓库中已经同时改动 Norm 和读出的 v10。

## 5. E3 与当前完整 v10 的关系

修正后的 E3 使用 `PureFlat + LayerNorm + [2048,2048,256]`，Dense 参数为 199,275,877，因此在算法拓扑、张量维度和参数量上已经到达当前完整 v10 端点：

```text
E3：        16384 → 2048 → 2048 → 256 → 1
当前 v10： 16384 → 2048 → 2048 → 256 → 1
```

E3 为了保证消融链路可审计，保留了 v6 的语义分组版本标签和部分 legacy variable scope；v10 使用不同的命名标签，但冻结字段 checksum、前向计算和 Dense 冷启动结构一致。因而无需再安排一个“扩宽任务塔”的补充任务。

## 6. 预注册比较与允许结论

### 6.1 两个主消融

```text
C1  Readout only：  AUC(E2) − AUC(E1)
C2  Norm only：     AUC(E3) − AUC(E2)
```

不要把 `E3−E1` 当作第三个独立模块效应；它只是 C1 与 C2 的组合结果。

### 6.2 Base 位置

按 C1/C2 的预注册规则在 E1～E3 中选出候选后，报告其相对 E0 的 AUC 与成本。因为候选是在同一测试集上选出的，`winner−E0` 默认作为描述性结果；若要正式声明赢家优于 Base，应在提交前划出未参与选型的固定哈希确认切片，或在后续测试日复核，不需要重新训练。

### 6.3 判定阈值

提交前如果团队有正式线上阈值，应替换下面默认值；结果出来后不得修改：

```text
δwin = 0.0002
δeq  = 0.0001
```

- `|ΔAUC|` 小于业务阈值但统计显著：没有足够业务价值；
- paired bootstrap 95% CI 完整位于 `[-δeq,+δeq]`：工程等价；
- 其余为灰区，仅当灰区会改变路线选择时，给相关 pair 各补一个相同新种子。

## 7. 服务器执行顺序

### 7.1 先做 500～1,000 step 预检

只检查工程正确性，不用预检 AUC 选模型：

1. E2/E3 图上 Dense 参数分别为 `199,367,013 / 199,275,877`；
2. E2 日志固定为 RMSNorm、E3 固定为 LayerNorm，且各自图参数断言通过；
3. E2/E3 Dense 均为随机初始化，Sparse 来源与既有 E0/E1 一致；
4. 无 NaN、OOM、变量未更新或错误 restore；
5. 实测 step time、吞吐和峰值内存被记录；
6. E2/E3 使用互不重叠的新模型目录，并且都不复用 E0/E1 目录。

### 7.2 再提交整日任务

E2/E3 使用同一代码 commit、同一文件列表和同一集群规格并尽量同时提交。不要先看 E2 结果再修改 E3。任务完成后，将它们与既有 E0/E1 的 commit、数据快照、worker/PS 和 Sparse 来源逐项核对；任何关键口径不一致时不得作正式消融结论。

## 8. 结果必须回填什么

| ID | Task ID | Commit | Dense Params | Step Time | Peak Memory | AUC | COPC | LogLoss | Prediction Path | 状态 |
|---|---|---|---:|---:|---:|---:|---:|---:|---|---|
| E0_BASE |  |  | 90,341,785 |  |  |  |  |  |  |  |
| E1_V6 |  |  | 177,217,126 |  |  |  |  |  |  |  |
| E2_TML_FLAT_RMS |  |  | 199,367,013 |  |  |  |  |  |  |  |
| E3_FLAT_LN |  |  | 199,275,877 |  |  |  |  |  |  |  |

主结果表：

| 对比 | ΔAUC | paired 95% CI | 判定 | 允许的解释 |
|---|---:|---|---|---|
| E2−E1 |  |  |  | 固定任务塔宽度后的完整 Readout 接口替换效应 |
| E3−E2 |  |  |  | 全链路 Norm-only 效应 |

## 9. 已实现文件

- E2 完整模型：[`cvr_bn_rankmixer_v6_e2.py`](../src/models/rankmixer/cvr_bn_rankmixer_v6_e2.py)
- E3 完整模型：[`cvr_bn_rankmixer_v6_e3.py`](../src/models/rankmixer/cvr_bn_rankmixer_v6_e3.py)
- 机器可读实验定义：[`manifest.json`](../bash/rankmixer_first_batch_20260814/manifest.json)
- E0：[`00-e0-base-args.txt`](../bash/rankmixer_first_batch_20260814/00-e0-base-args.txt)
- E1：[`01-e1-v6-args.txt`](../bash/rankmixer_first_batch_20260814/01-e1-v6-args.txt)
- E2：[`02-e2-tml-flat-rms-args.txt`](../bash/rankmixer_first_batch_20260814/02-e2-tml-flat-rms-args.txt)
- E3：[`03-e3-flat-ln-args.txt`](../bash/rankmixer_first_batch_20260814/03-e3-flat-ln-args.txt)
- E2 顶层启动参数：[`set-rankmixer-v6-e2-args.txt`](../bash/set-rankmixer-v6-e2-args.txt)
- E3 顶层启动参数：[`set-rankmixer-v6-e3-args.txt`](../bash/set-rankmixer-v6-e3-args.txt)

静态 FLOPs 估计沿用仓库现有扩展口径：E2 约 `398,962,687`、E3 约 `399,355,903` FLOPs/样本。FLOPs 不包含 Sparse lookup、反向、参数服务器通信和内存搬运，最终成本判断以服务器 step time、吞吐和峰值内存为准。
