# RankMixer 2026-08-14 Dense 冷启动消融实验报告

> 状态：**实验预注册版，尚未提交服务器训练**  
> 训练数据：`2026-08-14`；测试数据：`2026-08-15`  
> 首轮预算：**4 个任务（1 个 Base + 3 个 RankMixer）**

## 1. 结论先行

首轮不再同时比较 v7/v8/v9/v10/UniMixer，也不做残差、Norm、Mixing、SwiGLU、Self-Attention 等通用模块消融。当前最有价值、且历史结果无法回答的问题只有一个：

> v6 相对 v5 的首日提升，究竟来自 **D=1024→512 带来的单日冷启动优化/参数效率**，还是来自 **稳定哈希均衡分组→语义均衡分组**？

现有 v5→v6 同时改变了这两个变量。在 `2026-08-15` 历史冷启动结果中，v6 比 v5 高 `0.001854`，但这个差值不能归因。首轮采用三点 “L 型桥接”：

```text
E1：v5 固定均衡哈希分组，D=1024
                    │  仅改 D
                    ▼
E2：v5 固定均衡哈希分组，D=512    ← 新增的桥接点
                    │  仅改分组 ABI
                    ▼
E3：v6 语义均衡分组，D=512
```

再加入同日同协议的 `E0 Base`。这样用 4 个任务即可回答：

1. `E2−E1`：大模型是否在一天约 5.5 亿样本的 Dense 冷启动中反而欠收敛；
2. `E3−E2`：语义分组在严格同参数量下是否真的有效；
3. `max(E1,E2,E3)−E0`：RankMixer 当前距 Base 的真实差距。

## 2. 为什么必须重做这组对比

### 2.1 历史证据只适合排序优先级，不能替代本轮结果

[`background.md`](../docs/background.md) 记录的 `2026-08-15 → 2026-08-16` Dense 冷启动结果为：

| 模型 | AUC | 相对 Base |
|---|---:|---:|
| Base | 0.866960 | 0 |
| v5，哈希均衡，D=1024 | 0.864163 | -0.002797 |
| v6，语义均衡，D=512 | 0.866017 | -0.000943 |
| v9，Base/DCNM 混合方案 | 0.865254 | -0.001706 |

v6 是目前最好的 RankMixer 首日结果，但 v6−v5 的 `+0.001854` 同时混入“宽度”和“分组”两个效应；v9 又一次性改变了 Cross 输入、SwiGLU 宽度、Shortcut 和读出，不能作为 DCN 的单变量证据。

本轮用户指定 `2026-08-14` 为新的 Dense 冷启动日，因此 Base、v5、桥接点和 v6 都必须在这一天独立重跑。旧日期结果只用于确定实验优先级。

### 2.2 E2 不需要新增模型代码

[`cvr_bn_rankmixer_v5.py`](../src/models/rankmixer/cvr_bn_rankmixer_v5.py) 已经把 `rm_hidden_dim` 参数化。将 v5 的 `rm_hidden_dim` 从 1024 改为 512，就得到：

- 与 E1 完全相同的字段成员、Token 顺序和模型实现；
- 与 E3 相同的 `D=512`、主干超参数、读出、任务头和 Dense 参数量；
- Dense 参数从 `348,432,486` 降到 `177,217,126`，与 v6 严格持平。

因此 E2 是一个低风险的 args-only 桥接实验，不是新模型分支。

配套静态测试还对 v5/v6 的 `MLPModel` 做了 AST 结构复核：排除 `__init__` 中的默认 D/分组版本守卫，以及两段冻结分组构建与校验后，其余 **41 个方法在消除 v5/v6 日志字符串差异后结构完全一致**。这为 “E2→E3 的有效建模变量只有字段分组” 提供了代码级证据，而不只是依赖设计文档描述。

## 3. 文献边界：哪些不再重复验证

本轮只复用论文结论来删减实验，不直接把其他场景的结论当作本业务结论。

| 论文证据 | 已经较充分验证的内容 | 本轮处理 |
|---|---|---|
| [RankMixer](https://arxiv.org/html/2507.15551v3) | 原论文已消融残差、Token Mixing、LayerNorm、Per-token FFN，并比较 Self-Attention；还报告在其数据中按 L/D/T 扩展的效果主要由总参数量决定 | 不再消融残差、Mixing、Norm、共享 FFN、Self-Attention、纯深宽扩展 |
| [TokenMixer-Large](https://arxiv.org/html/2602.06563) | 已系统验证 Global Token、Mixing/Reverting、Residual、Per-token SwiGLU、小尺度 Down 初始化、Norm 位置和 Mixing 切分；RMSNorm 在其系统中保持效果并提升吞吐 | v5/v6 已采用其中核心设计，不重复做 v10 的 LayerNorm/PureFlat 组合实验 |
| [RankUp](https://arxiv.org/html/2604.17878) | 在 1,200+ 稀疏特征、32 任务广告 CVR 场景中，随机排列切分相对语义分组提高 Token 独立性/有效秩，并带来正向 Realtime AUC | 不照搬其结论；它与本地 v6 的历史方向相反，恰好说明必须用 E2/E3 做单任务搜索 CVR 的同预算迁移验证 |
| [RankElastor](https://arxiv.org/html/2605.23191) | 在 Criteo/Avazu 上用 Parameterized Full Mixing 与 GLU 改善有效秩；单独给 RankMixer 换 GLU 只有边际收益 | v6 已是 SwiGLU；Full Mixing 对当前 `T=32,D=512` 成本和改造跨度较大，首轮不做 |
| [UniMixer](https://arxiv.org/html/2604.00590) | 将规则 Mixing 参数化并统一多类 Mixing 结构 | 当前仓库 UniMixer 是整套架构替换，无法回答 v5/v6 的局部归因，暂缓 |

特别注意：RankMixer 论文中“`+0.0001` 可视为显著”的说法只适用于论文自己的数据和评估链路，不能直接作为本业务的统计阈值。本轮以本地逐样本 paired 统计为准。

## 4. 首轮实验矩阵

| ID | 模型 | 分组 | D | Dense 参数 | 作用 |
|---|---|---|---:|---:|---|
| E0_BASE | Base SENet + 2-layer DCNM500 | 不适用 | 不适用 | 90,341,785 | 同日绝对锚点 |
| E1_RANDOM_D1024 | RankMixer v5 | 固定均衡哈希分组 | 1024 | 348,432,486 | 历史 v5 结构复现 |
| E2_RANDOM_D512 | RankMixer v5 | 固定均衡哈希分组 | 512 | 177,217,126 | **新增桥接点；只改宽度** |
| E3_SEMANTIC_D512 | RankMixer v6 | 语义约束、容量均衡 | 512 | 177,217,126 | 相对 E2 **只改分组 ABI** |

首轮只排这 4 个任务。不要同时排 v7/v8/v9/v10/UniMixer。

### 4.1 预注册对比

设 `AUC(Ei)` 为同一测试集上的 fst_CVR AUC：

1. 宽度效应：`Δ_width = AUC(E2) − AUC(E1)`；
2. 分组效应：`Δ_group = AUC(E3) − AUC(E2)`；
3. 端到端差距：`Δ_base = max(AUC(E1), AUC(E2), AUC(E3)) − AUC(E0)`。

不把 `E3−E1` 解释成单模块收益，它只是 v5→v6 的组合变化。

### 4.2 为什么不是完整 2×2

完整矩阵还需要“语义分组、D=1024”这一格。首轮不排它，因为业务目标是从当前最佳 D512 路线找可落地版本，E1→E2→E3 已能顺序分解现有 v5→v6 差异。

只有出现下面的交互迹象时，才追加 1 个 `E4_SEMANTIC_D1024`：

- `E2 < E1`，说明哈希分组下 D512 变差；同时
- `E3 > E2` 且 E3 接近或超过 E1，说明语义分组可能改变宽度最优点。

除此之外，不补第四格。

E4 仅作为条件实验保留，不预先实现或排队。只有同时满足上面两条触发条件时，才基于 v6 将 `D` 调到 1024、其余条件保持不变，并提交这个任务。

## 5. 严格控制的训练协议

四个任务必须满足以下条件，否则整组实验无效：

1. 训练数据均为 `2026-08-14:2026-08-14`，测试均为 `2026-08-15:2026-08-15`；
2. `ignore_dense_checkpoint=True`，`enable_dense_warmup=false`；
3. 每个任务使用独立且为空的输出目录，防止 `auto_load_cp=true` 从自己的旧目录误载 Dense；
4. Sparse embedding 允许按现有协议热启动，但四个任务的主 sparse checkpoint 和 `2026-08-13` item embedding 来源必须完全一致；
5. 同一代码 commit、同一三路数据源、文件过滤/顺序、优化器、学习率、epoch、batch size、资源规格和测试文件数；
6. 测试全量执行：`test_batch_num=-1`、`test_file_num=600`；
7. 四个任务都保存 `search_id, example_id, label, pred`，用于 paired 检验；
8. 启动日志中保存：任务 ID、代码 commit、模型输出目录、Sparse checkpoint、实际训练/测试日期、Dense restore 摘要和最终参数量。

生成的配置保留了当前 v5/v6 模板共用的 sparse checkpoint：

```text
hdfs://pdd-data-ns/apps/nothive/warehouse/bsearch/bsearch_rank_cvr_fea_v10_fst_v3/pt=2026-07-01/checkpoint
```

这是根据 [`background.md`](../docs/background.md) 中“Dense 冷启动、Sparse 共用热启动”的规则做的保守选择，不代表该路径一定是 `2026-08-14` 队列当前批准的 sparse 基线。提交前只需确认一次：若应换路径，必须四个任务一起换，不能单独修改某一组。

## 6. 可直接排队的配置

配置目录：[`bash/ablation_20260814`](../bash/ablation_20260814)

| ID | 参数文件 | 建议唯一任务名 |
|---|---|---|
| E0 | [`00-base-dcnm-args.txt`](../bash/ablation_20260814/00-base-dcnm-args.txt) | `abl0814_base_dcnm` |
| E1 | [`10-rm-v5-balanced-random-d1024-args.txt`](../bash/ablation_20260814/10-rm-v5-balanced-random-d1024-args.txt) | `abl0814_rm_random_d1024` |
| E2 | [`11-rm-v5-balanced-random-d512-args.txt`](../bash/ablation_20260814/11-rm-v5-balanced-random-d512-args.txt) | `abl0814_rm_random_d512` |
| E3 | [`12-rm-v6-semantic-d512-args.txt`](../bash/ablation_20260814/12-rm-v6-semantic-d512-args.txt) | `abl0814_rm_semantic_d512` |

机器可读预注册信息在 [`manifest.json`](../bash/ablation_20260814/manifest.json)。配置由 [`prepare_configs.py`](../bash/ablation_20260814/prepare_configs.py) 生成；若需统一替换 sparse checkpoint：

```bash
python3 bash/ablation_20260814/prepare_configs.py \
  --checkpoint-import-dir '统一的新 sparse checkpoint 路径'
```

生成器只写本地文件，不提交任务、不访问 HDFS、不删除模型目录。

## 7. 指标与统计判定

### 7.1 指标优先级

1. 主指标：fst_CVR AUC；
2. 护栏指标：COPC、PR-AUC、bucket error；
3. 效率指标：Dense 参数量、总训练时长、step time、峰值显存/内存；
4. 诊断指标：v5/v6 已有的 Pool entropy、Flatten gate；若日志可得，一并保留。

### 7.2 Paired 检验

所有模型在同一测试样本上预测。不要对 1.1 亿行样本做普通 iid bootstrap，因为同一 `search_id` 下的候选相关。应按 `search_id` 成组做 paired block bootstrap，或采用等价的 cluster jackknife/hash-bucket jackknife，报告每个对比的：

- `ΔAUC`；
- 95% paired CI；
- `P(ΔAUC>0)` 或双侧 p-value；
- 有效 search 数和样本数。

### 7.3 省算力的复跑规则

首轮每个方案只跑一次，不预先做全量 3 seeds。

- **明确胜出**：paired 95% CI 下界 `>0`，且 `ΔAUC ≥ +0.0002`；
- **工程等价**：`|ΔAUC| < 0.0001` 且 CI 包含 0，选择参数更少、维护更简单者；
- **灰区**：CI 跨 0，或绝对差值在 `0.0001～0.0002`，只复跑这个不确定 pair 的两个方案；
- **反向明确**：CI 上界 `<0`，停止该方向。

`0.0002` 是本轮的业务晋级门槛，不宣称是普适统计显著线；统计显著仍由本地 paired CI 决定。

### 7.4 逐样本统计工具

[`paired_auc_analysis.py`](../src/models/rankmixer/tools/paired_auc_analysis.py) 可流式读取各任务上传的 `predictions-*.txt`，无需把约 1.1 亿条样本全部放入内存。它执行四件事：

1. 让同一 `search_id` 永远进入同一个固定 hash group；
2. 比较每个 group 的样本数、正例数及两种顺序无关指纹，确认四个实验的 `search_id/example_id/label` 集合一致；
3. 用 20,000 个 score bins 计算近似 AUC、PR-AUC 和 COPC；
4. 用 delete-one-hash-group jackknife 计算 paired ΔAUC、标准误和 95% CI。

在仓库根目录执行：

```bash
python3 -m src.models.rankmixer.tools.paired_auc_analysis \
  --run 'E0_BASE=/local/e0/predictions-*.txt' \
  --run 'E1_RANDOM_D1024=/local/e1/predictions-*.txt' \
  --run 'E2_RANDOM_D512=/local/e2/predictions-*.txt' \
  --run 'E3_SEMANTIC_D512=/local/e3/predictions-*.txt' \
  --output-dir /local/rankmixer_ablation_20260814_stats
```

输出包括：

- `paired_stats.md`：可直接回填本报告的对比表；
- `paired_stats.json`：完整的机器可读证据；
- `run_metrics.csv` 和 `paired_contrasts.csv`；
- 每个实验的 `.hist.npz` 缓存，重复分析时无需重扫预测文件。

两个口径不得混淆：最终报告中的单模型 AUC 仍以生产 validator 日志为准；工具的 histogram AUC 是近似值，主要用于 paired 差值和置信区间。如果样本一致性校验失败，工具会停止 paired 对比，此时禁止解释模型差异，必须先补齐或重导预测文件。

本地 100 万行合成预测基准中，默认 `200 groups × 20,000 bins` 的扫描速度约为 87.9 万行/秒，单实验 histogram 约占 64 MB；四个实验同时保留 histogram 约 256 MB。该数字只用于确认复杂度量级，正式耗时仍取决于预测文件所在存储和服务器 Python 环境。

### 7.5 最终报告终审器

完成 [`rankmixer_ablation_20260814_results.csv`](rankmixer_ablation_20260814_results.csv) 并生成 `paired_stats.json` 后，使用 [`finalize_ablation_report.py`](../src/models/rankmixer/tools/finalize_ablation_report.py) 生成独立的最终报告：

```bash
python3 -m src.models.rankmixer.tools.finalize_ablation_report \
  --paired-stats /local/rankmixer_ablation_20260814_stats/paired_stats.json
```

终审器不会覆盖本预注册报告。它会先检查：四个任务状态、训练/测试日期、Dense 冷启动标记、Dense 参数量、代码 commit、共用 sparse checkpoint、唯一 task/model/prediction 路径，以及 paired 样本一致性。全部通过后才生成 `introduce/rankmixer_ablation_20260814_final_report.md`；否则只生成 `.audit.json` 并列出缺失证据。若某个对比处于灰区，报告状态为 `needs_pair_rerun`，且只列出需要复跑的 pair。

配套端到端测试已用四组同样本合成预测完整走通“逐样本文件 → paired_stats.json → validator 结果 CSV → 最终报告”，覆盖默认三组对比和最佳 RankMixer 选择，避免仅分别测试两个工具却遗漏组合字段不兼容。

## 8. 结果判读矩阵

| 观察结果 | 结论 | 下一步 |
|---|---|---|
| `E2>E1`，`E3≈E2` | v6 历史收益主要来自更适合单日冷启动的 D512；人工语义分组无增益 | 选择 v5 哈希均衡 D512，减少人工分组维护 |
| `E2>E1`，`E3>E2` | 宽度和语义分组均有贡献 | 选择 v6 语义 D512 |
| `E2≈E1`，`E3>E2` | 收益主要来自语义分组 | 选择 v6 语义 D512；不做更多宽度扫参 |
| `E2<E1`，但 `E3>E2` 且接近/超过 E1 | 分组与宽度存在潜在交互 | 只追加 E4 语义 D1024 |
| E1/E2/E3 都稳定落后 Base | 核心差距不在分组/宽度 | 停止堆版本；进入第 9 节的单变量读出实验 |
| 任一 RankMixer 与 Base 工程等价 | 优先比较资源、稳定性和维护成本 | D512 优先；再决定是否做一个业务定制读出实验 |

## 9. 唯一预留的新结构实验：三桶层级池化

这个实验 **不与首轮一起排队**。只有首轮选出 D512 分组赢家，且它仍稳定落后 Base 或与 Base 接近时，才增加 1 个任务。

### 9.1 业务问题

当前 v6 的 Global-conditioned Pool 在 31 个 Local Token 上做一次全局 softmax。若初始化阶段各 Token score 接近，Common/Item/Creative 三桶得到的总先验质量约为：

```text
Common  = 10/31
Item    = 20/31
Creative=  1/31
```

Creative 桶只有一个 Token，容易在单日冷启动早期被 30 个其他 Token 的总质量压制；同时 Flatten 支路的 gate 初始化为 `sigmoid(-2)≈0.119`，早期补偿有限。这是当前三桶数据结构特有的问题，不是论文已经回答的通用模块消融。

### 9.2 严格单变量设计

仅替换 Pool，其他部分全部继承首轮 D512 赢家：

1. 仍用现有 Global query / Local key 计算 Token score；
2. 分别在 Common 10、Item 20、Creative 1 个 Token 内做 softmax，得到 3 个桶向量；
3. 用 Global Token 产生 3 维样本级 bucket gate；
4. gate bias 初始化为 `log([10,20,1])`，使 score 相等时复现当前 `10:20:1` 的总质量先验，再允许训练按样本改变三桶权重；
5. 输出仍为 512 维，Global/Flatten/任务头完全不变；新增参数约 `512×3+3=1,539`，相对 177.217M 可忽略。

预注册对比为：`层级三桶 Pool − 首轮 D512 赢家`。若没有正向 paired 证据，不继续扩展 Pool 变体。

### 9.3 为什么不先加 DCN

[TokenMixer-Large](https://arxiv.org/html/2602.06563) 报告 DCN 在 150M 规模仍可能有收益、随模型变大逐渐消失；本地 Base 也说明显式 Cross 很强。但当前 v8/v9 同时改变了 DCN、输入视图、SwiGLU 宽度、Shortcut 和读出，v9 又已经弱于 v6，因此直接重跑 v8/v9仍无法归因。先解决分组/宽度，再做三桶读出的单变量实验；只有这些都失败，才值得另建“v6 + 受控 DCN、其余完全不变”的对照。

<!-- AUTO_RESULTS_START -->

## 10. 结果回填区

运行信息同时填写到 [`rankmixer_ablation_20260814_results.csv`](rankmixer_ablation_20260814_results.csv)。

### 10.1 原始结果

| ID | 任务 ID | AUC | COPC | PR-AUC | Bucket Error | 训练时长 | 预测路径 | 状态 |
|---|---|---:|---:|---:|---:|---:|---|---|
| E0_BASE | 待填 | 待填 | 待填 | 待填 | 待填 | 待填 | 待填 | 未运行 |
| E1_RANDOM_D1024 | 待填 | 待填 | 待填 | 待填 | 待填 | 待填 | 待填 | 未运行 |
| E2_RANDOM_D512 | 待填 | 待填 | 待填 | 待填 | 待填 | 待填 | 待填 | 未运行 |
| E3_SEMANTIC_D512 | 待填 | 待填 | 待填 | 待填 | 待填 | 待填 | 待填 | 未运行 |

### 10.2 Paired 对比

| 对比 | ΔAUC | 95% CI | 判定 | 可支持的结论 |
|---|---:|---|---|---|
| E2 − E1（宽度） | 待填 | 待填 | 待填 | 待填 |
| E3 − E2（分组） | 待填 | 待填 | 待填 | 待填 |
| 最佳 RankMixer − E0 | 待填 | 待填 | 待填 | 待填 |

### 10.3 最终结论

待四个首轮任务及 paired 统计完成后填写。最终结论必须只回答三件事：

1. v5→v6 的收益由宽度、分组还是两者共同贡献；
2. 当前最佳 RankMixer 是否追平 Base，代价是多少；
3. 下一次只追加 E4、三桶层级池化，还是停止该方向。

<!-- AUTO_RESULTS_END -->

## 11. 提交前 2 分钟检查表

- [ ] 四个任务使用同一个代码 commit；
- [ ] 四个任务的主 sparse checkpoint 已确认并完全相同；
- [ ] 四个任务的输出目录不同且为空；
- [ ] 日志显示训练 `2026-08-14`、测试 `2026-08-15`；
- [ ] 日志显示 Dense 未 restore、Sparse 正常 restore；
- [ ] E1 参数量 `348,432,486`；E2/E3 均为 `177,217,126`；
- [ ] 四个任务均输出逐样本预测；
- [ ] 不在首轮夹带 v8/v9/v10/UniMixer 或其他结构变化。
