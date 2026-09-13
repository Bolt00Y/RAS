# Scale up 分析归档

日期：2026-09-13。综合三份实验工作簿、当前源码与args、已有报告/日志及五篇核心原始论文，共审计97条去重后的“方案—测试日”记录。本次没有新增训练。

**核心判断：存在局部增深与容量分配的正向证据，尚不足以建立普遍的scaling law。** Small由两层增到三层，五日平均AUC提高0.0001892；Small-3比更宽E2少31.24%的Dense参数，五日平均AUC高0.0001354，但仍未超过同日Base。

## 阅读顺序

1. [综合分析与可汇报结论](01_Scale_Up综合分析.md)：主要结论、证据分级、所有已产出方案的解释和三张对比图。
2. [实验结果与可比性审计](02_实验结果与可比性审计.md)：原始单元格、去重、日期对齐、缺失结果、数值冲突。
3. [代码容量与计算口径审计](03_代码容量与计算口径审计.md)：实际实现、参数/FLOPs、历史配置绑定限制和硬编码约束。
4. [论文依据与外推边界](04_论文依据与外推边界.md)：五篇原文的扩容证据、适用范围与有效秩诊断。
5. [后续验证方案](05_后续验证方案.md)：以约137M为核心的等预算深度/宽度/FFN对照，及data/compute扩展方法。

## 数据与复算

| 文件 | 用途 |
|---|---|
| [experiment_results.csv](data/experiment_results.csv) | 97条有效数值记录、同日Base、原始单元格和全部重复来源 |
| [model_result_summary.csv](data/model_result_summary.csv) | 各方案在各自已有窗口的概况；不同窗口不可直接相减 |
| [paired_comparisons.csv](data/paired_comparisons.csv) | 同日配对；差值方向为right_model−left_model |
| [matched_five_day_summary.csv](data/matched_five_day_summary.csv) | Base/Small/Small-3/E2/mature_v1统一五日比较 |
| [model_capacity.csv](data/model_capacity.csv) | 当前源码容量、FLOPs估算、源码/args哈希及历史绑定状态 |
| [headline_findings.json](data/headline_findings.json) | 结论中的差值、比例及首日2×2交互差分 |
| [proposed_capacity_grid.csv](data/proposed_capacity_grid.csv) | 新实验预算，全部标记proposal_not_run |
| [extraction_audit.json](data/extraction_audit.json) | 三表完整性、重复和公式校验 |
| [source_manifest.json](data/source_manifest.json) | 分析输入范围、仓库HEAD及文件哈希；不是历史训练commit证明 |

保留原始表，未回填或覆盖。空值不记0、不据此断定运行失败。mature_v5的199M表备注与135.775M当前源码冲突单列，未用于拟合规模关系；不存在跨架构scaling-law拟合或伪造置信区间。

在仓库根目录按顺序复算（提取脚本需要openpyxl，仅用于读取；其他数值脚本只需Python标准库）：

```bash
python3 introduce/scale_up_analysis_2026-09-13/scripts/extract_results.py
python3 introduce/scale_up_analysis_2026-09-13/scripts/audit_capacity.py
python3 introduce/scale_up_analysis_2026-09-13/scripts/analyze_scale_up.py
python3 introduce/scale_up_analysis_2026-09-13/scripts/build_source_manifest.py
```

`analyze_scale_up.py --plots`另外需要matplotlib，重建`assets/`的PNG和SVG。数值复算本身不需要matplotlib。`extract_results.py`只读xlsx，不创建或修改工作簿；具体依赖见脚本。图表纵横轴的bp统一指AUC绝对值0.0001。CSV使用UTF-8；提取表保留单元格来源。

## 本次验证

- 三表去重、AUC/COPC一致性、已填差值公式检查；独立复算所有配对均值及主结论。
- 当前源码静态参数复算，未来预算通式与逐模块算法交叉核对；不称为生产建图或实测profiler。
- 三张PNG逐图查看，保存SVG用于后续编辑；文档内部链接核验。
- 旧有用户修改保留，交付文件集中在本目录；未提交服务器训练、未生成新实验AUC。
