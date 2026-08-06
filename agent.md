# Agent Instructions

本文件中的规则适用于整个仓库及所有分支中的后续自动化编辑。

## GitHub Markdown 与数学公式：强制规则

所有提交到 GitHub 的 Markdown 文档都必须以 GitHub网页端 **Preview** 的实际解析结果为准。公式是否能被 MathJax 计算只是第二步；第一步是 GitHub 的 Markdown 解析器必须先把对应源码识别为数学元素。

## 1. 先区分“数学内容”和“代码内容”

以下内容属于数学表达式，必须使用 GitHub 数学语法：

- 概率、损失函数、归一化、范数和统计量；
- 参数量、FLOPs、维度和分组规模的算术计算；
- 张量所属空间、映射关系和矩阵形状；
- 模型递推、残差、门控、路由和交叉公式；
- 集合、上下界和超参数搜索空间。

以下内容可以保留为普通代码块：

- 可执行代码、命令和配置；
- 训练实验 ID 列表；
- 纯伪代码流程；
- 使用箭头和缩进表示的架构图。

**关键规则：** 普通反引号和 `text` 代码块永远按代码显示。即使其中包含 `×`、`≈`、`∈`、上下标或等号，GitHub 也不会把它自动转换为公式。

## 2. 块级公式只使用 `math` fenced block

块级公式必须写成下面的形式，围栏独占一行并从行首开始：

````markdown
```math
y = f(x)
```
````

多行公式可以使用 `aligned`：

````markdown
```math
\begin{aligned}
a &= b+c, \\
d &= e+f.
\end{aligned}
```
````

本仓库不使用双美元符号作为块级公式分隔符，也不使用旧式圆括号或方括号 TeX 数学分隔符。

## 3. 行内公式必须具有明确边界

简单行内公式使用 `$...$`。当公式前后紧邻中文或英文标点时，公式两侧必须增加空格，避免 GitHub 的行内分隔符识别失败。

正确示例：

```markdown
隐藏维度为 $D=768$ 。
模型保持 $T=H=16$ ，其输出属于 $\mathbb{R}^{B\times16\times768}$ 。
```

错误示例：

```markdown
隐藏维度为`D=768`。
模型保持，$T=H=16$，其输出为`[B,16,768]`。
```

当行内公式复杂、包含多个集合括号，或无法提供清晰空格边界时，应改成独立的 `math` block，而不是依赖紧贴标点的 `$...$`。

## 4. 禁止的写法

- 禁止把需要渲染的公式放入普通反引号；
- 禁止把计算式或张量空间放入 `text`、`latex` 或无语言标记的代码块；
- 禁止双美元符号块级分隔符；
- 禁止旧式圆括号或方括号 TeX 数学分隔符；
- 禁止把块级公式放入 Markdown 表格单元格、引用块或缩进列表；
- 禁止依赖 `\newcommand`、`\def`、外部 LaTeX package、交叉引用或编号命令；
- 禁止仅凭本地编辑器或单独 MathJax 解析成功就认定 GitHub Preview 可用。

## 5. 推荐的 MathJax 子集

优先使用：

- `\frac`、`\sqrt`、`\sum`、`\prod`；
- `\mathbf`、`\mathrm`、`\mathbb`；
- `\operatorname`、`\left`、`\right`；
- `\begin{aligned}` 与 `\end{aligned}`；
- 常用希腊字母、关系符号和集合符号。

复杂公式应拆成多个独立 `math` block，避免过深环境嵌套。

## 6. 提交前必须完成两层验证

第一层验证 GitHub Markdown 分类：源码中的每个公式都必须在 GitHub 官方 GFM Render API 返回的 HTML 中形成 `math-renderer`，且数学内容不得残留在普通代码块中。

第二层验证 MathJax 语法：每个 `math` block 必须能被 MathJax 正常解析，不得产生未知命令、未闭合环境或错误节点。

本地检查命令：

```bash
python .github/scripts/normalize_markdown_math.py .
GITHUB_TOKEN=<token> python .github/scripts/validate_github_markdown_render.py
npm install --no-save --ignore-scripts mathjax-full@3.2.2
node .github/scripts/validate_markdown_math.mjs .
```

GitHub Actions 工作流 `.github/workflows/markdown-math-preview.yml` 必须保持启用，并以只读方式重新执行上述检查。工作流未通过时，不得将 Markdown 公式修改视为完成。

## 7. 内容保护

- 格式修复不得改变公式的数学含义、符号定义、数值、实验配置或结论；
- 参数计算由代码或计算器复核后再提交；
- 修改已有文档时，应保持正文、标题层级、链接、表格和真正的代码块不变；
- 公式格式变更后必须检查 diff，确认只包含预期的分类、分隔符和必要的 MathJax 兼容调整；
- 新增公式时，必须确认其前后段落在 GitHub Preview 中没有被错误合并。
