# Agent Instructions

本文件中的规则适用于整个仓库及所有分支中的后续自动化编辑。

## GitHub Markdown 与数学公式：强制规则

所有提交到 GitHub 的 Markdown 文档都必须以 GitHub 网页端 **Preview** 的实际解析规则为准。GitHub 使用 MathJax 渲染数学表达式；本仓库采用比 GitHub 官方语法更严格的子集，以消除 Markdown 段落和换行解析歧义。

### 1. 块级公式只允许使用 `math` fenced block

块级公式必须写成下面的形式，围栏必须独占一行并从行首开始：

````markdown
```math
y = f(x)
```
````

多行公式可以在 `math` block 内使用 `aligned`：

````markdown
```math
\begin{aligned}
a &= b + c, \\
d &= e + f.
\end{aligned}
```
````

本仓库不再使用双美元符号作为块级公式分隔符。即使该语法被 GitHub 文档列为可选方式，也不得用于本仓库，因为它更容易受到 Markdown 段落边界、列表缩进和换行的影响。

### 2. 行内公式

简单行内公式使用单美元符号包围；若公式包含可能与 Markdown 冲突的字符，则使用 GitHub 支持的“美元符号加反引号”形式。示例：

````markdown
简单形式：$x_i + y_i$
稳健形式：$`x_i + y_i`$
````

不要把需要排版显示的公式写成普通行内代码。

### 3. 禁止的写法

- 禁止旧式圆括号或方括号 TeX 数学分隔符；
- 禁止双美元符号块级分隔符；
- 禁止把块级公式放入 Markdown 表格单元格、引用块或缩进列表；
- 禁止在公式中依赖 `\newcommand`、`\def`、外部 LaTeX package、交叉引用或编号命令；
- 禁止把数学公式包在普通 `text`、`latex` 或无语言标记的代码块中；
- 禁止仅凭本地编辑器渲染成功就认定 GitHub Preview 一定可用。

### 4. 推荐的 MathJax 子集

优先使用 MathJax 标准支持的命令，例如：

- `\frac`、`\sqrt`、`\sum`、`\prod`；
- `\mathbf`、`\mathrm`、`\mathbb`；
- `\operatorname`、`\left`、`\right`；
- `\begin{aligned}` 与 `\end{aligned}`；
- 常用希腊字母、关系符号和集合符号。

对复杂公式应尽量拆成多个独立 `math` block，避免过深的环境嵌套。

### 5. 提交前自动规范化与验证

每次修改 Markdown 后必须运行：

```bash
python .github/scripts/normalize_markdown_math.py --write .
npm install --no-save --ignore-scripts mathjax-full@3.2.2
node .github/scripts/validate_markdown_math.mjs .
```

规范化脚本会把遗留的块级双美元符号公式转换为 `math` fenced block，并拒绝旧式分隔符。验证脚本会提取所有 `math` block，使用 MathJax 逐个解析；任何未知命令、未闭合环境或 MathJax error node 都必须导致提交失败。

GitHub Actions 工作流 `.github/workflows/markdown-math-preview.yml` 必须保持启用，用于在远端再次执行同一套规范化与 MathJax 检查。

### 6. 内容保护

- 格式修复不得改变公式的数学含义、符号定义、数值、实验配置或结论；
- 修改已有文档时，应保持正文、标题层级、链接、表格和普通代码块不变；
- 公式格式变更后必须检查 `git diff`，确认差异仅包含预期的围栏或必要的 MathJax 兼容调整；
- 新增公式时，必须同时检查其前后段落在 GitHub Preview 中没有被错误合并。
