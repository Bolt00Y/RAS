#!/usr/bin/env python3
"""Apply small post-conversion corrections for GitHub math rendering."""

from pathlib import Path

root = Path(__file__).resolve().parents[2]

doc_path = root / "RankMixer/docs/02_modification_schemes.md"
doc = doc_path.read_text(encoding="utf-8")
old = r"""```math
\begin{aligned}
2\times78\times17 &= 1326, \\
14\times77\times17 &= 18{,}326.
\end{aligned}
```"""
new = r"""```math
\begin{aligned}
2\text{ groups}:&\quad 78\times17=1326, \\
14\text{ groups}:&\quad 77\times17=1309.
\end{aligned}
```"""
if old not in doc:
    raise RuntimeError("Expected generated random-split block was not found")
doc = doc.replace(old, new)
inline_fixes = {
    "`16×16`": r"$16\times16$",
    "`alpha≈2`": r"$\alpha\approx2$",
    "`alpha ∈ {1,2}`": r"$\alpha\in\{1,2\}$",
    "`4×1024`": r"$4\times1024$",
}
for source, target in inline_fixes.items():
    if source not in doc:
        raise RuntimeError(f"Expected inline expression was not found: {source}")
    doc = doc.replace(source, target)

# GitHub's inline-dollar recognizer is conservative when delimiters touch CJK
# punctuation. Delimit the affected expressions with spaces explicitly.
doc = doc.replace(
    "backbone、$T=H=16$、PFFN",
    "backbone、 $T=H=16$ 、PFFN",
)
doc = doc.replace(
    r"测试 $\alpha\in\{1,2\}$，不能",
    r"测试 $\alpha\in\{1,2\}$ ，不能",
)
doc = doc.replace(
    r"建议从 $\lambda_g\in\{0,10^{-5},10^{-4}\}$ 搜索",
    r"建议从 $\lambda_g\in\{0,10^{-5},10^{-4}\}$ 搜索",
)
doc_path.write_text(doc, encoding="utf-8")

readme_path = root / "RankMixer/README.md"
readme = readme_path.read_text(encoding="utf-8")
readme = readme.replace(
    r"展开输入：$\mathbb{R}^{B\times20{,}978}$；",
    r"展开输入： $\mathbb{R}^{B\times20{,}978}$ ；",
)
readme = readme.replace(
    r"RankMixer：2 个 block，$T=H=16$；",
    r"RankMixer：2 个 block， $T=H=16$ ；",
)
readme_path.write_text(readme, encoding="utf-8")
print("Applied grouped-dimension, inline-math, and delimiter-spacing corrections.")
