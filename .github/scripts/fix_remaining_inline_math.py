#!/usr/bin/env python3
"""Convert the final equation-like inline code spans in the RankMixer design doc."""

from pathlib import Path

root = Path(__file__).resolve().parents[2]
path = root / "RankMixer/docs/02_modification_schemes.md"
text = path.read_text(encoding="utf-8")
replacements = {
    "按 `log(cardinality)`、coverage": r"按 $\log(\mathrm{cardinality})$ 、coverage",
    "会使 `T=17`，而": r"会使 $T=17$ ，而",
    "原始 `H=T` 约束": r"原始 $H=T$ 约束",
    "将 `[B,20978]` 投影": r"将 $\mathbb{R}^{B\times20{,}978}$ 投影",
    "reshape 回 `[B,16,768]`": r"reshape 回 $\mathbb{R}^{B\times16\times768}$",
    "不在 `[B,20978]` 上": r"不在 $\mathbb{R}^{B\times20{,}978}$ 上",
}
for source, target in replacements.items():
    if source not in text:
        raise RuntimeError(f"Expected source fragment was not found: {source}")
    text = text.replace(source, target)
path.write_text(text, encoding="utf-8")
print(f"Converted {len(replacements)} remaining inline equations.")
