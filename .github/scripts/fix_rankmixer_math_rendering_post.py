#!/usr/bin/env python3
"""Correct a grouped-dimension label after the one-time math conversion."""

from pathlib import Path

root = Path(__file__).resolve().parents[2]
path = root / "RankMixer/docs/02_modification_schemes.md"
text = path.read_text(encoding="utf-8")
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
if old not in text:
    raise RuntimeError("Expected generated random-split block was not found")
path.write_text(text.replace(old, new), encoding="utf-8")
print("Corrected random-split per-group dimensions.")
