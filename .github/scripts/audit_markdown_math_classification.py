#!/usr/bin/env python3
"""Reject equation-like inline code in RankMixer research Markdown."""

from __future__ import annotations

import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
TARGETS = [
    ROOT / "RankMixer/README.md",
    *sorted((ROOT / "RankMixer/docs").glob("*.md")),
]
INLINE_CODE_RE = re.compile(r"(?<!`)`([^`\n]+)`(?!`)")
FENCE_RE = re.compile(r"^\s*(`{3,}|~{3,})(.*)$")
SIMPLE_EQUATION_RE = re.compile(
    r"^(?:"
    r"[A-Za-z](?:_[A-Za-z0-9]+)?\s*(?:=|>=|<=|>|<)\s*.+"
    r"|\[[A-Za-z0-9_,\s]+\]"
    r"|\d+(?:\.\d+)?\s*[×x]\s*\d+"
    r"|(?:log|exp|sigmoid|softmax)\([^)]*\)"
    r"|[A-Za-z0-9_]+\s*->\s*[A-Za-z0-9_]+"
    r")$"
)


def audit(path: Path) -> list[str]:
    errors: list[str] = []
    in_fence = False
    marker = ""
    for line_no, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        fence = FENCE_RE.match(line)
        if fence:
            current = fence.group(1)
            if in_fence and current[0] == marker[0] and len(current) >= len(marker):
                in_fence = False
                marker = ""
            elif not in_fence:
                in_fence = True
                marker = current
            continue
        if in_fence:
            continue
        for match in INLINE_CODE_RE.finditer(line):
            value = match.group(1).strip()
            if SIMPLE_EQUATION_RE.fullmatch(value):
                errors.append(
                    f"{path.relative_to(ROOT)}:{line_no}: equation-like inline code `{value}` must use math syntax"
                )
    return errors


def main() -> int:
    errors = [error for path in TARGETS for error in audit(path)]
    if errors:
        print("Markdown math classification audit failed:", file=sys.stderr)
        for error in errors:
            print(f"- {error}", file=sys.stderr)
        return 1
    print("No simple equations or tensor shapes remain in inline code.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
