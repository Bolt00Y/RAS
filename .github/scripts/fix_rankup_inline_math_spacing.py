#!/usr/bin/env python3
"""Add unambiguous whitespace around inline math in the RankUp walkthrough.

GitHub's GFM parser may leave `$...$` as ordinary text when a delimiter is
immediately adjacent to CJK punctuation. This one-time fixer only edits inline
math outside fenced blocks and leaves the TeX expression itself unchanged.
"""

from __future__ import annotations

import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
TARGET = ROOT / "RankMixer/docs/04_rankup_paper_walkthrough.md"
FENCE_RE = re.compile(r"^\s*(`{3,}|~{3,})(.*)$")
INLINE_MATH_RE = re.compile(r"(?<!\\)\$(?!\$)(.+?)(?<!\\)\$")


def add_boundaries(line: str) -> str:
    matches = list(INLINE_MATH_RE.finditer(line))
    if not matches:
        return line

    parts: list[str] = []
    cursor = 0
    for match in matches:
        prefix = line[cursor : match.start()]
        if prefix and not prefix[-1].isspace():
            prefix += " "
        parts.append(prefix)
        parts.append(match.group(0))
        cursor = match.end()
        if cursor < len(line) and not line[cursor].isspace():
            parts.append(" ")
    parts.append(line[cursor:])
    return "".join(parts)


def main() -> int:
    source = TARGET.read_text(encoding="utf-8")
    output: list[str] = []
    in_fence = False
    fence_marker = ""

    for line in source.splitlines():
        fence = FENCE_RE.match(line)
        if fence:
            marker = fence.group(1)
            if in_fence and marker[0] == fence_marker[0] and len(marker) >= len(fence_marker):
                in_fence = False
                fence_marker = ""
            elif not in_fence:
                in_fence = True
                fence_marker = marker
            output.append(line)
            continue
        output.append(line if in_fence else add_boundaries(line))

    updated = "\n".join(output) + "\n"
    if updated == source:
        print("RankUp inline math already has unambiguous boundaries.")
        return 0

    TARGET.write_text(updated, encoding="utf-8")
    print("Normalized inline-math boundaries in RankUp walkthrough.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
