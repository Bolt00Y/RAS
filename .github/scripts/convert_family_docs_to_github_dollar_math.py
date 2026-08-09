#!/usr/bin/env python3
"""Convert the new RankMixer-family documents to GitHub's dollar math syntax.

The conversion is intentionally limited to docs 07-12:
- standalone three-line `$$` blocks are collapsed to one physical
  `$$...$$` line;
- TeX line breaks such as `\\` and aligned environments are preserved;
- unambiguous spaces are added around inline `$...$` expressions;
- two malformed right-delimiter fragments introduced while serializing the
  original Markdown are restored to `\\right)`;
- fenced code blocks are left untouched.
"""

from __future__ import annotations

import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
TARGETS = [
    ROOT / "RankMixer/docs/07_rankmixer_paper_detailed_review.md",
    ROOT / "RankMixer/docs/08_tokenmixer_large_paper_detailed_review.md",
    ROOT / "RankMixer/docs/09_mixformer_paper_detailed_review.md",
    ROOT / "RankMixer/docs/10_unimixer_paper_detailed_review.md",
    ROOT / "RankMixer/docs/11_rankmixer_family_evolution_overview.md",
    ROOT / "RankMixer/docs/12_rankup_paper_detailed_review.md",
]
FENCE_RE = re.compile(r"^\s*(`{3,}|~{3,})(.*)$")
INLINE_MATH_RE = re.compile(r"(?<!\\)\$(?!\$)(.+?)(?<!\\)\$")


def add_inline_boundaries(line: str) -> str:
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


def collapse_formula(lines: list[str], path: Path, start_line: int) -> str:
    content = " ".join(part.strip() for part in lines if part.strip())
    if not content:
        raise RuntimeError(f"{path}:{start_line}: empty display formula")
    if "$$" in content:
        raise RuntimeError(
            f"{path}:{start_line}: nested double-dollar delimiter inside display formula"
        )
    return f"$${content}$$"


def repair_serialized_tex(source: str) -> str:
    # The original tool payload interpreted the `\\r` prefix of two
    # `\\right)` commands as a carriage-return boundary. Restore only the
    # exact malformed fragment observed in the two affected formulas.
    return source.replace(
        "\n\night).",
        "\n" + chr(92) + "right).",
    )


def convert(path: Path) -> bool:
    original = path.read_text(encoding="utf-8")
    source = repair_serialized_tex(original)
    source_lines = source.splitlines()
    output: list[str] = []

    in_fence = False
    fence_marker = ""
    display_open = False
    display_start = 0
    display_buffer: list[str] = []

    for line_no, line in enumerate(source_lines, start=1):
        fence = FENCE_RE.match(line)

        if display_open:
            if line.strip() == "$$":
                output.append(collapse_formula(display_buffer, path, display_start))
                display_open = False
                display_start = 0
                display_buffer = []
            else:
                display_buffer.append(line)
            continue

        if in_fence:
            output.append(line)
            if (
                fence
                and fence.group(1)[0] == fence_marker[0]
                and len(fence.group(1)) >= len(fence_marker)
                and fence.group(2).strip() == ""
            ):
                in_fence = False
                fence_marker = ""
            continue

        if fence:
            in_fence = True
            fence_marker = fence.group(1)
            output.append(line)
            continue

        if line.strip() == "$$":
            display_open = True
            display_start = line_no
            display_buffer = []
            continue

        output.append(add_inline_boundaries(line))

    if display_open:
        raise RuntimeError(f"{path}:{display_start}: unclosed display formula")
    if in_fence:
        raise RuntimeError(f"{path}: unclosed fenced code block")

    updated = "\n".join(output) + "\n"
    if updated == original:
        return False
    path.write_text(updated, encoding="utf-8")
    return True


def main() -> int:
    changed = [path for path in TARGETS if convert(path)]
    print(f"Converted {len(changed)} RankMixer-family document(s).")
    for path in changed:
        print(f"- {path.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
