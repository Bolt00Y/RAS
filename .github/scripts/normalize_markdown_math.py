#!/usr/bin/env python3
"""Validate Markdown math delimiters used by GitHub Preview.

Repository convention for new or edited Markdown:
- inline mathematics uses `$...$`;
- display mathematics uses standalone `$$` delimiter lines;
- legacy `math` fenced blocks remain readable so existing documents do not need
  a risky bulk rewrite, but new documents should not introduce them;
- TeX delimiters written as backslash-parenthesis or backslash-bracket are
  rejected outside code fences and inline-code examples.

The checker is deliberately non-mutating. It verifies balanced code fences and
balanced standalone double-dollar blocks, and reports ambiguous same-line
`$$...$$` constructs because GitHub Preview is most reliable when delimiters are
on their own lines.
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path
from typing import Iterable

FENCE_RE = re.compile(r"^(?P<indent>\s*)(?P<marker>`{3,}|~{3,})(?P<info>.*)$")
INLINE_CODE_RE = re.compile(r"(?<!`)`[^`\n]+`(?!`)")
LEGACY_DELIMITERS = (r"\(", r"\)", r"\[", r"\]")
SKIP_DIRS = {".git", "node_modules", ".venv", "venv", "dist", "build"}


def iter_markdown_files(root: Path) -> Iterable[Path]:
    for path in sorted(root.rglob("*.md")):
        if any(part in SKIP_DIRS for part in path.parts):
            continue
        if path.is_file():
            yield path


def validate_text(text: str, path: Path) -> list[str]:
    lines = text.splitlines()
    errors: list[str] = []

    code_fence_marker: str | None = None
    code_fence_start = 0
    display_math_open = False
    display_math_start = 0

    for line_no, line in enumerate(lines, start=1):
        stripped = line.strip()
        fence_match = FENCE_RE.match(line)

        if code_fence_marker is not None:
            if (
                fence_match
                and fence_match.group("marker")[0] == code_fence_marker[0]
                and len(fence_match.group("marker")) >= len(code_fence_marker)
                and fence_match.group("info").strip() == ""
            ):
                code_fence_marker = None
                code_fence_start = 0
            continue

        if display_math_open:
            if stripped == "$$":
                display_math_open = False
                display_math_start = 0
            elif fence_match:
                errors.append(
                    f"{path}:{line_no}: fenced code block cannot start inside a display-math block"
                )
            continue

        if fence_match:
            code_fence_marker = fence_match.group("marker")
            code_fence_start = line_no
            continue

        if stripped == "$$":
            display_math_open = True
            display_math_start = line_no
            continue

        line_without_inline_code = INLINE_CODE_RE.sub("", line)

        if "$$" in line_without_inline_code:
            errors.append(
                f"{path}:{line_no}: display-math delimiters must be standalone `$$` lines"
            )

        for delimiter in LEGACY_DELIMITERS:
            if delimiter in line_without_inline_code:
                errors.append(
                    f"{path}:{line_no}: legacy math delimiter {delimiter!r} is not allowed"
                )

    if display_math_open:
        errors.append(
            f"{path}:{display_math_start}: unclosed standalone double-dollar math block"
        )
    if code_fence_marker is not None:
        errors.append(
            f"{path}:{code_fence_start}: unclosed fenced code block starting with {code_fence_marker!r}"
        )

    return errors


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("root", nargs="?", default=".")
    parser.add_argument(
        "--write",
        action="store_true",
        help="kept for backwards compatibility; this validator never rewrites files",
    )
    args = parser.parse_args()

    root = Path(args.root).resolve()
    errors: list[str] = []

    for path in iter_markdown_files(root):
        relative = path.relative_to(root)
        errors.extend(validate_text(path.read_text(encoding="utf-8"), relative))

    if errors:
        print("Markdown math delimiter validation failed:", file=sys.stderr)
        for error in errors:
            print(f"- {error}", file=sys.stderr)
        return 1

    print("Markdown math delimiters checked: no errors found.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
