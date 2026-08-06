#!/usr/bin/env python3
"""Normalize Markdown display math to GitHub's fenced `math` syntax.

The script is intentionally conservative:
- only standalone double-dollar delimiter lines outside code fences are converted;
- legacy TeX delimiters outside code fences are rejected;
- unbalanced delimiters and code fences are reported with file/line context.
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path
from typing import Iterable

FENCE_RE = re.compile(r"^(?P<indent>\s*)(?P<marker>`{3,}|~{3,})(?P<info>.*)$")
LEGACY_DELIMITERS = (r"\(", r"\)", r"\[", r"\]")
SKIP_DIRS = {".git", "node_modules", ".venv", "venv", "dist", "build"}


def iter_markdown_files(root: Path) -> Iterable[Path]:
    for path in sorted(root.rglob("*.md")):
        if any(part in SKIP_DIRS for part in path.parts):
            continue
        if path.is_file():
            yield path


def normalize_text(text: str, path: Path) -> tuple[str, list[str]]:
    lines = text.splitlines()
    out: list[str] = []
    errors: list[str] = []

    code_fence_marker: str | None = None
    display_math_open = False
    display_math_start = 0

    for line_no, line in enumerate(lines, start=1):
        stripped = line.strip()

        if display_math_open:
            if stripped == "$$":
                out.append("```")
                display_math_open = False
            else:
                out.append(line)
            continue

        fence_match = FENCE_RE.match(line)
        if code_fence_marker is not None:
            out.append(line)
            if fence_match and fence_match.group("marker").startswith(code_fence_marker[0]) and len(
                fence_match.group("marker")
            ) >= len(code_fence_marker):
                code_fence_marker = None
            continue

        if fence_match:
            code_fence_marker = fence_match.group("marker")
            out.append(line)
            continue

        if stripped == "$$":
            out.append("```math")
            display_math_open = True
            display_math_start = line_no
            continue

        for delimiter in LEGACY_DELIMITERS:
            if delimiter in line:
                errors.append(
                    f"{path}:{line_no}: legacy math delimiter {delimiter!r} is not allowed"
                )

        out.append(line)

    if display_math_open:
        errors.append(
            f"{path}:{display_math_start}: unclosed standalone double-dollar math block"
        )
    if code_fence_marker is not None:
        errors.append(f"{path}: unclosed fenced code block starting with {code_fence_marker!r}")

    normalized = "\n".join(out)
    if text.endswith("\n") or normalized:
        normalized += "\n"
    return normalized, errors


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("root", nargs="?", default=".")
    parser.add_argument(
        "--write",
        action="store_true",
        help="rewrite files in place; without this flag, fail when normalization is needed",
    )
    args = parser.parse_args()

    root = Path(args.root).resolve()
    changed: list[Path] = []
    errors: list[str] = []

    for path in iter_markdown_files(root):
        original = path.read_text(encoding="utf-8")
        normalized, file_errors = normalize_text(original, path.relative_to(root))
        errors.extend(file_errors)
        if normalized != original:
            changed.append(path)
            if args.write:
                path.write_text(normalized, encoding="utf-8")

    if errors:
        print("Markdown math normalization failed:", file=sys.stderr)
        for error in errors:
            print(f"- {error}", file=sys.stderr)
        return 1

    if changed and not args.write:
        print("The following Markdown files require math normalization:", file=sys.stderr)
        for path in changed:
            print(f"- {path.relative_to(root)}", file=sys.stderr)
        return 1

    action = "normalized" if args.write else "checked"
    print(f"Markdown math {action}: {len(changed)} file(s) changed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
