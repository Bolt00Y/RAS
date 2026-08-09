#!/usr/bin/env python3
"""Verify that GitHub's official GFM renderer classifies every formula as math.

MathJax syntax validation alone is insufficient: GitHub's Markdown parser must
first turn the source delimiters into `math-renderer` elements. This script
counts source formulas written with `$...$`, standalone `$$` blocks, and legacy
`math` fences, renders each Markdown document through GitHub's Markdown API,
and requires exact agreement with the resulting inline/display math elements.
"""

from __future__ import annotations

import json
import os
import re
import sys
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
API_URL = "https://api.github.com/markdown"
SKIP_DIRS = {".git", "node_modules", ".venv", "venv", "dist", "build"}
FENCE_RE = re.compile(r"^(?P<indent>\s*)(?P<marker>`{3,}|~{3,})(?P<info>.*)$")
INLINE_CODE_RE = re.compile(r"(?<!`)`[^`\n]+`(?!`)")
INLINE_MATH_RE = re.compile(r"(?<!\\)\$(?!\$)(.+?)(?<!\\)\$")
DISPLAY_RENDER_RE = re.compile(
    r'<math-renderer\b[^>]*class="[^"]*\bjs-display-math\b[^"]*"[^>]*>',
    re.IGNORECASE,
)
INLINE_RENDER_RE = re.compile(
    r'<math-renderer\b[^>]*class="[^"]*\bjs-inline-math\b[^"]*"[^>]*>',
    re.IGNORECASE,
)


def iter_markdown_files() -> list[Path]:
    files: list[Path] = []
    for path in ROOT.rglob("*.md"):
        if any(part in SKIP_DIRS for part in path.parts):
            continue
        if path.is_file():
            files.append(path)
    return sorted(files)


def render_gfm(source: str) -> str:
    payload = json.dumps(
        {"text": source, "mode": "gfm", "context": "Bolt00Y/RAS"}
    ).encode("utf-8")
    headers = {
        "Accept": "text/html",
        "Content-Type": "application/json",
        "User-Agent": "RAS-markdown-math-validator",
        "X-GitHub-Api-Version": "2022-11-28",
    }
    token = os.environ.get("GITHUB_TOKEN")
    if token:
        headers["Authorization"] = f"Bearer {token}"
    request = urllib.request.Request(API_URL, data=payload, headers=headers, method="POST")
    with urllib.request.urlopen(request, timeout=60) as response:
        return response.read().decode("utf-8")


def analyze_source(source: str) -> tuple[int, int, list[str]]:
    display_count = 0
    inline_count = 0
    errors: list[str] = []

    open_fence: str | None = None
    open_info = ""
    open_fence_line = 0
    display_open = False
    display_line = 0

    for line_no, line in enumerate(source.splitlines(), start=1):
        match = FENCE_RE.match(line)

        if open_fence is not None:
            closes = (
                match is not None
                and match.group("marker")[0] == open_fence[0]
                and len(match.group("marker")) >= len(open_fence)
                and match.group("info").strip() == ""
            )
            if closes:
                if open_info == "math":
                    display_count += 1
                open_fence = None
                open_info = ""
                open_fence_line = 0
            continue

        if display_open:
            if line.strip() == "$$":
                display_count += 1
                display_open = False
                display_line = 0
            continue

        if match is not None:
            open_fence = match.group("marker")
            open_info = match.group("info").strip().lower()
            open_fence_line = line_no
            continue

        if line.strip() == "$$":
            display_open = True
            display_line = line_no
            continue

        line_without_code = INLINE_CODE_RE.sub("", line)
        if "$$" in line_without_code:
            errors.append(
                f"line {line_no}: display-math delimiters must be standalone `$$` lines"
            )

        inline_count += len(INLINE_MATH_RE.findall(line_without_code))

    if open_fence is not None:
        errors.append(f"line {open_fence_line}: unclosed fenced block")
    if display_open:
        errors.append(f"line {display_line}: unclosed double-dollar display block")

    return display_count, inline_count, errors


def emit_annotation(path: Path, message: str) -> None:
    safe = message.replace("%", "%25").replace("\r", "%0D").replace("\n", "%0A")
    print(f"::error file={path.as_posix()},line=1,title=GitHub math render mismatch::{safe}")


def main() -> int:
    failures: list[tuple[Path, str]] = []
    total_source_display = 0
    total_source_inline = 0
    total_rendered_display = 0
    total_rendered_inline = 0

    for path in iter_markdown_files():
        relative = path.relative_to(ROOT)
        source = path.read_text(encoding="utf-8")
        source_display, source_inline, source_errors = analyze_source(source)
        for error in source_errors:
            failures.append((relative, error))

        rendered = render_gfm(source)
        rendered_display = len(DISPLAY_RENDER_RE.findall(rendered))
        rendered_inline = len(INLINE_RENDER_RE.findall(rendered))

        print(
            f"{relative}: display {source_display}/{rendered_display}, "
            f"inline {source_inline}/{rendered_inline}"
        )

        total_source_display += source_display
        total_source_inline += source_inline
        total_rendered_display += rendered_display
        total_rendered_inline += rendered_inline

        if source_display != rendered_display:
            failures.append(
                (
                    relative,
                    f"{source_display} source display formula(s), but GitHub produced "
                    f"{rendered_display} display math renderer(s)",
                )
            )
        if source_inline != rendered_inline:
            failures.append(
                (
                    relative,
                    f"{source_inline} source inline formula(s), but GitHub produced "
                    f"{rendered_inline} inline math renderer(s)",
                )
            )

    print(
        "Totals: "
        f"display {total_source_display}/{total_rendered_display}, "
        f"inline {total_source_inline}/{total_rendered_inline}"
    )

    if failures:
        print("GitHub Markdown render validation failed:", file=sys.stderr)
        for relative, failure in failures:
            print(f"- {relative}: {failure}", file=sys.stderr)
            if os.environ.get("GITHUB_ACTIONS") == "true":
                emit_annotation(relative, failure)
        return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
