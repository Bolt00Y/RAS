#!/usr/bin/env python3
"""Verify that GitHub's official GFM renderer classifies every formula as math.

The checker compares both counts and normalized TeX contents for source formulas
and GitHub `math-renderer` elements.  On GitHub Actions, missing expressions are
reported as file annotations with their source line numbers.
"""

from __future__ import annotations

import html
import json
import os
import re
import sys
import urllib.request
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
API_URL = "https://api.github.com/markdown"
SKIP_DIRS = {".git", "node_modules", ".venv", "venv", "dist", "build"}
FENCE_RE = re.compile(r"^(?P<indent>\s*)(?P<marker>`{3,}|~{3,})(?P<info>.*)$")
INLINE_CODE_RE = re.compile(r"(?<!`)`[^`\n]+`(?!`)")
INLINE_MATH_RE = re.compile(r"(?<!\\)\$(?!\$)(.+?)(?<!\\)\$")
DISPLAY_RENDER_CONTENT_RE = re.compile(
    r'<math-renderer\b[^>]*class="[^"]*\bjs-display-math\b[^"]*"[^>]*>(.*?)</math-renderer>',
    re.IGNORECASE | re.DOTALL,
)
INLINE_RENDER_CONTENT_RE = re.compile(
    r'<math-renderer\b[^>]*class="[^"]*\bjs-inline-math\b[^"]*"[^>]*>(.*?)</math-renderer>',
    re.IGNORECASE | re.DOTALL,
)


@dataclass(frozen=True)
class Formula:
    expression: str
    line: int
    display: bool


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


def normalize_expression(value: str) -> str:
    value = html.unescape(re.sub(r"<[^>]+>", "", value)).strip()
    if value.startswith("$$") and value.endswith("$$") and len(value) >= 4:
        value = value[2:-2]
    elif value.startswith("$") and value.endswith("$") and len(value) >= 2:
        value = value[1:-1]
    value = value.strip()
    value = re.sub(r"\s+", " ", value)
    return value


def analyze_source(source: str) -> tuple[list[Formula], list[str]]:
    formulas: list[Formula] = []
    errors: list[str] = []

    open_fence: str | None = None
    open_info = ""
    open_fence_line = 0
    fence_buffer: list[str] = []

    display_open = False
    display_line = 0
    display_buffer: list[str] = []

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
                    formulas.append(
                        Formula(
                            normalize_expression("\n".join(fence_buffer)),
                            open_fence_line,
                            True,
                        )
                    )
                open_fence = None
                open_info = ""
                open_fence_line = 0
                fence_buffer = []
            elif open_info == "math":
                fence_buffer.append(line)
            continue

        if display_open:
            if line.strip() == "$$":
                formulas.append(
                    Formula(
                        normalize_expression("\n".join(display_buffer)),
                        display_line,
                        True,
                    )
                )
                display_open = False
                display_line = 0
                display_buffer = []
            else:
                display_buffer.append(line)
            continue

        if match is not None:
            open_fence = match.group("marker")
            open_info = match.group("info").strip().lower()
            open_fence_line = line_no
            fence_buffer = []
            continue

        if line.strip() == "$$":
            display_open = True
            display_line = line_no
            display_buffer = []
            continue

        line_without_code = INLINE_CODE_RE.sub("", line)
        if "$$" in line_without_code:
            errors.append(
                f"line {line_no}: display-math delimiters must be standalone `$$` lines"
            )

        for inline_match in INLINE_MATH_RE.finditer(line_without_code):
            formulas.append(
                Formula(normalize_expression(inline_match.group(1)), line_no, False)
            )

    if open_fence is not None:
        errors.append(f"line {open_fence_line}: unclosed fenced block")
    if display_open:
        errors.append(f"line {display_line}: unclosed double-dollar display block")

    return formulas, errors


def rendered_formulas(rendered: str, display: bool) -> Counter[str]:
    pattern = DISPLAY_RENDER_CONTENT_RE if display else INLINE_RENDER_CONTENT_RE
    return Counter(normalize_expression(raw) for raw in pattern.findall(rendered))


def emit_annotation(path: Path, line: int, message: str) -> None:
    safe = message.replace("%", "%25").replace("\r", "%0D").replace("\n", "%0A")
    print(
        f"::error file={path.as_posix()},line={line},title=GitHub math render mismatch::{safe}"
    )


def main() -> int:
    failures: list[tuple[Path, int, str]] = []
    total_source_display = 0
    total_source_inline = 0
    total_rendered_display = 0
    total_rendered_inline = 0

    for path in iter_markdown_files():
        relative = path.relative_to(ROOT)
        source = path.read_text(encoding="utf-8")
        source_formulas, source_errors = analyze_source(source)
        for error in source_errors:
            failures.append((relative, 1, error))

        rendered = render_gfm(source)
        actual_display = rendered_formulas(rendered, True)
        actual_inline = rendered_formulas(rendered, False)
        source_display = [item for item in source_formulas if item.display]
        source_inline = [item for item in source_formulas if not item.display]

        print(
            f"{relative}: display {len(source_display)}/{sum(actual_display.values())}, "
            f"inline {len(source_inline)}/{sum(actual_inline.values())}"
        )

        total_source_display += len(source_display)
        total_source_inline += len(source_inline)
        total_rendered_display += sum(actual_display.values())
        total_rendered_inline += sum(actual_inline.values())

        for display, source_items, actual in (
            (True, source_display, actual_display),
            (False, source_inline, actual_inline),
        ):
            remaining = actual.copy()
            for item in source_items:
                if remaining[item.expression] > 0:
                    remaining[item.expression] -= 1
                else:
                    kind = "display" if display else "inline"
                    excerpt = item.expression
                    if len(excerpt) > 180:
                        excerpt = excerpt[:177] + "..."
                    failures.append(
                        (
                            relative,
                            item.line,
                            f"GitHub did not classify this {kind} expression as math: {excerpt}",
                        )
                    )

    print(
        "Totals: "
        f"display {total_source_display}/{total_rendered_display}, "
        f"inline {total_source_inline}/{total_rendered_inline}"
    )

    if failures:
        print("GitHub Markdown render validation failed:", file=sys.stderr)
        for relative, line, failure in failures:
            print(f"- {relative}:{line}: {failure}", file=sys.stderr)
            if os.environ.get("GITHUB_ACTIONS") == "true":
                emit_annotation(relative, line, failure)
        return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
