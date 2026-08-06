#!/usr/bin/env python3
"""Validate Markdown mathematics with GitHub's official GFM renderer.

The local MathJax parser validates TeX syntax. This validator checks the other
half of the pipeline: whether GitHub's Markdown parser actually classifies each
source expression as mathematics instead of ordinary code.
"""

from __future__ import annotations

import html
import json
import os
import re
import sys
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
API_URL = "https://api.github.com/markdown"
SKIP_DIRS = {".git", "node_modules", ".venv", "venv", "dist", "build"}
AUDIT_PATHS = {
    Path("RankMixer/README.md"),
    Path("RankMixer/docs/01_literature_and_diagnosis.md"),
    Path("RankMixer/docs/02_modification_schemes.md"),
    Path("RankMixer/docs/03_experiment_protocol.md"),
}
FENCE_RE = re.compile(r"^(?P<indent>\s*)(?P<marker>`{3,}|~{3,})(?P<info>.*)$")
INLINE_CODE_RE = re.compile(r"(?<!`)`([^`\n]+)`(?!`)")
INLINE_MATH_RE = re.compile(r"(?<!\\)\$(?!\$)(.+?)(?<!\\)\$")
SUSPICIOUS_INLINE_RE = re.compile(
    r"(?:∈|⊙|≈|²|³|Δ|λ|σ|ρ|α|\\in|\\times|R\^\[|\|\|.+?\|\||P\([^`]*\||\d+\s*×\s*\d+)"
)
SUSPICIOUS_BLOCK_RE = re.compile(
    r"(?:∈|⊙|≈|²|³|Δ|λ|σ|ρ|α|R\^\[|\d+\s*×\s*\d+\s*(?:×\s*\d+\s*)*(?:=|≈)|^[A-Za-z][A-Za-z0-9_]*\s*[:=]\s*\[[^\]]+\])",
    re.MULTILINE,
)


def iter_markdown_files() -> list[Path]:
    files = []
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
        "X-GitHub-Api-Version": "2026-03-10",
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
    lines = source.splitlines()
    open_marker: str | None = None
    open_info = ""
    open_line = 0
    buffer: list[str] = []

    for index, line in enumerate(lines, start=1):
        match = FENCE_RE.match(line)
        if open_marker is not None:
            closes = (
                match is not None
                and match.group("marker")[0] == open_marker[0]
                and len(match.group("marker")) >= len(open_marker)
                and match.group("info").strip() == ""
            )
            if closes:
                content = "\n".join(buffer)
                if open_info == "math":
                    display_count += 1
                elif open_info in {"text", ""} and "↓" not in content:
                    if SUSPICIOUS_BLOCK_RE.search(content):
                        errors.append(
                            f"line {open_line}: equation-like content is inside a {open_info or 'plain'} code fence"
                        )
                open_marker = None
                open_info = ""
                open_line = 0
                buffer = []
            else:
                buffer.append(line)
            continue

        if match is not None:
            open_marker = match.group("marker")
            open_info = match.group("info").strip().lower()
            open_line = index
            buffer = []
            continue

        line_without_code = INLINE_CODE_RE.sub("", line)
        inline_count += len(INLINE_MATH_RE.findall(line_without_code))
        for code_match in INLINE_CODE_RE.finditer(line):
            code = code_match.group(1)
            if SUSPICIOUS_INLINE_RE.search(code):
                errors.append(
                    f"line {index}: equation-like inline code should use inline math: `{code}`"
                )

    if open_marker is not None:
        errors.append(f"line {open_line}: unclosed code fence")
    return display_count, inline_count, errors


def main() -> int:
    failures: list[str] = []
    total_source_display = 0
    total_rendered_display = 0
    total_source_inline = 0
    total_rendered_inline = 0

    for path in iter_markdown_files():
        relative = path.relative_to(ROOT)
        source = path.read_text(encoding="utf-8")
        source_display, source_inline, audit_errors = analyze_source(source)
        if relative not in AUDIT_PATHS:
            audit_errors = []

        rendered = render_gfm(source)
        rendered_display = rendered.count('class="js-display-math"')
        rendered_inline = rendered.count('class="js-inline-math"')

        total_source_display += source_display
        total_rendered_display += rendered_display
        total_source_inline += source_inline
        total_rendered_inline += rendered_inline

        if source_display != rendered_display:
            failures.append(
                f"{relative}: {source_display} source math fence(s), but GitHub produced {rendered_display} display math renderer(s)"
            )
        if source_inline != rendered_inline:
            failures.append(
                f"{relative}: {source_inline} source inline formula(s), but GitHub produced {rendered_inline} inline math renderer(s)"
            )
        if '<pre lang="math"' in rendered:
            failures.append(f"{relative}: GitHub rendered a math fence as ordinary code")
        failures.extend(f"{relative}:{error}" for error in audit_errors)

        print(
            f"{relative}: display {source_display}/{rendered_display}, "
            f"inline {source_inline}/{rendered_inline}"
        )

    print(
        "Totals: "
        f"display {total_source_display}/{total_rendered_display}, "
        f"inline {total_source_inline}/{total_rendered_inline}"
    )

    if failures:
        print("GitHub Markdown render validation failed:", file=sys.stderr)
        for failure in failures:
            print(f"- {html.unescape(failure)}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
