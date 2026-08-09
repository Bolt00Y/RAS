#!/usr/bin/env python3
"""Verify that GitHub's official GFM renderer classifies every formula as math.

The repository uses `$...$` for inline mathematics and `$$...$$` on one
physical line for display mathematics. Existing `math` fenced blocks remain
supported for legacy documents. The validator checks whole-document counts and,
when a mismatch occurs, probes each source formula in its local Markdown context
to produce actionable GitHub annotations.
"""

from __future__ import annotations

import json
import os
import re
import sys
import urllib.request
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
API_URL = "https://api.github.com/markdown"
SKIP_DIRS = {".git", "node_modules", ".venv", "venv", "dist", "build"}
FENCE_RE = re.compile(r"^(?P<indent>\s*)(?P<marker>`{3,}|~{3,})(?P<info>.*)$")
INLINE_CODE_RE = re.compile(r"(?<!`)`[^`\n]+`(?!`)")
DISPLAY_MATH_RE = re.compile(r"^\s*\$\$(?P<expr>.+?)\$\$\s*$")
INLINE_MATH_RE = re.compile(r"(?<!\\)\$(?!\$)(.+?)(?<!\\)\$")
DISPLAY_RENDER_RE = re.compile(
    r'<math-renderer\b[^>]*class="[^"]*\bjs-display-math\b[^"]*"[^>]*>',
    re.IGNORECASE,
)
INLINE_RENDER_RE = re.compile(
    r'<math-renderer\b[^>]*class="[^"]*\bjs-inline-math\b[^"]*"[^>]*>',
    re.IGNORECASE,
)


@dataclass(frozen=True)
class Formula:
    line: int
    display: bool
    probe_markdown: str
    excerpt: str


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


def renderer_counts(rendered: str) -> tuple[int, int]:
    return len(DISPLAY_RENDER_RE.findall(rendered)), len(INLINE_RENDER_RE.findall(rendered))


def analyze_source(source: str) -> tuple[list[Formula], list[str]]:
    formulas: list[Formula] = []
    errors: list[str] = []

    open_fence: str | None = None
    open_info = ""
    open_fence_line = 0
    fence_buffer: list[str] = []

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
                    content = "\n".join(fence_buffer).strip()
                    formulas.append(
                        Formula(
                            line=open_fence_line,
                            display=True,
                            probe_markdown=f"```math\n{content}\n```\n",
                            excerpt=content,
                        )
                    )
                open_fence = None
                open_info = ""
                open_fence_line = 0
                fence_buffer = []
            elif open_info == "math":
                fence_buffer.append(line)
            continue

        if match is not None:
            open_fence = match.group("marker")
            open_info = match.group("info").strip().lower()
            open_fence_line = line_no
            fence_buffer = []
            continue

        line_without_code = INLINE_CODE_RE.sub("", line)
        stripped = line_without_code.strip()
        display_match = DISPLAY_MATH_RE.fullmatch(line_without_code)

        if display_match is not None:
            expression = display_match.group("expr").strip()
            formulas.append(
                Formula(
                    line=line_no,
                    display=True,
                    probe_markdown=f"$${expression}$$\n",
                    excerpt=expression,
                )
            )
            continue

        if stripped == "$$" or "$$" in line_without_code:
            errors.append(
                f"line {line_no}: display math must use `$$formula$$` on one physical line"
            )
            continue

        for inline_match in INLINE_MATH_RE.finditer(line_without_code):
            expression = inline_match.group(1).strip()
            formulas.append(
                Formula(
                    line=line_no,
                    display=False,
                    probe_markdown=line + "\n",
                    excerpt=expression,
                )
            )

    if open_fence is not None:
        errors.append(f"line {open_fence_line}: unclosed fenced block")

    return formulas, errors


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
        formulas, source_errors = analyze_source(source)
        for error in source_errors:
            failures.append((relative, 1, error))

        source_display = sum(item.display for item in formulas)
        source_inline = len(formulas) - source_display
        rendered_display, rendered_inline = renderer_counts(render_gfm(source))

        print(
            f"{relative}: display {source_display}/{rendered_display}, "
            f"inline {source_inline}/{rendered_inline}"
        )

        total_source_display += source_display
        total_source_inline += source_inline
        total_rendered_display += rendered_display
        total_rendered_inline += rendered_inline

        if source_display != rendered_display or source_inline != rendered_inline:
            for item in formulas:
                probe_display, probe_inline = renderer_counts(render_gfm(item.probe_markdown))
                recognized = probe_display == 1 if item.display else probe_inline >= 1
                if not recognized:
                    kind = "display" if item.display else "inline"
                    excerpt = item.excerpt
                    if len(excerpt) > 180:
                        excerpt = excerpt[:177] + "..."
                    failures.append(
                        (
                            relative,
                            item.line,
                            f"GitHub did not classify this {kind} expression as math: {excerpt}",
                        )
                    )

            failures.append(
                (
                    relative,
                    1,
                    f"whole-file count mismatch: display {source_display}/{rendered_display}, "
                    f"inline {source_inline}/{rendered_inline}",
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
