#!/usr/bin/env python3
"""Print inline formulas that GitHub's GFM renderer did not classify as math."""

from __future__ import annotations

import html
import json
import os
import re
import urllib.request
from collections import Counter
from pathlib import Path

root = Path(__file__).resolve().parents[2]
paths = [
    root / "RankMixer/README.md",
    root / "RankMixer/docs/01_literature_and_diagnosis.md",
    root / "RankMixer/docs/02_modification_schemes.md",
    root / "RankMixer/docs/03_experiment_protocol.md",
]
inline_source_re = re.compile(r"(?<!\\)\$(?!\$)(.+?)(?<!\\)\$")
inline_rendered_re = re.compile(
    r'<math-renderer[^>]*class="js-inline-math"[^>]*>(.*?)</math-renderer>',
    re.DOTALL,
)


def source_inline(source: str) -> list[str]:
    out = []
    in_fence = False
    marker = ""
    for line in source.splitlines():
        stripped = line.lstrip()
        if stripped.startswith("```"):
            current = stripped.split(maxsplit=1)[0]
            if in_fence and current.startswith(marker[0]):
                in_fence = False
                marker = ""
            elif not in_fence:
                in_fence = True
                marker = current
            continue
        if in_fence:
            continue
        line = re.sub(r"(?<!`)`[^`\n]+`(?!`)", "", line)
        out.extend(inline_source_re.findall(line))
    return out


def render(source: str) -> str:
    body = json.dumps(
        {"text": source, "mode": "gfm", "context": "Bolt00Y/RAS"}
    ).encode()
    headers = {
        "Accept": "text/html",
        "Content-Type": "application/json",
        "User-Agent": "RAS-inline-math-diagnostic",
        "X-GitHub-Api-Version": "2026-03-10",
        "Authorization": f"Bearer {os.environ['GITHUB_TOKEN']}",
    }
    request = urllib.request.Request(
        "https://api.github.com/markdown", data=body, headers=headers, method="POST"
    )
    with urllib.request.urlopen(request, timeout=60) as response:
        return response.read().decode()


for path in paths:
    source = path.read_text(encoding="utf-8")
    expected = Counter(source_inline(source))
    rendered_html = render(source)
    actual = Counter()
    for raw in inline_rendered_re.findall(rendered_html):
        value = html.unescape(re.sub(r"<[^>]+>", "", raw)).strip()
        if value.startswith("$") and value.endswith("$"):
            value = value[1:-1]
        actual[value] += 1
    missing = expected - actual
    print(f"{path.relative_to(root)}: expected={sum(expected.values())}, rendered={sum(actual.values())}")
    for formula, count in missing.items():
        print(f"  MISSING x{count}: {formula}")
