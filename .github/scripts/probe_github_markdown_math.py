#!/usr/bin/env python3
"""Probe GitHub's official Markdown renderer for math HTML output."""

from __future__ import annotations

import json
import os
import urllib.request

API_URL = "https://api.github.com/markdown"
SAMPLES = {
    "single-line-math-fence": """# Probe\n\n```math\nx_i = W_i e_i + b_i\n```\n""",
    "aligned-math-fence": """# Probe\n\n```math\n\\begin{aligned}\na_i &= b_i + c_i, \\\\\nd_i &= e_i + f_i.\n\\end{aligned}\n```\n""",
    "ordinary-text-fence": """# Probe\n\n```text\n2 × 768 × 3072 = 4,718,592\n```\n""",
    "inline-code-versus-inline-math": """# Probe\n\nCode: `H ∈ R^[B,T,D]`\n\nMath: $H \\in \\mathbb{R}^{B \\times T \\times D}$\n""",
}


def render(markdown: str) -> str:
    payload = json.dumps(
        {"text": markdown, "mode": "gfm", "context": "Bolt00Y/RAS"}
    ).encode("utf-8")
    headers = {
        "Accept": "text/html",
        "Content-Type": "application/json",
        "User-Agent": "RAS-markdown-math-probe",
        "X-GitHub-Api-Version": "2026-03-10",
    }
    token = os.environ.get("GITHUB_TOKEN")
    if token:
        headers["Authorization"] = f"Bearer {token}"
    request = urllib.request.Request(API_URL, data=payload, headers=headers, method="POST")
    with urllib.request.urlopen(request, timeout=30) as response:
        return response.read().decode("utf-8")


def main() -> int:
    for name, source in SAMPLES.items():
        html = render(source)
        print(f"::group::{name}")
        print(html)
        print("::endgroup::")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
