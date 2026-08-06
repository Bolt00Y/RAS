#!/usr/bin/env python3
"""Convert RankMixer equations that were accidentally formatted as code into GitHub math.

This is a one-time, exact transformation. It deliberately leaves pseudocode,
configuration snippets, experiment IDs, and architecture diagrams as code.
"""

from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def required_replace(text: str, old: str, new: str, label: str) -> str:
    if old not in text:
        raise RuntimeError(f"Missing expected fragment for {label}: {old!r}")
    return text.replace(old, new)


def replace_backticks(text: str, replacements: dict[str, str]) -> str:
    for old, new in replacements.items():
        text = text.replace(f"`{old}`", new)
    return text


def fix_doc_01(text: str) -> str:
    text = replace_backticks(
        text,
        {
            "T=16": "$T=16$",
            "D=768": "$D=768$",
            "L=2": "$L=2$",
            "k": "$k$",
            "F=20978": "$F=20{,}978$",
            "k=4": "$k=4$",
            "H=T": "$H=T$",
            "L=4/6/8": "$L\\in\\{4,6,8\\}$",
            "x0 ⊙ f(xl)": "$x_0\\odot f(x_l)$",
            "20,978² ≈ 440M": "$20{,}978^2\\approx 440\\,\\mathrm{M}$",
            "[B,16,768]": "$\\mathbb{R}^{B\\times16\\times768}$",
            "1234²≈1.52M": "$1234^2\\approx1.52\\,\\mathrm{M}$",
        },
    )
    text = required_replace(
        text,
        """```text
PFFN / block ≈ 75.50M
2 blocks      ≈ 150.99M
Tokenizer     ≈ 16.11M
Dense total   ≈ 167.11M
```""",
        """```math
\\begin{aligned}
P_{\\mathrm{PFFN/block}} &\\approx 75.50\\,\\mathrm{M}, \\\\
P_{\\mathrm{2\\ blocks}} &\\approx 150.99\\,\\mathrm{M}, \\\\
P_{\\mathrm{tokenizer}} &\\approx 16.11\\,\\mathrm{M}, \\\\
P_{\\mathrm{dense,total}} &\\approx 167.11\\,\\mathrm{M}.
\\end{aligned}
```""",
        "doc01 parameter summary",
    )
    return text


def fix_doc_02(text: str) -> str:
    block_replacements = {
        """```text
B = 2048
F = 20978
T = 16
D = 768
L = 2
X0 = [B, T, D]
```""": """```math
\\begin{aligned}
B &= 2048, & F &= 20{,}978, & T &= 16, \\\\
D &= 768, & L &= 2, & X_0 &\\in \\mathbb{R}^{B\\times T\\times D}.
\\end{aligned}
```""",
        """```text
2 groups × 78 fields -> 1326 input dims
14 groups × 77 fields -> 1309 input dims
```""": """```math
\\begin{aligned}
2\\times78\\times17 &= 1326, \\\\
14\\times77\\times17 &= 18{,}326.
\\end{aligned}
```""",
        """```text
X0_random: [B, 16, 768]
```""": """```math
X_{0,\\mathrm{random}}\\in\\mathbb{R}^{B\\times16\\times768}.
```""",
        """```text
4 groups × 83 fields -> 1411 dims
11 groups × 82 fields -> 1394 dims
```""": """```math
\\begin{aligned}
4\\text{ groups}:&\\quad 83\\times17=1411, \\\\
11\\text{ groups}:&\\quad 82\\times17=1394.
\\end{aligned}
```""",
        """```text
X_local: [B, 15, 768]
```""": """```math
X_{\\mathrm{local}}\\in\\mathbb{R}^{B\\times15\\times768}.
```""",
        """```text
W1: 1536 -> 192
W2: 192  -> 768
```""": """```math
W_1:\mathbb{R}^{1536}\\to\\mathbb{R}^{192},\\qquad
W_2:\mathbb{R}^{192}\\to\\mathbb{R}^{768}.
```""",
        """```text
1536 × 192 + 192 × 768 = 442,368
```""": """```math
1536\\times192+192\\times768=442{,}368.
```""",
        """```text
X0 = concat([g, X_local], dim=1) -> [B, 16, 768]
```""": """```math
X_0=\\operatorname{Concat}(g,X_{\\mathrm{local}})\\in\\mathbb{R}^{B\\times16\\times768}.
```""",
        """```text
X: [B, 16, 768]
```""": """```math
X\\in\\mathbb{R}^{B\\times16\\times768}.
```""",
        """```text
H = Mix(X): [B, 16, 768]
```""": """```math
H=\\operatorname{Mix}(X)\\in\\mathbb{R}^{B\\times16\\times768}.
```""",
        """```text
H1 = H + pSwiGLU_mix(RMSNorm(H))
R  = Revert(H1)                  # [B,16,768]
X1 = X + pSwiGLU_orig(RMSNorm(R))
```""": """```math
\\begin{aligned}
H_1 &= H+\\operatorname{pSwiGLU}_{\\mathrm{mix}}(\\operatorname{RMSNorm}(H)), \\\\
R &= \\operatorname{Revert}(H_1)\\in\\mathbb{R}^{B\\times16\\times768}, \\\\
X_1 &= X+\\operatorname{pSwiGLU}_{\\mathrm{orig}}(\\operatorname{RMSNorm}(R)).
\\end{aligned}
```""",
        """```text
D -> 4D -> D
```""": """```math
D\\longrightarrow4D\\longrightarrow D.
```""",
        """```text
2 × 768 × 3072 = 4,718,592
```""": """```math
2\\times768\\times3072=4{,}718{,}592.
```""",
        """```text
3 × 768 × 2048 = 4,718,592
```""": """```math
3\\times768\\times2048=4{,}718{,}592.
```""",
        """```text
2 × 3 × 768 × 1024 = 4,718,592 / token / block
```""": """```math
2\\times3\\times768\\times1024=4{,}718{,}592
\\quad\\text{parameters per token per block}.
```""",
        """```text
L = 4, 6
```""": """```math
L\\in\\{4,6\\}.
```""",
        """```text
lambda_aux ∈ {0.05, 0.1, 0.2}
```""": """```math
\\lambda_{\\mathrm{aux}}\\in\\{0.05,0.1,0.2\\}.
```""",
        """```text
X: [B,16,768]
```""": """```math
X\\in\\mathbb{R}^{B\\times16\\times768}.
```""",
        """```text
v: [B,12288]
```""": """```math
v\\in\\mathbb{R}^{B\\times12{,}288}.
```""",
        """```text
M = 12288 / 48 = 256 blocks
V: [B,256,48]
```""": """```math
M=\\frac{12{,}288}{48}=256,\\qquad
V\\in\\mathbb{R}^{B\\times256\\times48}.
```""",
        """```text
Z_l: [48,48], l=1..4
```""": """```math
Z_l\\in\\mathbb{R}^{48\\times48},\\qquad l\\in\\{1,2,3,4\\}.
```""",
        """```text
A: [256,16]
B: [16,256]
W_global = Sinkhorn(A @ B): [256,256]
```""": """```math
A\\in\\mathbb{R}^{256\\times16},\\quad
B\\in\\mathbb{R}^{16\\times256},\\quad
W_{\\mathrm{global}}=\\operatorname{Sinkhorn}(AB)\\in\\mathbb{R}^{256\\times256}.
```""",
        """```text
Local bases:        4 × 48 × 48 = 9,216
Block coefficients:256 × 4      = 1,024
Global low-rank:    2 × 256 ×16 = 8,192
Total:                            18,432 parameters
```""": """```math
\\begin{aligned}
P_{\\mathrm{local}} &= 4\\times48\\times48=9{,}216, \\\\
P_{\\mathrm{coeff}} &= 256\\times4=1{,}024, \\\\
P_{\\mathrm{global}} &= 2\\times256\\times16=8{,}192, \\\\
P_{\\mathrm{total}} &= 18{,}432.
\\end{aligned}
```""",
        """```text
X0: [B,16,768]
C_i: 768 -> 32
Z0: [B,16,32]
z0: [B,512]
```""": """```math
\\begin{aligned}
X_0 &\\in\\mathbb{R}^{B\\times16\\times768}, \\\\
C_i &: \\mathbb{R}^{768}\\to\\mathbb{R}^{32}, \\\\
Z_0 &\\in\\mathbb{R}^{B\\times16\\times32}, \\\\
z_0 &\\in\\mathbb{R}^{B\\times512}.
\\end{aligned}
```""",
        """```text
h_rm = mean(XL, dim=1): [B,768]
```""": """```math
h_{\\mathrm{rm}}=\\operatorname{MeanPool}(X_L)\\in\\mathbb{R}^{B\\times768}.
```""",
        """```text
g_cross = W_o LayerNorm(z3): [B,768]
```""": """```math
g_{\\mathrm{cross}}=W_o\\operatorname{LayerNorm}(z_3)\\in\\mathbb{R}^{B\\times768}.
```""",
        """```text
Token compression: 16 × 768 × 32        = 393,216
3 low-rank layers: 3 × 2 × 512 × 64     = 196,608
Output projection: 512 × 768             = 393,216
Total ≈ 983,040 parameters
```""": """```math
\\begin{aligned}
P_{\\mathrm{compress}} &= 16\\times768\\times32=393{,}216, \\\\
P_{\\mathrm{cross}} &= 3\\times2\\times512\\times64=196{,}608, \\\\
P_{\\mathrm{output}} &= 512\\times768=393{,}216, \\\\
P_{\\mathrm{total}} &\\approx 983{,}040.
\\end{aligned}
```""",
        """```text
初始上限 0.1
稳定后搜索 {0.1, 0.2, 0.3}
```""": """```math
\\rho_{\\mathrm{init}}\\le 0.1,\\qquad
\\rho\\in\\{0.1,0.2,0.3\\}.
```""",
        """```text
Context: 1536 × 64         = 98,304
Token gate: 64 × 16        = 1,024
Channel gate: 64 × 768     = 49,152
Total ≈ 148,480 parameters
```""": """```math
\\begin{aligned}
P_{\\mathrm{context}} &= 1536\\times64=98{,}304, \\\\
P_{\\mathrm{token}} &= 64\\times16=1{,}024, \\\\
P_{\\mathrm{channel}} &= 64\\times768=49{,}152, \\\\
P_{\\mathrm{total}} &\\approx 148{,}480.
\\end{aligned}
```""",
        """```text
2 × 768 × 3072 = 4,718,592
```""": """```math
2\\times768\\times3072=4{,}718{,}592.
```""",
        """```text
3 × 768 × 1024 = 2,359,296
```""": """```math
3\\times768\\times1024=2{,}359{,}296.
```""",
        """```text
4 × 2,359,296 = 9,437,184
```""": """```math
4\\times2{,}359{,}296=9{,}437{,}184.
```""",
        """```text
2 × 2,359,296 = 4,718,592
```""": """```math
2\\times2{,}359{,}296=4{,}718{,}592.
```""",
        """```text
router_t: R^768 -> R^3
```""": """```math
\\operatorname{router}_t:\\mathbb{R}^{768}\\to\\mathbb{R}^{3}.
```""",
        """```text
XL: [B,16,768]
```""": """```math
X_L\\in\\mathbb{R}^{B\\times16\\times768}.
```""",
    }
    for old, new in block_replacements.items():
        if old in text:
            text = text.replace(old, new)

    text = replace_backticks(
        text,
        {
            "T=H=16": "$T=H=16$",
            "H ∈ R^[B,T,D]": "$H\\in\\mathbb{R}^{B\\times T\\times D}$",
            "T=16": "$T=16$",
            "16×768": "$16\\times768$",
            "k=4": "$k=4$",
            "L=2": "$L=2$",
            "L>=4": "$L\\ge4$",
            "omega_i ∈ R^4": "$\\omega_i\\in\\mathbb{R}^{4}$",
            "Bc=48": "$B_c=48$",
            "M=256": "$M=256$",
            "b=4": "$b=4$",
            "r=16": "$r=16$",
            "m=512, r=64, N_cross=3": "$m=512,\\ r=64,\\ N_{\\mathrm{cross}}=3$",
            "U_l,V_l ∈ R^[512,64]": "$U_l,V_l\\in\\mathbb{R}^{512\\times64}$",
            "E=4": "$E=4$",
            "rho": "$\\rho$",
            "M=1": "$M=1$",
            "[16,768]": "$16\\times768$",
            "R=4": "$R=4$",
            "lambda_g ∈ {0, 1e-5, 1e-4}": "$\\lambda_g\\in\\{0,10^{-5},10^{-4}\\}$",
            "sum(pred)/sum(label)": "$\\sum_i\\hat p_i/\\sum_i y_i$",
        },
    )
    return text


def fix_doc_03(text: str) -> str:
    text = required_replace(
        text,
        """```text
Train:      [t0, t1)
Validation: [t1, t2)
Test:       [t2, t3)
```""",
        """```math
\\mathcal{D}_{\\mathrm{train}}=[t_0,t_1),\\qquad
\\mathcal{D}_{\\mathrm{valid}}=[t_1,t_2),\\qquad
\\mathcal{D}_{\\mathrm{test}}=[t_2,t_3).
```""",
        "doc03 temporal split",
    )
    text = replace_backticks(
        text,
        {
            "P(conversion | click, x)": "$P(\\mathrm{conversion}\\mid\\mathrm{click},x)$",
            "H_b ∈ R^[16,768]": "$H_b\\in\\mathbb{R}^{16\\times768}$",
            "||ΔW||/||W||": "$\\lVert\\Delta W\\rVert/\\lVert W\\rVert$",
            "gamma=0": "$\\gamma=0$",
            "r": "$r$",
            "16×16": "$16\\times16$",
            "L=2": "$L=2$",
        },
    )
    text = text.replace("2×2 factorial design", "$2\\times2$ factorial design")
    return text


def fix_readme(text: str) -> str:
    return replace_backticks(
        text,
        {
            "[B, 20978]": "$\\mathbb{R}^{B\\times20{,}978}$",
            "T=H=16": "$T=H=16$",
        },
    )


def main() -> int:
    targets = {
        ROOT / "RankMixer/docs/01_literature_and_diagnosis.md": fix_doc_01,
        ROOT / "RankMixer/docs/02_modification_schemes.md": fix_doc_02,
        ROOT / "RankMixer/docs/03_experiment_protocol.md": fix_doc_03,
        ROOT / "RankMixer/README.md": fix_readme,
    }
    changed = []
    for path, fixer in targets.items():
        original = path.read_text(encoding="utf-8")
        updated = fixer(original)
        if updated != original:
            path.write_text(updated, encoding="utf-8")
            changed.append(path.relative_to(ROOT))
    print(f"Updated {len(changed)} file(s):")
    for path in changed:
        print(f"- {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
