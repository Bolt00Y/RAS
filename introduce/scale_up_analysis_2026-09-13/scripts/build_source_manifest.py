#!/usr/bin/env python3
"""Inventory local analysis context without reading credentials or changing sources.

Hashing a file records provenance, not an assertion that every line was reviewed.
Generated analysis files are excluded to avoid a self-referential manifest.
"""
import hashlib
import json
import subprocess
from pathlib import Path

OUT = Path(__file__).resolve().parents[1]
ROOT = OUT.parents[1]
SUFFIXES = {'.md', '.py', '.txt', '.json', '.csv', '.xlsx', '.pdf', '.log'}


def main():
    paths = []
    for dirname in ('docs', 'introduce', 'bash', 'scripts', 'src', 'new_rankmixer_0831'):
        for p in (ROOT / dirname).rglob('*'):
            if p.is_file() and p.suffix in SUFFIXES and OUT not in p.parents and '__pycache__' not in p.parts:
                paths.append(p)
    paths += [ROOT/'README.md', ROOT/'outputs/rankmixer_v6_e2_small_3_20260908.tar.gz']
    head = subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip()
    status = subprocess.check_output(['git','status','--short'],cwd=ROOT,text=True).splitlines()
    result = {
        'analysis_date': '2026-09-13', 'repository_head_at_analysis': head,
        'note': 'Current workspace provenance inventory, not historical training-run attestation or per-line review coverage.',
        'existing_user_changes_preserved': [x for x in status if 'scale_up_analysis_2026-09-13' not in x],
        'context_files': [{'path':str(p.relative_to(ROOT)), 'size_bytes':p.stat().st_size,
                           'sha256':hashlib.sha256(p.read_bytes()).hexdigest()} for p in sorted(set(paths))],
        'result_sources': ['docs/experiments/RankMixer-汇总-0909.xlsx','docs/experiments/副本RankMixer-汇总-0909.xlsx','docs/experiments/RankMixer-汇总.xlsx'],
        'core_papers_reviewed': ['RankMixer','TokenMixer-Large','RankUp','RankElastor','UniMixer'],
        'other_papers_scope': 'MixFormer/OneRec and previous surveys are context inventory, not sources of project AUC or a separate new literature review.',
        'archive_scope': 'Small-3 output tar contains code, args, learning guide/demo and manifest; no new experiment predictions.'
    }
    (OUT/'data/source_manifest.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
    print(f'Inventoried {len(result["context_files"])} context files; HEAD={head}')


if __name__=='__main__':
    main()
