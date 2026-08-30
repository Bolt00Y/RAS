#!/usr/bin/env python3
"""Materialize the pre-registered 2026-08-14 RankMixer ablation configs.

This script only writes local argument files and a manifest.  It never submits a
job, touches HDFS, or deletes an existing model directory.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, Iterable, List


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT_DIR = ROOT / "bash" / "ablation_20260814"
V5_TEMPLATE = ROOT / "bash" / "set-rankmixer-v5-args.txt"
V6_TEMPLATE = ROOT / "bash" / "set-rankmixer-v6-args.txt"

TRAIN_DATE = "2026-08-14:2026-08-14"
TEST_DATE = "2026-08-15:2026-08-15"
ADDITIONAL_CHECKPOINT_DATE = "2026-08-13:2026-08-13"


BASE_MODEL_ARGS: Dict[str, Any] = {
    "use_senet": True,
    "use_senet_bn": True,
    "enable_dense_warmup": False,
    "use_gate_seq_model": False,
    "optimizer": "flood_adam",
    "opt_goal": "first_cvr",
    "save_predict_result": True,
    "skip_tensors": "dcnm-cross;mlp0;bn_input;senet",
    "warm_up_tensors": "dcnm-cross;mlp0;bn_input;senet",
    "dense_bn": False,
    "dcnm_layer": 500,
    "cvr_layers": [2048, 2048, 256],
    "mlt_cvr_layers": [512, 256, 128],
    "ppnet_input": False,
    "use_dcnm_ln": True,
    "showclick_coef": 0,
    "drop_last_files": 2,
    "learning_rate": 0.00002,
    "batch_size": 2048,
    "compression_type": "GZIP",
    "act_type": "prelu",
    "mlp_act_type": "gelu_2",
    "epochs": 1,
    "eval_batch_size": 2048,
    "embedding_size": 17,
    "batch_norm": True,
    "batch_norm_input": 1,
    "sample_format": "parquet",
    "init_type": "normal",
    "sparse_lr": 0.05,
    "sparse_optimizer": "downpour_sgd_opt",
    "train_reset_interval": 10000,
    "upload_log": True,
    "lookup_fuse_num": 20,
    "example_default_value": None,
    "prefetch_num": 100,
    "interleave": 6,
    "test_interleave": 8,
    "dense_scale": 0.01,
    "feature_version": "data.cvr.cvr_fea_v10_base_cold",
    "feature_version_old": "data.cvr.cvr_fea_v10_base_cold",
    "use_riemann_bn": True,
    "enable_wide_cvr": False,
    "enable_mlt_loss": False,
    "enable_last_cvr": False,
    "enable_delay_train_mode": False,
    "mlt_tuning_v2": False,
    "layer_norm_opt": True,
    "din_gated_ffn": True,
}


def _extract_model_args(template: Path) -> Dict[str, Any]:
    for line in template.read_text(encoding="utf-8").splitlines():
        if line.startswith("--model_args='") and line.endswith("'"):
            return json.loads(line[len("--model_args='") : -1])
    raise ValueError(f"No --model_args JSON found in {template}")


def _extract_flag(template: Path, flag: str) -> str:
    prefix = f"--{flag}="
    for line in template.read_text(encoding="utf-8").splitlines():
        if line.startswith(prefix):
            return line[len(prefix) :]
    raise ValueError(f"No --{flag} found in {template}")


def _render_args(
    template: Path,
    module: str,
    model_args: Dict[str, Any],
    checkpoint_import_dir: str,
) -> str:
    replacements = {
        "checkpoint_import_dir": checkpoint_import_dir,
        "model_args": "'" + json.dumps(model_args, ensure_ascii=False, separators=(",", ":")) + "'",
        "train_dates": TRAIN_DATE,
        "test_date": TEST_DATE,
        "additional_checkpoint_dates": ADDITIONAL_CHECKPOINT_DATE,
        "ignore_dense_checkpoint": "True",
        "ignore_sparse_checkpoint": "False",
    }
    seen = set()
    lines: List[str] = []
    source_lines = template.read_text(encoding="utf-8").splitlines()
    if not source_lines:
        raise ValueError(f"Empty template: {template}")

    lines.append(module)
    for line in source_lines[1:]:
        if not line.startswith("--") or "=" not in line:
            lines.append(line)
            continue
        flag = line[2:].split("=", 1)[0]
        if flag in replacements:
            lines.append(f"--{flag}={replacements[flag]}")
            seen.add(flag)
        else:
            lines.append(line)

    missing = set(replacements) - seen
    if missing:
        raise ValueError(f"Template {template} is missing required flags: {sorted(missing)}")
    return "\n".join(lines) + "\n"


def _rankmixer_args(template: Path, hidden_dim: int, group_version: str) -> Dict[str, Any]:
    args = _extract_model_args(template)
    args["rm_hidden_dim"] = hidden_dim
    # v5 accepts extra kwargs; spelling out the frozen group ABI makes the run auditable.
    args["rm_group_version"] = group_version
    args["save_predict_result"] = True
    args["enable_dense_warmup"] = False
    return args


def _experiment_specs(checkpoint_import_dir: str) -> Iterable[Dict[str, Any]]:
    common = {
        "train_dates": TRAIN_DATE,
        "test_date": TEST_DATE,
        "additional_checkpoint_dates": ADDITIONAL_CHECKPOINT_DATE,
        "checkpoint_import_dir": checkpoint_import_dir,
        "dense_start": "cold",
        "save_predictions": True,
    }
    yield {
        **common,
        "id": "E0_BASE",
        "task_name": "abl0814_base_dcnm",
        "file": "00-base-dcnm-args.txt",
        "template": V5_TEMPLATE,
        "module": "models.seq_model.cvr_bn_senet_dcnm_fst.MLPModel",
        "model_args": dict(BASE_MODEL_ARGS),
        "grouping": None,
        "hidden_dim": None,
        "dense_params": 90_341_785,
        "role": "same-day absolute anchor",
    }
    yield {
        **common,
        "id": "E1_RANDOM_D1024",
        "task_name": "abl0814_rm_random_d1024",
        "file": "10-rm-v5-balanced-random-d1024-args.txt",
        "template": V5_TEMPLATE,
        "module": "models.rankmixer.cvr_bn_rankmixer_v5.MLPModel",
        "model_args": _rankmixer_args(
            V5_TEMPLATE, 1024, "rankmixer_v5_balanced_v1"
        ),
        "grouping": "v5 frozen balanced hash grouping",
        "hidden_dim": 1024,
        "dense_params": 348_432_486,
        "role": "upper-left endpoint of the bridge",
    }
    yield {
        **common,
        "id": "E2_RANDOM_D512",
        "task_name": "abl0814_rm_random_d512",
        "file": "11-rm-v5-balanced-random-d512-args.txt",
        "template": V5_TEMPLATE,
        "module": "models.rankmixer.cvr_bn_rankmixer_v5.MLPModel",
        "model_args": _rankmixer_args(
            V5_TEMPLATE, 512, "rankmixer_v5_balanced_v1"
        ),
        "grouping": "v5 frozen balanced hash grouping",
        "hidden_dim": 512,
        "dense_params": 177_217_126,
        "role": "width-only bridge; new args-only arm",
    }
    yield {
        **common,
        "id": "E3_SEMANTIC_D512",
        "task_name": "abl0814_rm_semantic_d512",
        "file": "12-rm-v6-semantic-d512-args.txt",
        "template": V6_TEMPLATE,
        "module": "models.rankmixer.cvr_bn_rankmixer_v6.MLPModel",
        "model_args": _rankmixer_args(
            V6_TEMPLATE, 512, "rankmixer_v6_semantic_balanced_v1"
        ),
        "grouping": "v6 frozen semantic-balanced grouping",
        "hidden_dim": 512,
        "dense_params": 177_217_126,
        "role": "grouping-only endpoint of the bridge",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Local output directory (default: this ablation directory)",
    )
    parser.add_argument(
        "--checkpoint-import-dir",
        default=_extract_flag(V5_TEMPLATE, "checkpoint_import_dir"),
        help=(
            "One common sparse checkpoint source for all four arms. The default "
            "preserves the source already used by the v5/v6 templates."
        ),
    )
    args = parser.parse_args()

    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    specs = list(_experiment_specs(args.checkpoint_import_dir))
    for spec in specs:
        rendered = _render_args(
            spec["template"],
            spec["module"],
            spec["model_args"],
            args.checkpoint_import_dir,
        )
        (output_dir / spec["file"]).write_text(rendered, encoding="utf-8")

    manifest_experiments = []
    for spec in specs:
        item = {key: value for key, value in spec.items() if key not in {"template", "model_args"}}
        manifest_experiments.append(item)
    manifest = {
        "title": "RankMixer 2026-08-14 dense-cold L-shaped bridge ablation",
        "status": "pre_registered_not_submitted",
        "submission_side_effects": False,
        "protocol": {
            "train_dates": TRAIN_DATE,
            "test_date": TEST_DATE,
            "additional_checkpoint_dates": ADDITIONAL_CHECKPOINT_DATE,
            "dense_checkpoint_ignored": True,
            "dense_warmup": False,
            "sparse_checkpoint_shared": True,
            "checkpoint_import_dir": args.checkpoint_import_dir,
        },
        "primary_contrasts": [
            {
                "name": "width_at_v5_grouping",
                "formula": "AUC(E2_RANDOM_D512) - AUC(E1_RANDOM_D1024)",
            },
            {
                "name": "semantic_grouping_at_D512",
                "formula": "AUC(E3_SEMANTIC_D512) - AUC(E2_RANDOM_D512)",
            },
            {
                "name": "rankmixer_vs_base",
                "formula": "AUC(best_of_E1_E2_E3) - AUC(E0_BASE)",
            },
        ],
        "prediction_analysis": {
            "tool": "src/models/rankmixer/tools/paired_auc_analysis.py",
            "finalization_tool": "src/models/rankmixer/tools/finalize_ablation_report.py",
            "input_columns": ["search_id", "example_id", "label", "prediction"],
            "sample_integrity_gate": (
                "per-search-hash-group counts, positives, XOR fingerprint, and SUM fingerprint"
            ),
            "paired_inference": "delete-one-search-hash-group jackknife",
            "official_point_metric": "production validator fst_CVR AUC",
            "note": (
                "The tool's histogram AUC is an approximation used for paired deltas and CI; "
                "do not replace the validator AUC in the final report."
            ),
        },
        "experiments": manifest_experiments,
    }
    (output_dir / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(f"Wrote {len(specs)} configs and manifest to {output_dir}")


if __name__ == "__main__":
    main()
