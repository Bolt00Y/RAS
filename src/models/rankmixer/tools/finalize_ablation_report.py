#!/usr/bin/env python3
"""Audit RankMixer ablation evidence and materialize the final report.

The pre-registration report is intentionally kept immutable.  This tool reads
the completed validator result CSV plus ``paired_stats.json``, enforces the
experiment protocol, recomputes the pre-registered decisions, and writes a new
final report.  It refuses to state a model conclusion when metadata, samples,
or paired evidence are incomplete.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Mapping, MutableMapping, Sequence, Tuple


ROOT = Path(__file__).resolve().parents[4]
DEFAULT_MANIFEST = ROOT / "bash" / "ablation_20260814" / "manifest.json"
DEFAULT_RESULTS = ROOT / "introduce" / "rankmixer_ablation_20260814_results.csv"
DEFAULT_PREREG_REPORT = ROOT / "introduce" / "rankmixer_ablation_20260814_report.md"
DEFAULT_FINAL_REPORT = ROOT / "introduce" / "rankmixer_ablation_20260814_final_report.md"

RESULTS_START = "<!-- AUTO_RESULTS_START -->"
RESULTS_END = "<!-- AUTO_RESULTS_END -->"
REQUIRED_IDS = (
    "E0_BASE",
    "E1_RANDOM_D1024",
    "E2_RANDOM_D512",
    "E3_SEMANTIC_D512",
)
CANDIDATE_IDS = REQUIRED_IDS[1:]
SUCCESS_STATUSES = {"complete", "completed", "success", "succeeded"}
PRACTICAL_WIN = 0.0002
EQUIVALENCE_BAND = 0.0001
METRIC_DELTA_TOLERANCE = 0.0001


def _load_json(path: Path) -> Mapping[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))


def _read_csv(path: Path) -> Tuple[List[str], List[MutableMapping[str, str]]]:
    with path.open(encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise ValueError(f"CSV has no header: {path}")
        return list(reader.fieldnames), [dict(row) for row in reader]


def _first_date(value: object) -> str:
    return str(value).split(":", 1)[0]


def _float_value(
    row: Mapping[str, str], field: str, errors: List[str], run_id: str
) -> float | None:
    raw = row.get(field, "").strip()
    if not raw:
        errors.append(f"{run_id}: missing {field}")
        return None
    try:
        value = float(raw)
    except ValueError:
        errors.append(f"{run_id}: invalid {field}={raw!r}")
        return None
    if not math.isfinite(value):
        errors.append(f"{run_id}: non-finite {field}={raw!r}")
        return None
    return value


def _int_value(
    row: Mapping[str, str], field: str, errors: List[str], run_id: str
) -> int | None:
    value = _float_value(row, field, errors, run_id)
    if value is None:
        return None
    integer = int(value)
    if value != integer:
        errors.append(f"{run_id}: {field} must be an integer, got {value}")
        return None
    return integer


def audit_validator_results(
    rows: Sequence[Mapping[str, str]], manifest: Mapping[str, object]
) -> Dict[str, object]:
    errors: List[str] = []
    warnings: List[str] = []
    by_id: Dict[str, Mapping[str, str]] = {}
    for row in rows:
        run_id = row.get("experiment_id", "").strip()
        if not run_id:
            errors.append("results CSV contains a row without experiment_id")
            continue
        if run_id in by_id:
            errors.append(f"duplicate experiment_id: {run_id}")
        by_id[run_id] = row
    missing_ids = sorted(set(REQUIRED_IDS) - set(by_id))
    extra_ids = sorted(set(by_id) - set(REQUIRED_IDS))
    if missing_ids:
        errors.append(f"missing experiments: {missing_ids}")
    if extra_ids:
        errors.append(f"unexpected first-wave experiments: {extra_ids}")

    protocol = manifest["protocol"]
    expected_train = _first_date(protocol["train_dates"])
    expected_test = _first_date(protocol["test_date"])
    expected_additional = _first_date(protocol["additional_checkpoint_dates"])
    expected_checkpoint = str(protocol["checkpoint_import_dir"])
    manifest_runs = {
        item["id"]: item for item in manifest["experiments"]  # type: ignore[index]
    }

    parsed: Dict[str, Dict[str, object]] = {}
    required_metadata = (
        "git_commit",
        "task_id",
        "model_dir",
        "checkpoint_import_dir",
        "prediction_path",
    )
    for run_id in REQUIRED_IDS:
        if run_id not in by_id:
            continue
        row = by_id[run_id]
        status = row.get("status", "").strip().lower()
        if status not in SUCCESS_STATUSES:
            errors.append(
                f"{run_id}: status must be one of {sorted(SUCCESS_STATUSES)}, got {status!r}"
            )
        for field in required_metadata:
            if not row.get(field, "").strip():
                errors.append(f"{run_id}: missing {field}")
        if row.get("train_date", "").strip() != expected_train:
            errors.append(
                f"{run_id}: train_date must be {expected_train}, got {row.get('train_date')!r}"
            )
        if row.get("test_date", "").strip() != expected_test:
            errors.append(
                f"{run_id}: test_date must be {expected_test}, got {row.get('test_date')!r}"
            )
        if row.get("dense_start", "").strip().lower() != "cold":
            errors.append(f"{run_id}: dense_start must be cold")
        if row.get("additional_checkpoint_date", "").strip() != expected_additional:
            errors.append(
                f"{run_id}: additional_checkpoint_date must be {expected_additional}"
            )
        checkpoint = row.get("checkpoint_import_dir", "").strip()
        if checkpoint and checkpoint != expected_checkpoint:
            errors.append(
                f"{run_id}: checkpoint_import_dir differs from manifest; regenerate all "
                "configs and manifest together if the shared sparse source changed"
            )

        auc = _float_value(row, "auc", errors, run_id)
        dense_params = _int_value(row, "dense_params", errors, run_id)
        expected_params = int(manifest_runs[run_id]["dense_params"])
        if dense_params is not None and dense_params != expected_params:
            errors.append(
                f"{run_id}: dense_params={dense_params} but manifest expects {expected_params}"
            )
        if auc is not None and not 0.5 < auc < 1.0:
            errors.append(f"{run_id}: AUC must be in (0.5, 1.0), got {auc}")

        optional_metrics: Dict[str, float | None] = {}
        for field in (
            "copc",
            "pr_auc",
            "bucket_error",
            "train_wall_time_min",
            "step_time_ms",
            "peak_memory_gb",
        ):
            raw = row.get(field, "").strip()
            if not raw:
                optional_metrics[field] = None
                warnings.append(f"{run_id}: optional metric {field} is missing")
                continue
            try:
                optional_metrics[field] = float(raw)
            except ValueError:
                errors.append(f"{run_id}: invalid optional metric {field}={raw!r}")
                optional_metrics[field] = None

        parsed[run_id] = {
            "experiment_id": run_id,
            "auc": auc,
            "dense_params": dense_params,
            "git_commit": row.get("git_commit", "").strip(),
            "task_id": row.get("task_id", "").strip(),
            "model_dir": row.get("model_dir", "").strip(),
            "checkpoint_import_dir": checkpoint,
            "prediction_path": row.get("prediction_path", "").strip(),
            "status": status,
            **optional_metrics,
        }

    for field in ("git_commit", "checkpoint_import_dir"):
        values = {str(parsed[run_id][field]) for run_id in parsed if parsed[run_id][field]}
        if len(values) > 1:
            errors.append(f"all experiments must share one {field}, got {sorted(values)}")
    for field in ("task_id", "model_dir", "prediction_path"):
        values = [str(parsed[run_id][field]) for run_id in parsed if parsed[run_id][field]]
        if len(values) != len(set(values)):
            errors.append(f"{field} must be unique for every experiment")

    return {
        "passed": not errors,
        "errors": errors,
        "warnings": warnings,
        "runs": parsed,
        "expected_protocol": {
            "train_date": expected_train,
            "test_date": expected_test,
            "additional_checkpoint_date": expected_additional,
            "checkpoint_import_dir": expected_checkpoint,
        },
    }


def _find_contrast(
    paired: Mapping[str, object], left: str, right: str
) -> Mapping[str, object] | None:
    for contrast in paired.get("contrasts", []):  # type: ignore[union-attr]
        if contrast.get("left") == left and contrast.get("right") == right:
            return contrast
    return None


def _decision(delta: float, ci_low: float, ci_high: float) -> str:
    if ci_low > 0.0 and delta >= PRACTICAL_WIN:
        return "clear_win"
    if ci_high < 0.0:
        return "clear_loss"
    if abs(delta) < EQUIVALENCE_BAND and ci_low <= 0.0 <= ci_high:
        return "engineering_tie"
    return "gray_zone_rerun_pair_only"


def audit_paired_evidence(
    paired: Mapping[str, object], official_runs: Mapping[str, Mapping[str, object]]
) -> Dict[str, object]:
    errors: List[str] = []
    warnings: List[str] = []
    if paired.get("status") != "ok":
        errors.append(f"paired_stats status must be ok, got {paired.get('status')!r}")
    integrity = paired.get("integrity", {})
    if not isinstance(integrity, Mapping) or not integrity.get("verified"):
        errors.append("paired sample integrity is not verified")
    paired_run_ids = {
        str(item.get("run_id"))
        for item in paired.get("runs", [])  # type: ignore[union-attr]
        if item.get("run_id")
    }
    if not set(REQUIRED_IDS).issubset(paired_run_ids):
        errors.append(
            f"paired_stats must contain all runs {list(REQUIRED_IDS)}, got {sorted(paired_run_ids)}"
        )

    if any(official_runs[run_id].get("auc") is None for run_id in REQUIRED_IDS):
        errors.append("official validator AUC is missing; cannot select the best RankMixer")
        return {"passed": False, "errors": errors, "warnings": warnings, "contrasts": {}}
    official_auc = {run_id: float(official_runs[run_id]["auc"]) for run_id in REQUIRED_IDS}
    best_id = max(CANDIDATE_IDS, key=lambda run_id: official_auc[run_id])
    specs = {
        "width_E2_minus_E1": ("E2_RANDOM_D512", "E1_RANDOM_D1024"),
        "group_E3_minus_E2": ("E3_SEMANTIC_D512", "E2_RANDOM_D512"),
        "best_rankmixer_minus_base": (best_id, "E0_BASE"),
    }
    audited_contrasts: Dict[str, Dict[str, object]] = {}
    for name, (left, right) in specs.items():
        evidence = _find_contrast(paired, left, right)
        if evidence is None:
            errors.append(
                f"missing paired contrast {name}: {left} minus {right}; rerun the paired tool "
                f"with --contrast '{name}={left}:{right}'"
            )
            continue
        try:
            ci_low = float(evidence["ci_95_low"])
            ci_high = float(evidence["ci_95_high"])
            histogram_delta = float(evidence["full_delta_auc"])
        except (KeyError, TypeError, ValueError) as exc:
            errors.append(f"invalid paired evidence for {name}: {exc}")
            continue
        official_delta = official_auc[left] - official_auc[right]
        delta_gap = official_delta - histogram_delta
        if abs(delta_gap) > METRIC_DELTA_TOLERANCE:
            errors.append(
                f"{name}: validator ΔAUC={official_delta:.8f} and histogram "
                f"ΔAUC={histogram_delta:.8f} differ by {delta_gap:.8f}, exceeding "
                f"the {METRIC_DELTA_TOLERANCE:.4f} consistency tolerance"
            )
        elif abs(delta_gap) > METRIC_DELTA_TOLERANCE / 2:
            warnings.append(
                f"{name}: validator and histogram ΔAUC differ by {delta_gap:.8f}"
            )
        decision = _decision(official_delta, ci_low, ci_high)
        audited_contrasts[name] = {
            "name": name,
            "left": left,
            "right": right,
            "official_delta_auc": official_delta,
            "histogram_delta_auc": histogram_delta,
            "delta_gap": delta_gap,
            "ci_95_low": ci_low,
            "ci_95_high": ci_high,
            "decision": decision,
        }

    return {
        "passed": not errors,
        "errors": errors,
        "warnings": warnings,
        "best_rankmixer": best_id,
        "official_auc": official_auc,
        "contrasts": audited_contrasts,
    }


def derive_conclusion(
    validator_audit: Mapping[str, object], paired_audit: Mapping[str, object]
) -> Dict[str, object]:
    runs = validator_audit["runs"]
    contrasts = paired_audit["contrasts"]
    width = contrasts["width_E2_minus_E1"]
    group = contrasts["group_E3_minus_E2"]
    base = contrasts["best_rankmixer_minus_base"]
    best_id = str(paired_audit["best_rankmixer"])

    uncertain = [
        item
        for item in (width, group, base)
        if item["decision"] == "gray_zone_rerun_pair_only"
    ]
    if uncertain:
        rerun_pairs = [f"{item['left']} + {item['right']}" for item in uncertain]
        return {
            "status": "needs_pair_rerun",
            "attribution": "当前 paired 证据处于灰区，不能给出最终结构归因。",
            "base_result": "最佳 RankMixer 与 Base 的关系尚未稳定。",
            "selected_model": None,
            "next_action": "仅复跑不确定 pair：" + "；".join(rerun_pairs),
            "rerun_pairs": rerun_pairs,
        }

    width_decision = width["decision"]
    group_decision = group["decision"]
    auc = paired_audit["official_auc"]
    interaction = (
        width_decision == "clear_loss"
        and group_decision == "clear_win"
        and float(auc["E3_SEMANTIC_D512"])
        >= float(auc["E1_RANDOM_D1024"]) - EQUIVALENCE_BAND
    )

    if width_decision == "clear_win" and group_decision == "clear_win":
        attribution = "D512 的单日冷启动/参数效率与语义分组均有独立正贡献。"
    elif width_decision == "clear_win" and group_decision in {
        "engineering_tie",
        "clear_loss",
    }:
        attribution = "v5→v6 的主要收益来自 D1024→D512；语义分组没有独立增益。"
    elif group_decision == "clear_win" and width_decision == "engineering_tie":
        attribution = "v5→v6 的主要收益来自语义分组；宽度在本轮工程等价。"
    elif interaction:
        attribution = "D512 在哈希分组下变差，但语义分组可恢复效果，存在宽度×分组交互迹象。"
    elif group_decision == "clear_win":
        attribution = "语义分组有独立正贡献；宽度变化本身不是正向来源。"
    elif width_decision == "engineering_tie" and group_decision == "engineering_tie":
        attribution = "宽度与分组在本轮均工程等价，历史 v5/v6 差异未稳定复现。"
    else:
        attribution = "D512 与语义分组均未形成正向证据，v5→v6 的历史差异不可复现。"

    base_decision = base["decision"]
    if base_decision == "clear_win":
        base_result = "最佳 RankMixer 明确超过同日 Base。"
    elif base_decision == "engineering_tie":
        base_result = "最佳 RankMixer 与同日 Base 工程等价。"
    else:
        base_result = "最佳 RankMixer 仍明确落后同日 Base。"

    if interaction:
        selected_model = None
        next_action = "只追加 E4_SEMANTIC_D1024，用于确认宽度×分组交互。"
    elif base_decision == "clear_loss":
        d512_best = max(
            ("E2_RANDOM_D512", "E3_SEMANTIC_D512"),
            key=lambda run_id: float(auc[run_id]),
        )
        selected_model = best_id
        next_action = (
            f"停止扩展旧版本；若继续，只在 {d512_best} 上追加一个三桶层级池化单变量实验。"
        )
    else:
        selected_model = best_id
        next_action = f"首轮归因完成，选择 {best_id}；不再追加结构实验。"

    best_params = int(runs[best_id]["dense_params"])
    base_params = int(runs["E0_BASE"]["dense_params"])
    return {
        "status": "finalized",
        "attribution": attribution,
        "base_result": base_result,
        "selected_model": selected_model,
        "best_rankmixer": best_id,
        "best_dense_params": best_params,
        "base_dense_params": base_params,
        "parameter_ratio_vs_base": best_params / base_params,
        "next_action": next_action,
        "interaction_followup": interaction,
        "rerun_pairs": [],
    }


def _metric(value: object, digits: int = 6) -> str:
    if value is None or value == "":
        return "未记录"
    return f"{float(value):.{digits}f}"


def render_results_section(
    validator_audit: Mapping[str, object],
    paired_audit: Mapping[str, object],
    conclusion: Mapping[str, object],
    paired_stats_path: Path,
) -> str:
    runs = validator_audit["runs"]
    contrasts = paired_audit["contrasts"]
    generated_at = datetime.now(timezone.utc).isoformat()
    lines = [
        "## 10. 最终实验结果",
        "",
        f"> 自动终审状态：`{conclusion['status']}`  ",
        f"> 生成时间（UTC）：`{generated_at}`",
        "",
        "### 10.1 Validator 原始指标",
        "",
        "| ID | AUC | ΔBase | COPC | PR-AUC | Dense 参数 | 训练时长/min | 任务 ID |",
        "|---|---:|---:|---:|---:|---:|---:|---|",
    ]
    base_auc = float(runs["E0_BASE"]["auc"])
    for run_id in REQUIRED_IDS:
        row = runs[run_id]
        auc = float(row["auc"])
        lines.append(
            f"| {run_id} | {auc:.6f} | {auc - base_auc:+.6f} | "
            f"{_metric(row.get('copc'))} | {_metric(row.get('pr_auc'))} | "
            f"{int(row['dense_params']):,} | {_metric(row.get('train_wall_time_min'), 2)} | "
            f"`{row['task_id']}` |"
        )
    lines.extend(
        [
            "",
            "### 10.2 Paired 成组统计",
            "",
            "| 对比 | Validator ΔAUC | Hist ΔAUC | 95% CI | 判定 |",
            "|---|---:|---:|---|---|",
        ]
    )
    labels = {
        "width_E2_minus_E1": "E2−E1（宽度）",
        "group_E3_minus_E2": "E3−E2（分组）",
        "best_rankmixer_minus_base": "最佳 RankMixer−Base",
    }
    for key in labels:
        item = contrasts[key]
        lines.append(
            f"| {labels[key]} | {item['official_delta_auc']:+.8f} | "
            f"{item['histogram_delta_auc']:+.8f} | "
            f"[{item['ci_95_low']:+.8f}, {item['ci_95_high']:+.8f}] | "
            f"`{item['decision']}` |"
        )
    lines.extend(
        [
            "",
            "### 10.3 最终归因",
            "",
            f"1. **宽度/分组归因**：{conclusion['attribution']}",
            f"2. **相对 Base**：{conclusion['base_result']}",
            f"3. **下一步**：{conclusion['next_action']}",
            "",
            "### 10.4 证据与复现信息",
            "",
            f"- Paired 统计源：`{paired_stats_path}`；",
            f"- 样本一致性：`{paired_audit.get('passed')}`，原始 paired integrity 已通过；",
            f"- 代码 commit：`{runs['E0_BASE']['git_commit']}`；",
            f"- 共用 sparse checkpoint：`{runs['E0_BASE']['checkpoint_import_dir']}`；",
            "- 四个任务的 model_dir、task_id 和 prediction_path 已检查为非空且互不重复；",
            "- 单模型 AUC 取 validator；CI 取逐样本 hash-group jackknife。",
            "",
        ]
    )
    if conclusion["status"] == "needs_pair_rerun":
        lines.extend(
            [
                "> 当前为中期报告，不是最终模型结论；只允许复跑上面列出的灰区 pair。",
                "",
            ]
        )
    return "\n".join(lines)


def materialize_report(prereg_report: Path, output_report: Path, section: str) -> None:
    text = prereg_report.read_text(encoding="utf-8")
    if text.count(RESULTS_START) != 1 or text.count(RESULTS_END) != 1:
        raise ValueError(
            f"Pre-registration report must contain exactly one {RESULTS_START} and {RESULTS_END}"
        )
    start = text.index(RESULTS_START) + len(RESULTS_START)
    end = text.index(RESULTS_END)
    if start >= end:
        raise ValueError("Result markers are in the wrong order")
    final_text = text[:start] + "\n\n" + section.rstrip() + "\n\n" + text[end:]
    output_report.parent.mkdir(parents=True, exist_ok=True)
    output_report.write_text(final_text, encoding="utf-8")


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--results-csv", type=Path, default=DEFAULT_RESULTS)
    parser.add_argument("--paired-stats", type=Path, required=True)
    parser.add_argument("--prereg-report", type=Path, default=DEFAULT_PREREG_REPORT)
    parser.add_argument("--output-report", type=Path, default=DEFAULT_FINAL_REPORT)
    parser.add_argument(
        "--audit-json",
        type=Path,
        help="Defaults to <output-report>.audit.json",
    )
    args = parser.parse_args(argv)

    manifest = _load_json(args.manifest.resolve())
    _, rows = _read_csv(args.results_csv.resolve())
    paired = _load_json(args.paired_stats.resolve())
    validator_audit = audit_validator_results(rows, manifest)
    audit_path = (
        args.audit_json.resolve()
        if args.audit_json
        else args.output_report.resolve().with_suffix(".audit.json")
    )
    audit: Dict[str, object] = {
        "status": "validator_audit_failed" if not validator_audit["passed"] else "pending_paired_audit",
        "validator": validator_audit,
        "paired": None,
        "conclusion": None,
    }
    audit_path.parent.mkdir(parents=True, exist_ok=True)
    if not validator_audit["passed"]:
        audit_path.write_text(
            json.dumps(audit, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
        print(f"Validator audit failed; see {audit_path}", file=sys.stderr)
        return 2

    paired_audit = audit_paired_evidence(paired, validator_audit["runs"])
    audit["paired"] = paired_audit
    if not paired_audit["passed"]:
        audit["status"] = "paired_audit_failed"
        audit_path.write_text(
            json.dumps(audit, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
        print(f"Paired audit failed; see {audit_path}", file=sys.stderr)
        return 2

    conclusion = derive_conclusion(validator_audit, paired_audit)
    audit["status"] = conclusion["status"]
    audit["conclusion"] = conclusion
    audit_path.write_text(
        json.dumps(audit, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    section = render_results_section(
        validator_audit, paired_audit, conclusion, args.paired_stats.resolve()
    )
    materialize_report(
        args.prereg_report.resolve(), args.output_report.resolve(), section
    )
    print(f"Wrote {conclusion['status']} report to {args.output_report.resolve()}")
    return 3 if conclusion["status"] == "needs_pair_rerun" else 0


if __name__ == "__main__":
    raise SystemExit(main())
