#!/usr/bin/env python3
"""Streaming paired AUC analysis for the RankMixer 2026-08-14 ablation.

The model prediction files contain four tab-separated columns without a header:

    search_id    example_id    label    prediction

The implementation keeps every search request in one deterministic hash group,
builds score histograms in a single streaming pass, verifies that all runs used
the same samples, and estimates paired confidence intervals with a delete-one-
hash-group jackknife.  It is intended for local copies or mounts of the HDFS
``predictions-*.txt`` files and never writes to HDFS.

The reported AUC is histogram-approximated.  Keep the production validator AUC
as the official point metric and use this tool for paired deltas, confidence
intervals, integrity checks, and directional decisions.
"""

from __future__ import annotations

import argparse
import csv
import glob
import gzip
import hashlib
import json
import math
import os
import sys
import time
import zlib
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Mapping, MutableMapping, Sequence, Tuple

try:
    import numpy as np
except ImportError as exc:  # pragma: no cover - depends on the execution image.
    raise SystemExit(
        "numpy is required. Use the Codex workspace Python or the server training Python."
    ) from exc


CACHE_VERSION = 1
MASK64 = (1 << 64) - 1
DEFAULT_BUCKETS = 200
DEFAULT_SCORE_BINS = 20_000
Z_975 = 1.959963984540054

DEFAULT_CONTRASTS = (
    ("width_E2_minus_E1", "E2_RANDOM_D512", "E1_RANDOM_D1024"),
    ("group_E3_minus_E2", "E3_SEMANTIC_D512", "E2_RANDOM_D512"),
)


@dataclass
class RunHistogram:
    run_id: str
    files: Tuple[str, ...]
    file_signature: str
    buckets: int
    score_bins: int
    positive_hist: np.ndarray
    negative_hist: np.ndarray
    sample_count: np.ndarray
    positive_count: np.ndarray
    fingerprint_xor: np.ndarray
    fingerprint_sum: np.ndarray
    prediction_sum: np.ndarray
    hll_registers: np.ndarray
    elapsed_seconds: float

    @property
    def total_samples(self) -> int:
        return int(self.sample_count.sum(dtype=np.uint64))

    @property
    def total_positives(self) -> int:
        return int(self.positive_count.sum(dtype=np.uint64))

    @property
    def total_negatives(self) -> int:
        return self.total_samples - self.total_positives


def _open_prediction(path: str):
    if path.endswith(".gz"):
        return gzip.open(path, "rb")
    return open(path, "rb")


def _parse_scalar(raw: bytes, name: str, path: str, line_number: int) -> float:
    normalized = raw.strip().strip(b"[]")
    try:
        value = float(normalized)
    except ValueError as exc:
        raise ValueError(
            f"{path}:{line_number}: invalid {name} value {raw[:80]!r}"
        ) from exc
    if not math.isfinite(value):
        raise ValueError(f"{path}:{line_number}: non-finite {name}={value}")
    return value


def _canonical_files(patterns: Sequence[str]) -> Tuple[str, ...]:
    matches: List[str] = []
    for pattern in patterns:
        expanded = sorted(glob.glob(os.path.expanduser(pattern)))
        if not expanded and Path(os.path.expanduser(pattern)).is_file():
            expanded = [os.path.expanduser(pattern)]
        if not expanded:
            raise FileNotFoundError(f"Prediction pattern matched no local files: {pattern}")
        matches.extend(expanded)
    files = tuple(sorted({str(Path(path).resolve()) for path in matches}))
    if not files:
        raise FileNotFoundError("No prediction files resolved")
    return files


def _file_signature(files: Sequence[str], buckets: int, score_bins: int) -> str:
    digest = hashlib.sha256()
    digest.update(f"cache={CACHE_VERSION};buckets={buckets};bins={score_bins}\n".encode())
    for path in files:
        stat = os.stat(path)
        digest.update(
            f"{path}\t{stat.st_size}\t{stat.st_mtime_ns}\n".encode("utf-8")
        )
    return digest.hexdigest()


def _row_fingerprint(search_id: bytes, example_id: bytes, label: int) -> int:
    crc = zlib.crc32(search_id)
    crc = zlib.crc32(example_id, crc)
    crc = zlib.crc32(bytes((label,)), crc) & 0xFFFFFFFF
    adler = zlib.adler32(search_id)
    adler = zlib.adler32(example_id, adler)
    adler = zlib.adler32(bytes((label,)), adler) & 0xFFFFFFFF
    return (crc << 32) | adler


def _update_hll(registers: np.ndarray, search_hash: int) -> None:
    """Update a small 32-bit HyperLogLog sketch (p=14)."""
    precision = 14
    index = search_hash >> (32 - precision)
    remainder_bits = 32 - precision
    remainder = search_hash & ((1 << remainder_bits) - 1)
    if remainder == 0:
        rank = remainder_bits + 1
    else:
        rank = remainder_bits - remainder.bit_length() + 1
    if rank > int(registers[index]):
        registers[index] = rank


def _estimate_hll(registers: np.ndarray) -> float:
    m = int(registers.size)
    alpha = 0.7213 / (1.0 + 1.079 / m)
    inv_sum = np.exp2(-registers.astype(np.float64)).sum()
    estimate = alpha * m * m / inv_sum
    zeros = int(np.count_nonzero(registers == 0))
    if zeros and estimate <= 2.5 * m:
        estimate = m * math.log(m / zeros)
    return estimate


def build_histogram(
    run_id: str,
    patterns: Sequence[str],
    buckets: int = DEFAULT_BUCKETS,
    score_bins: int = DEFAULT_SCORE_BINS,
    progress_every: int = 10_000_000,
) -> RunHistogram:
    if buckets < 20:
        raise ValueError("Use at least 20 hash groups for grouped jackknife inference")
    if score_bins < 2_000:
        raise ValueError("Use at least 2,000 score bins for AUC approximation")

    files = _canonical_files(patterns)
    signature = _file_signature(files, buckets, score_bins)
    positive_hist = np.zeros((buckets, score_bins), dtype=np.uint64)
    negative_hist = np.zeros((buckets, score_bins), dtype=np.uint64)
    sample_count = np.zeros(buckets, dtype=np.uint64)
    positive_count = np.zeros(buckets, dtype=np.uint64)
    prediction_sum = np.zeros(buckets, dtype=np.float64)
    fingerprint_xor = [0] * buckets
    fingerprint_sum = [0] * buckets
    hll_registers = np.zeros(1 << 14, dtype=np.uint8)

    started = time.monotonic()
    processed = 0
    next_progress = progress_every if progress_every > 0 else sys.maxsize
    for path in files:
        with _open_prediction(path) as handle:
            for line_number, line in enumerate(handle, start=1):
                stripped = line.rstrip(b"\r\n")
                if not stripped:
                    continue
                parts = stripped.split(b"\t")
                if len(parts) != 4:
                    raise ValueError(
                        f"{path}:{line_number}: expected 4 tab-separated columns, got {len(parts)}"
                    )
                search_id, example_id, raw_label, raw_prediction = parts
                if line_number == 1 and search_id.lower() == b"search_id":
                    continue

                label_value = _parse_scalar(raw_label, "label", path, line_number)
                label = int(label_value)
                if label_value != label or label not in (0, 1):
                    raise ValueError(
                        f"{path}:{line_number}: label must be exactly 0 or 1, got {label_value}"
                    )
                prediction = _parse_scalar(
                    raw_prediction, "prediction", path, line_number
                )
                if prediction < 0.0 or prediction > 1.0:
                    raise ValueError(
                        f"{path}:{line_number}: prediction outside [0,1]: {prediction}"
                    )

                search_hash = zlib.crc32(search_id) & 0xFFFFFFFF
                bucket = search_hash % buckets
                score_bin = min(int(prediction * score_bins), score_bins - 1)
                if label:
                    positive_hist[bucket, score_bin] += 1
                    positive_count[bucket] += 1
                else:
                    negative_hist[bucket, score_bin] += 1
                sample_count[bucket] += 1
                prediction_sum[bucket] += prediction

                row_fingerprint = _row_fingerprint(search_id, example_id, label)
                fingerprint_xor[bucket] ^= row_fingerprint
                fingerprint_sum[bucket] = (
                    fingerprint_sum[bucket] + row_fingerprint
                ) & MASK64
                _update_hll(hll_registers, search_hash)

                processed += 1
                if processed >= next_progress:
                    elapsed = time.monotonic() - started
                    rate = processed / max(elapsed, 1e-9)
                    print(
                        f"[{run_id}] processed {processed:,} rows "
                        f"({rate:,.0f} rows/s)",
                        file=sys.stderr,
                    )
                    next_progress += progress_every

    elapsed = time.monotonic() - started
    if processed == 0:
        raise ValueError(f"Run {run_id} has no prediction rows")
    if int(positive_count.sum()) == 0 or int(positive_count.sum()) == processed:
        raise ValueError(f"Run {run_id} must contain both positive and negative labels")

    return RunHistogram(
        run_id=run_id,
        files=files,
        file_signature=signature,
        buckets=buckets,
        score_bins=score_bins,
        positive_hist=positive_hist,
        negative_hist=negative_hist,
        sample_count=sample_count,
        positive_count=positive_count,
        fingerprint_xor=np.asarray(fingerprint_xor, dtype=np.uint64),
        fingerprint_sum=np.asarray(fingerprint_sum, dtype=np.uint64),
        prediction_sum=prediction_sum,
        hll_registers=hll_registers,
        elapsed_seconds=elapsed,
    )


def save_cache(run: RunHistogram, path: Path) -> None:
    metadata = {
        "cache_version": CACHE_VERSION,
        "run_id": run.run_id,
        "files": list(run.files),
        "file_signature": run.file_signature,
        "buckets": run.buckets,
        "score_bins": run.score_bins,
        "elapsed_seconds": run.elapsed_seconds,
    }
    np.savez(
        path,
        metadata=np.asarray(json.dumps(metadata, ensure_ascii=False)),
        positive_hist=run.positive_hist,
        negative_hist=run.negative_hist,
        sample_count=run.sample_count,
        positive_count=run.positive_count,
        fingerprint_xor=run.fingerprint_xor,
        fingerprint_sum=run.fingerprint_sum,
        prediction_sum=run.prediction_sum,
        hll_registers=run.hll_registers,
    )


def load_cache(
    path: Path,
    run_id: str,
    patterns: Sequence[str],
    buckets: int,
    score_bins: int,
) -> RunHistogram | None:
    if not path.exists():
        return None
    files = _canonical_files(patterns)
    expected_signature = _file_signature(files, buckets, score_bins)
    with np.load(path, allow_pickle=False) as cached:
        metadata = json.loads(str(cached["metadata"].item()))
        if (
            metadata.get("cache_version") != CACHE_VERSION
            or metadata.get("run_id") != run_id
            or metadata.get("file_signature") != expected_signature
            or metadata.get("buckets") != buckets
            or metadata.get("score_bins") != score_bins
        ):
            return None
        return RunHistogram(
            run_id=run_id,
            files=tuple(metadata["files"]),
            file_signature=metadata["file_signature"],
            buckets=buckets,
            score_bins=score_bins,
            positive_hist=cached["positive_hist"].copy(),
            negative_hist=cached["negative_hist"].copy(),
            sample_count=cached["sample_count"].copy(),
            positive_count=cached["positive_count"].copy(),
            fingerprint_xor=cached["fingerprint_xor"].copy(),
            fingerprint_sum=cached["fingerprint_sum"].copy(),
            prediction_sum=cached["prediction_sum"].copy(),
            hll_registers=cached["hll_registers"].copy(),
            elapsed_seconds=float(metadata.get("elapsed_seconds", 0.0)),
        )


def auc_from_histograms(positive: np.ndarray, negative: np.ndarray) -> float:
    positive = np.asarray(positive, dtype=np.float64)
    negative = np.asarray(negative, dtype=np.float64)
    positives = float(positive.sum())
    negatives = float(negative.sum())
    if positives <= 0.0 or negatives <= 0.0:
        return float("nan")
    negatives_below = np.cumsum(negative) - negative
    concordant = np.sum(positive * (negatives_below + 0.5 * negative))
    return float(concordant / (positives * negatives))


def pr_auc_from_histograms(positive: np.ndarray, negative: np.ndarray) -> float:
    positive = np.asarray(positive, dtype=np.float64)
    negative = np.asarray(negative, dtype=np.float64)
    positives = float(positive.sum())
    if positives <= 0.0:
        return float("nan")
    true_positive = np.cumsum(positive[::-1])
    false_positive = np.cumsum(negative[::-1])
    recall = true_positive / positives
    precision = np.divide(
        true_positive,
        true_positive + false_positive,
        out=np.ones_like(true_positive),
        where=(true_positive + false_positive) > 0,
    )
    recall = np.concatenate(([0.0], recall))
    precision = np.concatenate(([1.0], precision))
    # Write the trapezoid rule explicitly for compatibility with older server
    # NumPy versions where ``np.trapezoid`` is not yet available.
    return float(np.sum((precision[1:] + precision[:-1]) * np.diff(recall) * 0.5))


def summarize_run(run: RunHistogram) -> Dict[str, object]:
    positive = run.positive_hist.sum(axis=0, dtype=np.uint64)
    negative = run.negative_hist.sum(axis=0, dtype=np.uint64)
    positives = run.total_positives
    return {
        "run_id": run.run_id,
        "files": list(run.files),
        "sample_count": run.total_samples,
        "positive_count": positives,
        "negative_count": run.total_negatives,
        "positive_rate": positives / run.total_samples,
        "estimated_distinct_search_ids": round(_estimate_hll(run.hll_registers)),
        "histogram_auc": auc_from_histograms(positive, negative),
        "histogram_pr_auc": pr_auc_from_histograms(positive, negative),
        "copc": float(run.prediction_sum.sum()) / positives,
        "elapsed_seconds": run.elapsed_seconds,
        "file_signature": run.file_signature,
    }


def sample_mismatch_details(
    reference: RunHistogram, candidate: RunHistogram
) -> List[Dict[str, object]]:
    if reference.buckets != candidate.buckets:
        return [{"reason": "different bucket counts"}]
    mismatch_mask = (
        (reference.sample_count != candidate.sample_count)
        | (reference.positive_count != candidate.positive_count)
        | (reference.fingerprint_xor != candidate.fingerprint_xor)
        | (reference.fingerprint_sum != candidate.fingerprint_sum)
    )
    details = []
    for bucket in np.flatnonzero(mismatch_mask)[:20]:
        index = int(bucket)
        details.append(
            {
                "bucket": index,
                "reference_samples": int(reference.sample_count[index]),
                "candidate_samples": int(candidate.sample_count[index]),
                "reference_positives": int(reference.positive_count[index]),
                "candidate_positives": int(candidate.positive_count[index]),
                "reference_xor": int(reference.fingerprint_xor[index]),
                "candidate_xor": int(candidate.fingerprint_xor[index]),
                "reference_sum": int(reference.fingerprint_sum[index]),
                "candidate_sum": int(candidate.fingerprint_sum[index]),
            }
        )
    return details


def verify_same_samples(runs: Mapping[str, RunHistogram]) -> Dict[str, object]:
    run_ids = list(runs)
    if len(run_ids) < 2:
        return {"verified": True, "reference": run_ids[0], "comparisons": {}}
    reference = runs[run_ids[0]]
    comparisons = {}
    verified = True
    for run_id in run_ids[1:]:
        details = sample_mismatch_details(reference, runs[run_id])
        comparisons[run_id] = {
            "matches": not details,
            "mismatch_count_shown": len(details),
            "details": details,
        }
        verified = verified and not details
    return {
        "verified": verified,
        "reference": reference.run_id,
        "comparisons": comparisons,
        "fingerprint_note": (
            "Exact per-bucket counts plus commutative CRC32/Adler32 fingerprints; "
            "collision risk is non-zero but very small."
        ),
    }


def _normal_cdf(value: float) -> float:
    return 0.5 * (1.0 + math.erf(value / math.sqrt(2.0)))


def paired_grouped_jackknife(
    left: RunHistogram,
    right: RunHistogram,
    contrast_name: str,
    practical_win: float = 0.0002,
    equivalence_band: float = 0.0001,
) -> Dict[str, object]:
    mismatch = sample_mismatch_details(right, left)
    if mismatch:
        raise ValueError(
            f"Cannot compute paired contrast {contrast_name}: sample fingerprints differ"
        )
    if left.score_bins != right.score_bins:
        raise ValueError("Paired runs must use the same score bin count")

    left_positive = left.positive_hist.sum(axis=0, dtype=np.uint64)
    left_negative = left.negative_hist.sum(axis=0, dtype=np.uint64)
    right_positive = right.positive_hist.sum(axis=0, dtype=np.uint64)
    right_negative = right.negative_hist.sum(axis=0, dtype=np.uint64)
    left_auc = auc_from_histograms(left_positive, left_negative)
    right_auc = auc_from_histograms(right_positive, right_negative)
    full_delta = left_auc - right_auc

    leave_one_out = []
    invalid_buckets = []
    for bucket in range(left.buckets):
        left_loo = auc_from_histograms(
            left_positive - left.positive_hist[bucket],
            left_negative - left.negative_hist[bucket],
        )
        right_loo = auc_from_histograms(
            right_positive - right.positive_hist[bucket],
            right_negative - right.negative_hist[bucket],
        )
        if not math.isfinite(left_loo) or not math.isfinite(right_loo):
            invalid_buckets.append(bucket)
            continue
        leave_one_out.append(left_loo - right_loo)
    if invalid_buckets:
        raise ValueError(
            "Some delete-one-group samples have only one class; increase data volume "
            f"or reduce --buckets. Invalid groups: {invalid_buckets[:20]}"
        )

    group_count = len(leave_one_out)
    loo = np.asarray(leave_one_out, dtype=np.float64)
    pseudovalues = group_count * full_delta - (group_count - 1) * loo
    jackknife_estimate = float(pseudovalues.mean())
    standard_error = float(pseudovalues.std(ddof=1) / math.sqrt(group_count))
    if standard_error == 0.0:
        ci_low = ci_high = jackknife_estimate
        probability_positive = 1.0 if jackknife_estimate > 0 else 0.0
        p_value = 0.0 if jackknife_estimate != 0 else 1.0
    else:
        ci_low = jackknife_estimate - Z_975 * standard_error
        ci_high = jackknife_estimate + Z_975 * standard_error
        z_score = jackknife_estimate / standard_error
        probability_positive = _normal_cdf(z_score)
        p_value = 2.0 * (1.0 - _normal_cdf(abs(z_score)))

    if ci_low > 0.0 and full_delta >= practical_win:
        decision = "clear_win"
    elif ci_high < 0.0:
        decision = "clear_loss"
    elif abs(full_delta) < equivalence_band and ci_low <= 0.0 <= ci_high:
        decision = "engineering_tie"
    else:
        decision = "gray_zone_rerun_pair_only"

    return {
        "contrast": contrast_name,
        "left": left.run_id,
        "right": right.run_id,
        "histogram_auc_left": left_auc,
        "histogram_auc_right": right_auc,
        "full_delta_auc": full_delta,
        "jackknife_bias_corrected_delta": jackknife_estimate,
        "standard_error": standard_error,
        "ci_95_low": ci_low,
        "ci_95_high": ci_high,
        "probability_delta_positive_normal_approx": probability_positive,
        "two_sided_p_value_normal_approx": p_value,
        "hash_groups": group_count,
        "decision": decision,
        "practical_win_threshold": practical_win,
        "equivalence_band": equivalence_band,
    }


def _parse_key_value(values: Sequence[str], option: str) -> Dict[str, List[str]]:
    parsed: Dict[str, List[str]] = {}
    for value in values:
        if "=" not in value:
            raise ValueError(f"{option} expects KEY=VALUE, got: {value}")
        key, item = value.split("=", 1)
        key = key.strip()
        item = item.strip()
        if not key or not item:
            raise ValueError(f"{option} expects non-empty KEY=VALUE, got: {value}")
        parsed.setdefault(key, []).append(item)
    return parsed


def _parse_contrasts(values: Sequence[str]) -> List[Tuple[str, str, str]]:
    contrasts = []
    for value in values:
        if "=" not in value or ":" not in value:
            raise ValueError(
                "--contrast expects NAME=LEFT_ID:RIGHT_ID, for example "
                "width=E2_RANDOM_D512:E1_RANDOM_D1024"
            )
        name, pair = value.split("=", 1)
        left, right = pair.split(":", 1)
        contrasts.append((name.strip(), left.strip(), right.strip()))
    return contrasts


def _default_contrasts(runs: Mapping[str, RunHistogram]) -> List[Tuple[str, str, str]]:
    contrasts = [item for item in DEFAULT_CONTRASTS if item[1] in runs and item[2] in runs]
    candidates = [
        run_id
        for run_id in ("E1_RANDOM_D1024", "E2_RANDOM_D512", "E3_SEMANTIC_D512")
        if run_id in runs
    ]
    if "E0_BASE" in runs and candidates:
        best = max(candidates, key=lambda run_id: summarize_run(runs[run_id])["histogram_auc"])
        contrasts.append(("best_rankmixer_minus_base", best, "E0_BASE"))
    return contrasts


def _write_csv(path: Path, rows: Sequence[Mapping[str, object]]) -> None:
    if not rows:
        return
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def _write_markdown(
    path: Path,
    run_summaries: Sequence[Mapping[str, object]],
    contrasts: Sequence[Mapping[str, object]],
    integrity: Mapping[str, object],
) -> None:
    lines = [
        "# RankMixer 2026-08-14 Paired 统计输出",
        "",
        "> 本文件由逐样本预测生成。AUC 为 score histogram 近似值；正式点指标仍以线上 validator 日志为准。",
        "",
        "## 样本一致性",
        "",
        f"- 校验结果：`{'通过' if integrity['verified'] else '失败'}`",
        f"- 参考实验：`{integrity['reference']}`",
        "- 校验口径：每个 search hash group 的样本数、正例数、XOR 与 SUM 双指纹。",
        "",
        "## 单实验指标",
        "",
        "| 实验 | 样本数 | 正例率 | 估算 Search 数 | Hist AUC | Hist PR-AUC | COPC |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in run_summaries:
        lines.append(
            "| {run_id} | {sample_count:,} | {positive_rate:.8f} | "
            "{estimated_distinct_search_ids:,} | {histogram_auc:.8f} | "
            "{histogram_pr_auc:.8f} | {copc:.8f} |".format(**row)
        )
    lines.extend(
        [
            "",
            "## Paired 对比",
            "",
            "| 对比 | 左−右 | ΔAUC | 95% CI | 判定 |",
            "|---|---|---:|---|---|",
        ]
    )
    for row in contrasts:
        lines.append(
            "| {contrast} | {left} − {right} | {delta:.8f} | "
            "[{low:.8f}, {high:.8f}] | {decision} |".format(
                contrast=row["contrast"],
                left=row["left"],
                right=row["right"],
                delta=row["full_delta_auc"],
                low=row["ci_95_low"],
                high=row["ci_95_high"],
                decision=row["decision"],
            )
        )
    lines.extend(
        [
            "",
            "## 方法说明",
            "",
            "- 同一 `search_id` 始终进入同一固定 hash group。",
            "- CI 使用 delete-one-hash-group jackknife；这是一种可扩展的成组近似，不是逐请求完整 bootstrap。",
            "- 如果样本一致性失败，禁止解释任何 paired 差异。",
            "- `clear_win` 还要求 ΔAUC 达到预注册的 +0.0002 实用门槛。",
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--run",
        action="append",
        required=True,
        metavar="ID=GLOB",
        help="Prediction glob for one run; repeat the same ID for multiple globs",
    )
    parser.add_argument(
        "--contrast",
        action="append",
        default=[],
        metavar="NAME=LEFT:RIGHT",
        help="Custom left-minus-right contrast; defaults to the pre-registered contrasts",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--buckets", type=int, default=DEFAULT_BUCKETS)
    parser.add_argument("--score-bins", type=int, default=DEFAULT_SCORE_BINS)
    parser.add_argument("--progress-every", type=int, default=10_000_000)
    parser.add_argument("--no-cache", action="store_true")
    parser.add_argument("--practical-win", type=float, default=0.0002)
    parser.add_argument("--equivalence-band", type=float, default=0.0001)
    args = parser.parse_args(argv)

    run_patterns = _parse_key_value(args.run, "--run")
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    runs: MutableMapping[str, RunHistogram] = {}
    for run_id, patterns in run_patterns.items():
        cache_path = output_dir / f"{run_id}.hist.npz"
        run = None
        if not args.no_cache:
            run = load_cache(
                cache_path, run_id, patterns, args.buckets, args.score_bins
            )
            if run is not None:
                print(f"[{run_id}] reused cache {cache_path}", file=sys.stderr)
        if run is None:
            run = build_histogram(
                run_id,
                patterns,
                buckets=args.buckets,
                score_bins=args.score_bins,
                progress_every=args.progress_every,
            )
            save_cache(run, cache_path)
            print(f"[{run_id}] wrote cache {cache_path}", file=sys.stderr)
        runs[run_id] = run

    integrity = verify_same_samples(runs)
    if not integrity["verified"]:
        result = {
            "status": "sample_integrity_failed",
            "integrity": integrity,
            "runs": [summarize_run(run) for run in runs.values()],
            "contrasts": [],
        }
        (output_dir / "paired_stats.json").write_text(
            json.dumps(result, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        print(
            "Sample integrity failed; wrote diagnostics and skipped paired contrasts.",
            file=sys.stderr,
        )
        return 2

    contrast_specs = (
        _parse_contrasts(args.contrast)
        if args.contrast
        else _default_contrasts(runs)
    )
    for _, left, right in contrast_specs:
        if left not in runs or right not in runs:
            raise ValueError(f"Unknown run in contrast: {left}:{right}")

    run_summaries = [summarize_run(run) for run in runs.values()]
    contrast_results = [
        paired_grouped_jackknife(
            runs[left],
            runs[right],
            name,
            practical_win=args.practical_win,
            equivalence_band=args.equivalence_band,
        )
        for name, left, right in contrast_specs
    ]
    result = {
        "status": "ok",
        "method": {
            "score_bins": args.score_bins,
            "hash_groups": args.buckets,
            "auc": "uniform-score histogram approximation",
            "inference": "paired delete-one-hash-group jackknife",
            "official_point_metric": "production validator AUC",
        },
        "integrity": integrity,
        "runs": run_summaries,
        "contrasts": contrast_results,
    }
    (output_dir / "paired_stats.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    _write_csv(output_dir / "run_metrics.csv", run_summaries)
    _write_csv(output_dir / "paired_contrasts.csv", contrast_results)
    _write_markdown(
        output_dir / "paired_stats.md", run_summaries, contrast_results, integrity
    )
    print(f"Wrote paired analysis to {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
