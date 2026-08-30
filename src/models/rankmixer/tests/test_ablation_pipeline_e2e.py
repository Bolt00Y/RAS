import csv
import importlib.util
import json
import tempfile
import unittest
from pathlib import Path

from src.models.rankmixer.tools.finalize_ablation_report import main as finalize_main


HAS_NUMPY = importlib.util.find_spec("numpy") is not None
if HAS_NUMPY:
    from src.models.rankmixer.tools.paired_auc_analysis import main as paired_main


ROOT = Path(__file__).resolve().parents[4]
MANIFEST = ROOT / "bash" / "ablation_20260814" / "manifest.json"
PREREG_REPORT = ROOT / "introduce" / "rankmixer_ablation_20260814_report.md"
RESULT_TEMPLATE = ROOT / "introduce" / "rankmixer_ablation_20260814_results.csv"


@unittest.skipUnless(HAS_NUMPY, "end-to-end paired pipeline requires numpy")
class AblationPipelineEndToEndTest(unittest.TestCase):
    def test_predictions_to_final_report(self):
        run_margins = {
            "E0_BASE": 0.14,
            "E1_RANDOM_D1024": 0.05,
            "E2_RANDOM_D512": 0.10,
            "E3_SEMANTIC_D512": 0.18,
        }
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            prediction_paths = {}
            for run_id, margin in run_margins.items():
                path = root / f"{run_id}.txt"
                prediction_paths[run_id] = path
                lines = []
                for search_index in range(800):
                    noise = ((search_index * 37) % 809) / 808.0 * 0.72 - 0.36
                    for label in (0, 1):
                        score = 0.5 + (margin if label else -margin) + noise
                        score = min(max(score, 0.001), 0.999)
                        lines.append(
                            f"search-{search_index}\texample-{search_index}-{label}\t"
                            f"{label}.0\t{score:.8f}\n"
                        )
                path.write_text("".join(lines), encoding="utf-8")

            stats_dir = root / "stats"
            paired_args = []
            for run_id, path in prediction_paths.items():
                paired_args.extend(["--run", f"{run_id}={path}"])
            paired_args.extend(
                [
                    "--output-dir",
                    str(stats_dir),
                    "--buckets",
                    "20",
                    "--score-bins",
                    "10000",
                    "--progress-every",
                    "0",
                ]
            )
            self.assertEqual(paired_main(paired_args), 0)
            paired_path = stats_dir / "paired_stats.json"
            paired = json.loads(paired_path.read_text())
            self.assertTrue(paired["integrity"]["verified"])

            run_metrics = {item["run_id"]: item for item in paired["runs"]}
            with RESULT_TEMPLATE.open(encoding="utf-8", newline="") as handle:
                reader = csv.DictReader(handle)
                fieldnames = list(reader.fieldnames)
                rows = [dict(row) for row in reader]
            for index, row in enumerate(rows):
                run_id = row["experiment_id"]
                metric = run_metrics[run_id]
                row.update(
                    {
                        "git_commit": "e2e-commit",
                        "task_id": f"e2e-task-{index}",
                        "model_dir": f"hdfs://e2e/models/{run_id}",
                        "train_wall_time_min": str(120 + index),
                        "step_time_ms": str(12 + index),
                        "peak_memory_gb": str(30 + index),
                        # Use the same AUC estimator in this integration test so the
                        # finalizer's validator-vs-histogram consistency gate is exact.
                        "auc": str(metric["histogram_auc"]),
                        "copc": str(metric["copc"]),
                        "pr_auc": str(metric["histogram_pr_auc"]),
                        "bucket_error": "0.001",
                        "prediction_path": f"hdfs://e2e/predictions/{run_id}",
                        "status": "complete",
                    }
                )
            results_path = root / "results.csv"
            with results_path.open("w", encoding="utf-8", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerows(rows)

            final_report = root / "final_report.md"
            audit_path = root / "final_report.audit.json"
            code = finalize_main(
                [
                    "--manifest",
                    str(MANIFEST),
                    "--results-csv",
                    str(results_path),
                    "--paired-stats",
                    str(paired_path),
                    "--prereg-report",
                    str(PREREG_REPORT),
                    "--output-report",
                    str(final_report),
                    "--audit-json",
                    str(audit_path),
                ]
            )
            self.assertEqual(code, 0)
            final_text = final_report.read_text(encoding="utf-8")
            self.assertIn("## 10. 最终实验结果", final_text)
            self.assertIn(
                "D512 的单日冷启动/参数效率与语义分组均有独立正贡献",
                final_text,
            )
            self.assertIn("最佳 RankMixer 明确超过同日 Base", final_text)
            audit = json.loads(audit_path.read_text())
            self.assertEqual(audit["status"], "finalized")
            self.assertEqual(
                audit["conclusion"]["best_rankmixer"], "E3_SEMANTIC_D512"
            )


if __name__ == "__main__":
    unittest.main()
