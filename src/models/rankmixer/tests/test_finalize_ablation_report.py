import csv
import json
import tempfile
import unittest
from pathlib import Path

from src.models.rankmixer.tools.finalize_ablation_report import main


ROOT = Path(__file__).resolve().parents[4]
MANIFEST = ROOT / "bash" / "ablation_20260814" / "manifest.json"
PREREG_REPORT = ROOT / "introduce" / "rankmixer_ablation_20260814_report.md"
TEMPLATE_RESULTS = ROOT / "introduce" / "rankmixer_ablation_20260814_results.csv"


def completed_rows(aucs):
    with TEMPLATE_RESULTS.open(encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        fieldnames = list(reader.fieldnames)
        rows = [dict(row) for row in reader]
    for index, row in enumerate(rows):
        run_id = row["experiment_id"]
        row.update(
            {
                "git_commit": "abc123",
                "task_id": f"task-{index}",
                "model_dir": f"hdfs://models/{run_id}",
                "train_wall_time_min": str(100 + index),
                "step_time_ms": str(10 + index),
                "peak_memory_gb": str(20 + index),
                "auc": str(aucs[run_id]),
                "copc": "1.01",
                "pr_auc": "0.12",
                "bucket_error": "0.001",
                "prediction_path": f"hdfs://predictions/{run_id}",
                "status": "complete",
            }
        )
    return fieldnames, rows


def paired_payload(aucs, integrity=True, override=None):
    best = max(
        ("E1_RANDOM_D1024", "E2_RANDOM_D512", "E3_SEMANTIC_D512"),
        key=lambda run_id: aucs[run_id],
    )
    specs = [
        ("width_E2_minus_E1", "E2_RANDOM_D512", "E1_RANDOM_D1024"),
        ("group_E3_minus_E2", "E3_SEMANTIC_D512", "E2_RANDOM_D512"),
        ("best_rankmixer_minus_base", best, "E0_BASE"),
    ]
    contrasts = []
    for name, left, right in specs:
        delta = aucs[left] - aucs[right]
        half_width = min(max(abs(delta) * 0.25, 0.00002), 0.00008)
        contrasts.append(
            {
                "contrast": name,
                "left": left,
                "right": right,
                "full_delta_auc": delta,
                "ci_95_low": delta - half_width,
                "ci_95_high": delta + half_width,
            }
        )
    if override:
        for item in contrasts:
            item.update(override.get(item["contrast"], {}))
    return {
        "status": "ok",
        "integrity": {"verified": integrity},
        "runs": [{"run_id": run_id} for run_id in aucs],
        "contrasts": contrasts,
    }


class FinalizeAblationReportTest(unittest.TestCase):
    def setUp(self):
        self.tempdir = tempfile.TemporaryDirectory()
        self.root = Path(self.tempdir.name)

    def tearDown(self):
        self.tempdir.cleanup()

    def write_inputs(self, aucs, integrity=True, override=None):
        fieldnames, rows = completed_rows(aucs)
        results = self.root / "results.csv"
        with results.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)
        paired = self.root / "paired.json"
        paired.write_text(
            json.dumps(paired_payload(aucs, integrity, override)), encoding="utf-8"
        )
        return results, paired

    def run_finalizer(self, results, paired):
        output = self.root / "final.md"
        audit = self.root / "audit.json"
        code = main(
            [
                "--manifest",
                str(MANIFEST),
                "--results-csv",
                str(results),
                "--paired-stats",
                str(paired),
                "--prereg-report",
                str(PREREG_REPORT),
                "--output-report",
                str(output),
                "--audit-json",
                str(audit),
            ]
        )
        return code, output, audit

    def test_complete_evidence_materializes_final_report(self):
        aucs = {
            "E0_BASE": 0.8660,
            "E1_RANDOM_D1024": 0.8640,
            "E2_RANDOM_D512": 0.8650,
            "E3_SEMANTIC_D512": 0.8663,
        }
        results, paired = self.write_inputs(aucs)
        code, output, audit = self.run_finalizer(results, paired)
        self.assertEqual(code, 0)
        text = output.read_text(encoding="utf-8")
        self.assertIn("## 10. 最终实验结果", text)
        self.assertIn("D512 的单日冷启动/参数效率与语义分组均有独立正贡献", text)
        self.assertIn("最佳 RankMixer 明确超过同日 Base", text)
        self.assertEqual(text.count("<!-- AUTO_RESULTS_START -->"), 1)
        payload = json.loads(audit.read_text())
        self.assertEqual(payload["status"], "finalized")

    def test_unfinished_result_csv_is_rejected_without_final_report(self):
        paired = self.root / "paired.json"
        paired.write_text(
            json.dumps(
                paired_payload(
                    {
                        "E0_BASE": 0.8660,
                        "E1_RANDOM_D1024": 0.8640,
                        "E2_RANDOM_D512": 0.8650,
                        "E3_SEMANTIC_D512": 0.8663,
                    }
                )
            )
        )
        code, output, audit = self.run_finalizer(TEMPLATE_RESULTS, paired)
        self.assertEqual(code, 2)
        self.assertFalse(output.exists())
        payload = json.loads(audit.read_text())
        self.assertEqual(payload["status"], "validator_audit_failed")
        self.assertTrue(payload["validator"]["errors"])

    def test_failed_sample_integrity_is_rejected(self):
        aucs = {
            "E0_BASE": 0.8660,
            "E1_RANDOM_D1024": 0.8640,
            "E2_RANDOM_D512": 0.8650,
            "E3_SEMANTIC_D512": 0.8663,
        }
        results, paired = self.write_inputs(aucs, integrity=False)
        code, output, audit = self.run_finalizer(results, paired)
        self.assertEqual(code, 2)
        self.assertFalse(output.exists())
        payload = json.loads(audit.read_text())
        self.assertEqual(payload["status"], "paired_audit_failed")

    def test_gray_zone_generates_interim_report_and_pair_only_rerun(self):
        aucs = {
            "E0_BASE": 0.8660,
            "E1_RANDOM_D1024": 0.8657,
            "E2_RANDOM_D512": 0.86585,
            "E3_SEMANTIC_D512": 0.86615,
        }
        override = {
            "width_E2_minus_E1": {
                "ci_95_low": -0.00005,
                "ci_95_high": 0.00035,
            },
            "best_rankmixer_minus_base": {
                "ci_95_low": -0.00005,
                "ci_95_high": 0.00035,
            },
        }
        results, paired = self.write_inputs(aucs, override=override)
        code, output, audit = self.run_finalizer(results, paired)
        self.assertEqual(code, 3)
        self.assertIn("needs_pair_rerun", output.read_text())
        payload = json.loads(audit.read_text())
        self.assertEqual(payload["status"], "needs_pair_rerun")
        self.assertTrue(payload["conclusion"]["rerun_pairs"])


if __name__ == "__main__":
    unittest.main()
