import importlib.util
import json
import tempfile
import unittest
from pathlib import Path

HAS_NUMPY = importlib.util.find_spec("numpy") is not None
if HAS_NUMPY:
    from src.models.rankmixer.tools.paired_auc_analysis import (
        auc_from_histograms,
        build_histogram,
        main,
        paired_grouped_jackknife,
        verify_same_samples,
    )


def exact_auc(labels, scores):
    positives = [score for label, score in zip(labels, scores) if label == 1]
    negatives = [score for label, score in zip(labels, scores) if label == 0]
    wins = 0.0
    for positive in positives:
        for negative in negatives:
            if positive > negative:
                wins += 1.0
            elif positive == negative:
                wins += 0.5
    return wins / (len(positives) * len(negatives))


def prediction_rows(stronger):
    rows = []
    labels = []
    scores = []
    for search_index in range(240):
        noise = ((search_index * 37) % 101) / 100.0 * 0.5 - 0.25
        margin = 0.16 if stronger else 0.07
        for label in (0, 1):
            center = 0.5 + (margin if label else -margin)
            score = min(max(center + noise, 0.001), 0.999)
            rows.append(
                f"search-{search_index}\texample-{search_index}-{label}\t{label}.0\t{score:.8f}\n"
            )
            labels.append(label)
            scores.append(score)
    return rows, labels, scores


@unittest.skipUnless(HAS_NUMPY, "paired analysis requires numpy")
class PairedAucAnalysisTest(unittest.TestCase):
    def setUp(self):
        self.tempdir = tempfile.TemporaryDirectory()
        self.root = Path(self.tempdir.name)
        left_rows, self.labels, self.left_scores = prediction_rows(stronger=True)
        right_rows, _, self.right_scores = prediction_rows(stronger=False)
        self.left_path = self.root / "left.txt"
        self.right_path = self.root / "right.txt"
        self.left_path.write_text("".join(left_rows), encoding="utf-8")
        self.right_path.write_text("".join(right_rows), encoding="utf-8")

    def tearDown(self):
        self.tempdir.cleanup()

    def build(self, run_id, path):
        return build_histogram(
            run_id,
            [str(path)],
            buckets=20,
            score_bins=10_000,
            progress_every=0,
        )

    def test_histogram_auc_tracks_exact_auc(self):
        left = self.build("LEFT", self.left_path)
        positive = left.positive_hist.sum(axis=0)
        negative = left.negative_hist.sum(axis=0)
        histogram_auc = auc_from_histograms(positive, negative)
        parsed_scores = [
            float(line.split("\t")[3])
            for line in self.left_path.read_text(encoding="utf-8").splitlines()
        ]
        self.assertAlmostEqual(
            histogram_auc, exact_auc(self.labels, parsed_scores), places=4
        )

    def test_sample_integrity_and_positive_paired_delta(self):
        left = self.build("LEFT", self.left_path)
        right = self.build("RIGHT", self.right_path)
        integrity = verify_same_samples({"LEFT": left, "RIGHT": right})
        self.assertTrue(integrity["verified"])
        result = paired_grouped_jackknife(
            left,
            right,
            "left_minus_right",
            practical_win=0.0,
            equivalence_band=0.0,
        )
        self.assertGreater(result["full_delta_auc"], 0.0)
        self.assertGreater(result["ci_95_low"], 0.0)
        self.assertEqual(result["decision"], "clear_win")

    def test_integrity_fails_after_one_label_change(self):
        rows = self.right_path.read_text(encoding="utf-8").splitlines()
        parts = rows[0].split("\t")
        parts[2] = "1.0"
        rows[0] = "\t".join(parts)
        mismatch_path = self.root / "mismatch.txt"
        mismatch_path.write_text("\n".join(rows) + "\n", encoding="utf-8")
        left = self.build("LEFT", self.left_path)
        mismatch = self.build("MISMATCH", mismatch_path)
        integrity = verify_same_samples({"LEFT": left, "MISMATCH": mismatch})
        self.assertFalse(integrity["verified"])

    def test_cli_writes_machine_and_human_readable_outputs(self):
        output_dir = self.root / "output"
        exit_code = main(
            [
                "--run",
                f"LEFT={self.left_path}",
                "--run",
                f"RIGHT={self.right_path}",
                "--contrast",
                "left_minus_right=LEFT:RIGHT",
                "--output-dir",
                str(output_dir),
                "--buckets",
                "20",
                "--score-bins",
                "10000",
                "--progress-every",
                "0",
            ]
        )
        self.assertEqual(exit_code, 0)
        for name in (
            "paired_stats.json",
            "paired_stats.md",
            "run_metrics.csv",
            "paired_contrasts.csv",
            "LEFT.hist.npz",
            "RIGHT.hist.npz",
        ):
            self.assertTrue((output_dir / name).is_file(), name)
        result = json.loads((output_dir / "paired_stats.json").read_text())
        self.assertEqual(result["status"], "ok")
        self.assertTrue(result["integrity"]["verified"])
        self.assertGreater(result["contrasts"][0]["full_delta_auc"], 0.0)


if __name__ == "__main__":
    unittest.main()
