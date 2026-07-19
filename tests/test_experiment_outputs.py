from __future__ import annotations

import csv
import math
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DATASETS = {
    "parler_emotion_v1": "parler_emotion",
    "boundary_metric_v1": "boundary",
}


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8-sig") as handle:
        return list(csv.DictReader(handle))


class ExperimentOutputTests(unittest.TestCase):
    def test_pipeline_ids_audio_and_scores_are_consistent(self) -> None:
        total = 0
        for dataset, prefix in DATASETS.items():
            base = ROOT / "experiments" / dataset
            manifest_name = "parler_emotion_manifest.csv" if dataset.startswith("parler") else "boundary_manifest.csv"
            manifest = read_csv(base / "inputs" / manifest_name)
            asr = read_csv(base / "metrics" / "intelligibility" / "asr_wer.csv")
            sanity = read_csv(base / "metrics" / "naturalness" / "naturalness_proxy.csv")
            style = read_csv(base / "metrics" / "style_emotion" / "emotion_prosody.csv")
            combined = read_csv(base / "combined" / f"{prefix}_main_metrics.csv")
            scored = read_csv(base / "combined" / f"{prefix}_scored_main_metric.csv")

            expected_ids = {row["id"] for row in manifest}
            for rows in (asr, sanity, style, combined, scored):
                self.assertEqual({row["id"] for row in rows}, expected_ids)

            for row in manifest:
                audio = Path(row["audio_path"])
                if not audio.is_absolute():
                    audio = ROOT / audio
                self.assertTrue(audio.is_file(), audio)
                self.assertGreater(audio.stat().st_size, 44, audio)

            previous = math.inf
            for row in scored:
                self.assertEqual(row["metric_version"], "provisional_teacher_v2")
                self.assertEqual(row["metric_status"], "valid_provisional")
                self.assertEqual(row["quality_component_source"], "acoustic_sanity_fallback")
                self.assertIn("Q_sanity=diagnostic_only", row["teacher_active_weights"])
                intelligibility = float(row["intelligibility_component_0_1"])
                emotion = float(row["emotion_component_0_1"])
                expected = (0.55 * intelligibility + 0.35 * emotion) / 0.90
                actual = float(row["main_metric_0_1"])
                self.assertAlmostEqual(actual, expected, places=5)
                self.assertLessEqual(actual, previous)
                previous = actual

            for row in style:
                for field in ("f0_std_semitones", "f0_mad_semitones", "f0_range_semitones_p90_p10"):
                    self.assertTrue(math.isfinite(float(row[field])), (row["id"], field))
            total += len(scored)
        self.assertEqual(total, 26)

    def test_surrogate_reports_contain_finite_ranking_metrics(self) -> None:
        paths = [
            ROOT / "surrogate_exploration_v1" / "outputs_v3" / "surrogate_candidates_v3.csv",
            ROOT / "surrogate_exploration_v1" / "outputs_v3" / "sim_like" / "sim_like_candidates.csv",
        ]
        for path in paths:
            rows = read_csv(path)
            self.assertGreater(len(rows), 0)
            for row in rows:
                for field in ("pearson", "spearman", "kendall_tau_b", "pairwise_accuracy", "mae"):
                    value = float(row[field])
                    self.assertTrue(math.isfinite(value), (path, row["candidate"], field))
                self.assertGreaterEqual(float(row["pairwise_accuracy"]), 0.0)
                self.assertLessEqual(float(row["pairwise_accuracy"]), 1.0)

    def test_multisystem_grouped_outputs_and_blinding(self) -> None:
        base = ROOT / "experiments" / "multisystem_generalization_v1"
        manifest = read_csv(base / "inputs" / "multisystem_manifest.csv")
        self.assertEqual(len(manifest), 18)
        self.assertEqual({row["tts_system"] for row in manifest}, {"parler", "bark", "sapi"})
        self.assertEqual(sum(int(row["is_boundary"]) for row in manifest), 9)

        by_text: dict[str, set[str]] = {}
        for row in manifest:
            by_text.setdefault(row["text_id"], set()).add(row["tts_system"])
            audio = ROOT / row["audio_path"]
            self.assertTrue(audio.is_file(), audio)
            self.assertGreater(audio.stat().st_size, 44, audio)
        self.assertEqual(len(by_text), 6)
        self.assertTrue(all(systems == {"parler", "bark", "sapi"} for systems in by_text.values()))

        expected_ids = {row["id"] for row in manifest}
        outputs = [
            base / "metrics" / "intelligibility" / "asr_wer.csv",
            base / "metrics" / "naturalness" / "acoustic_sanity.csv",
            base / "metrics" / "style_emotion" / "emotion_prosody.csv",
            base / "combined" / "multisystem_main_metrics.csv",
            base / "combined" / "multisystem_scored_main_metric.csv",
            base / "analysis" / "per_clip_scores.csv",
        ]
        for path in outputs:
            self.assertEqual({row["id"] for row in read_csv(path)}, expected_ids, path)

        candidates = {row["candidate"]: row for row in read_csv(base / "analysis" / "surrogate_grouped_candidates.csv")}
        for name in ("loso_low_dsp_ridge", "loso_neural_dsp_ridge"):
            self.assertIn(name, candidates)
            self.assertTrue(math.isfinite(float(candidates[name]["spearman"])))

        public = read_csv(base / "human_evaluation" / "ratings_template.csv")
        private = read_csv(base / "human_evaluation" / "private_blind_key.csv")
        self.assertEqual(len(public), 18)
        self.assertEqual({row["blind_id"] for row in public}, {row["blind_id"] for row in private})
        self.assertNotIn("tts_system", public[0])
        self.assertNotIn("model", public[0])

    def test_automatic_consensus_outputs(self) -> None:
        base = ROOT / "experiments" / "automatic_emotion_consensus_v1"
        scores = read_csv(base / "analysis" / "per_clip_scores.csv")
        self.assertEqual(len(scores), 52)
        self.assertEqual({row["tts_system"] for row in scores}, {"parler", "bark", "sapi"})
        self.assertEqual(sum(row["dataset"] == "controlled_emotion_intensity_v1" for row in scores), 8)

        for row in scores:
            intelligibility = float(row["intelligibility_auto_0_1"])
            emotion = float(row["emotion_consensus_0_1"])
            sanity = float(row["acoustic_sanity_0_1"])
            expected = intelligibility**0.55 * emotion**0.35 * sanity**0.10
            self.assertAlmostEqual(float(row["main_auto_v3_0_1"]), expected, places=10)
            self.assertGreaterEqual(float(row["emotion_model_disagreement_0_1"]), 0.0)

        candidates = read_csv(base / "analysis" / "surrogate_candidates.csv")
        self.assertEqual(len(candidates), 15)
        for row in candidates:
            for field in ("spearman", "kendall_tau_b", "pairwise_accuracy", "mae"):
                self.assertTrue(math.isfinite(float(row[field])), (row["candidate"], field))

        controlled = {row["id"]: row for row in scores if row["dataset"] == "controlled_emotion_intensity_v1"}
        self.assertGreater(
            float(controlled["happy_obvious"]["emotion_consensus_0_1"]),
            float(controlled["happy_subtle"]["emotion_consensus_0_1"]),
        )


if __name__ == "__main__":
    unittest.main()
