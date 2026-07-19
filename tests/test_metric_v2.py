from __future__ import annotations

import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "surrogate_exploration_v1"))

from score_emotion_tts_main_metric import score_row
from analyze_surrogates_v2 import kendall_tau_b, pairwise_ranking_accuracy


class TeacherV2Tests(unittest.TestCase):
    def test_formula_uses_continuous_emotion_and_wer(self) -> None:
        result = score_row(
            {
                "wer": "0.20",
                "cer": "0.10",
                "target_emotion_prob": "0.40",
                "target_emotion_match": "1",
                "acoustic_sanity_score_0_1": "0.90",
            }
        )
        self.assertEqual(result["metric_status"], "valid_provisional")
        self.assertAlmostEqual(float(result["main_metric_0_1"]), 0.58 / 0.90, places=6)
        self.assertEqual(result["emotion_component_0_1"], "0.400000")

    def test_missing_metric_is_invalid_not_zero(self) -> None:
        result = score_row({"wer": "0.1", "acoustic_sanity_score_0_1": "1"})
        self.assertEqual(result["metric_status"], "invalid")
        self.assertEqual(result["main_metric_0_1"], "")
        self.assertIn("target_emotion_prob", result["metric_missing_fields"])

    def test_legacy_sanity_alias_is_supported(self) -> None:
        result = score_row({"wer": "0", "target_emotion_prob": "1", "naturalness_proxy_1_5": "5"})
        self.assertEqual(result["quality_component_source"], "acoustic_sanity_fallback")
        self.assertEqual(result["main_metric_0_1"], "1.000000")

    def test_learned_mos_enters_score(self) -> None:
        result = score_row(
            {
                "wer": "0.20",
                "target_emotion_prob": "0.40",
                "acoustic_sanity_score_0_1": "0.90",
                "utmos_score_1_5": "3.0",
            }
        )
        self.assertEqual(result["quality_component_source"], "learned_mos")
        self.assertAlmostEqual(float(result["main_metric_0_1"]), 0.63, places=6)


class RankingMetricTests(unittest.TestCase):
    def test_rank_metrics(self) -> None:
        target = [0.1, 0.2, 0.3, 0.4]
        self.assertAlmostEqual(kendall_tau_b(target, target), 1.0)
        self.assertAlmostEqual(pairwise_ranking_accuracy(target, target), 1.0)
        reverse = list(reversed(target))
        self.assertAlmostEqual(kendall_tau_b(reverse, target), -1.0)
        self.assertAlmostEqual(pairwise_ranking_accuracy(reverse, target), 0.0)


if __name__ == "__main__":
    unittest.main()
