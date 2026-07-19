from __future__ import annotations

import argparse
import csv
import math
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "surrogate_exploration_v1"))

from analyze_surrogates import acoustic_features, text_difficulty  # noqa: E402
from analyze_surrogates_v2 import evaluate, ridge_fit, ridge_predict  # noqa: E402


EMOTIONS = ("neutral", "happy", "angry", "sad")
# Independent, interpretable VAD anchors. They are hypotheses, not fitted labels.
VAD_ANCHORS = {
    "neutral": (0.40, 0.50, 0.50),
    "happy": (0.70, 0.70, 0.75),
    "angry": (0.75, 0.70, 0.25),
    "sad": (0.30, 0.35, 0.30),
}
VAD_SCALES = np.asarray((0.25, 0.25, 0.30), dtype="float64")
RIDGE_ALPHA = 10.0


def parse_args() -> argparse.Namespace:
    base = ROOT / "experiments" / "automatic_emotion_consensus_v1"
    parser = argparse.ArgumentParser(description="Build an automatic-only emotion-aware TTS teacher and surrogates.")
    parser.add_argument(
        "--input",
        type=Path,
        default=base / "model_outputs" / "emotion_model_outputs_all_52.csv",
        help="Combined emotion-model output. Build it with combine_automatic_emotion_outputs.py.",
    )
    parser.add_argument("--model-costs", type=Path, default=base / "model_outputs" / "emotion_model_costs.csv")
    parser.add_argument("--output-dir", type=Path, default=base / "analysis")
    return parser.parse_args()


def clamp(value: float) -> float:
    return max(0.0, min(1.0, value))


def f(row: dict[str, str], key: str) -> float:
    return float(row[key])


def vad_probabilities(row: dict[str, str]) -> dict[str, float]:
    point = np.asarray((f(row, "vad_arousal"), f(row, "vad_dominance"), f(row, "vad_valence")))
    logits = []
    for emotion in EMOTIONS:
        delta = (point - np.asarray(VAD_ANCHORS[emotion])) / VAD_SCALES
        logits.append(-0.5 * float(delta @ delta))
    values = np.exp(np.asarray(logits) - max(logits))
    values /= values.sum()
    return dict(zip(EMOTIONS, values.tolist()))


def median3(a: float, b: float, c: float) -> float:
    return sorted((a, b, c))[1]


def automatic_main(row: dict[str, str]) -> dict[str, float | str]:
    target = row["target_emotion"].lower()
    vad = vad_probabilities(row)
    e2v = f(row, "e2v_target_prob")
    superb = f(row, "superb_target_prob")
    emotion = median3(e2v, superb, vad[target])
    disagreement = max(e2v, superb, vad[target]) - min(e2v, superb, vad[target])
    intelligibility = clamp(1.0 - f(row, "wer"))
    sanity = clamp(f(row, "acoustic_sanity_score_0_1"))
    eps = 1e-6
    score = math.exp(
        0.55 * math.log(max(intelligibility, eps))
        + 0.35 * math.log(max(emotion, eps))
        + 0.10 * math.log(max(sanity, eps))
    )
    eligible = intelligibility >= 0.70 and sanity >= 0.50
    return {
        "intelligibility_auto_0_1": intelligibility,
        "emotion_consensus_0_1": emotion,
        "emotion_model_disagreement_0_1": disagreement,
        "vad_target_prob": vad[target],
        "vad_top_label": max(vad, key=vad.get),
        "acoustic_sanity_0_1": sanity,
        "main_auto_v3_0_1": score,
        "ranking_eligible": "1" if eligible else "0",
    }


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else math.nan


def group_summary(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    grouped: dict[tuple[str, str], list[dict[str, object]]] = defaultdict(list)
    for row in rows:
        grouped[(str(row["dataset"]), str(row["target_emotion"]))].append(row)
        grouped[("ALL", str(row["target_emotion"]))].append(row)
    output = []
    for (dataset, emotion), group in sorted(grouped.items()):
        output.append(
            {
                "dataset": dataset,
                "target_emotion": emotion,
                "n": len(group),
                "e2v_target_prob_mean": fmean(group, "e2v_target_prob"),
                "e2v_top_accuracy": accuracy(group, "e2v_top_label"),
                "superb_target_prob_mean": fmean(group, "superb_target_prob"),
                "superb_top_accuracy": accuracy(group, "superb_top_label"),
                "vad_target_prob_mean": fmean(group, "vad_target_prob"),
                "vad_top_accuracy": accuracy(group, "vad_top_label"),
                "emotion_consensus_mean": fmean(group, "emotion_consensus_0_1"),
                "vad_arousal_mean": fmean(group, "vad_arousal"),
                "vad_dominance_mean": fmean(group, "vad_dominance"),
                "vad_valence_mean": fmean(group, "vad_valence"),
                "main_auto_v3_mean": fmean(group, "main_auto_v3_0_1"),
            }
        )
    return output


def fmean(rows: list[dict[str, object]], key: str) -> float:
    return mean([float(row[key]) for row in rows])


def accuracy(rows: list[dict[str, object]], key: str) -> float:
    return mean([float(str(row[key]).lower() == str(row["target_emotion"]).lower()) for row in rows])


def split_predictions(
    rows: list[dict[str, float | str]], target: list[float], features: list[str], split_key: str
) -> list[float]:
    predictions = [math.nan] * len(rows)
    groups = list(range(len(rows))) if split_key == "LOOCV" else sorted({str(row[split_key]) for row in rows})
    for group in groups:
        if split_key == "LOOCV":
            test = [int(group)]
        else:
            test = [i for i, row in enumerate(rows) if str(row[split_key]) == group]
        test_set = set(test)
        train = [i for i in range(len(rows)) if i not in test_set]
        x_train = [[float(rows[i][name]) for name in features] for i in train]
        x_test = [[float(rows[i][name]) for name in features] for i in test]
        coef, center, scale = ridge_fit(x_train, [target[i] for i in train], alpha=RIDGE_ALPHA)
        for idx, prediction in zip(test, ridge_predict(x_test, coef, center, scale)):
            predictions[idx] = prediction
    return predictions


def add_low_cost_features(rows: list[dict[str, object]]) -> tuple[float, list[str]]:
    start = time.perf_counter()
    for row in rows:
        cheap = acoustic_features({key: str(value) for key, value in row.items()})
        text = text_difficulty(str(row["text"]))
        for key in ("signal_quality", "rate_fit", "silence_ratio", "energy_cv", "voiced_ratio", "spectral_flatness"):
            row[key] = cheap[key]
        row["text_ease"] = text["text_ease"]
    elapsed = time.perf_counter() - start
    return elapsed, [
        "signal_quality",
        "rate_fit",
        "silence_ratio",
        "energy_cv",
        "voiced_ratio",
        "spectral_flatness",
        "text_ease",
    ]


def build_surrogates(rows: list[dict[str, object]], low_features: list[str]) -> tuple[list[dict[str, object]], dict[str, list[float]]]:
    typed = rows  # The numeric feature accesses below are explicit.
    target = [float(row["main_auto_v3_0_1"]) for row in rows]
    feature_sets = {
        "low_dsp_ridge": low_features,
        "e2v_plus_dsp_ridge": low_features + ["e2v_target_prob", "e2v_entropy_norm"],
        "superb_plus_dsp_ridge": low_features + ["superb_target_prob", "superb_entropy_norm"],
        "vad_plus_dsp_ridge": low_features + ["vad_target_prob", "vad_arousal", "vad_dominance", "vad_valence"],
        "all_emotion_plus_dsp_ridge": low_features
        + ["e2v_target_prob", "e2v_entropy_norm", "superb_target_prob", "superb_entropy_norm", "vad_target_prob"],
    }
    results: list[dict[str, object]] = []
    predictions: dict[str, list[float]] = {}
    for name, features in feature_sets.items():
        for validation, split_key in (("LOOCV", "LOOCV"), ("leave_dataset_out", "dataset"), ("leave_system_out", "tts_system")):
            if split_key != "LOOCV" and len({str(row[split_key]) for row in rows}) < 2:
                continue
            pred = split_predictions(typed, target, features, split_key)  # type: ignore[arg-type]
            key = f"{name}__{validation}"
            predictions[key] = pred
            row = evaluate(name, pred, target, "", "+".join(features), f"alpha={RIDGE_ALPHA:g}")
            row["validation"] = validation
            results.append(row)
    return results, predictions


def cost_table(model_cost_path: Path, low_dsp_seconds: float, n: int) -> list[dict[str, object]]:
    with model_cost_path.open(newline="", encoding="utf-8-sig") as handle:
        model_rows = list(csv.DictReader(handle))
    warm = {row["model"]: float(row["seconds_per_clip_warm"]) for row in model_rows}
    low = low_dsp_seconds / n
    # Whisper and sanity figures are measured on this machine in the prior 26-clip cost run.
    asr = 0.597766
    sanity = 0.046026
    e2v = warm["emotion2vec_plus_base"]
    superb = warm["superb/wav2vec2-base-superb-er"]
    vad = warm["audeering/wav2vec2-large-robust-12-ft-emotion-msp-dim"]
    main = asr + sanity + e2v + superb + vad
    entries = [
        ("main_auto_v3", main, "Whisper+sanity+emotion2vec+SUPERB+MSP-Dim"),
        ("low_dsp_ridge", low, "waveform/text DSP"),
        ("e2v_plus_dsp_ridge", low + e2v, "emotion2vec+DSP"),
        ("superb_plus_dsp_ridge", low + superb, "SUPERB+DSP"),
        ("vad_plus_dsp_ridge", low + vad, "MSP-Dim+DSP"),
        ("all_emotion_plus_dsp_ridge", low + e2v + superb + vad, "three emotion models+DSP"),
    ]
    return [
        {
            "metric": name,
            "warm_seconds_per_clip": seconds,
            "speedup_vs_main": main / seconds,
            "ingredients": ingredients,
            "measurement_note": "DSP/emotion measured on 44 clips; Whisper/sanity reused from prior 26-clip same-machine benchmark",
        }
        for name, seconds, ingredients in entries
    ]


def report(
    rows: list[dict[str, object]], groups: list[dict[str, object]], candidates: list[dict[str, object]], costs: list[dict[str, object]]
) -> str:
    lines = [
        "# Automatic Emotion Consensus Experiment",
        "",
        "No human labels are used. Target emotion means the generation prompt label, so this experiment measures automatic-model agreement, not human emotion validity.",
        "",
        "## Main Metric",
        "",
        "```text",
        "I = clamp(1 - WER)",
        "E = median(P_emotion2vec(target), P_SUPERB(target), P_MSP-VAD-anchor(target))",
        "S = acoustic_sanity_score",
        "Main_auto_v3 = I^0.55 * E^0.35 * S^0.10",
        "eligible = I >= 0.70 and S >= 0.50",
        "```",
        "",
        "The geometric mean is non-compensatory. Model disagreement remains a diagnostic field and high disagreement must not be presented as confident emotion correctness.",
        "",
        "## Emotion Separability",
        "",
        "| emotion | n | e2v target P | e2v acc | SUPERB target P | SUPERB acc | VAD target P | VAD acc | consensus | arousal | valence |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in groups:
        if row["dataset"] == "ALL":
            lines.append(
                f"| {row['target_emotion']} | {row['n']} | {row['e2v_target_prob_mean']:.3f} | {row['e2v_top_accuracy']:.3f} | "
                f"{row['superb_target_prob_mean']:.3f} | {row['superb_top_accuracy']:.3f} | {row['vad_target_prob_mean']:.3f} | "
                f"{row['vad_top_accuracy']:.3f} | {row['emotion_consensus_mean']:.3f} | {row['vad_arousal_mean']:.3f} | {row['vad_valence_mean']:.3f} |"
            )
    lines += [
        "",
        "## Surrogate Agreement",
        "",
        "| candidate | validation | Spearman | Kendall tau-b | pairwise accuracy | MAE |",
        "| --- | --- | ---: | ---: | ---: | ---: |",
    ]
    for row in candidates:
        lines.append(
            f"| {row['candidate']} | {row['validation']} | {row['spearman']} | {row['kendall_tau_b']} | {row['pairwise_accuracy']} | {row['mae']} |"
        )
    boundary_groups = {
        label: [row for row in rows if str(row.get("is_boundary", "0")) == flag]
        for label, flag in (("non-boundary", "0"), ("boundary", "1"))
    }
    lines += [
        "",
        "## Aggregate Sanity Check",
        "",
        "| group | n | mean I | mean E | mean Main |",
        "| --- | ---: | ---: | ---: | ---: |",
    ]
    for label, group in boundary_groups.items():
        if group:
            lines.append(
                f"| {label} | {len(group)} | {fmean(group, 'intelligibility_auto_0_1'):.3f} | "
                f"{fmean(group, 'emotion_consensus_0_1'):.3f} | {fmean(group, 'main_auto_v3_0_1'):.3f} |"
            )
    lines += [
        "",
        "A higher boundary mean is not evidence of better audio. Label composition and the weak acoustic-sanity detector confound this aggregate; it must not be used as a system-quality ranking.",
    ]
    lines += [
        "",
        "## Warm Cost",
        "",
        "| metric | sec/clip | speedup vs Main | ingredients |",
        "| --- | ---: | ---: | --- |",
    ]
    for row in costs:
        lines.append(f"| {row['metric']} | {row['warm_seconds_per_clip']:.4f} | {row['speedup_vs_main']:.1f}x | {row['ingredients']} |")
    lines += [
        "",
        "## Per-Clip Scores",
        "",
        "| sample | system | target | WER | E consensus | disagreement | Main v3 | low-DSP surrogate | best neural surrogate |",
        "| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in sorted(rows, key=lambda item: float(item["main_auto_v3_0_1"]), reverse=True):
        lines.append(
            f"| {row['sample_key']} | {row['tts_system']} | {row['target_emotion']} | {float(row['wer']):.3f} | "
            f"{float(row['emotion_consensus_0_1']):.3f} | {float(row['emotion_model_disagreement_0_1']):.3f} | "
            f"{float(row['main_auto_v3_0_1']):.3f} | {float(row['surrogate_low_dsp_loocv']):.3f} | "
            f"{float(row['surrogate_all_emotion_loocv']):.3f} |"
        )
    return "\n".join(lines) + "\n"


def main() -> None:
    args = parse_args()
    with args.input.open(newline="", encoding="utf-8-sig") as handle:
        raw = list(csv.DictReader(handle))
    rows: list[dict[str, object]] = []
    for source in raw:
        merged: dict[str, object] = dict(source)
        merged.update(automatic_main(source))
        rows.append(merged)

    low_elapsed, low_features = add_low_cost_features(rows)
    groups = group_summary(rows)
    candidates, predictions = build_surrogates(rows, low_features)
    for index, row in enumerate(rows):
        row["surrogate_low_dsp_loocv"] = predictions["low_dsp_ridge__LOOCV"][index]
        row["surrogate_all_emotion_loocv"] = predictions["all_emotion_plus_dsp_ridge__LOOCV"][index]
    costs = cost_table(args.model_costs, low_elapsed, len(rows))

    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(args.output_dir / "per_clip_scores.csv", rows)
    write_csv(args.output_dir / "emotion_group_summary.csv", groups)
    write_csv(args.output_dir / "surrogate_candidates.csv", candidates)
    write_csv(args.output_dir / "metric_costs.csv", costs)
    (args.output_dir / "automatic_metric_report.md").write_text(report(rows, groups, candidates, costs), encoding="utf-8")
    print(f"wrote {args.output_dir} ({len(rows)} clips; low-DSP {low_elapsed:.3f}s)")


if __name__ == "__main__":
    main()
