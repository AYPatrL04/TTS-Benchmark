from __future__ import annotations

import csv
import math
import sys
import time
from pathlib import Path

import numpy as np


HERE = Path(__file__).resolve().parent
PROJECT_ROOT = HERE.parent
sys.path.insert(0, str(HERE))

from analyze_sim_like_surrogates import cosine, read_audio_16k, resolve_audio_path, setup_model, target_prob_from_logits
from analyze_surrogates import acoustic_features, clamp, text_difficulty
from analyze_surrogates_v2 import evaluate, mean, ridge_fit, ridge_predict, write_csv


EXPERIMENT = PROJECT_ROOT / "experiments" / "multisystem_generalization_v1"
SCORED_CSV = EXPERIMENT / "combined" / "multisystem_scored_main_metric.csv"
OUTPUT_DIR = EXPERIMENT / "analysis"
LOW_DSP_FEATURES = ["signal_quality", "rate_fit", "text_ease"]
NEURAL_FEATURES = ["ser_target_prob", "cross_source_sim", *LOW_DSP_FEATURES]
RIDGE_ALPHA = 10.0


def parse_float(row: dict[str, str], key: str) -> float:
    try:
        return float(row.get(key, ""))
    except (TypeError, ValueError):
        return math.nan


def read_rows() -> tuple[list[dict[str, object]], float]:
    with SCORED_CSV.open(newline="", encoding="utf-8-sig") as handle:
        raw_rows = list(csv.DictReader(handle))
    rows: list[dict[str, object]] = []
    start = time.perf_counter()
    for raw in raw_rows:
        item: dict[str, object] = dict(raw)
        item.update(text_difficulty(raw["text"]))
        item.update(acoustic_features(raw))
        item["main_metric_0_1"] = parse_float(raw, "main_metric_0_1")
        item["intelligibility_component_0_1"] = parse_float(raw, "intelligibility_component_0_1")
        item["emotion_component_0_1"] = parse_float(raw, "emotion_component_0_1")
        item["acoustic_sanity_score_0_1"] = parse_float(raw, "acoustic_sanity_score_0_1")
        item["is_boundary"] = int(raw.get("is_boundary", "0") or 0)
        rows.append(item)
    return rows, time.perf_counter() - start


def extract_neural(rows: list[dict[str, object]]) -> tuple[np.ndarray, list[float], float, str]:
    start = time.perf_counter()
    modules = setup_model("cuda" if __import__("torch").cuda.is_available() else "cpu")
    torch = modules["torch"]
    model = modules["model"]
    extractor = modules["feature_extractor"]
    device = modules["device"]
    embeddings: list[np.ndarray] = []
    target_probs: list[float] = []
    with torch.no_grad():
        for row in rows:
            audio = read_audio_16k(resolve_audio_path(str(row["audio_path"])))
            inputs = extractor(audio, sampling_rate=16_000, return_tensors="pt", padding=True)
            inputs = {key: value.to(device) for key, value in inputs.items()}
            outputs = model(**inputs, output_hidden_states=True)
            pooled = outputs.hidden_states[-1][0].mean(dim=0).detach().float().cpu().numpy()
            pooled /= max(float(np.linalg.norm(pooled)), 1e-12)
            embeddings.append(pooled)
            target_probs.append(
                target_prob_from_logits(row, outputs.logits[0].detach().float().cpu().numpy(), dict(model.config.id2label))
            )
    return np.vstack(embeddings), target_probs, time.perf_counter() - start, device


def reference_similarity(
    rows: list[dict[str, object]], embeddings: np.ndarray, query_idx: int, reference_indices: list[int]
) -> float:
    same_text = [idx for idx in reference_indices if rows[idx]["text_id"] == rows[query_idx]["text_id"]]
    refs = same_text or reference_indices
    if not refs:
        return 0.5
    center = np.mean(embeddings[refs], axis=0)
    center /= max(float(np.linalg.norm(center)), 1e-12)
    return clamp((cosine(embeddings[query_idx], center) + 1.0) / 2.0)


def fold_rows(
    rows: list[dict[str, object]], embeddings: np.ndarray, target_probs: list[float], indices: list[int], references: list[int]
) -> list[dict[str, object]]:
    out = []
    for idx in indices:
        item = dict(rows[idx])
        item["ser_target_prob"] = target_probs[idx]
        item["cross_source_sim"] = reference_similarity(rows, embeddings, idx, [ref for ref in references if ref != idx])
        out.append(item)
    return out


def grouped_ridge_predictions(
    rows: list[dict[str, object]],
    embeddings: np.ndarray,
    target_probs: list[float],
    features: list[str],
    group_key: str,
) -> list[float]:
    predictions = [math.nan] * len(rows)
    groups = sorted({str(row[group_key]) for row in rows})
    for group in groups:
        test_idx = [idx for idx, row in enumerate(rows) if str(row[group_key]) == group]
        train_idx = [idx for idx in range(len(rows)) if idx not in set(test_idx)]
        train_rows = fold_rows(rows, embeddings, target_probs, train_idx, train_idx)
        test_rows = fold_rows(rows, embeddings, target_probs, test_idx, train_idx)
        x_train = [[float(row[name]) for name in features] for row in train_rows]
        x_test = [[float(row[name]) for name in features] for row in test_rows]
        coef, mean_x, std_x = ridge_fit(x_train, [float(rows[idx]["main_metric_0_1"]) for idx in train_idx], alpha=RIDGE_ALPHA)
        for idx, prediction in zip(test_idx, ridge_predict(x_test, coef, mean_x, std_x)):
            predictions[idx] = prediction
    return predictions


def fixed_low_dsp(row: dict[str, object]) -> float:
    return clamp(0.40 * float(row["signal_quality"]) + 0.35 * float(row["rate_fit"]) + 0.25 * float(row["text_ease"]))


def group_summary(rows: list[dict[str, object]], predictions: dict[str, list[float]], key: str) -> list[dict[str, str]]:
    out = []
    for group in sorted({str(row[key]) for row in rows}):
        indices = [idx for idx, row in enumerate(rows) if str(row[key]) == group]
        item = {
            "group_by": key,
            "group": group,
            "samples": str(len(indices)),
            "main_mean": f"{mean(float(rows[idx]['main_metric_0_1']) for idx in indices):.6f}",
            "intelligibility_mean": f"{mean(float(rows[idx]['intelligibility_component_0_1']) for idx in indices):.6f}",
            "emotion_teacher_mean": f"{mean(float(rows[idx]['emotion_component_0_1']) for idx in indices):.6f}",
            "sanity_mean": f"{mean(float(rows[idx]['acoustic_sanity_score_0_1']) for idx in indices):.6f}",
        }
        for name, values in predictions.items():
            item[f"{name}_mean"] = f"{mean(values[idx] for idx in indices):.6f}"
            item[f"{name}_mae"] = f"{mean(abs(values[idx] - float(rows[idx]['main_metric_0_1'])) for idx in indices):.6f}"
        out.append(item)
    return out


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    rows, dsp_seconds = read_rows()
    embeddings, target_probs, neural_seconds, device = extract_neural(rows)
    target = [float(row["main_metric_0_1"]) for row in rows]

    full_refs = list(range(len(rows)))
    all_features = fold_rows(rows, embeddings, target_probs, full_refs, full_refs)
    fixed = [fixed_low_dsp(row) for row in all_features]
    predictions = {
        "fixed_low_dsp": fixed,
        "ser_target_prob": target_probs,
        "loso_low_dsp_ridge": grouped_ridge_predictions(rows, embeddings, target_probs, LOW_DSP_FEATURES, "tts_system"),
        "loso_neural_dsp_ridge": grouped_ridge_predictions(rows, embeddings, target_probs, NEURAL_FEATURES, "tts_system"),
        "loto_low_dsp_ridge": grouped_ridge_predictions(rows, embeddings, target_probs, LOW_DSP_FEATURES, "text_id"),
        "loto_neural_dsp_ridge": grouped_ridge_predictions(rows, embeddings, target_probs, NEURAL_FEATURES, "text_id"),
        "lobc_low_dsp_ridge": grouped_ridge_predictions(rows, embeddings, target_probs, LOW_DSP_FEATURES, "is_boundary"),
        "lobc_neural_dsp_ridge": grouped_ridge_predictions(rows, embeddings, target_probs, NEURAL_FEATURES, "is_boundary"),
    }

    candidate_rows = []
    for name, values in predictions.items():
        notes = {
            "fixed_low_dsp": "no fitted parameters",
            "ser_target_prob": "raw SER signal; directly reused by the teacher",
        }.get(name, f"fold-pure grouped ridge; alpha={RIDGE_ALPHA:g}")
        tier = "low_dsp" if "neural" not in name and name != "ser_target_prob" else "medium_neural"
        candidate_rows.append(evaluate(name, values, target, tier, "see report", notes))
    write_csv(OUTPUT_DIR / "surrogate_grouped_candidates.csv", candidate_rows)

    per_clip = []
    for idx, row in enumerate(rows):
        item = {
            "id": row["id"],
            "text_id": row["text_id"],
            "tts_system": row["tts_system"],
            "voice": row["voice"],
            "is_boundary": row["is_boundary"],
            "boundary_type": row["boundary_type"],
            "main_metric_0_1": f"{target[idx]:.6f}",
            "intelligibility": f"{float(row['intelligibility_component_0_1']):.6f}",
            "emotion_teacher": f"{float(row['emotion_component_0_1']):.6f}",
            "acoustic_sanity": f"{float(row['acoustic_sanity_score_0_1']):.6f}",
        }
        for name, values in predictions.items():
            item[name] = f"{values[idx]:.6f}"
        per_clip.append(item)
    write_csv(OUTPUT_DIR / "per_clip_scores.csv", per_clip)

    summaries = []
    for key in ("tts_system", "is_boundary", "text_id"):
        summaries.extend(group_summary(rows, predictions, key))
    write_csv(OUTPUT_DIR / "group_summary.csv", summaries)

    candidate_by_name = {row["candidate"]: row for row in candidate_rows}
    system_rows = [row for row in summaries if row["group_by"] == "tts_system"]
    boundary_rows = [row for row in summaries if row["group_by"] == "is_boundary"]
    lines = [
        "# Multi-system Generalization and Overfitting Audit",
        "",
        "This audit uses the same six texts across Parler-TTS Mini, Bark Small, and Windows SAPI Zira. Three texts are regular and three are boundary cases. All targets are neutral so cross-system behavior is not confounded by requested emotion labels.",
        "",
        "## Validation controls",
        "",
        f"- Learned surrogates use fixed features and strong ridge regularization (`alpha={RIDGE_ALPHA:g}`).",
        "- Standardization and fitting are repeated inside every outer fold.",
        "- LOSO holds out an entire TTS system; LOTO holds out a text; LOBC trains on one boundary condition and tests on the other.",
        "- SIM-like references are rebuilt from training-fold audio only.",
        "- `ser_target_prob` is reported as teacher replication, not independent validation.",
        "",
        "## Overall agreement",
        "",
        "| candidate | validation | Spearman | pairwise accuracy | MAE |",
        "| --- | --- | ---: | ---: | ---: |",
    ]
    for name in predictions:
        row = candidate_by_name[name]
        validation = "none" if name in {"fixed_low_dsp", "ser_target_prob"} else name.split("_")[0].upper()
        lines.append(f"| `{name}` | {validation} | {row['spearman']} | {row['pairwise_accuracy']} | {row['mae']} |")
    lines.extend(["", "## Main metric by system", "", "| system | n | Main | I | E (SER teacher) | sanity | LOSO low-DSP MAE | LOSO neural+DSP MAE |", "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |"])
    for row in system_rows:
        lines.append(f"| {row['group']} | {row['samples']} | {row['main_mean']} | {row['intelligibility_mean']} | {row['emotion_teacher_mean']} | {row['sanity_mean']} | {row['loso_low_dsp_ridge_mae']} | {row['loso_neural_dsp_ridge_mae']} |")
    lines.extend(["", "## Regular versus boundary", "", "| boundary | n | Main | I | E | sanity | fixed low-DSP | LOSO neural+DSP |", "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |"])
    for row in boundary_rows:
        label = "yes" if row["group"] == "1" else "no"
        lines.append(f"| {label} | {row['samples']} | {row['main_mean']} | {row['intelligibility_mean']} | {row['emotion_teacher_mean']} | {row['sanity_mean']} | {row['fixed_low_dsp_mean']} | {row['loso_neural_dsp_ridge_mean']} |")
    lines.extend([
        "",
        "## Cost observed in this analysis",
        "",
        f"- low-DSP extraction: `{dsp_seconds:.3f}` s total, `{dsp_seconds / len(rows):.3f}` s/clip",
        f"- neural SER/embedding extraction: `{neural_seconds:.3f}` s total, `{neural_seconds / len(rows):.3f}` s/clip on `{device}`",
        "- Main metric additionally requires Whisper ASR; generation time is excluded from metric cost.",
        "",
        "## Interpretation",
        "",
        "A large drop from text-level validation to LOSO indicates source overfitting. A neural surrogate can agree with this teacher by copying its SER component, but that does not establish human perceptual validity. Cross-system conclusions must therefore use LOSO results and retain I/E/sanity as separate dimensions.",
        "",
        "The automatic teacher cannot determine whether Bark's highly rated output is actually more natural than Zira's synthetic timbre, or how objectionable Parler's acronym errors are to listeners. Use the blind rating template in `human_evaluation/ratings_template.csv` before calibrating the Main scalar or claiming a final surrogate.",
    ])
    (OUTPUT_DIR / "generalization_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"wrote {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
