from __future__ import annotations

import csv
import math
import os
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
from scipy.io import wavfile
from scipy.signal import resample_poly

from analyze_surrogates_v2 import (
    PROJECT_ROOT,
    EXPLORE_DIR,
    clamp,
    evaluate,
    mean,
    pearson,
    ridge_fit,
    ridge_predict,
    spearman,
    topk_overlap,
    write_csv,
)
from analyze_surrogates_v3 import load_rows_with_audio_features


OUTPUT_DIR = EXPLORE_DIR / "outputs_v3" / "sim_like"
COST_SUMMARY = EXPLORE_DIR / "outputs_v3" / "cost_measurement" / "metric_cost_summary.csv"
MODEL_NAME = "superb/wav2vec2-base-superb-er"
EMOTIONS = ["neutral", "happy", "angry", "sad"]


def load_cost_baseline() -> tuple[float, float]:
    if not COST_SUMMARY.exists():
        return math.nan, math.nan
    with COST_SUMMARY.open(newline="", encoding="utf-8-sig") as handle:
        rows = {row["scenario"]: row for row in csv.DictReader(handle)}
    try:
        main_total = float(rows["main_metric_current_pipeline"]["seconds_total_26"])
        low_dsp_total = float(rows["low_dsp_base_plus_v3_features"]["seconds_total_26"])
    except (KeyError, TypeError, ValueError):
        return math.nan, math.nan
    return main_total, low_dsp_total


def format_cost(value: float) -> str:
    return f"{value:.6f}" if math.isfinite(value) else ""


def relative_cost(value: float, baseline: float) -> str:
    return f"{value / baseline:.6f}" if math.isfinite(baseline) and baseline > 0 else ""


def cosine(a: np.ndarray, b: np.ndarray) -> float:
    denom = float(np.linalg.norm(a) * np.linalg.norm(b))
    if denom <= 1e-12:
        return 0.0
    return float(np.dot(a, b) / denom)


def read_audio_16k(path: Path) -> np.ndarray:
    sample_rate, data = wavfile.read(path)
    audio = np.asarray(data)
    if audio.ndim == 2:
        audio = audio.mean(axis=1)
    if np.issubdtype(audio.dtype, np.integer):
        audio = audio.astype("float32") / float(np.iinfo(data.dtype).max)
    else:
        audio = np.clip(audio.astype("float32"), -1.0, 1.0)
    if sample_rate != 16_000:
        gcd = math.gcd(sample_rate, 16_000)
        audio = resample_poly(audio, 16_000 // gcd, sample_rate // gcd).astype("float32")
    return audio


def resolve_audio_path(path_text: str) -> Path:
    path = Path(path_text)
    if path.is_absolute():
        return path
    return PROJECT_ROOT / path


def scored_csv_paths() -> list[tuple[str, Path]]:
    return [
        (
            "parler_emotion_v1",
            PROJECT_ROOT / "experiments" / "parler_emotion_v1" / "combined" / "parler_emotion_scored_main_metric.csv",
        ),
        (
            "boundary_metric_v1",
            PROJECT_ROOT / "experiments" / "boundary_metric_v1" / "combined" / "boundary_scored_main_metric.csv",
        ),
    ]


def load_audio_paths() -> dict[tuple[str, str], str]:
    out = {}
    for dataset, path in scored_csv_paths():
        with path.open(newline="", encoding="utf-8-sig") as handle:
            for row in csv.DictReader(handle):
                out[(dataset, row["id"])] = row["audio_path"]
    return out


def setup_model(device: str) -> dict[str, Any]:
    os.environ.setdefault("TRANSFORMERS_NO_TF", "1")
    os.environ.setdefault("USE_TF", "0")
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

    import torch
    from transformers import AutoFeatureExtractor, AutoModelForAudioClassification

    feature_extractor = AutoFeatureExtractor.from_pretrained(MODEL_NAME, local_files_only=True)
    model = AutoModelForAudioClassification.from_pretrained(MODEL_NAME, local_files_only=True)
    model.to(device)
    model.eval()
    return {
        "torch": torch,
        "feature_extractor": feature_extractor,
        "model": model,
        "device": device,
    }


def target_prob_from_logits(row: dict[str, float], logits: np.ndarray, id2label: dict[int, str]) -> float:
    exp = np.exp(logits - np.max(logits))
    probs = exp / max(float(exp.sum()), 1e-12)
    target = str(row.get("target_emotion", "neutral")).lower()
    aliases = {
        "neutral": {"neu", "neutral"},
        "happy": {"hap", "happy"},
        "angry": {"ang", "angry"},
        "sad": {"sad"},
    }
    best = 0.0
    for idx, label in id2label.items():
        normalized = str(label).lower()
        if normalized in aliases.get(target, {target}):
            best = max(best, float(probs[int(idx)]))
    return best


def extract_embeddings(rows: list[dict[str, float]]) -> tuple[np.ndarray, list[float], float, str]:
    import torch

    device = "cuda" if torch.cuda.is_available() else "cpu"
    audio_paths = load_audio_paths()

    start = time.perf_counter()
    modules = setup_model(device)
    model = modules["model"]
    feature_extractor = modules["feature_extractor"]
    id2label = dict(model.config.id2label)

    embeddings = []
    target_probs = []
    with modules["torch"].no_grad():
        for row in rows:
            audio = read_audio_16k(resolve_audio_path(audio_paths[(row["dataset"], row["id"])]))
            inputs = feature_extractor(audio, sampling_rate=16_000, return_tensors="pt", padding=True)
            inputs = {key: value.to(device) for key, value in inputs.items()}
            outputs = model(**inputs, output_hidden_states=True)
            hidden = outputs.hidden_states[-1][0]
            mask = inputs.get("attention_mask")
            if mask is not None:
                frame_mask = mask[0].to(hidden.device).float()
                if frame_mask.shape[0] != hidden.shape[0]:
                    frame_mask = torch.nn.functional.interpolate(
                        frame_mask[None, None, :], size=hidden.shape[0], mode="nearest"
                    )[0, 0]
                pooled = (hidden * frame_mask[:, None]).sum(dim=0) / torch.clamp(frame_mask.sum(), min=1.0)
            else:
                pooled = hidden.mean(dim=0)
            pooled_np = pooled.detach().float().cpu().numpy()
            norm = float(np.linalg.norm(pooled_np))
            if norm > 1e-12:
                pooled_np = pooled_np / norm
            embeddings.append(pooled_np)
            target_probs.append(target_prob_from_logits(row, outputs.logits[0].detach().cpu().numpy(), id2label))

    elapsed = time.perf_counter() - start
    return np.vstack(embeddings), target_probs, elapsed, device


def centroid(vectors: list[np.ndarray]) -> np.ndarray:
    if not vectors:
        return np.zeros(1, dtype="float32")
    center = np.mean(np.vstack(vectors), axis=0)
    norm = float(np.linalg.norm(center))
    if norm > 1e-12:
        center = center / norm
    return center


def loo_centroid_features(rows: list[dict[str, float]], embeddings: np.ndarray) -> dict[str, list[float]]:
    target_cos = []
    target_margin = []
    target_rank_score = []
    target_softmax_score = []

    for held_out, row in enumerate(rows):
        train_idx = [idx for idx in range(len(rows)) if idx != held_out]
        centroids = {}
        for emotion in EMOTIONS:
            vectors = [
                embeddings[idx]
                for idx in train_idx
                if str(rows[idx].get("target_emotion", "")).lower() == emotion
            ]
            centroids[emotion] = centroid(vectors)

        sims = {emotion: cosine(embeddings[held_out], center) for emotion, center in centroids.items() if center.size == embeddings.shape[1]}
        target = str(row.get("target_emotion", "neutral")).lower()
        target_value = sims.get(target, 0.0)
        other_values = [value for emotion, value in sims.items() if emotion != target]
        other_max = max(other_values) if other_values else 0.0
        sorted_sims = sorted(sims.items(), key=lambda item: item[1], reverse=True)
        rank = next((idx for idx, (emotion, _value) in enumerate(sorted_sims) if emotion == target), len(sorted_sims) - 1)
        exp_values = np.exp(np.asarray(list(sims.values()), dtype="float64") * 20.0)
        softmax_den = float(exp_values.sum())
        target_idx = list(sims.keys()).index(target) if target in sims else 0
        target_soft = float(exp_values[target_idx] / softmax_den) if softmax_den > 0 else 0.0

        target_cos.append(clamp((target_value + 1.0) / 2.0))
        target_margin.append(clamp(0.5 + (target_value - other_max) / 0.10))
        target_rank_score.append(clamp(1.0 - rank / max(len(sorted_sims) - 1, 1)))
        target_softmax_score.append(clamp(target_soft))

    return {
        "sim_target_cos_loo": target_cos,
        "sim_target_margin_loo": target_margin,
        "sim_target_rank_loo": target_rank_score,
        "sim_target_softmax_loo": target_softmax_score,
    }


def pairwise_consistency(rows: list[dict[str, float]], embeddings: np.ndarray) -> list[float]:
    scores = []
    for idx, row in enumerate(rows):
        target = str(row.get("target_emotion", "neutral")).lower()
        same = []
        diff = []
        for jdx, other in enumerate(rows):
            if idx == jdx:
                continue
            value = cosine(embeddings[idx], embeddings[jdx])
            if str(other.get("target_emotion", "")).lower() == target:
                same.append(value)
            else:
                diff.append(value)
        if not same or not diff:
            scores.append(0.5)
        else:
            scores.append(clamp(0.5 + (mean(same) - mean(diff)) / 0.10))
    return scores


def sim_features_for_query(
    rows: list[dict[str, float]],
    embeddings: np.ndarray,
    query_idx: int,
    reference_indices: list[int],
) -> dict[str, float]:
    query = embeddings[query_idx]
    sims: dict[str, float] = {}
    for emotion in EMOTIONS:
        vectors = [
            embeddings[idx]
            for idx in reference_indices
            if str(rows[idx].get("target_emotion", "")).lower() == emotion
        ]
        sims[emotion] = cosine(query, centroid(vectors)) if vectors else 0.0

    target = str(rows[query_idx].get("target_emotion", "neutral")).lower()
    target_value = sims.get(target, 0.0)
    other_max = max((value for emotion, value in sims.items() if emotion != target), default=0.0)
    sorted_sims = sorted(sims.items(), key=lambda item: item[1], reverse=True)
    rank = next((idx for idx, (emotion, _value) in enumerate(sorted_sims) if emotion == target), len(sorted_sims) - 1)
    sim_values = np.asarray(list(sims.values()), dtype="float64")
    exp_values = np.exp((sim_values - np.max(sim_values)) * 20.0)
    target_idx = list(sims.keys()).index(target) if target in sims else 0
    target_soft = float(exp_values[target_idx] / max(float(exp_values.sum()), 1e-12))

    same = [
        cosine(query, embeddings[idx])
        for idx in reference_indices
        if str(rows[idx].get("target_emotion", "")).lower() == target
    ]
    diff = [
        cosine(query, embeddings[idx])
        for idx in reference_indices
        if str(rows[idx].get("target_emotion", "")).lower() != target
    ]
    consistency = 0.5 if not same or not diff else clamp(0.5 + (mean(same) - mean(diff)) / 0.10)

    return {
        "sim_target_cos_loo": clamp((target_value + 1.0) / 2.0),
        "sim_target_margin_loo": clamp(0.5 + (target_value - other_max) / 0.10),
        "sim_target_rank_loo": clamp(1.0 - rank / max(len(sorted_sims) - 1, 1)),
        "sim_target_softmax_loo": clamp(target_soft),
        "sim_pairwise_consistency": consistency,
    }


def neural_cv_predictions(
    rows: list[dict[str, float]],
    embeddings: np.ndarray,
    target_probs: list[float],
    target: list[float],
    features: list[str],
    test_folds: list[list[int]],
    alpha: float = 5e-2,
) -> list[float]:
    predictions = [math.nan] * len(rows)
    all_indices = list(range(len(rows)))
    for test_idx in test_folds:
        test_set = set(test_idx)
        train_idx = [idx for idx in all_indices if idx not in test_set]
        train_rows = []
        for idx in train_idx:
            item = dict(rows[idx])
            item.update(sim_features_for_query(rows, embeddings, idx, [ref for ref in train_idx if ref != idx]))
            item["sim_ser_target_prob"] = target_probs[idx]
            train_rows.append(item)
        test_rows = []
        for idx in test_idx:
            item = dict(rows[idx])
            item.update(sim_features_for_query(rows, embeddings, idx, train_idx))
            item["sim_ser_target_prob"] = target_probs[idx]
            test_rows.append(item)

        x_train = [[row[name] for name in features] for row in train_rows]
        x_test = [[row[name] for name in features] for row in test_rows]
        coef, mean_x, std_x = ridge_fit(x_train, [target[idx] for idx in train_idx], alpha=alpha)
        fold_predictions = ridge_predict(x_test, coef, mean_x, std_x)
        for idx, prediction in zip(test_idx, fold_predictions):
            predictions[idx] = prediction
    return predictions


def build_rows(rows: list[dict[str, float]], sim_features: dict[str, list[float]], target_probs: list[float]) -> list[dict[str, float]]:
    out = []
    for idx, row in enumerate(rows):
        item = dict(row)
        for key, values in sim_features.items():
            item[key] = values[idx]
        item["sim_pairwise_consistency"] = sim_features["sim_pairwise_consistency"][idx]
        item["sim_ser_target_prob"] = target_probs[idx]
        out.append(item)
    return out


def prediction_table(rows: list[dict[str, float]], predictions: dict[str, list[float]]) -> list[dict[str, str]]:
    out = []
    for idx, row in enumerate(rows):
        item = {
            "dataset": row["dataset"],
            "id": row["id"],
            "target_emotion": row.get("target_emotion", ""),
            "main_metric_0_1": f"{row['main_metric_0_1']:.6f}",
        }
        for name, values in predictions.items():
            item[name] = f"{values[idx]:.6f}"
            item[f"{name}_abs_error"] = f"{abs(values[idx] - row['main_metric_0_1']):.6f}"
        out.append(item)
    return out


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    rows, low_dsp_elapsed = load_rows_with_audio_features()
    target = [row["main_metric_0_1"] for row in rows]
    embeddings, target_probs, embedding_elapsed, device = extract_embeddings(rows)

    sim_features = loo_centroid_features(rows, embeddings)
    sim_features["sim_pairwise_consistency"] = pairwise_consistency(rows, embeddings)
    rows = build_rows(rows, sim_features, target_probs)

    prediction_map: dict[str, list[float]] = {
        "sim_target_cos_loo": [row["sim_target_cos_loo"] for row in rows],
        "sim_target_margin_loo": [row["sim_target_margin_loo"] for row in rows],
        "sim_target_rank_loo": [row["sim_target_rank_loo"] for row in rows],
        "sim_target_softmax_loo": [row["sim_target_softmax_loo"] for row in rows],
        "sim_pairwise_consistency": [row["sim_pairwise_consistency"] for row in rows],
        "sim_ser_target_prob": [row["sim_ser_target_prob"] for row in rows],
    }

    candidate_specs = [
        ("sim_target_cos_loo", "medium_neural", "wav2vec2 embedding cosine to target-emotion centroid"),
        ("sim_target_margin_loo", "medium_neural", "target centroid cosine minus nearest non-target centroid"),
        ("sim_target_rank_loo", "medium_neural", "rank of target centroid among emotion centroids"),
        ("sim_target_softmax_loo", "medium_neural", "softmax over emotion centroid cosine similarities"),
        ("sim_pairwise_consistency", "medium_neural", "same-emotion pairwise embedding similarity minus different-emotion similarity"),
        ("sim_ser_target_prob", "medium_neural", "same model classifier target probability"),
    ]

    ridge_specs = {
        "ridge_sim_centroid_loo": [
            "sim_target_cos_loo",
            "sim_target_margin_loo",
            "sim_target_rank_loo",
            "sim_target_softmax_loo",
            "sim_pairwise_consistency",
        ],
        "ridge_sim_plus_low_dsp_loo": [
            "sim_target_cos_loo",
            "sim_target_margin_loo",
            "sim_pairwise_consistency",
            "target_style_fit_v1",
            "prosody_fit_light",
            "rate_fit",
            "text_ease",
        ],
        "ridge_sim_ser_plus_low_dsp_loo": [
            "sim_ser_target_prob",
            "sim_target_margin_loo",
            "sim_pairwise_consistency",
            "target_style_fit_v1",
            "prosody_fit_light",
            "rate_fit",
            "text_ease",
        ],
        "lodo_sim_plus_low_dsp": [
            "sim_target_cos_loo",
            "sim_target_margin_loo",
            "sim_pairwise_consistency",
            "target_style_fit_v1",
            "prosody_fit_light",
            "rate_fit",
            "text_ease",
        ],
        "lodo_sim_ser_plus_low_dsp": [
            "sim_ser_target_prob",
            "sim_target_margin_loo",
            "sim_pairwise_consistency",
            "target_style_fit_v1",
            "prosody_fit_light",
            "rate_fit",
            "text_ease",
        ],
    }

    candidates = []
    for name, tier, ingredients in candidate_specs:
        candidates.append(evaluate(name, prediction_map[name], target, tier, ingredients))

    for name, features in ridge_specs.items():
        if name.startswith("lodo_"):
            datasets = sorted({str(row["dataset"]) for row in rows})
            folds = [[idx for idx, row in enumerate(rows) if row["dataset"] == dataset] for dataset in datasets]
            preds = neural_cv_predictions(rows, embeddings, target_probs, target, features, folds)
            notes = "fold-pure leave-one-dataset-out"
        else:
            folds = [[idx] for idx in range(len(rows))]
            preds = neural_cv_predictions(rows, embeddings, target_probs, target, features, folds)
            notes = "fold-pure LOOCV ridge"
        prediction_map[name] = preds
        candidates.append(evaluate(name, preds, target, "medium_neural", "+".join(features), notes))

    candidates.sort(key=lambda row: (float(row["spearman"]), float(row["pearson"])), reverse=True)

    write_csv(OUTPUT_DIR / "sim_like_candidates.csv", candidates)
    write_csv(OUTPUT_DIR / "sim_like_per_sample.csv", prediction_table(rows, prediction_map))

    main_total, low_dsp_total = load_cost_baseline()
    sim_total = embedding_elapsed
    sim_plus_low_total = embedding_elapsed + low_dsp_elapsed
    cost_rows = [
        {
            "scenario": "main_metric_current_pipeline",
            "seconds_total_26": format_cost(main_total),
            "seconds_per_clip": format_cost(main_total / len(rows)) if math.isfinite(main_total) else "",
            "relative_to_main": "1.000000" if math.isfinite(main_total) else "",
            "notes": f"loaded from {COST_SUMMARY.relative_to(PROJECT_ROOT)}",
        },
        {
            "scenario": "low_dsp_base_plus_v3_features",
            "seconds_total_26": format_cost(low_dsp_total),
            "seconds_per_clip": format_cost(low_dsp_total / len(rows)) if math.isfinite(low_dsp_total) else "",
            "relative_to_main": relative_cost(low_dsp_total, main_total),
            "notes": f"loaded from {COST_SUMMARY.relative_to(PROJECT_ROOT)}",
        },
        {
            "scenario": "sim_like_embedding_only",
            "seconds_total_26": f"{sim_total:.6f}",
            "seconds_per_clip": f"{sim_total / len(rows):.6f}",
            "relative_to_main": relative_cost(sim_total, main_total),
            "notes": f"{MODEL_NAME}; device={device}; local cached model",
        },
        {
            "scenario": "sim_like_plus_low_dsp",
            "seconds_total_26": f"{sim_plus_low_total:.6f}",
            "seconds_per_clip": f"{sim_plus_low_total / len(rows):.6f}",
            "relative_to_main": relative_cost(sim_plus_low_total, main_total),
            "notes": "embedding extraction plus enhanced low-DSP feature extraction",
        },
    ]
    write_csv(OUTPUT_DIR / "sim_like_costs.csv", cost_rows)

    top = candidates[:10]
    lines = [
        "# SIM-like Neural Surrogate Exploration",
        "",
        f"Model: `{MODEL_NAME}`. Device: `{device}`. Samples: `{len(rows)}`.",
        "",
        "The SIM-like signal here is not a production speaker-SIM metric with a separate reference speaker recording. It is a local neural audio-embedding proxy: wav2vec2 hidden embeddings are compared with target-emotion/style centroids using cosine similarity.",
        "",
        "## Agreement With Main Metric",
        "",
        "| candidate | Pearson | Spearman | Kendall | pairwise acc. | MAE | top5 | bottom5 | notes |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |",
    ]
    for row in top:
        lines.append(
            f"| `{row['candidate']}` | {row['pearson']} | {row['spearman']} | {row['kendall_tau_b']} | {row['pairwise_accuracy']} | {row['mae']} | {row['top5_overlap']} | {row['bottom5_overlap']} | {row['notes']} |"
        )
    lines.extend(
        [
            "",
            "## Cost",
            "",
            "| scenario | seconds / 26 clips | seconds / clip | relative to main |",
            "| --- | ---: | ---: | ---: |",
        ]
    )
    for row in cost_rows:
        lines.append(
            f"| `{row['scenario']}` | {row['seconds_total_26']} | {row['seconds_per_clip']} | {row['relative_to_main']} |"
        )
    lines.extend(
        [
            "",
            "## Takeaway",
            "",
            "- Raw centroid SIM-like cosine scores do not fit the current main metric on this dataset.",
            "- The useful neural signal is the classifier/logit side of the same wav2vec2 model, especially when combined with SIM-like margin and low-DSP features.",
            "- This is still much cheaper than the current full main metric because it avoids Whisper ASR, but it is more expensive than pure low-DSP features.",
            "- The strongest candidate reuses the same SER model family as the teacher. Its agreement is teacher replication, not independent perceptual validation.",
            "- Leave-dataset-out still holds out only a test-set type; both sets share Parler-TTS and the same speaker, so no cross-system generalization is established.",
        ]
    )
    (OUTPUT_DIR / "sim_like_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")

    print(f"wrote {OUTPUT_DIR / 'sim_like_candidates.csv'}")
    print(f"wrote {OUTPUT_DIR / 'sim_like_per_sample.csv'}")
    print(f"wrote {OUTPUT_DIR / 'sim_like_costs.csv'}")
    print(f"wrote {OUTPUT_DIR / 'sim_like_report.md'}")


if __name__ == "__main__":
    main()
