from __future__ import annotations

import csv
import itertools
import math
import time
from collections import Counter
from pathlib import Path

import numpy as np
from scipy.io import wavfile

from analyze_surrogates_v2 import (
    FEATURE_CSV,
    PROJECT_ROOT,
    EXPLORE_DIR,
    SCORED_DATASETS,
    clamp,
    evaluate,
    exhaustive_subset_search,
    mean,
    pearson,
    per_sample_errors,
    ridge_fit,
    ridge_leave_dataset_out,
    ridge_loo,
    ridge_predict,
    spearman,
    topk_overlap,
    write_csv,
)


OUTPUT_DIR = EXPLORE_DIR / "outputs_v3"


BASE_LOW_FEATURES = [
    "text_ease",
    "rate_fit",
    "duration_sec",
    "speech_rate_wps",
    "signal_quality",
    "silence_ratio",
    "energy_cv",
    "f0_std_semitones",
    "f0_range_semitones_p90_p10",
    "voiced_ratio",
    "zcr",
    "spectral_flatness",
    "spectral_centroid_hz",
    "high_freq_ratio",
    "prosody_fit_light",
    "prosody_activity_light",
]

NEW_LOW_FEATURES = [
    "target_style_fit_v1",
    "emotion_arousal_fit_v1",
    "pause_naturalness",
    "envelope_stability",
    "dynamic_range_fit",
    "spectral_balance_fit",
    "voice_presence_fit",
    "articulation_risk_inverse",
    "delivery_fit_v1",
]

SUBSET_SEARCH_FEATURES = [
    "text_ease",
    "rate_fit",
    "duration_sec",
    "speech_rate_wps",
    "silence_ratio",
    "energy_cv",
    "f0_range_semitones_p90_p10",
    "voiced_ratio",
    "zcr",
    "spectral_centroid_hz",
    "high_freq_ratio",
    "prosody_fit_light",
    "prosody_activity_light",
    "target_style_fit_v1",
    "emotion_arousal_fit_v1",
    "pause_naturalness",
    "spectral_balance_fit",
    "articulation_risk_inverse",
    "delivery_fit_v1",
]

NESTED_SUBSET_FEATURES = [
    "text_ease",
    "rate_fit",
    "energy_cv",
    "f0_range_semitones_p90_p10",
    "high_freq_ratio",
    "prosody_fit_light",
    "prosody_activity_light",
    "target_style_fit_v1",
    "pause_naturalness",
    "spectral_balance_fit",
    "articulation_risk_inverse",
    "delivery_fit_v1",
]

MEDIUM_FEATURES = [
    "emotion_component_0_1",
    "target_style_fit_v1",
    "prosody_fit_light",
    "rate_fit",
    "text_ease",
]

HIGH_REFERENCE_FEATURES = [
    "intelligibility_component_0_1",
    "emotion_component_0_1",
    "quality_component_0_1",
]


def parse_float(row: dict[str, str], key: str, default: float = math.nan) -> float:
    try:
        return float(row.get(key, ""))
    except ValueError:
        return default


def resolve_audio_path(path_text: str) -> Path:
    path = Path(path_text)
    if path.is_absolute():
        return path
    return PROJECT_ROOT / path


def read_audio(path: Path) -> tuple[int, np.ndarray]:
    sample_rate, data = wavfile.read(path)
    audio = np.asarray(data)
    if audio.ndim == 2:
        audio = audio.mean(axis=1)
    if np.issubdtype(audio.dtype, np.integer):
        audio = audio.astype("float64") / float(np.iinfo(data.dtype).max)
    else:
        audio = audio.astype("float64")
    audio = np.nan_to_num(audio)
    if audio.size == 0:
        return sample_rate, audio
    peak = float(np.max(np.abs(audio)))
    if peak > 1.5:
        audio = audio / peak
    return sample_rate, audio


def framed(audio: np.ndarray, sample_rate: int, frame_ms: float = 40.0, hop_ms: float = 10.0) -> np.ndarray:
    frame = max(1, int(round(sample_rate * frame_ms / 1000.0)))
    hop = max(1, int(round(sample_rate * hop_ms / 1000.0)))
    if audio.size < frame:
        audio = np.pad(audio, (0, frame - audio.size))
    count = 1 + max(0, (audio.size - frame) // hop)
    out = np.empty((count, frame), dtype="float64")
    for idx in range(count):
        start = idx * hop
        out[idx] = audio[start : start + frame]
    return out


def fit_to_target(value: float, target: float, tolerance: float) -> float:
    return clamp(1.0 - abs(value - target) / max(tolerance, 1e-8))


def norm_range(value: float, low: float, high: float) -> float:
    return clamp((value - low) / max(high - low, 1e-8))


def silence_segments(mask: np.ndarray, hop_sec: float) -> tuple[int, float]:
    if mask.size == 0:
        return 0, 0.0
    count = 0
    total = 0.0
    run = 0
    for active in mask:
        if active:
            if run * hop_sec >= 0.12:
                count += 1
                total += run * hop_sec
            run = 0
        else:
            run += 1
    if run * hop_sec >= 0.12:
        count += 1
        total += run * hop_sec
    return count, total


def audio_shape_features(audio: np.ndarray, sample_rate: int) -> dict[str, float]:
    if audio.size == 0:
        return {
            "pause_count": 0.0,
            "pause_rate_per_sec": 0.0,
            "active_ratio_env": 0.0,
            "dynamic_range_db": 0.0,
            "envelope_jitter": 0.0,
            "spectral_rolloff_hz": 0.0,
            "spectral_bandwidth_hz": 0.0,
            "low_band_ratio": 0.0,
            "mid_band_ratio": 0.0,
            "very_high_band_ratio": 0.0,
        }

    frames = framed(audio, sample_rate)
    window = np.hanning(frames.shape[1])
    rms = np.sqrt(np.mean(frames * frames, axis=1) + 1e-12)
    db = 20.0 * np.log10(rms + 1e-8)
    active_threshold = max(-55.0, float(np.percentile(db, 80)) - 28.0)
    active = db > active_threshold
    hop_sec = 0.010
    pause_count, pause_total = silence_segments(active, hop_sec)
    duration = audio.size / float(sample_rate)

    spectra = np.abs(np.fft.rfft(frames * window, axis=1))
    freqs = np.fft.rfftfreq(frames.shape[1], d=1.0 / sample_rate)
    power = spectra + 1e-10
    power_sum = power.sum(axis=1)
    centroid = (power * freqs).sum(axis=1) / power_sum
    bandwidth = np.sqrt(((freqs - centroid[:, None]) ** 2 * power).sum(axis=1) / power_sum)
    cdf = np.cumsum(power, axis=1) / power_sum[:, None]
    rolloff_idx = np.argmax(cdf >= 0.85, axis=1)
    rolloff = freqs[rolloff_idx]

    total_power = float(power.sum())
    low_band = float(power[:, freqs < 300.0].sum()) / total_power
    mid_band = float(power[:, (freqs >= 300.0) & (freqs < 3000.0)].sum()) / total_power
    very_high_band = float(power[:, freqs >= 6000.0].sum()) / total_power

    return {
        "pause_count": float(pause_count),
        "pause_rate_per_sec": pause_count / max(duration, 1e-8),
        "active_ratio_env": float(np.mean(active)),
        "dynamic_range_db": float(np.percentile(db, 95) - np.percentile(db, 10)),
        "envelope_jitter": float(np.mean(np.abs(np.diff(db))) / 20.0) if len(db) > 1 else 0.0,
        "spectral_rolloff_hz": float(np.mean(rolloff)),
        "spectral_bandwidth_hz": float(np.mean(bandwidth)),
        "low_band_ratio": low_band,
        "mid_band_ratio": mid_band,
        "very_high_band_ratio": very_high_band,
        "long_pause_total_sec": pause_total,
    }


def style_targets(target_emotion: str) -> dict[str, float]:
    profiles = {
        "happy": {
            "activity": 0.82,
            "rate": 3.25,
            "loudness": -17.0,
            "pitch_range": 8.0,
            "energy_cv": 0.88,
            "pause_rate": 0.20,
            "spectral_centroid": 740.0,
        },
        "angry": {
            "activity": 0.88,
            "rate": 3.65,
            "loudness": -17.5,
            "pitch_range": 8.5,
            "energy_cv": 0.95,
            "pause_rate": 0.18,
            "spectral_centroid": 760.0,
        },
        "sad": {
            "activity": 0.28,
            "rate": 2.65,
            "loudness": -20.5,
            "pitch_range": 4.0,
            "energy_cv": 0.62,
            "pause_rate": 0.45,
            "spectral_centroid": 520.0,
        },
        "neutral": {
            "activity": 0.55,
            "rate": 3.15,
            "loudness": -18.5,
            "pitch_range": 5.5,
            "energy_cv": 0.78,
            "pause_rate": 0.28,
            "spectral_centroid": 620.0,
        },
    }
    return profiles.get(target_emotion.lower(), profiles["neutral"])


def add_derived_features(row: dict[str, float]) -> None:
    target = style_targets(str(row.get("target_emotion", "neutral")))
    rate = row["speech_rate_wps"]
    activity = row["prosody_activity_light"]
    pitch_range = row["f0_range_semitones_p90_p10"]
    energy_cv = row["energy_cv"]
    loudness = row["rms_dbfs"]
    pause_rate = row["pause_rate_per_sec"]
    centroid = row["spectral_centroid_hz"]

    activity_fit = fit_to_target(activity, target["activity"], 0.42)
    rate_fit_emotion = fit_to_target(rate, target["rate"], 1.25)
    loudness_fit = fit_to_target(loudness, target["loudness"], 7.0)
    pitch_fit = fit_to_target(pitch_range, target["pitch_range"], 5.0)
    energy_fit = fit_to_target(energy_cv, target["energy_cv"], 0.55)
    pause_fit = fit_to_target(pause_rate, target["pause_rate"], 0.65)
    centroid_fit = fit_to_target(centroid, target["spectral_centroid"], 650.0)

    row["emotion_arousal_fit_v1"] = clamp(
        0.35 * activity_fit
        + 0.20 * energy_fit
        + 0.15 * pitch_fit
        + 0.15 * loudness_fit
        + 0.15 * rate_fit_emotion
    )
    row["target_style_fit_v1"] = clamp(
        0.28 * activity_fit
        + 0.18 * rate_fit_emotion
        + 0.16 * pitch_fit
        + 0.16 * energy_fit
        + 0.12 * loudness_fit
        + 0.10 * pause_fit
    )
    row["pause_naturalness"] = clamp(
        0.50 * fit_to_target(row["silence_ratio"], 0.16, 0.18)
        + 0.35 * pause_fit
        + 0.15 * fit_to_target(row["active_ratio_env"], 0.72, 0.25)
    )
    row["envelope_stability"] = clamp(
        0.55 * fit_to_target(row["envelope_jitter"], 0.08, 0.10)
        + 0.45 * fit_to_target(row["dynamic_range_db"], 22.0, 18.0)
    )
    row["dynamic_range_fit"] = fit_to_target(row["dynamic_range_db"], 22.0, 20.0)
    row["spectral_balance_fit"] = clamp(
        0.45 * centroid_fit
        + 0.25 * fit_to_target(row["high_freq_ratio"], 0.06, 0.08)
        + 0.20 * fit_to_target(row["very_high_band_ratio"], 0.018, 0.035)
        + 0.10 * fit_to_target(row["low_band_ratio"], 0.22, 0.20)
    )
    row["voice_presence_fit"] = clamp(
        0.45 * fit_to_target(row["mid_band_ratio"], 0.62, 0.28)
        + 0.35 * fit_to_target(row["voiced_ratio"], 0.64, 0.22)
        + 0.20 * fit_to_target(row["active_ratio_env"], 0.72, 0.25)
    )

    text_risk = clamp(
        0.45 * row["text_difficulty"]
        + 0.20 * norm_range(row["speech_rate_wps"], 3.6, 5.0)
        + 0.20 * norm_range(row["zcr"], 0.11, 0.22)
        + 0.15 * norm_range(row["spectral_centroid_hz"], 950.0, 1650.0)
    )
    row["articulation_risk_inverse"] = clamp(1.0 - text_risk)
    row["delivery_fit_v1"] = clamp(
        0.36 * row["target_style_fit_v1"]
        + 0.22 * row["prosody_fit_light"]
        + 0.16 * row["pause_naturalness"]
        + 0.14 * row["spectral_balance_fit"]
        + 0.12 * row["articulation_risk_inverse"]
    )


def load_rows_with_audio_features() -> tuple[list[dict[str, float]], float]:
    base_rows: dict[tuple[str, str], dict[str, float]] = {}
    with FEATURE_CSV.open(newline="", encoding="utf-8-sig") as handle:
        for raw in csv.DictReader(handle):
            item: dict[str, float] = {
                "dataset": raw["dataset"],
                "id": raw["id"],
                "case_type": raw.get("case_type", ""),
                "target_emotion": raw.get("target_emotion", ""),
            }
            for key, value in raw.items():
                if key in item:
                    continue
                item[key] = parse_float(raw, key)
            base_rows[(raw["dataset"], raw["id"])] = item

    rows = []
    start = time.perf_counter()
    for dataset_name, csv_path in SCORED_DATASETS:
        with csv_path.open(newline="", encoding="utf-8-sig") as handle:
            for raw in csv.DictReader(handle):
                item = dict(base_rows[(dataset_name, raw["id"])])
                for key, value in raw.items():
                    if key in {"id", "case_type", "target_emotion", "text", "description", "audio_path", "asr_transcript", "model"}:
                        continue
                    item[key] = parse_float(raw, key)
                audio_path = resolve_audio_path(raw["audio_path"])
                sample_rate, audio = read_audio(audio_path)
                item.update(audio_shape_features(audio, sample_rate))
                add_derived_features(item)
                rows.append(item)
    elapsed = time.perf_counter() - start
    return rows, elapsed


def ridge_predictions(rows: list[dict[str, float]], target: list[float], features: list[str], alpha: float = 5e-2) -> list[float]:
    return ridge_loo(rows, target, features, alpha=alpha)


def ridge_train_predict(train_rows: list[dict[str, float]], train_y: list[float], test_rows: list[dict[str, float]], features: list[str]) -> list[float]:
    coef, mean_x, std_x = ridge_fit([[row[name] for name in features] for row in train_rows], train_y, alpha=5e-2)
    return ridge_predict([[row[name] for name in features] for row in test_rows], coef, mean_x, std_x)


def nested_subset_loo(
    rows: list[dict[str, float]],
    target: list[float],
    features: list[str],
    max_size: int = 4,
) -> tuple[list[float], Counter[str]]:
    preds = []
    selected = Counter()
    for held_out in range(len(rows)):
        train_idx = [idx for idx in range(len(rows)) if idx != held_out]
        train_rows = [rows[idx] for idx in train_idx]
        train_y = [target[idx] for idx in train_idx]
        subset_rows = exhaustive_subset_search(train_rows, train_y, features, max_size=max_size)
        chosen = subset_rows[0]["features"].split("+")
        selected.update(chosen)
        pred = ridge_train_predict(train_rows, train_y, [rows[held_out]], chosen)[0]
        preds.append(pred)
    return preds, selected


def bootstrap_ci(preds: list[float], target: list[float], metric_fn, seed: int = 2026, repeats: int = 2000) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    values = []
    n = len(target)
    for _ in range(repeats):
        idx = rng.integers(0, n, size=n)
        pred_sample = [preds[int(i)] for i in idx]
        target_sample = [target[int(i)] for i in idx]
        value = metric_fn(pred_sample, target_sample)
        if math.isfinite(value):
            values.append(value)
    if not values:
        return math.nan, math.nan
    return float(np.percentile(values, 5)), float(np.percentile(values, 95))


def candidate_row_with_ci(
    name: str,
    preds: list[float],
    target: list[float],
    cost_tier: str,
    ingredients: str,
    notes: str = "",
) -> dict[str, str]:
    row = evaluate(name, preds, target, cost_tier, ingredients, notes)
    p_low, p_high = bootstrap_ci(preds, target, pearson)
    s_low, s_high = bootstrap_ci(preds, target, spearman)
    row["pearson_ci90"] = f"[{p_low:.3f},{p_high:.3f}]"
    row["spearman_ci90"] = f"[{s_low:.3f},{s_high:.3f}]"
    return row


def build_candidates(rows: list[dict[str, float]], target: list[float]) -> tuple[list[dict[str, str]], dict[str, list[float]], Counter[str]]:
    predictions: dict[str, list[float]] = {}
    candidates: list[dict[str, str]] = []

    hand = {
        "target_style_fit_v1": ("low_dsp", "emotion target + f0/rate/energy/pause"),
        "delivery_fit_v1": ("low_dsp", "target_style + prosody + pause + spectrum + articulation risk"),
        "pause_naturalness": ("low_dsp", "silence ratio + pause count + active ratio"),
        "spectral_balance_fit": ("low_dsp", "centroid + high-frequency balance"),
        "voice_presence_fit": ("low_dsp", "mid-band energy + voiced/active ratio"),
        "articulation_risk_inverse": ("very_low", "text difficulty + rate + zcr/high-frequency risk"),
        "text_ease": ("very_low", "text difficulty only"),
        "rate_fit": ("very_low", "text + duration speech-rate fit"),
        "prosody_fit_light": ("low_dsp", "existing f0 + energy proxy"),
        "signal_quality": ("low_dsp", "existing signal-quality proxy"),
        "emotion_component_0_1": ("medium_neural", "SER target-emotion component only"),
        "intelligibility_component_0_1": ("high_reference", "ASR/WER-derived component only"),
    }
    for name, (tier, ingredients) in hand.items():
        preds = [row[name] for row in rows]
        predictions[name] = preds
        candidates.append(candidate_row_with_ci(name, preds, target, tier, ingredients))

    fixed_candidates = {
        "fixed_delivery_surrogate_v1": (
            "low_dsp",
            [
                ("target_style_fit_v1", 0.36),
                ("prosody_fit_light", 0.22),
                ("pause_naturalness", 0.16),
                ("spectral_balance_fit", 0.14),
                ("articulation_risk_inverse", 0.12),
            ],
            "hand-weighted low-DSP delivery metric",
        ),
        "fixed_main_shape_proxy_v1": (
            "low_dsp",
            [
                ("delivery_fit_v1", 0.38),
                ("text_ease", 0.18),
                ("rate_fit", 0.16),
                ("voice_presence_fit", 0.14),
                ("signal_quality", 0.14),
            ],
            "hand-weighted main-shape approximation without ASR/SER",
        ),
    }
    for name, (tier, weighted_features, notes) in fixed_candidates.items():
        preds = [clamp(sum(row[feature] * weight for feature, weight in weighted_features)) for row in rows]
        predictions[name] = preds
        ingredients = "+".join(f"{weight:.2f}*{feature}" for feature, weight in weighted_features)
        candidates.append(candidate_row_with_ci(name, preds, target, tier, ingredients, notes))

    groups = {
        "ridge_delivery_low_dsp_loo": (
            "low_dsp",
            ["target_style_fit_v1", "prosody_fit_light", "pause_naturalness", "spectral_balance_fit", "articulation_risk_inverse"],
        ),
        "ridge_emotion_dsp_text_loo": (
            "low_dsp",
            ["target_style_fit_v1", "emotion_arousal_fit_v1", "prosody_fit_light", "rate_fit", "text_ease"],
        ),
        "ridge_new_shape_low_dsp_loo": ("low_dsp", NEW_LOW_FEATURES),
        "ridge_expanded_low_dsp_loo": ("low_dsp", BASE_LOW_FEATURES + NEW_LOW_FEATURES),
        "ridge_ser_delivery_loo": ("medium_neural", MEDIUM_FEATURES),
        "ridge_reference_components_loo": ("high_reference", HIGH_REFERENCE_FEATURES),
    }
    for name, (tier, features) in groups.items():
        preds = ridge_predictions(rows, target, features)
        predictions[name] = preds
        candidates.append(candidate_row_with_ci(name, preds, target, tier, "+".join(features), "LOOCV ridge"))

    for name, features in {
        "lodo_delivery_low_dsp": ["target_style_fit_v1", "prosody_fit_light", "pause_naturalness", "spectral_balance_fit", "articulation_risk_inverse"],
        "lodo_emotion_dsp_text": ["target_style_fit_v1", "emotion_arousal_fit_v1", "prosody_fit_light", "rate_fit", "text_ease"],
        "lodo_ser_delivery": MEDIUM_FEATURES,
    }.items():
        tier = "medium_neural" if "ser" in name else "low_dsp"
        preds = ridge_leave_dataset_out(rows, target, features, alpha=5e-2)
        predictions[name] = preds
        candidates.append(candidate_row_with_ci(name, preds, target, tier, "+".join(features), "leave-one-dataset-out"))

    nested_preds, selected_counter = nested_subset_loo(rows, target, NESTED_SUBSET_FEATURES, max_size=3)
    predictions["nested_subset_low_dsp_loo"] = nested_preds
    candidates.append(
        candidate_row_with_ci(
            "nested_subset_low_dsp_loo",
            nested_preds,
            target,
            "low_dsp",
            "nested selected subset from low-DSP feature pool",
            "nested LOOCV; less optimistic than global subset search",
        )
    )

    candidates.sort(key=lambda row: (float(row["spearman"]), float(row["pearson"])), reverse=True)
    return candidates, predictions, selected_counter


def resource_rows(feature_elapsed: float) -> list[dict[str, str]]:
    return [
        {
            "tier": "very_low",
            "uses": "text, manifest duration, precomputed generation metadata",
            "extra_models": "none",
            "expected_cost": "negligible; milliseconds per clip",
            "current_run_observation": "included inside Python table pass",
        },
        {
            "tier": "low_dsp",
            "uses": "wav read, frame RMS, FFT, coarse f0/prosody features",
            "extra_models": "none",
            "expected_cost": "CPU seconds for tens/hundreds of clips; scales linearly with audio length",
            "current_run_observation": f"enhanced waveform feature extraction for 26 clips: {feature_elapsed:.3f}s",
        },
        {
            "tier": "medium_neural",
            "uses": "emotion classifier or speaker/style embedding such as SIM",
            "extra_models": "one neural audio encoder",
            "expected_cost": "much cheaper than full objective if batched, but requires model load/GPU or slower CPU",
            "current_run_observation": "not rerun here; reused existing SER columns from main metric outputs",
        },
        {
            "tier": "high_reference",
            "uses": "ASR WER plus SER and quality/sanity components",
            "extra_models": "ASR plus SER/proxy models",
            "expected_cost": "baseline/main metric cost; not suitable as surrogate except sanity upper bound",
            "current_run_observation": "not rerun here; reused existing main metric columns",
        },
    ]


def feature_snapshot(rows: list[dict[str, float]]) -> list[dict[str, str]]:
    keys = BASE_LOW_FEATURES + NEW_LOW_FEATURES + ["emotion_component_0_1", "intelligibility_component_0_1"]
    out = []
    for key in keys:
        values = [float(row[key]) for row in rows]
        out.append(
            {
                "feature": key,
                "mean": f"{mean(values):.6f}",
                "min": f"{min(values):.6f}",
                "max": f"{max(values):.6f}",
            }
        )
    return out


def selection_rows(counter: Counter[str], folds: int) -> list[dict[str, str]]:
    return [
        {"feature": feature, "selected_folds": str(count), "selected_fraction": f"{count / max(folds, 1):.6f}"}
        for feature, count in counter.most_common()
    ]


def report_lines(
    candidates: list[dict[str, str]],
    subset_rows: list[dict[str, str]],
    feature_elapsed: float,
    sample_count: int,
) -> list[str]:
    top = candidates[:14]
    best_low = next(row for row in candidates if row["cost_tier"] == "low_dsp")
    best_very_low = next(row for row in candidates if row["cost_tier"] == "very_low")
    best_medium = next(row for row in candidates if row["cost_tier"] == "medium_neural")
    best_lodo_low = next(row for row in candidates if row["candidate"].startswith("lodo_") and row["cost_tier"] == "low_dsp")

    lines = [
        "# Surrogate Metric Exploration V3",
        "",
        f"Samples: {sample_count}. Target: current `main_metric_0_1`.",
        "",
        "## What Was Added",
        "",
        "- Target-emotion DSP style fit: compares rate, loudness, f0 range, energy variation, pause rate, and activity against simple happy/sad/angry/neutral acoustic profiles.",
        "- Pause/envelope/spectral features: pause rate, active ratio, dynamic range, envelope jitter, rolloff, bandwidth, low/mid/high band ratios.",
        "- More conservative validation: LOOCV, leave-one-dataset-out, and nested subset LOOCV.",
        "",
        "## Top Candidates",
        "",
        "| candidate | tier | Pearson | 90% CI | Spearman | 90% CI | Kendall | pairwise acc. | MAE | top5 | bottom5 |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in top:
        lines.append(
            f"| {row['candidate']} | {row['cost_tier']} | {row['pearson']} | {row['pearson_ci90']} | {row['spearman']} | {row['spearman_ci90']} | {row['kendall_tau_b']} | {row['pairwise_accuracy']} | {row['mae']} | {row['top5_overlap']} | {row['bottom5_overlap']} |"
        )

    lines.extend(
        [
            "",
            "## Current Readout",
            "",
            f"- Best very-low-cost option: `{best_very_low['candidate']}` with Pearson `{best_very_low['pearson']}`, Spearman `{best_very_low['spearman']}`, MAE `{best_very_low['mae']}`.",
            f"- Best low-DSP option by ranking: `{best_low['candidate']}` with Pearson `{best_low['pearson']}`, Spearman `{best_low['spearman']}`, MAE `{best_low['mae']}`.",
            f"- Best low-DSP leave-dataset-out check: `{best_lodo_low['candidate']}` with Pearson `{best_lodo_low['pearson']}`, Spearman `{best_lodo_low['spearman']}`, MAE `{best_lodo_low['mae']}`.",
            f"- Best medium-neural option: `{best_medium['candidate']}` with Pearson `{best_medium['pearson']}`, Spearman `{best_medium['spearman']}`, MAE `{best_medium['mae']}`.",
            f"- Enhanced low-DSP waveform feature extraction took `{feature_elapsed:.3f}s` for `{sample_count}` clips in this run.",
            "",
            "## Best Low-DSP Subsets",
            "",
            "| features | size | Pearson | Spearman | MAE | bottom5 |",
            "| --- | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for row in subset_rows[:12]:
        lines.append(
            f"| `{row['features']}` | {row['size']} | {row['pearson']} | {row['spearman']} | {row['mae']} | {row['bottom5_overlap']} |"
        )

    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "- The best cheap combinations remain useful as a local ranking filter, but they are not strong enough to replace the current main metric.",
            "- The new target-emotion DSP features help expose style misses, especially sad/happy/angry delivery mismatches, but they still cannot see lexical intelligibility failures as reliably as ASR.",
            "- Medium-neural SER/SIM-like signals are the most promising surrogate family if the goal is to approach the high correlation Yufan reported for SIM versus WER.",
            "- Pure text and duration features are too weak for this task. They can flag difficult prompts, but they do not know whether the generated audio actually pronounced the text or conveyed emotion.",
            "- Leave-dataset-out results are the caution sign: the sample set is still too small and too Parler-specific for a final surrogate claim.",
            "- Both datasets use the same TTS system and speaker. Leave-dataset-out is therefore a boundary-set shift check, not held-out-system validation.",
            "",
            "## Local Outputs",
            "",
            "- `outputs_v3/surrogate_candidates_v3.csv`",
            "- `outputs_v3/subset_search_top30_v3.csv`",
            "- `outputs_v3/nested_subset_selection_counts.csv`",
            "- `outputs_v3/surrogate_error_analysis_v3.csv`",
            "- `outputs_v3/resource_estimate.csv`",
            "- `outputs_v3/feature_snapshot.csv`",
        ]
    )
    return lines


def main() -> None:
    rows, feature_elapsed = load_rows_with_audio_features()
    target = [row["main_metric_0_1"] for row in rows]
    candidates, predictions, selected_counter = build_candidates(rows, target)
    subset_rows = exhaustive_subset_search(rows, target, SUBSET_SEARCH_FEATURES, max_size=4)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    write_csv(OUTPUT_DIR / "surrogate_candidates_v3.csv", candidates)
    write_csv(OUTPUT_DIR / "subset_search_top30_v3.csv", subset_rows[:30])
    write_csv(OUTPUT_DIR / "nested_subset_selection_counts.csv", selection_rows(selected_counter, len(rows)))
    write_csv(OUTPUT_DIR / "resource_estimate.csv", resource_rows(feature_elapsed))
    write_csv(OUTPUT_DIR / "feature_snapshot.csv", feature_snapshot(rows))
    write_csv(
        OUTPUT_DIR / "surrogate_error_analysis_v3.csv",
        per_sample_errors(
            rows,
            target,
            {
                "target_style_fit_v1": predictions["target_style_fit_v1"],
                "delivery_fit_v1": predictions["delivery_fit_v1"],
                "ridge_delivery_low_dsp_loo": predictions["ridge_delivery_low_dsp_loo"],
                "nested_subset_low_dsp_loo": predictions["nested_subset_low_dsp_loo"],
                "ridge_ser_delivery_loo": predictions["ridge_ser_delivery_loo"],
            },
        ),
    )
    (OUTPUT_DIR / "surrogate_report_v3.md").write_text(
        "\n".join(report_lines(candidates, subset_rows, feature_elapsed, len(rows))) + "\n",
        encoding="utf-8",
    )

    print(f"wrote {OUTPUT_DIR / 'surrogate_candidates_v3.csv'}")
    print(f"wrote {OUTPUT_DIR / 'subset_search_top30_v3.csv'}")
    print(f"wrote {OUTPUT_DIR / 'nested_subset_selection_counts.csv'}")
    print(f"wrote {OUTPUT_DIR / 'surrogate_error_analysis_v3.csv'}")
    print(f"wrote {OUTPUT_DIR / 'surrogate_report_v3.md'}")


if __name__ == "__main__":
    main()
