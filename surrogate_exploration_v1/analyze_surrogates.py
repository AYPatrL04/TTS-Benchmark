from __future__ import annotations

import csv
import json
import math
import re
from pathlib import Path
from typing import Iterable


PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = PROJECT_ROOT / "surrogate_exploration_v1" / "outputs"

DATASETS = [
    (
        "parler_emotion_v1",
        PROJECT_ROOT / "experiments" / "parler_emotion_v1" / "combined" / "parler_emotion_scored_main_metric.csv",
    ),
    (
        "boundary_metric_v1",
        PROJECT_ROOT / "experiments" / "boundary_metric_v1" / "combined" / "boundary_scored_main_metric.csv",
    ),
]

NUMBER_WORDS = {
    "zero",
    "one",
    "two",
    "three",
    "four",
    "five",
    "six",
    "seven",
    "eight",
    "nine",
    "ten",
    "eleven",
    "twelve",
    "thirteen",
    "fourteen",
    "fifteen",
    "sixteen",
    "seventeen",
    "eighteen",
    "nineteen",
    "twenty",
    "thirty",
    "forty",
    "fifty",
}

PROSODY_TARGETS = {
    "happy": (0.85, 0.35),
    "angry": (0.90, 0.35),
    "sad": (0.60, 0.40),
    "neutral": (0.80, 0.40),
}

RATE_TARGETS = {
    "happy": (2.75, 1.25),
    "angry": (2.95, 1.35),
    "sad": (2.05, 1.20),
    "neutral": (2.55, 1.20),
}


def clamp(value: float, low: float = 0.0, high: float = 1.0) -> float:
    if not math.isfinite(value):
        return low
    return max(low, min(high, value))


def parse_float(row: dict[str, str], key: str, default: float = math.nan) -> float:
    try:
        return float(row.get(key, ""))
    except ValueError:
        return default


def words(text: str) -> list[str]:
    return re.findall(r"[A-Za-z0-9]+(?:'[A-Za-z]+)?", text.lower())


def read_rows() -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for dataset_name, csv_path in DATASETS:
        with csv_path.open(newline="", encoding="utf-8-sig") as handle:
            for row in csv.DictReader(handle):
                item = dict(row)
                item["dataset"] = dataset_name
                rows.append(item)
    return rows


def resolve_audio(path_text: str) -> Path:
    path = Path(path_text)
    if path.is_absolute():
        return path
    return PROJECT_ROOT / path


def load_audio_16k(path: Path):
    import numpy as np
    from scipy.io import wavfile
    from scipy.signal import resample_poly

    sample_rate, data = wavfile.read(path)
    audio = data.astype("float32")
    if audio.ndim == 2:
        audio = audio.mean(axis=1)
    if data.dtype.kind in {"i", "u"}:
        audio = audio / float(np.iinfo(data.dtype).max)
    else:
        audio = np.clip(audio, -1.0, 1.0)
    if sample_rate != 16_000:
        gcd = math.gcd(sample_rate, 16_000)
        audio = resample_poly(audio, 16_000 // gcd, sample_rate // gcd).astype("float32")
    return audio


def frame_audio(audio, sample_rate: int = 16_000, frame_ms: float = 30.0, hop_ms: float = 10.0):
    import numpy as np

    frame_size = max(1, int(sample_rate * frame_ms / 1000.0))
    hop = max(1, int(sample_rate * hop_ms / 1000.0))
    frames = []
    for start in range(0, max(1, len(audio) - frame_size + 1), hop):
        frame = audio[start : start + frame_size]
        if len(frame) == frame_size:
            frames.append(frame)
    return frames


def rms_values(audio):
    import numpy as np

    return np.asarray([float(np.sqrt(np.mean(np.square(frame)))) for frame in frame_audio(audio)], dtype="float32")


def spectral_features(audio):
    import numpy as np

    sample_rate = 16_000
    frame_size = int(sample_rate * 0.04)
    hop = int(sample_rate * 0.02)
    window = np.hanning(frame_size).astype("float32")
    flatness_values = []
    centroid_values = []
    high_ratio_values = []
    freqs = np.fft.rfftfreq(frame_size, 1.0 / sample_rate)
    for start in range(0, max(1, len(audio) - frame_size + 1), hop):
        frame = audio[start : start + frame_size]
        if len(frame) != frame_size:
            continue
        power = np.square(np.abs(np.fft.rfft(frame * window))) + 1e-12
        total = float(np.sum(power))
        flatness_values.append(float(np.exp(np.mean(np.log(power))) / max(np.mean(power), 1e-12)))
        centroid_values.append(float(np.sum(freqs * power) / max(total, 1e-12)))
        high_ratio_values.append(float(np.sum(power[freqs >= 4_000]) / max(total, 1e-12)))
    return {
        "spectral_flatness": mean(flatness_values),
        "spectral_centroid_hz": mean(centroid_values),
        "high_freq_ratio": mean(high_ratio_values),
    }


def estimate_f0_autocorr(audio, sample_rate: int = 16_000) -> list[float]:
    import numpy as np

    frame_size = int(sample_rate * 0.04)
    hop = int(sample_rate * 0.01)
    min_lag = int(sample_rate / 400.0)
    max_lag = int(sample_rate / 70.0)
    values: list[float] = []
    rms = rms_values(audio)
    if rms.size == 0:
        return values
    voiced_threshold = max(1e-4, float(np.percentile(rms, 65)) * 0.35)
    for start in range(0, max(1, len(audio) - frame_size + 1), hop):
        frame = audio[start : start + frame_size]
        if len(frame) != frame_size:
            continue
        frame_energy = float(np.sqrt(np.mean(np.square(frame))))
        if frame_energy < voiced_threshold:
            continue
        frame = frame - float(np.mean(frame))
        corr = np.correlate(frame, frame, mode="full")[frame_size - 1 :]
        if corr[0] <= 1e-8:
            continue
        search = corr[min_lag:max_lag]
        if search.size == 0:
            continue
        lag = int(np.argmax(search) + min_lag)
        clarity = float(corr[lag] / corr[0])
        if clarity > 0.25:
            values.append(float(sample_rate / lag))
    return values


def mean(values: Iterable[float]) -> float:
    finite = [value for value in values if math.isfinite(value)]
    return sum(finite) / len(finite) if finite else math.nan


def text_difficulty(text: str) -> dict[str, float]:
    token_list = words(text)
    word_count = max(1, len(token_list))
    acronym_count = len(re.findall(r"\b[A-Z]{2,}\b", text))
    number_word_count = sum(1 for token in token_list if token in NUMBER_WORDS or token.isdigit())
    repeated_adjacent = sum(1 for a, b in zip(token_list, token_list[1:]) if a == b)
    repeated_ratio = repeated_adjacent / word_count
    sibilants = sum(text.lower().count(unit) for unit in ["s", "sh", "ch", "x", "z"])
    sibilant_density = sibilants / max(1, len(text))
    initials = [token[0] for token in token_list if token]
    max_initial_ratio = max((initials.count(initial) for initial in set(initials)), default=0) / word_count
    punctuation_ratio = sum(1 for char in text if char in ",;:!?") / max(1, len(text))
    difficulty = clamp(
        0.10 * acronym_count
        + 0.035 * number_word_count
        + 0.45 * repeated_ratio
        + 1.40 * max(0.0, sibilant_density - 0.12)
        + 0.25 * max(0.0, max_initial_ratio - 0.35)
        + 1.00 * max(0.0, punctuation_ratio - 0.06)
    )
    return {
        "word_count": float(word_count),
        "acronym_count": float(acronym_count),
        "number_word_count": float(number_word_count),
        "repeated_ratio": repeated_ratio,
        "sibilant_density": sibilant_density,
        "max_initial_ratio": max_initial_ratio,
        "text_difficulty": difficulty,
        "text_ease": 1.0 - difficulty,
    }


def acoustic_features(row: dict[str, str]) -> dict[str, float]:
    import numpy as np

    audio = load_audio_16k(resolve_audio(row["audio_path"]))
    duration = len(audio) / 16_000.0
    rms = float(np.sqrt(np.mean(np.square(audio)))) if len(audio) else math.nan
    rms_dbfs = 20.0 * math.log10(max(rms, 1e-8))
    peak_abs = float(np.max(np.abs(audio))) if len(audio) else math.nan
    clipping_ratio = float(np.mean(np.abs(audio) >= 0.98)) if len(audio) else math.nan
    frame_rms = rms_values(audio)
    silence_threshold = max(1e-4, rms * 0.10) if math.isfinite(rms) else 1e-4
    silence_ratio = float(np.mean(frame_rms < silence_threshold)) if frame_rms.size else math.nan
    rms_mean = float(np.mean(frame_rms)) if frame_rms.size else math.nan
    energy_cv = float(np.std(frame_rms) / max(rms_mean, 1e-8)) if math.isfinite(rms_mean) else math.nan
    f0_values = estimate_f0_autocorr(audio)
    f0_std = float(np.std(f0_values)) if f0_values else 0.0
    f0_range = float(np.max(f0_values) - np.min(f0_values)) if f0_values else 0.0
    voiced_ratio = float(len(f0_values) / len(frame_rms)) if frame_rms.size else math.nan
    zcr_values = []
    for frame in frame_audio(audio):
        if len(frame) > 1:
            zcr_values.append(float(np.mean(np.abs(np.diff(np.signbit(frame))))))
    spec = spectral_features(audio)

    loudness_penalty = 0.0
    if rms_dbfs < -32.0:
        loudness_penalty = clamp((-32.0 - rms_dbfs) / 18.0)
    elif rms_dbfs > -8.0:
        loudness_penalty = clamp((rms_dbfs + 8.0) / 8.0)
    silence_penalty = 0.0
    if math.isfinite(silence_ratio):
        if silence_ratio > 0.45:
            silence_penalty = clamp((silence_ratio - 0.45) / 0.35)
        elif silence_ratio < 0.02:
            silence_penalty = clamp((0.02 - silence_ratio) / 0.02)
    clipping_penalty = clamp(clipping_ratio * 100.0) if math.isfinite(clipping_ratio) else 0.0
    flatness_penalty = clamp((spec["spectral_flatness"] - 0.35) / 0.40) if spec["spectral_flatness"] > 0.35 else 0.0
    duration_penalty = clamp((1.5 - duration) / 1.5) if duration < 1.5 else 0.0
    signal_quality = 1.0 - clamp(
        0.30 * loudness_penalty
        + 0.30 * silence_penalty
        + 0.20 * clipping_penalty
        + 0.15 * flatness_penalty
        + 0.05 * duration_penalty
    )

    target_emotion = (row.get("target_emotion") or "neutral").strip().lower()
    word_count = text_difficulty(row.get("text", ""))["word_count"]
    speech_rate_wps = word_count / max(duration, 1e-8)
    rate_target, rate_tolerance = RATE_TARGETS.get(target_emotion, RATE_TARGETS["neutral"])
    rate_fit = clamp(1.0 - abs(speech_rate_wps - rate_target) / rate_tolerance)
    prosody_activity = 0.5 * min(1.0, f0_std / 45.0) + 0.5 * min(1.0, energy_cv / 0.9)
    prosody_target, prosody_tolerance = PROSODY_TARGETS.get(target_emotion, PROSODY_TARGETS["neutral"])
    prosody_fit = clamp(1.0 - abs(prosody_activity - prosody_target) / prosody_tolerance)

    return {
        "duration_sec": duration,
        "speech_rate_wps": speech_rate_wps,
        "rms_dbfs": rms_dbfs,
        "peak_abs": peak_abs,
        "clipping_ratio": clipping_ratio,
        "silence_ratio": silence_ratio,
        "energy_cv": energy_cv,
        "f0_std_hz": f0_std,
        "f0_range_hz": f0_range,
        "voiced_ratio": voiced_ratio,
        "zcr": mean(zcr_values),
        "prosody_activity_light": prosody_activity,
        "prosody_fit_light": prosody_fit,
        "rate_fit": rate_fit,
        "signal_quality": signal_quality,
        **spec,
    }


def rankdata(values: list[float]) -> list[float]:
    indexed = sorted(enumerate(values), key=lambda item: item[1])
    ranks = [0.0] * len(values)
    i = 0
    while i < len(indexed):
        j = i
        while j + 1 < len(indexed) and indexed[j + 1][1] == indexed[i][1]:
            j += 1
        rank = (i + j + 2) / 2.0
        for k in range(i, j + 1):
            ranks[indexed[k][0]] = rank
        i = j + 1
    return ranks


def pearson(x: list[float], y: list[float]) -> float:
    import numpy as np

    if len(x) < 2:
        return math.nan
    x_arr = np.asarray(x, dtype="float64")
    y_arr = np.asarray(y, dtype="float64")
    if float(np.std(x_arr)) == 0.0 or float(np.std(y_arr)) == 0.0:
        return math.nan
    return float(np.corrcoef(x_arr, y_arr)[0, 1])


def spearman(x: list[float], y: list[float]) -> float:
    return pearson(rankdata(x), rankdata(y))


def topk_overlap(candidate: list[float], target: list[float], k: int, largest: bool = True) -> float:
    order_candidate = sorted(range(len(candidate)), key=lambda i: candidate[i], reverse=largest)[:k]
    order_target = sorted(range(len(target)), key=lambda i: target[i], reverse=largest)[:k]
    return len(set(order_candidate) & set(order_target)) / float(k)


def ridge_fit_predict(x, y, alpha: float = 1e-3):
    import numpy as np

    x_arr = np.asarray(x, dtype="float64")
    y_arr = np.asarray(y, dtype="float64")
    mean_x = x_arr.mean(axis=0)
    std_x = x_arr.std(axis=0)
    std_x[std_x == 0] = 1.0
    x_std = (x_arr - mean_x) / std_x
    design = np.column_stack([np.ones(len(x_std)), x_std])
    penalty = np.eye(design.shape[1]) * alpha
    penalty[0, 0] = 0.0
    coef = np.linalg.solve(design.T @ design + penalty, design.T @ y_arr)
    pred = design @ coef
    return pred, coef, mean_x, std_x


def ridge_loo_predict(feature_rows: list[dict[str, float]], y: list[float], names: list[str], alpha: float = 1e-2) -> list[float]:
    import numpy as np

    x = np.asarray([[row[name] for name in names] for row in feature_rows], dtype="float64")
    y_arr = np.asarray(y, dtype="float64")
    preds = []
    for held_out in range(len(y)):
        train = [idx for idx in range(len(y)) if idx != held_out]
        pred_train, coef, mean_x, std_x = ridge_fit_predict(x[train], y_arr[train], alpha=alpha)
        held_x = (x[held_out] - mean_x) / std_x
        pred = coef[0] + held_x @ coef[1:]
        preds.append(float(clamp(pred)))
    return preds


def candidate_scores(feature_rows: list[dict[str, float]]) -> dict[str, list[float]]:
    candidates: dict[str, list[float]] = {}
    for name in [
        "signal_quality",
        "prosody_fit_light",
        "rate_fit",
        "text_ease",
    ]:
        candidates[name] = [row[name] for row in feature_rows]
    candidates["signal_rate_text"] = [
        clamp(0.40 * row["signal_quality"] + 0.35 * row["rate_fit"] + 0.25 * row["text_ease"])
        for row in feature_rows
    ]
    candidates["prosody_rate_text"] = [
        clamp(0.45 * row["prosody_fit_light"] + 0.30 * row["rate_fit"] + 0.25 * row["text_ease"])
        for row in feature_rows
    ]
    candidates["cheap_composite_v0"] = [
        clamp(
            0.30 * row["signal_quality"]
            + 0.30 * row["prosody_fit_light"]
            + 0.25 * row["rate_fit"]
            + 0.15 * row["text_ease"]
        )
        for row in feature_rows
    ]
    return candidates


def evaluate_candidates(candidates: dict[str, list[float]], target: list[float]) -> list[dict[str, str]]:
    rows = []
    for name, scores in candidates.items():
        mae = mean(abs(score - label) for score, label in zip(scores, target))
        rows.append(
            {
                "candidate": name,
                "pearson": f"{pearson(scores, target):.6f}",
                "spearman": f"{spearman(scores, target):.6f}",
                "mae": f"{mae:.6f}",
                "top3_overlap": f"{topk_overlap(scores, target, min(3, len(target))):.6f}",
                "top5_overlap": f"{topk_overlap(scores, target, min(5, len(target))):.6f}",
                "bottom5_overlap": f"{topk_overlap(scores, target, min(5, len(target)), largest=False):.6f}",
                "cost": "low",
            }
        )
    rows.sort(key=lambda row: (float(row["spearman"]), float(row["pearson"])), reverse=True)
    return rows


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    rows = read_rows()
    feature_rows = []
    for row in rows:
        features = {
            "dataset": row["dataset"],
            "id": row["id"],
            "case_type": row.get("case_type", ""),
            "target_emotion": row.get("target_emotion", ""),
            "main_metric_0_1": parse_float(row, "main_metric_0_1"),
        }
        features.update(text_difficulty(row.get("text", "")))
        features.update(acoustic_features(row))
        feature_rows.append(features)

    target = [row["main_metric_0_1"] for row in feature_rows]
    candidates = candidate_scores(feature_rows)
    feature_set_4 = ["signal_quality", "prosody_fit_light", "rate_fit", "text_ease"]
    x4 = [[row[name] for name in feature_set_4] for row in feature_rows]
    ridge4_pred, ridge4_coef, ridge4_mean, ridge4_std = ridge_fit_predict(x4, target)
    candidates["ridge4_in_sample"] = [
        clamp(value) for value in ridge4_pred
    ]
    candidates["ridge4_leave_one_out"] = ridge_loo_predict(feature_rows, target, feature_set_4)
    feature_set_7 = [
        "signal_quality",
        "prosody_fit_light",
        "rate_fit",
        "text_ease",
        "silence_ratio",
        "energy_cv",
        "spectral_flatness",
    ]
    candidates["ridge7_leave_one_out"] = ridge_loo_predict(feature_rows, target, feature_set_7)

    candidate_rows = evaluate_candidates(candidates, target)
    feature_output = []
    for idx, row in enumerate(feature_rows):
        item = dict(row)
        for name, scores in candidates.items():
            item[name] = scores[idx]
        feature_output.append(item)

    write_csv(OUTPUT_DIR / "surrogate_features.csv", feature_output)
    write_csv(OUTPUT_DIR / "surrogate_candidates.csv", candidate_rows)
    ridge4_model = {
        "name": "ridge4_surrogate",
        "target": "main_metric_0_1",
        "features": feature_set_4,
        "alpha": 1e-3,
        "standardization": {
            name: {"mean": float(mean), "std": float(std)}
            for name, mean, std in zip(feature_set_4, ridge4_mean, ridge4_std)
        },
        "intercept": float(ridge4_coef[0]),
        "coefficients_standardized": {
            name: float(coef) for name, coef in zip(feature_set_4, ridge4_coef[1:])
        },
        "prediction_formula": "clip(intercept + sum(coef_i * ((feature_i - mean_i) / std_i)), 0, 1)",
    }
    (OUTPUT_DIR / "ridge4_surrogate_model.json").write_text(
        json.dumps(ridge4_model, indent=2) + "\n",
        encoding="utf-8",
    )

    best = candidate_rows[0] if candidate_rows else {}
    lines = [
        "# Surrogate Metric Exploration V1",
        "",
        "This local-only exploration tests low-compute surrogate candidates against the current `main_metric_0_1` reference.",
        "",
        "Slack reference from Yufan: WER and SIM were reported as highly correlated for the TTS autoresearch artifact, with SIM around `0.930` correlation while requiring less computation. This run treats that as a design prior: prefer confidence-like or cheap signals that preserve ranking against the main metric.",
        "",
        "## Data",
        "",
        f"- samples: {len(feature_rows)}",
        "- sources: `parler_emotion_v1` and `boundary_metric_v1` scored outputs",
        "- target: `main_metric_0_1`",
        "- excluded from surrogate features: Whisper WER/CER and SER target emotion probability",
        "",
        "## Candidate Families",
        "",
        "| candidate | ingredients | rationale |",
        "| --- | --- | --- |",
        "| `signal_quality` | loudness, silence, clipping, spectral flatness, duration | cheapest audio usability guard |",
        "| `prosody_fit_light` | f0 variance and energy dynamics vs target emotion | SER-free style/prosody approximation |",
        "| `rate_fit` | text word count divided by audio duration vs target emotion | cheap intelligibility/style proxy |",
        "| `text_ease` | acronyms, number words, repetition, sibilant density, punctuation | prompt difficulty prior for WER-sensitive cases |",
        "| `cheap_composite_v0` | fixed weighted mix of the above | deployable first surrogate candidate |",
        "| `ridge*_leave_one_out` | tiny linear model over cheap features | tests whether weights can be learned instead of hand-set |",
        "",
        "## Results",
        "",
        "| candidate | Pearson | Spearman | MAE | top3 | top5 | bottom5 | cost |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |",
    ]
    for row in candidate_rows:
        lines.append(
            f"| {row['candidate']} | {row['pearson']} | {row['spearman']} | {row['mae']} | {row['top3_overlap']} | {row['top5_overlap']} | {row['bottom5_overlap']} | {row['cost']} |"
        )

    lines.extend(
        [
            "",
        "## Initial Readout",
        "",
        f"- Best ranked candidate in this run: `{best.get('candidate', 'n/a')}`.",
        "- Best practical fitted candidate: `ridge4_leave_one_out`, because it uses only four cheap features and avoids the over-optimism of the in-sample score.",
        "- The fitted deployable coefficients are saved in `outputs/ridge4_surrogate_model.json`.",
        "- This is a very small dataset, so in-sample fitted results should be treated as over-optimistic.",
            "- Leave-one-out results are more useful than in-sample results, but still fragile with only 26 samples.",
            "- A practical surrogate should be judged by Spearman/top-k agreement, not only Pearson correlation.",
            "",
            "## Files",
            "",
            "- `outputs/surrogate_features.csv`: per-sample cheap features and candidate scores.",
            "- `outputs/surrogate_candidates.csv`: correlation, ranking, and MAE summary.",
            "- `outputs/ridge4_surrogate_model.json`: tiny fitted surrogate formula.",
            "- `outputs/surrogate_report.md`: this report.",
            "",
            "## Next Step",
            "",
            "Expand the sample count and add a true SIM/codec-logit feature if the model exposes token probabilities. For the current waveform-only setting, start with a fixed cheap composite or a 4-feature ridge surrogate, then validate it on new boundary cases before using it inside autoresearch.",
        ]
    )
    (OUTPUT_DIR / "surrogate_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"wrote {OUTPUT_DIR / 'surrogate_features.csv'}")
    print(f"wrote {OUTPUT_DIR / 'surrogate_candidates.csv'}")
    print(f"wrote {OUTPUT_DIR / 'ridge4_surrogate_model.json'}")
    print(f"wrote {OUTPUT_DIR / 'surrogate_report.md'}")


if __name__ == "__main__":
    main()
