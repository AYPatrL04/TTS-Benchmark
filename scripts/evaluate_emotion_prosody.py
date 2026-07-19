from __future__ import annotations

import argparse
import csv
import math
import os
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate speech emotion classifier output and lightweight prosody.")
    parser.add_argument("--input", required=True, type=Path, help="CSV with audio_path/generated_audio column.")
    parser.add_argument("--output-csv", required=True, type=Path)
    parser.add_argument("--output-md", required=True, type=Path)
    parser.add_argument("--model", default="superb/wav2vec2-base-superb-er")
    parser.add_argument("--device", default="auto", choices=["auto", "cuda", "cpu"])
    parser.add_argument("--target-column", default="target_emotion")
    parser.add_argument("--max-rows", type=int)
    return parser.parse_args()


def setup_imports() -> dict[str, Any]:
    os.environ.setdefault("TRANSFORMERS_NO_TF", "1")
    os.environ.setdefault("USE_TF", "0")
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

    import numpy as np
    import torch
    from scipy.io import wavfile
    from scipy.signal import resample_poly
    from transformers import AutoFeatureExtractor, AutoModelForAudioClassification

    return {
        "np": np,
        "torch": torch,
        "wavfile": wavfile,
        "resample_poly": resample_poly,
        "AutoFeatureExtractor": AutoFeatureExtractor,
        "AutoModelForAudioClassification": AutoModelForAudioClassification,
    }


def resolve_audio_path(path_text: str, csv_path: Path) -> Path:
    path = Path(path_text)
    if path.is_absolute():
        return path
    candidates = [
        csv_path.resolve().parent / path,
        csv_path.resolve().parent.parent / path,
        PROJECT_ROOT / path,
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return candidates[-1]


def read_wav_mono_16k(modules: dict[str, Any], path: Path) -> Any:
    np = modules["np"]
    wavfile = modules["wavfile"]
    resample_poly = modules["resample_poly"]
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


def frame_rms(modules: dict[str, Any], audio: Any, sample_rate: int = 16_000, frame_ms: float = 30.0, hop_ms: float = 10.0) -> Any:
    np = modules["np"]
    frame_size = max(1, int(sample_rate * frame_ms / 1000.0))
    hop = max(1, int(sample_rate * hop_ms / 1000.0))
    values = []
    for start in range(0, max(1, len(audio) - frame_size + 1), hop):
        frame = audio[start : start + frame_size]
        if len(frame) == frame_size:
            values.append(float(np.sqrt(np.mean(np.square(frame)))))
    return np.asarray(values, dtype="float32")


def estimate_f0_autocorr(modules: dict[str, Any], audio: Any, sample_rate: int = 16_000) -> list[float]:
    np = modules["np"]
    frame_size = int(sample_rate * 0.04)
    hop = int(sample_rate * 0.01)
    min_lag = int(sample_rate / 400.0)
    max_lag = int(sample_rate / 70.0)
    f0_values: list[float] = []
    rms_values = frame_rms(modules, audio, sample_rate=sample_rate, frame_ms=40.0, hop_ms=10.0)
    if rms_values.size == 0:
        return f0_values
    voiced_threshold = max(1e-4, float(np.percentile(rms_values, 65)) * 0.35)
    frame_index = 0
    for start in range(0, max(1, len(audio) - frame_size + 1), hop):
        frame = audio[start : start + frame_size]
        if len(frame) != frame_size:
            continue
        frame_energy = float(np.sqrt(np.mean(np.square(frame))))
        if frame_energy < voiced_threshold:
            frame_index += 1
            continue
        frame = frame - float(np.mean(frame))
        corr = np.correlate(frame, frame, mode="full")[frame_size - 1 :]
        if corr[0] <= 1e-8:
            frame_index += 1
            continue
        search = corr[min_lag:max_lag]
        if search.size == 0:
            frame_index += 1
            continue
        lag = int(np.argmax(search) + min_lag)
        clarity = float(corr[lag] / corr[0])
        if clarity > 0.25:
            f0_values.append(float(sample_rate / lag))
        frame_index += 1
    return f0_values


def prosody_features(modules: dict[str, Any], audio: Any) -> dict[str, str]:
    np = modules["np"]
    duration_sec = float(len(audio) / 16_000.0) if len(audio) else math.nan
    rms_values = frame_rms(modules, audio)
    rms_mean = float(np.mean(rms_values)) if rms_values.size else math.nan
    rms_std = float(np.std(rms_values)) if rms_values.size else math.nan
    energy_cv = float(rms_std / max(rms_mean, 1e-8)) if math.isfinite(rms_mean) else math.nan
    silence_threshold = max(1e-4, rms_mean * 0.10) if math.isfinite(rms_mean) else 1e-4
    silence_ratio = float(np.mean(rms_values < silence_threshold)) if rms_values.size else math.nan
    f0_values = estimate_f0_autocorr(modules, audio)
    f0_mean = float(np.mean(f0_values)) if f0_values else math.nan
    f0_std = float(np.std(f0_values)) if f0_values else math.nan
    f0_range = float(np.max(f0_values) - np.min(f0_values)) if f0_values else math.nan
    f0_median = float(np.median(f0_values)) if f0_values else math.nan
    semitone_values = (
        12.0 * np.log2(np.asarray(f0_values, dtype="float64") / max(f0_median, 1e-8))
        if f0_values
        else np.asarray([], dtype="float64")
    )
    f0_std_st = float(np.std(semitone_values)) if semitone_values.size else math.nan
    f0_mad_st = (
        float(np.median(np.abs(semitone_values - np.median(semitone_values))))
        if semitone_values.size
        else math.nan
    )
    f0_range_st = (
        float(np.percentile(semitone_values, 90) - np.percentile(semitone_values, 10))
        if semitone_values.size
        else math.nan
    )
    voiced_ratio = float(len(f0_values) / len(rms_values)) if rms_values.size else math.nan
    pitch_activity = 1.0 - math.exp(-max(f0_std_st if math.isfinite(f0_std_st) else 0.0, 0.0) / 3.0)
    energy_activity = 1.0 - math.exp(-max(energy_cv if math.isfinite(energy_cv) else 0.0, 0.0) / 0.65)
    prosody_activity = 0.5 * pitch_activity + 0.5 * energy_activity
    return {
        "style_duration_sec": f"{duration_sec:.6f}",
        "f0_mean_hz": f"{f0_mean:.6f}" if math.isfinite(f0_mean) else "",
        "f0_std_hz": f"{f0_std:.6f}" if math.isfinite(f0_std) else "",
        "f0_range_hz": f"{f0_range:.6f}" if math.isfinite(f0_range) else "",
        "f0_median_hz": f"{f0_median:.6f}" if math.isfinite(f0_median) else "",
        "f0_std_semitones": f"{f0_std_st:.6f}" if math.isfinite(f0_std_st) else "",
        "f0_mad_semitones": f"{f0_mad_st:.6f}" if math.isfinite(f0_mad_st) else "",
        "f0_range_semitones_p90_p10": f"{f0_range_st:.6f}" if math.isfinite(f0_range_st) else "",
        "voiced_ratio": f"{voiced_ratio:.6f}" if math.isfinite(voiced_ratio) else "",
        "energy_cv": f"{energy_cv:.6f}" if math.isfinite(energy_cv) else "",
        "style_silence_ratio": f"{silence_ratio:.6f}" if math.isfinite(silence_ratio) else "",
        "prosody_activity_0_1": f"{prosody_activity:.6f}",
    }


def normalize_label(label: str) -> str:
    normalized = label.strip().lower()
    mapping = {
        "hap": "happy",
        "happy": "happy",
        "happiness": "happy",
        "ang": "angry",
        "angry": "angry",
        "anger": "angry",
        "sad": "sad",
        "sadness": "sad",
        "neu": "neutral",
        "neutral": "neutral",
    }
    return mapping.get(normalized, normalized)


def main() -> None:
    args = parse_args()
    modules = setup_imports()
    np = modules["np"]
    torch = modules["torch"]
    AutoFeatureExtractor = modules["AutoFeatureExtractor"]
    AutoModelForAudioClassification = modules["AutoModelForAudioClassification"]

    with args.input.open(newline="", encoding="utf-8-sig") as handle:
        rows = list(csv.DictReader(handle))
    if args.max_rows is not None:
        rows = rows[: args.max_rows]

    device = "cuda" if args.device == "auto" and torch.cuda.is_available() else args.device
    if device == "auto":
        device = "cpu"

    print(f"loading emotion model {args.model} on {device}")
    feature_extractor = AutoFeatureExtractor.from_pretrained(args.model)
    model = AutoModelForAudioClassification.from_pretrained(args.model).to(device).eval()

    output_rows = []
    label_columns: list[str] = []
    for row in rows:
        audio_column = row.get("audio_path") or row.get("generated_audio")
        if not audio_column:
            raise ValueError(f"row {row.get('id', '')} has no audio_path/generated_audio")
        audio_path = resolve_audio_path(audio_column, args.input)
        audio = read_wav_mono_16k(modules, audio_path)
        inputs = feature_extractor(audio, sampling_rate=16_000, return_tensors="pt", padding=True)
        inputs = {key: value.to(device) for key, value in inputs.items()}
        with torch.no_grad():
            logits = model(**inputs).logits[0].float().cpu()
        probs = torch.softmax(logits, dim=-1).numpy()
        labels = {normalize_label(model.config.id2label[index]): float(prob) for index, prob in enumerate(probs)}
        top_label = max(labels, key=labels.get)
        top_prob = labels[top_label]
        entropy = -sum(prob * math.log(max(prob, 1e-12)) for prob in labels.values())
        target = normalize_label(row.get(args.target_column, ""))
        target_prob = labels.get(target, math.nan) if target else math.nan
        result = dict(row)
        result.update(
            {
                "emotion_model": args.model,
                "emotion_top_label": top_label,
                "emotion_top_prob": f"{top_prob:.6f}",
                "emotion_entropy": f"{entropy:.6f}",
                "target_emotion_normalized": target,
                "target_emotion_prob": f"{target_prob:.6f}" if math.isfinite(target_prob) else "",
                "target_emotion_match": "1" if target and top_label == target else ("0" if target else ""),
            }
        )
        for label, prob in labels.items():
            key = f"emotion_prob_{label}"
            result[key] = f"{prob:.6f}"
            if key not in label_columns:
                label_columns.append(key)
        result.update(prosody_features(modules, audio))
        output_rows.append(result)
        print(f"{row.get('id', audio_path.name)}: emotion={top_label} p={top_prob:.3f} prosody={result['prosody_activity_0_1']}")

    base_fields = list(rows[0].keys()) if rows else []
    extra_fields = [
        "emotion_model",
        "emotion_top_label",
        "emotion_top_prob",
        "emotion_entropy",
        "target_emotion_normalized",
        "target_emotion_prob",
        "target_emotion_match",
        *label_columns,
        "style_duration_sec",
        "f0_mean_hz",
        "f0_std_hz",
        "f0_range_hz",
        "f0_median_hz",
        "f0_std_semitones",
        "f0_mad_semitones",
        "f0_range_semitones_p90_p10",
        "voiced_ratio",
        "energy_cv",
        "style_silence_ratio",
        "prosody_activity_0_1",
    ]
    fieldnames = base_fields + [field for field in extra_fields if field not in base_fields]
    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    with args.output_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(output_rows)

    top_counts: dict[str, int] = {}
    for row in output_rows:
        top_counts[row["emotion_top_label"]] = top_counts.get(row["emotion_top_label"], 0) + 1
    mean_target_prob_values = [
        float(row["target_emotion_prob"]) for row in output_rows if row.get("target_emotion_prob")
    ]
    mean_prosody = sum(float(row["prosody_activity_0_1"]) for row in output_rows) / len(output_rows) if output_rows else math.nan
    mean_target_prob = (
        sum(mean_target_prob_values) / len(mean_target_prob_values) if mean_target_prob_values else math.nan
    )
    lines = [
        "# Emotion And Prosody Evaluation",
        "",
        f"- input: `{args.input}`",
        f"- model: `{args.model}`",
        f"- samples: {len(output_rows)}",
        f"- mean prosody activity: {mean_prosody:.6f}",
        f"- mean target emotion probability: {mean_target_prob:.6f}" if math.isfinite(mean_target_prob) else "- mean target emotion probability: n/a",
        f"- top-label counts: {top_counts}",
        "",
        "Emotion is an utterance-level classifier proxy. Prosody activity combines pitch variance and energy dynamics. These are automatic style proxies, not final human preference.",
        "",
        "| id | top_emotion | top_prob | target_prob | f0_std | energy_cv | prosody_activity |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in output_rows:
        lines.append(
            "| {id} | {emotion_top_label} | {emotion_top_prob} | {target_emotion_prob} | {f0_std_hz} | {energy_cv} | {prosody_activity_0_1} |".format(
                **row
            )
        )
    args.output_md.parent.mkdir(parents=True, exist_ok=True)
    args.output_md.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"wrote {args.output_csv}")
    print(f"wrote {args.output_md}")


if __name__ == "__main__":
    main()
