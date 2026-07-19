from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compute a lightweight no-reference acoustic naturalness proxy.")
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output-csv", required=True, type=Path)
    parser.add_argument("--output-md", required=True, type=Path)
    parser.add_argument("--max-rows", type=int)
    return parser.parse_args()


def setup_imports() -> dict[str, Any]:
    import numpy as np
    from scipy.io import wavfile
    from scipy.signal import resample_poly

    return {"np": np, "wavfile": wavfile, "resample_poly": resample_poly}


def resolve_audio_path(path_text: str, csv_path: Path) -> Path:
    path = Path(path_text)
    if path.is_absolute():
        return path
    candidates = [csv_path.resolve().parent / path, csv_path.resolve().parent.parent / path, PROJECT_ROOT / path]
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


def frame_rms(modules: dict[str, Any], audio: Any, sample_rate: int = 16_000) -> Any:
    np = modules["np"]
    frame_size = int(sample_rate * 0.03)
    hop = int(sample_rate * 0.01)
    values = []
    for start in range(0, max(1, len(audio) - frame_size + 1), hop):
        frame = audio[start : start + frame_size]
        if len(frame) == frame_size:
            values.append(float(np.sqrt(np.mean(np.square(frame)))))
    return np.asarray(values, dtype="float32")


def spectral_flatness(modules: dict[str, Any], audio: Any, sample_rate: int = 16_000) -> float:
    np = modules["np"]
    frame_size = int(sample_rate * 0.04)
    hop = int(sample_rate * 0.02)
    values = []
    window = np.hanning(frame_size).astype("float32")
    for start in range(0, max(1, len(audio) - frame_size + 1), hop):
        frame = audio[start : start + frame_size]
        if len(frame) != frame_size:
            continue
        power = np.square(np.abs(np.fft.rfft(frame * window))) + 1e-12
        geometric = float(np.exp(np.mean(np.log(power))))
        arithmetic = float(np.mean(power))
        values.append(geometric / max(arithmetic, 1e-12))
    return float(np.mean(values)) if values else math.nan


def clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


def score_audio(modules: dict[str, Any], audio: Any) -> dict[str, str]:
    np = modules["np"]
    duration = float(len(audio) / 16_000.0) if len(audio) else math.nan
    rms = float(np.sqrt(np.mean(np.square(audio)))) if len(audio) else math.nan
    rms_dbfs = 20.0 * math.log10(max(rms, 1e-8))
    peak = float(np.max(np.abs(audio))) if len(audio) else math.nan
    clipping_ratio = float(np.mean(np.abs(audio) >= 0.98)) if len(audio) else math.nan
    rms_values = frame_rms(modules, audio)
    silence_threshold = max(1e-4, rms * 0.10) if math.isfinite(rms) else 1e-4
    silence_ratio = float(np.mean(rms_values < silence_threshold)) if rms_values.size else math.nan
    flatness = spectral_flatness(modules, audio)

    loudness_penalty = 0.0
    if rms_dbfs < -32.0:
        loudness_penalty = clamp((-32.0 - rms_dbfs) / 18.0, 0.0, 1.0)
    elif rms_dbfs > -8.0:
        loudness_penalty = clamp((rms_dbfs + 8.0) / 8.0, 0.0, 1.0)

    silence_penalty = 0.0
    if math.isfinite(silence_ratio):
        if silence_ratio > 0.45:
            silence_penalty = clamp((silence_ratio - 0.45) / 0.35, 0.0, 1.0)
        elif silence_ratio < 0.02:
            silence_penalty = clamp((0.02 - silence_ratio) / 0.02, 0.0, 1.0)

    clipping_penalty = clamp(clipping_ratio * 100.0, 0.0, 1.0) if math.isfinite(clipping_ratio) else 0.0
    flatness_penalty = 0.0
    if math.isfinite(flatness) and flatness > 0.35:
        flatness_penalty = clamp((flatness - 0.35) / 0.40, 0.0, 1.0)

    duration_penalty = 0.0
    if math.isfinite(duration) and duration < 1.5:
        duration_penalty = clamp((1.5 - duration) / 1.5, 0.0, 1.0)

    penalty = (
        0.30 * loudness_penalty
        + 0.30 * silence_penalty
        + 0.20 * clipping_penalty
        + 0.15 * flatness_penalty
        + 0.05 * duration_penalty
    )
    score_1_5 = clamp(5.0 - 4.0 * penalty, 1.0, 5.0)

    return {
        "naturalness_proxy_1_5": f"{score_1_5:.6f}",
        "naturalness_penalty": f"{penalty:.6f}",
        "nat_duration_sec": f"{duration:.6f}",
        "nat_rms_dbfs": f"{rms_dbfs:.6f}",
        "nat_peak_abs": f"{peak:.6f}",
        "nat_clipping_ratio": f"{clipping_ratio:.8f}",
        "nat_silence_ratio": f"{silence_ratio:.6f}",
        "nat_spectral_flatness": f"{flatness:.6f}",
    }


def main() -> None:
    args = parse_args()
    modules = setup_imports()
    with args.input.open(newline="", encoding="utf-8-sig") as handle:
        rows = list(csv.DictReader(handle))
    if args.max_rows is not None:
        rows = rows[: args.max_rows]

    output_rows = []
    for row in rows:
        audio_column = row.get("audio_path") or row.get("generated_audio")
        if not audio_column:
            raise ValueError(f"row {row.get('id', '')} has no audio_path/generated_audio")
        audio_path = resolve_audio_path(audio_column, args.input)
        audio = read_wav_mono_16k(modules, audio_path)
        result = dict(row)
        result.update(score_audio(modules, audio))
        output_rows.append(result)
        print(f"{row.get('id', audio_path.name)}: naturalness_proxy={result['naturalness_proxy_1_5']}")

    base_fields = list(rows[0].keys()) if rows else []
    extra_fields = [
        "naturalness_proxy_1_5",
        "naturalness_penalty",
        "nat_duration_sec",
        "nat_rms_dbfs",
        "nat_peak_abs",
        "nat_clipping_ratio",
        "nat_silence_ratio",
        "nat_spectral_flatness",
    ]
    fieldnames = base_fields + [field for field in extra_fields if field not in base_fields]
    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    with args.output_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(output_rows)

    values = [float(row["naturalness_proxy_1_5"]) for row in output_rows]
    mean_score = sum(values) / len(values) if values else math.nan
    sorted_rows = sorted(output_rows, key=lambda row: float(row["naturalness_proxy_1_5"]), reverse=True)
    lines = [
        "# Acoustic Naturalness Proxy",
        "",
        f"- input: `{args.input}`",
        f"- samples: {len(output_rows)}",
        f"- mean naturalness proxy: {mean_score:.6f}",
        "",
        "This is a lightweight no-reference fallback, not a learned MOS model. It is intended to keep the main-metric pipeline runnable when UTMOS/NISQA weights are unavailable.",
        "",
        "| rank | id | naturalness_proxy | rms_dbfs | silence_ratio | clipping_ratio | flatness |",
        "| ---: | --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for rank, row in enumerate(sorted_rows, start=1):
        lines.append(
            "| {rank} | {id} | {naturalness_proxy_1_5} | {nat_rms_dbfs} | {nat_silence_ratio} | {nat_clipping_ratio} | {nat_spectral_flatness} |".format(
                rank=rank, **row
            )
        )
    args.output_md.parent.mkdir(parents=True, exist_ok=True)
    args.output_md.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"wrote {args.output_csv}")
    print(f"wrote {args.output_md}")


if __name__ == "__main__":
    main()
