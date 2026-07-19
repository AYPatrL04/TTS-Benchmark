from __future__ import annotations

import argparse
import csv
import math
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from tts_metric_eval.text_metrics import edit_distance, normalize_text


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Transcribe generated TTS audio with Transformers Whisper and compute WER/CER."
    )
    parser.add_argument("--input", required=True, type=Path, help="CSV with text and audio_path/generated_audio columns.")
    parser.add_argument("--output-csv", required=True, type=Path)
    parser.add_argument("--output-md", required=True, type=Path)
    parser.add_argument("--model", default="openai/whisper-tiny.en")
    parser.add_argument("--device", default="auto", choices=["auto", "cuda", "cpu"])
    parser.add_argument("--language", default="english")
    parser.add_argument("--task", default="transcribe")
    parser.add_argument("--max-rows", type=int)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def setup_imports() -> dict[str, Any]:
    os.environ.setdefault("TRANSFORMERS_NO_TF", "1")
    os.environ.setdefault("USE_TF", "0")
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

    import numpy as np
    import torch
    from scipy.io import wavfile
    from scipy.signal import resample_poly
    from transformers import WhisperForConditionalGeneration, WhisperProcessor

    return {
        "np": np,
        "torch": torch,
        "wavfile": wavfile,
        "resample_poly": resample_poly,
        "WhisperForConditionalGeneration": WhisperForConditionalGeneration,
        "WhisperProcessor": WhisperProcessor,
    }


@dataclass
class WordDiff:
    substitutions: int
    deletions: int
    insertions: int
    reference_words: int
    diff: str


def normalized_wer(reference_text: str, hypothesis_text: str, normalizer: Any | None = None) -> float:
    reference_normalized = normalizer(reference_text) if normalizer else normalize_text(reference_text)
    hypothesis_normalized = normalizer(hypothesis_text) if normalizer else normalize_text(hypothesis_text)
    reference_words = reference_normalized.split()
    hypothesis_words = hypothesis_normalized.split()
    if not reference_words:
        return 0.0 if not hypothesis_words else 1.0
    return edit_distance(reference_words, hypothesis_words) / len(reference_words)


def normalized_cer(reference_text: str, hypothesis_text: str, normalizer: Any | None = None) -> float:
    reference_normalized = normalizer(reference_text) if normalizer else normalize_text(reference_text)
    hypothesis_normalized = normalizer(hypothesis_text) if normalizer else normalize_text(hypothesis_text)
    reference_chars = list(reference_normalized.replace(" ", ""))
    hypothesis_chars = list(hypothesis_normalized.replace(" ", ""))
    if not reference_chars:
        return 0.0 if not hypothesis_chars else 1.0
    return edit_distance(reference_chars, hypothesis_chars) / len(reference_chars)


def word_diff(reference_text: str, hypothesis_text: str, normalizer: Any | None = None) -> WordDiff:
    reference_normalized = normalizer(reference_text) if normalizer else normalize_text(reference_text)
    hypothesis_normalized = normalizer(hypothesis_text) if normalizer else normalize_text(hypothesis_text)
    reference = reference_normalized.split()
    hypothesis = hypothesis_normalized.split()
    n = len(reference)
    m = len(hypothesis)
    costs = [[0] * (m + 1) for _ in range(n + 1)]
    ops = [[""] * (m + 1) for _ in range(n + 1)]

    for i in range(1, n + 1):
        costs[i][0] = i
        ops[i][0] = "D"
    for j in range(1, m + 1):
        costs[0][j] = j
        ops[0][j] = "I"

    for i in range(1, n + 1):
        for j in range(1, m + 1):
            if reference[i - 1] == hypothesis[j - 1]:
                choices = [(costs[i - 1][j - 1], "M")]
            else:
                choices = [(costs[i - 1][j - 1] + 1, "S")]
            choices.extend(
                [
                    (costs[i - 1][j] + 1, "D"),
                    (costs[i][j - 1] + 1, "I"),
                ]
            )
            costs[i][j], ops[i][j] = min(choices, key=lambda item: item[0])

    i, j = n, m
    substitutions = deletions = insertions = 0
    parts = []
    while i > 0 or j > 0:
        op = ops[i][j]
        if op == "M":
            parts.append(reference[i - 1])
            i -= 1
            j -= 1
        elif op == "S":
            substitutions += 1
            parts.append(f"{reference[i - 1]}->{hypothesis[j - 1]}")
            i -= 1
            j -= 1
        elif op == "D":
            deletions += 1
            parts.append(f"-{reference[i - 1]}")
            i -= 1
        elif op == "I":
            insertions += 1
            parts.append(f"+{hypothesis[j - 1]}")
            j -= 1
        else:
            break
    parts.reverse()
    return WordDiff(substitutions, deletions, insertions, n, " ".join(parts))


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
        max_value = float(np.iinfo(data.dtype).max)
        audio = audio / max_value
    elif data.dtype.kind == "f":
        audio = np.clip(audio, -1.0, 1.0)

    target_rate = 16_000
    if sample_rate != target_rate:
        gcd = math.gcd(sample_rate, target_rate)
        audio = resample_poly(audio, target_rate // gcd, sample_rate // gcd).astype("float32")
    return audio


def cheap_audio_features(modules: dict[str, Any], audio: Any, sample_rate: int, reference_words: int) -> dict[str, str]:
    np = modules["np"]
    duration_sec = float(len(audio) / sample_rate) if sample_rate else math.nan
    rms = float(np.sqrt(np.mean(np.square(audio)))) if len(audio) else math.nan
    peak_abs = float(np.max(np.abs(audio))) if len(audio) else math.nan
    speech_rate_wps = float(reference_words / duration_sec) if duration_sec and duration_sec > 0 else math.nan

    frame_size = max(1, int(sample_rate * 0.02))
    hop = max(1, int(sample_rate * 0.01))
    frame_rms_values = []
    for start in range(0, max(1, len(audio) - frame_size + 1), hop):
        frame = audio[start : start + frame_size]
        if len(frame) == frame_size:
            frame_rms_values.append(float(np.sqrt(np.mean(np.square(frame)))))
    silence_threshold = max(1e-4, rms * 0.10) if math.isfinite(rms) else 1e-4
    silence_ratio = (
        sum(1 for value in frame_rms_values if value < silence_threshold) / len(frame_rms_values)
        if frame_rms_values
        else math.nan
    )

    return {
        "asr_audio_duration_sec": f"{duration_sec:.6f}",
        "speech_rate_wps": f"{speech_rate_wps:.6f}",
        "speech_rate_abs_delta_2_5": f"{abs(speech_rate_wps - 2.5):.6f}" if math.isfinite(speech_rate_wps) else "",
        "rms": f"{rms:.8f}",
        "peak_abs": f"{peak_abs:.8f}",
        "silence_ratio": f"{silence_ratio:.8f}",
    }


def load_previous(path: Path) -> dict[str, dict[str, str]]:
    if not path.exists():
        return {}
    with path.open(newline="", encoding="utf-8-sig") as handle:
        rows = list(csv.DictReader(handle))
    return {row.get("id", ""): row for row in rows if row.get("id")}


def main() -> None:
    args = parse_args()
    modules = setup_imports()
    np = modules["np"]
    torch = modules["torch"]
    WhisperForConditionalGeneration = modules["WhisperForConditionalGeneration"]
    WhisperProcessor = modules["WhisperProcessor"]

    with args.input.open(newline="", encoding="utf-8-sig") as handle:
        rows = list(csv.DictReader(handle))
    if args.max_rows is not None:
        rows = rows[: args.max_rows]

    device = "cuda" if args.device == "auto" and torch.cuda.is_available() else args.device
    if device == "auto":
        device = "cpu"
    torch_dtype = torch.float16 if device == "cuda" else torch.float32

    print(f"loading ASR model {args.model} on {device}")
    processor = WhisperProcessor.from_pretrained(args.model)
    model = WhisperForConditionalGeneration.from_pretrained(args.model, torch_dtype=torch_dtype).to(device)
    model.eval()
    forced_decoder_ids = processor.get_decoder_prompt_ids(language=args.language, task=args.task)
    text_normalizer = processor.tokenizer.normalize

    previous = load_previous(args.output_csv) if not args.overwrite else {}
    output_rows: list[dict[str, str]] = []
    for row in rows:
        item_id = row.get("id", "")
        if item_id in previous and previous[item_id].get("asr_transcript"):
            output_rows.append(previous[item_id])
            print(f"skip existing {item_id}: {previous[item_id]['asr_transcript']}")
            continue

        audio_column = row.get("audio_path") or row.get("generated_audio")
        if not audio_column:
            raise ValueError(f"row {item_id} has no audio_path/generated_audio")
        audio_path = resolve_audio_path(audio_column, args.input)
        audio = read_wav_mono_16k(modules, audio_path)

        inputs = processor(audio, sampling_rate=16_000, return_tensors="pt", return_attention_mask=True)
        input_features = inputs.input_features.to(device=device, dtype=torch_dtype)
        attention_mask = getattr(inputs, "attention_mask", None)
        if attention_mask is not None:
            attention_mask = attention_mask.to(device=device)
        with torch.no_grad():
            predicted_ids = model.generate(
                input_features,
                attention_mask=attention_mask,
                forced_decoder_ids=forced_decoder_ids,
            )
        transcript = processor.batch_decode(predicted_ids, skip_special_tokens=True)[0].strip()

        reference_text = row.get("text", "")
        row_wer = normalized_wer(reference_text, transcript, text_normalizer)
        row_cer = normalized_cer(reference_text, transcript, text_normalizer)
        diff = word_diff(reference_text, transcript, text_normalizer)
        audio_features = cheap_audio_features(modules, audio, 16_000, diff.reference_words)
        result = dict(row)
        result.update(
            {
                "asr_model": args.model,
                "asr_transcript": transcript,
                "wer": f"{row_wer:.8f}",
                "cer": f"{row_cer:.8f}",
                "word_substitutions": str(diff.substitutions),
                "word_deletions": str(diff.deletions),
                "word_insertions": str(diff.insertions),
                "reference_words": str(diff.reference_words),
                "word_diff": diff.diff,
            }
        )
        result.update(audio_features)
        output_rows.append(result)
        print(f"{item_id}: WER={row_wer:.4f} CER={row_cer:.4f} transcript={transcript}")

    fieldnames = list(rows[0].keys()) if rows else []
    for extra in [
        "asr_model",
        "asr_transcript",
        "wer",
        "cer",
        "word_substitutions",
        "word_deletions",
        "word_insertions",
        "reference_words",
        "word_diff",
        "asr_audio_duration_sec",
        "speech_rate_wps",
        "speech_rate_abs_delta_2_5",
        "rms",
        "peak_abs",
        "silence_ratio",
    ]:
        if extra not in fieldnames:
            fieldnames.append(extra)

    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    with args.output_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(output_rows)

    wer_values = [float(row["wer"]) for row in output_rows if row.get("wer")]
    cer_values = [float(row["cer"]) for row in output_rows if row.get("cer")]
    mean_wer = sum(wer_values) / len(wer_values) if wer_values else math.nan
    mean_cer = sum(cer_values) / len(cer_values) if cer_values else math.nan
    sorted_rows = sorted(output_rows, key=lambda item: float(item.get("wer", "nan")))

    lines = [
        "# ASR-WER Evaluation",
        "",
        f"- input: `{args.input}`",
        f"- ASR model: `{args.model}`",
        f"- samples: {len(output_rows)}",
        f"- mean WER: {mean_wer:.8f}",
        f"- mean CER: {mean_cer:.8f}",
        "",
        "WER is computed by transcribing generated audio with ASR, normalizing text, then comparing the ASR transcript against the input text.",
        "This is an automatic intelligibility proxy, not a replacement for small human listening checks.",
        "",
        "| id | WER | CER | S/D/I | transcript |",
        "| --- | ---: | ---: | --- | --- |",
    ]
    for row in sorted_rows:
        sdi = f"{row.get('word_substitutions', '')}/{row.get('word_deletions', '')}/{row.get('word_insertions', '')}"
        transcript = row.get("asr_transcript", "").replace("|", "\\|")
        lines.append(f"| {row.get('id', '')} | {row.get('wer', '')} | {row.get('cer', '')} | {sdi} | {transcript} |")

    args.output_md.parent.mkdir(parents=True, exist_ok=True)
    args.output_md.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"wrote {args.output_csv}")
    print(f"wrote {args.output_md}")


if __name__ == "__main__":
    main()
