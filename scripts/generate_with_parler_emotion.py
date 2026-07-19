from __future__ import annotations

import argparse
import csv
import math
import os
from pathlib import Path
from typing import Any


EMOTION_DESCRIPTIONS = {
    "happy": (
        "Jenna's voice sounds bright, cheerful, and excited, with high energy, "
        "a slightly higher pitch, expressive intonation, moderate speed, and very clear close-up audio."
    ),
    "sad": (
        "Jenna's voice sounds sad, soft, and low energy, with a lower pitch, slower pacing, "
        "gentle pauses, restrained emotion, and very clear close-up audio."
    ),
    "angry": (
        "Jenna's voice sounds angry, firm, and intense, with sharp articulation, higher energy, "
        "a slightly faster pace, strong emphasis, and very clear close-up audio."
    ),
    "neutral": (
        "Jenna's voice sounds calm and neutral, with moderate pitch, moderate speed, plain narration, "
        "minimal emotion, and very clear close-up audio."
    ),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate an emotion-controlled TTS set with Parler-TTS and write a metrics-ready manifest."
    )
    parser.add_argument("--texts", required=True, type=Path, help="CSV with id,target_emotion,text columns.")
    parser.add_argument("--output-dir", required=True, type=Path, help="Directory for generated wav files.")
    parser.add_argument("--manifest-csv", required=True, type=Path, help="Output manifest for downstream metrics.")
    parser.add_argument("--report-md", type=Path, help="Optional Markdown generation report.")
    parser.add_argument("--model", default="parler-tts/parler-tts-mini-v1")
    parser.add_argument("--device", default="auto", choices=["auto", "cuda", "cpu"])
    parser.add_argument("--dtype", default="auto", choices=["auto", "float16", "float32"])
    parser.add_argument("--max-prompts", type=int, default=8)
    parser.add_argument("--max-text-chars", type=int, default=260)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--do-sample", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def setup_imports() -> dict[str, Any]:
    # Avoid unrelated TensorFlow/protobuf imports in a mixed ML environment.
    os.environ.setdefault("TRANSFORMERS_NO_TF", "1")
    os.environ.setdefault("USE_TF", "0")
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    os.environ.setdefault("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION", "python")

    import numpy as np
    import torch
    from parler_tts import ParlerTTSForConditionalGeneration
    from scipy.io import wavfile
    from transformers import AutoTokenizer

    return {
        "np": np,
        "torch": torch,
        "wavfile": wavfile,
        "AutoTokenizer": AutoTokenizer,
        "ParlerTTSForConditionalGeneration": ParlerTTSForConditionalGeneration,
    }


def load_rows(path: Path, max_prompts: int, max_text_chars: int) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8-sig") as handle:
        rows = list(csv.DictReader(handle))

    selected: list[dict[str, str]] = []
    for row in rows:
        item_id = row.get("id", "").strip()
        text = row.get("text", "").strip()
        if not item_id or not text:
            continue
        if len(text) > max_text_chars:
            text = text[: max_text_chars - 1].rstrip() + "."
        selected_row = dict(row)
        selected_row["id"] = item_id
        selected_row["text"] = text
        selected.append(selected_row)
        if len(selected) >= max_prompts:
            break
    return selected


def choose_description(row: dict[str, str]) -> str:
    explicit = row.get("description", "").strip()
    if explicit:
        return explicit
    target_emotion = row.get("target_emotion", "").strip().lower()
    return EMOTION_DESCRIPTIONS.get(target_emotion, EMOTION_DESCRIPTIONS["neutral"])


def write_wav(np: Any, wavfile: Any, path: Path, audio: Any, sample_rate: int) -> float:
    array = np.asarray(audio, dtype=np.float32).squeeze()
    array = np.nan_to_num(array)
    if array.ndim > 1:
        array = array.mean(axis=0)
    peak = float(np.max(np.abs(array))) if array.size else 0.0
    if peak > 1.0:
        array = array / peak
    wavfile.write(path, sample_rate, (array * 32767.0).astype(np.int16))
    return float(array.shape[0] / sample_rate) if array.ndim == 1 and sample_rate else math.nan


def main() -> None:
    args = parse_args()
    modules = setup_imports()
    np = modules["np"]
    torch = modules["torch"]
    wavfile = modules["wavfile"]
    AutoTokenizer = modules["AutoTokenizer"]
    ParlerTTSForConditionalGeneration = modules["ParlerTTSForConditionalGeneration"]

    device = "cuda" if args.device == "auto" and torch.cuda.is_available() else args.device
    if device == "auto":
        device = "cpu"
    dtype = torch.float16 if args.dtype == "float16" or (args.dtype == "auto" and device == "cuda") else torch.float32

    torch.manual_seed(args.seed)
    if device == "cuda":
        torch.cuda.manual_seed_all(args.seed)

    rows = load_rows(args.texts, args.max_prompts, args.max_text_chars)
    if not rows:
        raise ValueError(f"No usable rows found in {args.texts}")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    args.manifest_csv.parent.mkdir(parents=True, exist_ok=True)
    if args.report_md:
        args.report_md.parent.mkdir(parents=True, exist_ok=True)

    print(f"loading {args.model} on {device} with dtype={dtype}")
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = ParlerTTSForConditionalGeneration.from_pretrained(args.model, torch_dtype=dtype).to(device)
    model.eval()
    sample_rate = int(model.config.sampling_rate)

    output_rows: list[dict[str, str]] = []
    for row in rows:
        item_id = row["id"]
        text = row["text"]
        description = choose_description(row)
        wav_path = args.output_dir / f"{item_id}.wav"
        if wav_path.exists() and not args.overwrite:
            print(f"skip existing {wav_path}")
            duration_sec = math.nan
        else:
            print(f"generating {item_id}: {row.get('target_emotion', '')} | {text}")
            description_inputs = tokenizer(description, return_tensors="pt")
            prompt_inputs = tokenizer(text, return_tensors="pt")
            description_ids = description_inputs.input_ids.to(device)
            description_attention_mask = description_inputs.attention_mask.to(device)
            prompt_ids = prompt_inputs.input_ids.to(device)
            with torch.no_grad():
                generation = model.generate(
                    input_ids=description_ids,
                    attention_mask=description_attention_mask,
                    prompt_input_ids=prompt_ids,
                    do_sample=args.do_sample,
                    temperature=args.temperature,
                )
            audio = generation.detach().float().cpu().numpy().squeeze()
            duration_sec = write_wav(np, wavfile, wav_path, audio, sample_rate)

        result = dict(row)
        result.update(
            {
                "description": description,
                "audio_path": str(wav_path),
                "model": args.model,
                "generation_seed": str(args.seed),
                "generation_temperature": f"{args.temperature:.4f}",
                "generation_do_sample": str(args.do_sample),
                "sample_rate": str(sample_rate),
                "audio_duration_sec": f"{duration_sec:.3f}" if math.isfinite(duration_sec) else "",
            }
        )
        output_rows.append(result)

    fieldnames: list[str] = []
    for row in output_rows:
        for key in row.keys():
            if key not in fieldnames:
                fieldnames.append(key)
    with args.manifest_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(output_rows)

    if args.report_md:
        lines = [
            "# Parler Emotion TTS Generation",
            "",
            "This run uses Parler-TTS with one fixed speaker description style per target emotion.",
            "",
            f"- model: `{args.model}`",
            f"- device: `{device}`",
            f"- dtype: `{dtype}`",
            f"- samples: {len(output_rows)}",
            f"- seed: {args.seed}",
            f"- temperature: {args.temperature:.4f}",
            "",
            "| id | target_emotion | duration_sec | audio |",
            "| --- | --- | ---: | --- |",
        ]
        for output_row in output_rows:
            lines.append(
                f"| {output_row.get('id', '')} | {output_row.get('target_emotion', '')} | "
                f"{output_row.get('audio_duration_sec', '')} | `{output_row.get('audio_path', '')}` |"
            )
        args.report_md.write_text("\n".join(lines) + "\n", encoding="utf-8")

    print(f"wrote {args.manifest_csv}")
    if args.report_md:
        print(f"wrote {args.report_md}")


if __name__ == "__main__":
    main()
