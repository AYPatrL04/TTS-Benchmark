from __future__ import annotations

import argparse
import csv
import gc
import math
import os
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SYSTEMS = ("parler", "bark", "sapi")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate a shared-text TTS set with multiple local systems and voices.")
    parser.add_argument("--cases", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--manifest-csv", required=True, type=Path)
    parser.add_argument("--systems", nargs="+", choices=SYSTEMS, default=list(SYSTEMS))
    parser.add_argument("--device", choices=["auto", "cuda", "cpu"], default="auto")
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def setup_imports() -> dict[str, Any]:
    os.environ.setdefault("TRANSFORMERS_NO_TF", "1")
    os.environ.setdefault("USE_TF", "0")
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    os.environ.setdefault("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION", "python")

    import numpy as np
    import torch
    from scipy.io import wavfile

    return {"np": np, "torch": torch, "wavfile": wavfile}


def read_cases(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8-sig") as handle:
        rows = list(csv.DictReader(handle))
    if not rows or any(not row.get("text_id") or not row.get("text") for row in rows):
        raise ValueError("Cases CSV must contain non-empty text_id and text columns")
    return rows


def write_wav(modules: dict[str, Any], path: Path, audio: Any, sample_rate: int) -> float:
    np = modules["np"]
    wavfile = modules["wavfile"]
    array = np.nan_to_num(np.asarray(audio, dtype="float32").squeeze())
    if array.ndim > 1:
        array = array.mean(axis=0)
    peak = float(np.max(np.abs(array))) if array.size else 0.0
    if peak > 1.0:
        array = array / peak
    wavfile.write(path, sample_rate, (array * 32767.0).astype("int16"))
    return float(len(array) / sample_rate)


def wav_duration(modules: dict[str, Any], path: Path) -> tuple[int, float]:
    sample_rate, audio = modules["wavfile"].read(path)
    return int(sample_rate), float(len(audio) / sample_rate)


def generate_parler(cases: list[dict[str, str]], output_dir: Path, modules: dict[str, Any], args: argparse.Namespace) -> list[dict[str, str]]:
    from parler_tts import ParlerTTSForConditionalGeneration
    from transformers import AutoTokenizer

    torch = modules["torch"]
    model_name = "parler-tts/parler-tts-mini-v1"
    device = "cuda" if args.device == "auto" and torch.cuda.is_available() else args.device
    if device == "auto":
        device = "cpu"
    dtype = torch.float16 if device == "cuda" else torch.float32
    tokenizer = AutoTokenizer.from_pretrained(model_name, local_files_only=True)
    model = ParlerTTSForConditionalGeneration.from_pretrained(
        model_name, torch_dtype=dtype, local_files_only=True
    ).to(device).eval()
    sample_rate = int(model.config.sampling_rate)
    rows = []
    for case in cases:
        path = output_dir / "parler_jenna" / f"{case['text_id']}.wav"
        path.parent.mkdir(parents=True, exist_ok=True)
        if args.overwrite or not path.exists():
            description = (
                "Jenna's voice sounds calm and neutral, with moderate pitch, moderate speed, plain narration, "
                "minimal emotion, and very clear close-up audio. " + case.get("description", "")
            )
            description_inputs = tokenizer(description, return_tensors="pt")
            prompt_inputs = tokenizer(case["text"], return_tensors="pt")
            with torch.no_grad():
                generation = model.generate(
                    input_ids=description_inputs.input_ids.to(device),
                    attention_mask=description_inputs.attention_mask.to(device),
                    prompt_input_ids=prompt_inputs.input_ids.to(device),
                    do_sample=True,
                    temperature=1.0,
                )
            duration = write_wav(modules, path, generation.detach().float().cpu().numpy(), sample_rate)
        else:
            _existing_rate, duration = wav_duration(modules, path)
        rows.append(result_row(case, "parler", model_name, "Jenna", "seeded-sampling-neutral", path, sample_rate, duration))
    del model, tokenizer
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return rows


def generate_bark(cases: list[dict[str, str]], output_dir: Path, modules: dict[str, Any], args: argparse.Namespace) -> list[dict[str, str]]:
    from transformers import AutoProcessor, BarkModel

    torch = modules["torch"]
    model_name = "suno/bark-small"
    voice = "v2/en_speaker_6"
    device = "cuda" if args.device == "auto" and torch.cuda.is_available() else args.device
    if device == "auto":
        device = "cpu"
    processor = AutoProcessor.from_pretrained(model_name, local_files_only=True)
    model = BarkModel.from_pretrained(model_name, local_files_only=True).to(device).eval()
    sample_rate = int(model.generation_config.sample_rate)
    rows = []
    for case in cases:
        path = output_dir / "bark_speaker_6" / f"{case['text_id']}.wav"
        path.parent.mkdir(parents=True, exist_ok=True)
        if args.overwrite or not path.exists():
            inputs = processor(case["text"], voice_preset=voice, return_tensors="pt")
            inputs = {key: value.to(device) for key, value in inputs.items()}
            with torch.no_grad():
                generation = model.generate(**inputs)
            duration = write_wav(modules, path, generation.detach().float().cpu().numpy(), sample_rate)
        else:
            _existing_rate, duration = wav_duration(modules, path)
        rows.append(result_row(case, "bark", model_name, voice, "seeded-default-sampling", path, sample_rate, duration))
    del model, processor
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return rows


def generate_sapi(cases: list[dict[str, str]], output_dir: Path, modules: dict[str, Any], args: argparse.Namespace) -> list[dict[str, str]]:
    import win32com.client

    voice_name = "Microsoft Zira Desktop"
    rows = []
    for case in cases:
        path = (output_dir / "sapi_zira" / f"{case['text_id']}.wav").resolve()
        path.parent.mkdir(parents=True, exist_ok=True)
        if args.overwrite or not path.exists():
            speaker = win32com.client.Dispatch("SAPI.SpVoice")
            matching = [token for token in speaker.GetVoices() if voice_name.lower() in token.GetDescription().lower()]
            if not matching:
                raise RuntimeError(f"SAPI voice not installed: {voice_name}")
            speaker.Voice = matching[0]
            speaker.Rate = 0
            stream = win32com.client.Dispatch("SAPI.SpFileStream")
            stream.Open(str(path), 3, False)
            speaker.AudioOutputStream = stream
            speaker.Speak(case["text"])
            stream.Close()
        sample_rate, audio = modules["wavfile"].read(path)
        duration = float(len(audio) / sample_rate)
        rows.append(result_row(case, "sapi", "Windows SAPI", voice_name, "rate=0", path, int(sample_rate), duration))
    return rows


def result_row(case: dict[str, str], system: str, model: str, voice: str, config: str, path: Path, sample_rate: int, duration: float) -> dict[str, str]:
    relative_path = path.resolve().relative_to(PROJECT_ROOT.resolve())
    return {
        "id": f"{system}__{case['text_id']}",
        "text_id": case["text_id"],
        "tts_system": system,
        "model": model,
        "voice": voice,
        "source_config": config,
        "is_boundary": case.get("is_boundary", "0"),
        "boundary_type": case.get("boundary_type", "none"),
        "case_type": "boundary" if case.get("is_boundary", "0") == "1" else "regular",
        "target_emotion": case.get("target_emotion", "neutral"),
        "text": case["text"],
        "description": case.get("description", ""),
        "audio_path": str(relative_path),
        "generation_seed": "" if system == "sapi" else str(2026),
        "sample_rate": str(sample_rate),
        "audio_duration_sec": f"{duration:.3f}" if math.isfinite(duration) else "",
    }


def main() -> None:
    args = parse_args()
    modules = setup_imports()
    modules["torch"].manual_seed(args.seed)
    cases = read_cases(args.cases)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    generators = {"parler": generate_parler, "bark": generate_bark, "sapi": generate_sapi}
    rows: list[dict[str, str]] = []
    for system in args.systems:
        print(f"generating system={system}")
        rows.extend(generators[system](cases, args.output_dir, modules, args))
    args.manifest_csv.parent.mkdir(parents=True, exist_ok=True)
    with args.manifest_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"wrote {args.manifest_csv} ({len(rows)} clips)")


if __name__ == "__main__":
    main()
