from __future__ import annotations

import argparse
import csv
import logging
import math
import os
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
LABELS = ("angry", "disgusted", "fearful", "happy", "neutral", "other", "sad", "surprised", "unknown")
TARGETS = {"angry", "happy", "neutral", "sad"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate categorical and dimensional automatic speech-emotion teachers.")
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output-csv", required=True, type=Path)
    parser.add_argument("--embeddings-npz", required=True, type=Path)
    parser.add_argument("--cost-csv", required=True, type=Path)
    parser.add_argument("--device", default="auto", choices=["auto", "cuda", "cpu"])
    return parser.parse_args()


def resolve_audio(path_text: str) -> Path:
    path = Path(path_text)
    return path if path.is_absolute() else PROJECT_ROOT / path


def normalize_e2v_label(label: str) -> str:
    normalized = label.split("/")[-1].strip().lower()
    return "unknown" if normalized in {"<unk>", "unk"} else normalized


def entropy(probs: list[float]) -> float:
    return -sum(value * math.log(max(value, 1e-12)) for value in probs) / math.log(max(len(probs), 2))


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


def evaluate_emotion2vec(rows: list[dict[str, str]], device: str) -> tuple[list[dict[str, object]], np.ndarray, dict[str, object]]:
    os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
    logging.getLogger().setLevel(logging.WARNING)
    from funasr import AutoModel

    load_start = time.perf_counter()
    model = AutoModel(
        model="emotion2vec_plus_base",
        hub="hf",
        device=device,
        disable_update=True,
    )
    load_seconds = time.perf_counter() - load_start
    output_rows: list[dict[str, object]] = []
    embeddings = []
    inference_start = time.perf_counter()
    for row in rows:
        result = model.generate(
            input=str(resolve_audio(row["audio_path"])),
            granularity="utterance",
            extract_embedding=True,
        )[0]
        scores = {normalize_e2v_label(label): float(score) for label, score in zip(result["labels"], result["scores"])}
        values = [scores.get(label, 0.0) for label in LABELS]
        target = row.get("target_emotion", "").lower()
        top = max(scores, key=scores.get)
        item: dict[str, object] = {
            "sample_key": row["sample_key"],
            "e2v_top_label": top,
            "e2v_top_prob": scores[top],
            "e2v_target_prob": scores.get(target, math.nan),
            "e2v_target_match": int(top == target) if target in TARGETS else "",
            "e2v_entropy_norm": entropy(values),
        }
        for label in LABELS:
            item[f"e2v_prob_{label}"] = scores.get(label, 0.0)
        output_rows.append(item)
        embedding = np.asarray(result["feats"], dtype="float32").reshape(-1)
        embedding /= max(float(np.linalg.norm(embedding)), 1e-12)
        embeddings.append(embedding)
    inference_seconds = time.perf_counter() - inference_start
    return output_rows, np.vstack(embeddings), {
        "model": "emotion2vec_plus_base",
        "load_seconds": load_seconds,
        "inference_seconds": inference_seconds,
        "total_seconds": load_seconds + inference_seconds,
        "seconds_per_clip_warm": inference_seconds / len(rows),
        "device": device,
    }


def setup_superb(device: str) -> dict[str, Any]:
    import torch
    from transformers import AutoFeatureExtractor, AutoModelForAudioClassification

    name = "superb/wav2vec2-base-superb-er"
    extractor = AutoFeatureExtractor.from_pretrained(name, local_files_only=True)
    model = AutoModelForAudioClassification.from_pretrained(name, local_files_only=True).to(device).eval()
    return {"torch": torch, "extractor": extractor, "model": model, "name": name}


def read_audio_16k(path: Path) -> np.ndarray:
    from scipy.io import wavfile
    from scipy.signal import resample_poly

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


def normalize_superb_label(label: str) -> str:
    return {"hap": "happy", "ang": "angry", "neu": "neutral"}.get(label.lower(), label.lower())


def evaluate_superb(rows: list[dict[str, str]], device: str) -> tuple[list[dict[str, object]], np.ndarray, dict[str, object]]:
    load_start = time.perf_counter()
    modules = setup_superb(device)
    load_seconds = time.perf_counter() - load_start
    torch = modules["torch"]
    model = modules["model"]
    extractor = modules["extractor"]
    output_rows = []
    embeddings = []
    inference_start = time.perf_counter()
    with torch.no_grad():
        for row in rows:
            audio = read_audio_16k(resolve_audio(row["audio_path"]))
            inputs = extractor(audio, sampling_rate=16_000, return_tensors="pt", padding=True)
            inputs = {key: value.to(device) for key, value in inputs.items()}
            outputs = model(**inputs, output_hidden_states=True)
            probs = torch.softmax(outputs.logits[0], dim=-1).detach().float().cpu().numpy()
            scores = {
                normalize_superb_label(model.config.id2label[index]): float(prob)
                for index, prob in enumerate(probs)
            }
            target = row.get("target_emotion", "").lower()
            top = max(scores, key=scores.get)
            output_rows.append(
                {
                    "sample_key": row["sample_key"],
                    "superb_top_label": top,
                    "superb_top_prob": scores[top],
                    "superb_target_prob": scores.get(target, math.nan),
                    "superb_target_match": int(top == target) if target in TARGETS else "",
                    "superb_entropy_norm": entropy(list(scores.values())),
                }
            )
            pooled = outputs.hidden_states[-1][0].mean(dim=0).detach().float().cpu().numpy()
            pooled /= max(float(np.linalg.norm(pooled)), 1e-12)
            embeddings.append(pooled)
    inference_seconds = time.perf_counter() - inference_start
    return output_rows, np.vstack(embeddings), {
        "model": modules["name"],
        "load_seconds": load_seconds,
        "inference_seconds": inference_seconds,
        "total_seconds": load_seconds + inference_seconds,
        "seconds_per_clip_warm": inference_seconds / len(rows),
        "device": device,
    }


def setup_msp_dim(device: str) -> dict[str, Any]:
    import torch
    import torch.nn as nn
    from transformers import Wav2Vec2PreTrainedModel, Wav2Vec2Processor
    from transformers.models.wav2vec2.modeling_wav2vec2 import Wav2Vec2Model

    class RegressionHead(nn.Module):
        def __init__(self, config: Any):
            super().__init__()
            self.dense = nn.Linear(config.hidden_size, config.hidden_size)
            self.dropout = nn.Dropout(config.final_dropout)
            self.out_proj = nn.Linear(config.hidden_size, config.num_labels)

        def forward(self, features: Any) -> Any:
            value = self.dropout(features)
            value = torch.tanh(self.dense(value))
            return self.out_proj(self.dropout(value))

    class EmotionModel(Wav2Vec2PreTrainedModel):
        def __init__(self, config: Any):
            super().__init__(config)
            self.wav2vec2 = Wav2Vec2Model(config)
            self.classifier = RegressionHead(config)
            self.post_init()

        def forward(self, input_values: Any) -> tuple[Any, Any]:
            hidden = self.wav2vec2(input_values).last_hidden_state.mean(dim=1)
            return hidden, self.classifier(hidden)

    name = "audeering/wav2vec2-large-robust-12-ft-emotion-msp-dim"
    processor = Wav2Vec2Processor.from_pretrained(name)
    model = EmotionModel.from_pretrained(name).to(device).eval()
    return {"torch": torch, "processor": processor, "model": model, "name": name}


def evaluate_msp_dim(rows: list[dict[str, str]], device: str) -> tuple[list[dict[str, object]], np.ndarray, dict[str, object]]:
    load_start = time.perf_counter()
    modules = setup_msp_dim(device)
    load_seconds = time.perf_counter() - load_start
    torch = modules["torch"]
    output_rows = []
    embeddings = []
    inference_start = time.perf_counter()
    with torch.no_grad():
        for row in rows:
            audio = read_audio_16k(resolve_audio(row["audio_path"]))
            values = modules["processor"](audio, sampling_rate=16_000)["input_values"][0]
            inputs = torch.from_numpy(np.asarray(values)).reshape(1, -1).to(device)
            embedding, prediction = modules["model"](inputs)
            prediction = prediction[0].detach().float().cpu().numpy()
            output_rows.append(
                {
                    "sample_key": row["sample_key"],
                    "vad_arousal": float(prediction[0]),
                    "vad_dominance": float(prediction[1]),
                    "vad_valence": float(prediction[2]),
                }
            )
            pooled = embedding[0].detach().float().cpu().numpy()
            pooled /= max(float(np.linalg.norm(pooled)), 1e-12)
            embeddings.append(pooled)
    inference_seconds = time.perf_counter() - inference_start
    return output_rows, np.vstack(embeddings), {
        "model": modules["name"],
        "load_seconds": load_seconds,
        "inference_seconds": inference_seconds,
        "total_seconds": load_seconds + inference_seconds,
        "seconds_per_clip_warm": inference_seconds / len(rows),
        "device": device,
    }


def main() -> None:
    args = parse_args()
    os.environ.setdefault("TRANSFORMERS_NO_TF", "1")
    os.environ.setdefault("USE_TF", "0")
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    import torch

    device = "cuda" if args.device == "auto" and torch.cuda.is_available() else args.device
    if device == "auto":
        device = "cpu"
    with args.input.open(newline="", encoding="utf-8-sig") as handle:
        rows = list(csv.DictReader(handle))

    e2v_rows, e2v_embeddings, e2v_cost = evaluate_emotion2vec(rows, device)
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    superb_rows, superb_embeddings, superb_cost = evaluate_superb(rows, device)
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    vad_rows, vad_embeddings, vad_cost = evaluate_msp_dim(rows, device)

    indexed = [{row["sample_key"]: row for row in model_rows} for model_rows in (e2v_rows, superb_rows, vad_rows)]
    output_rows = []
    for row in rows:
        item: dict[str, object] = dict(row)
        for model_index in indexed:
            item.update(model_index[row["sample_key"]])
        output_rows.append(item)
    write_csv(args.output_csv, output_rows)
    args.embeddings_npz.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.embeddings_npz,
        sample_keys=np.asarray([row["sample_key"] for row in rows]),
        emotion2vec=e2v_embeddings,
        superb=superb_embeddings,
        msp_dim=vad_embeddings,
    )
    write_csv(args.cost_csv, [e2v_cost, superb_cost, vad_cost])
    print(f"wrote {args.output_csv}")
    print(f"wrote {args.embeddings_npz}")
    print(f"wrote {args.cost_csv}")


if __name__ == "__main__":
    main()
