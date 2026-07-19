from __future__ import annotations

import csv
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SOURCES = (
    (
        "parler_emotion_v1",
        PROJECT_ROOT / "experiments" / "parler_emotion_v1" / "combined" / "parler_emotion_scored_main_metric.csv",
    ),
    (
        "boundary_metric_v1",
        PROJECT_ROOT / "experiments" / "boundary_metric_v1" / "combined" / "boundary_scored_main_metric.csv",
    ),
    (
        "multisystem_generalization_v1",
        PROJECT_ROOT
        / "experiments"
        / "multisystem_generalization_v1"
        / "combined"
        / "multisystem_scored_main_metric.csv",
    ),
)
OUTPUT = (
    PROJECT_ROOT
    / "experiments"
    / "automatic_emotion_consensus_v1"
    / "inputs"
    / "evaluation_manifest.csv"
)


def main() -> None:
    rows: list[dict[str, str]] = []
    for dataset, path in SOURCES:
        with path.open(newline="", encoding="utf-8-sig") as handle:
            for raw in csv.DictReader(handle):
                row = dict(raw)
                row["dataset"] = dataset
                row["sample_key"] = f"{dataset}::{raw['id']}"
                row["tts_system"] = raw.get("tts_system") or "parler"
                row["voice"] = raw.get("voice") or "Jenna"
                row["is_boundary"] = raw.get("is_boundary") or (
                    "1" if dataset == "boundary_metric_v1" and raw.get("case_type") != "control" else "0"
                )
                rows.append(row)

    fields = [
        "sample_key",
        "dataset",
        "id",
        "tts_system",
        "voice",
        "case_type",
        "is_boundary",
        "boundary_type",
        "target_emotion",
        "text",
        "audio_path",
        "wer",
        "cer",
        "acoustic_sanity_score_0_1",
        "prosody_activity_0_1",
        "f0_std_semitones",
        "f0_range_semitones_p90_p10",
        "energy_cv",
        "silence_ratio",
        "asr_transcript",
        "main_metric_0_1",
        "emotion_component_0_1",
    ]
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    with OUTPUT.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    print(f"wrote {OUTPUT} ({len(rows)} clips)")


if __name__ == "__main__":
    main()
