from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path


METRIC_VERSION = "provisional_teacher_v2"
REQUIRED_FIELDS = ("wer", "target_emotion_prob")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Score emotion-aware TTS samples with a provisional teacher metric.")
    parser.add_argument("--input", required=True, type=Path, help="Combined metrics CSV.")
    parser.add_argument("--output-csv", required=True, type=Path)
    parser.add_argument("--output-md", required=True, type=Path)
    parser.add_argument("--experiment-name", default="emotion_tts")
    return parser.parse_args()


def parse_float(row: dict[str, str], key: str) -> float:
    try:
        return float(row.get(key, ""))
    except (TypeError, ValueError):
        return math.nan


def clamp(value: float, low: float = 0.0, high: float = 1.0) -> float:
    if not math.isfinite(value):
        return math.nan
    return max(low, min(high, value))


def mean(values: list[float]) -> float:
    finite = [value for value in values if math.isfinite(value)]
    return sum(finite) / len(finite) if finite else math.nan


def first_finite(row: dict[str, str], keys: tuple[str, ...]) -> float:
    for key in keys:
        value = parse_float(row, key)
        if math.isfinite(value):
            return value
    return math.nan


def acoustic_sanity(row: dict[str, str]) -> float:
    direct = parse_float(row, "acoustic_sanity_score_0_1")
    if math.isfinite(direct):
        return clamp(direct)
    legacy = parse_float(row, "naturalness_proxy_1_5")
    return clamp((legacy - 1.0) / 4.0) if math.isfinite(legacy) else math.nan


def quality_signal(row: dict[str, str]) -> tuple[float, str]:
    mos_1_5 = first_finite(row, ("utmos_score_1_5", "mos_prediction_1_5"))
    if math.isfinite(mos_1_5):
        return clamp((mos_1_5 - 1.0) / 4.0), "learned_mos"
    return acoustic_sanity(row), "acoustic_sanity_fallback"


def invalid_reason(row: dict[str, str]) -> str:
    missing = [key for key in REQUIRED_FIELDS if not math.isfinite(parse_float(row, key))]
    if not math.isfinite(acoustic_sanity(row)):
        missing.append("acoustic_sanity_score_0_1")
    return ",".join(missing)


def score_row(row: dict[str, str]) -> dict[str, str]:
    missing = invalid_reason(row)
    result = dict(row)
    if missing:
        result.update(
            {
                "metric_version": METRIC_VERSION,
                "metric_status": "invalid",
                "metric_missing_fields": missing,
                "main_metric_0_1": "",
                "provisional_teacher_score_0_1": "",
            }
        )
        return result

    wer = clamp(parse_float(row, "wer"))
    cer = clamp(parse_float(row, "cer"))
    target_prob = clamp(parse_float(row, "target_emotion_prob"))
    sanity = acoustic_sanity(row)
    quality, quality_source = quality_signal(row)

    intelligibility = clamp(1.0 - wer)
    emotion = target_prob
    # This scalar remains a provisional teacher. Prosody is diagnostic until its
    # emotion-conditional distributions are calibrated on human speech.
    if quality_source == "learned_mos":
        teacher = clamp(0.55 * intelligibility + 0.35 * emotion + 0.10 * quality)
        active_weights = "I=0.55;E=0.35;Q_mos=0.10"
    else:
        teacher = clamp((0.55 * intelligibility + 0.35 * emotion) / 0.90)
        active_weights = "I=0.611111;E=0.388889;Q_sanity=diagnostic_only"
    eligible = intelligibility >= 0.70 and sanity >= 0.50

    result.update(
        {
            "metric_version": METRIC_VERSION,
            "metric_status": "valid_provisional",
            "metric_missing_fields": "",
            "main_metric_0_1": f"{teacher:.6f}",
            "provisional_teacher_score_0_1": f"{teacher:.6f}",
            "ranking_eligible": "1" if eligible else "0",
            "intelligibility_component_0_1": f"{intelligibility:.6f}",
            "cer_diagnostic_0_1": f"{cer:.6f}" if math.isfinite(cer) else "",
            "quality_component_0_1": f"{quality:.6f}",
            "quality_component_source": quality_source,
            "acoustic_sanity_score_0_1": f"{sanity:.6f}",
            "emotion_component_0_1": f"{emotion:.6f}",
            "prosody_diagnostic_0_1": row.get("prosody_activity_0_1", ""),
            "teacher_active_weights": active_weights,
            "main_metric_formula": "with learned MOS: 0.55*I+0.35*E+0.10*Q; otherwise: (0.55*I+0.35*E)/0.90; sanity/prosody diagnostic only",
        }
    )
    return result


def score_key(row: dict[str, str]) -> float:
    value = parse_float(row, "main_metric_0_1")
    return value if math.isfinite(value) else -math.inf


def main() -> None:
    args = parse_args()
    with args.input.open(newline="", encoding="utf-8-sig") as handle:
        rows = list(csv.DictReader(handle))
    scored_rows = sorted((score_row(row) for row in rows), key=score_key, reverse=True)

    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    fieldnames: list[str] = []
    for row in scored_rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with args.output_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(scored_rows)

    valid = [row for row in scored_rows if row.get("metric_status") == "valid_provisional"]
    scores = [float(row["main_metric_0_1"]) for row in valid]
    fallback_count = sum(row.get("quality_component_source") == "acoustic_sanity_fallback" for row in valid)
    lines = [
        f"# {args.experiment_name} Provisional Teacher V2",
        "",
        "The benchmark is vector-first. The scalar is retained for experiments, not treated as human-grounded truth.",
        "",
        "```text",
        "I = 1 - normalized_WER                 # CER is diagnostic",
        "E = target_emotion_prob                # uncalibrated SER; no argmax bonus",
        "Q = learned_MOS when available; acoustic sanity is diagnostic only",
        "teacher_v2 = 0.55*I + 0.35*E + 0.10*Q_mos  # with learned MOS",
        "teacher_v2 = (0.55*I + 0.35*E) / 0.90      # without learned MOS",
        "eligible = I >= 0.70 and acoustic_sanity >= 0.50",
        "```",
        "",
        f"Coverage: {len(valid)}/{len(scored_rows)} ({len(valid) / max(len(scored_rows), 1):.1%}).",
        f"Acoustic-sanity quality fallbacks: {fallback_count}/{len(valid)}.",
        f"Mean provisional teacher: {mean(scores):.6f}" if scores else "Mean provisional teacher: n/a",
        "",
        "| rank | id | status | eligible | teacher | I | Q | Q source | E | WER | CER | prosody diagnostic |",
        "| ---: | --- | --- | ---: | ---: | ---: | ---: | --- | ---: | ---: | ---: | ---: |",
    ]
    for rank, row in enumerate(scored_rows, start=1):
        lines.append(
            f"| {rank} | {row.get('id', '')} | {row.get('metric_status', '')} | {row.get('ranking_eligible', '')} | "
            f"{row.get('main_metric_0_1', '')} | {row.get('intelligibility_component_0_1', '')} | "
            f"{row.get('quality_component_0_1', '')} | {row.get('quality_component_source', '')} | "
            f"{row.get('emotion_component_0_1', '')} | {row.get('wer', '')} | {row.get('cer', '')} | "
            f"{row.get('prosody_diagnostic_0_1', '')} |"
        )
    args.output_md.parent.mkdir(parents=True, exist_ok=True)
    args.output_md.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"wrote {args.output_csv}")
    print(f"wrote {args.output_md}")


if __name__ == "__main__":
    main()
